"""
CGLM Modal deployment.
  - fetch_corpus : download Simple English Wikipedia articles → Volume
  - train        : train SemanticCorticalGridLM with per-epoch metrics → Volume
  - WebApp       : ASGI FastAPI serving chat UI + inference endpoints
  - notebook     : JupyterLab analytics at port 8888
"""
from __future__ import annotations
import json
import pickle
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Version — bump this to create a new independent Modal deployment + volume dir
# ---------------------------------------------------------------------------

VERSION = "v1"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BEST_CONFIG = {
    "n_columns": 3,
    "dim": 2048,
    "fp_bits": 21,
    "index_bits": 7,
    "window": 2,
    "cells_per_col": 8,
    "epochs": 5,
    "max_sequences": 3000,
    "activation_threshold": 10,
    "new_synapses": 24,
    "periods": (7, 11, 13, 17, 19, 23),
}

# ---------------------------------------------------------------------------
# Modal plumbing
# ---------------------------------------------------------------------------

app = modal.App(f"cglm-chat-{VERSION}")
vol = modal.Volume.from_name("cglm-data", create_if_missing=True)
_VOL_MOUNT = "/data"                      # where the volume is mounted
VOL_PATH = Path(_VOL_MOUNT) / VERSION     # versioned subdirectory

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "fastapi", "uvicorn[standard]", "requests")
    .add_local_file("model_core.py", "/root/model_core.py")
)

nb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "jupyterlab", "matplotlib", "requests")
    .add_local_file("model_core.py", "/root/model_core.py")
)

# ---------------------------------------------------------------------------
# Corpus: Simple English Wikipedia via public API
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=300)
def fetch_corpus(n_articles: int = 2000, max_chars: int = 400_000) -> str:
    import re
    import requests

    VOL_PATH.mkdir(parents=True, exist_ok=True)
    out_path = VOL_PATH / "corpus.txt"
    api = "https://simple.wikipedia.org/w/api.php"
    all_text: list[str] = []
    total_chars = 0
    rvcontinue = None

    while len(all_text) < n_articles and total_chars < max_chars:
        params: dict = {
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "formatversion": 2,
            "format": "json",
        }
        if rvcontinue:
            params["rvcontinue"] = rvcontinue

        resp = requests.get(api, params=params, timeout=30)
        data = resp.json()

        for page in data.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            content = (
                page.get("revisions", [{}])[0]
                .get("slots", {})
                .get("main", {})
                .get("content", "")
            )
            content = re.sub(r"\{\{[^}]*\}\}", " ", content)
            content = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", content)
            content = re.sub(r"[=]{2,}[^=]*[=]{2,}", " ", content)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > 100:
                all_text.append(content)
                total_chars += len(content)

        rvcontinue = data.get("continue", {}).get("rvcontinue")
        if not rvcontinue:
            rvcontinue = None

        if total_chars >= max_chars or len(all_text) >= n_articles:
            break

    corpus = " ".join(all_text)[:max_chars]
    out_path.write_text(corpus, encoding="utf-8")
    vol.commit()
    return f"Fetched {len(all_text)} articles, {len(corpus):,} chars → corpus.txt"


# ---------------------------------------------------------------------------
# Training with per-epoch metrics
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={_VOL_MOUNT: vol},
    timeout=1800,
    cpu=4,
)
def train(test_fraction: float = 0.15) -> dict:
    import sys
    import random
    sys.path.insert(0, "/root")
    from model_core import SemanticCorticalGridLM, tokenize, accuracy

    VOL_PATH.mkdir(parents=True, exist_ok=True)
    corpus_path = VOL_PATH / "corpus.txt"
    if not corpus_path.exists():
        return {"error": "corpus.txt not found — run fetch_corpus first"}

    corpus = corpus_path.read_text(encoding="utf-8")
    sequences = tokenize(corpus)
    sequences = [s for s in sequences if len(s) >= 2]
    random.shuffle(sequences)
    sequences = sequences[:BEST_CONFIG["max_sequences"]]

    split = max(1, int(len(sequences) * test_fraction))
    test_seq = sequences[:split]
    train_seq = sequences[split:]

    print(f"Train={len(train_seq)} seqs, Test={len(test_seq)} seqs")

    cfg = {k: v for k, v in BEST_CONFIG.items() if k not in ("epochs", "max_sequences")}
    cfg["periods"] = tuple(cfg["periods"])
    model = SemanticCorticalGridLM(**cfg)

    epoch_metrics: list[dict] = []
    for epoch in range(1, BEST_CONFIG["epochs"] + 1):
        t0 = time.time()
        model.train_one_epoch(train_seq)
        elapsed = time.time() - t0

        t1, t3 = accuracy(model, test_seq, max_probes=200)
        tr1, tr3 = accuracy(model, train_seq[:100], max_probes=100)
        stats = model.stats()

        rec = {
            "epoch": epoch,
            "test_top1": t1,
            "test_top3": t3,
            "train_top1": tr1,
            "train_top3": tr3,
            "seconds": round(elapsed, 1),
            "vocab": stats["vocab"],
            "segments": stats["segments_per_unit"],
            "synapses": stats["synapses_per_unit"],
        }
        epoch_metrics.append(rec)
        print(f"  epoch {epoch}: test_top1={t1:.1f}% test_top3={t3:.1f}% ({elapsed:.0f}s)")

    for u in model.units:
        u.finalize()

    (VOL_PATH / "model.pkl").write_bytes(pickle.dumps(model))
    (VOL_PATH / "metrics.json").write_text(json.dumps(epoch_metrics, indent=2))

    _write_notebook(epoch_metrics)

    vol.commit()
    final = epoch_metrics[-1]
    return {
        "test_top1": final["test_top1"],
        "test_top3": final["test_top3"],
        "epochs": BEST_CONFIG["epochs"],
        "vocab": final["vocab"],
        "train_sequences": len(train_seq),
        "epoch_metrics": epoch_metrics,
    }


def _write_notebook(epoch_metrics: list[dict]):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# CGLM Analytics\n", "Training curves and model performance."],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json, sys\n",
                "sys.path.insert(0, '/root')\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                f"metrics = {json.dumps(epoch_metrics)}\n",
                "epochs = [m['epoch'] for m in metrics]\n",
                "top1   = [m['test_top1'] for m in metrics]\n",
                "top3   = [m['test_top3'] for m in metrics]\n",
                "tr1    = [m['train_top1'] for m in metrics]\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                "\n",
                "axes[0].plot(epochs, top1, 'o-', label='test top-1')\n",
                "axes[0].plot(epochs, top3, 's-', label='test top-3')\n",
                "axes[0].plot(epochs, tr1, '^--', label='train top-1')\n",
                "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('%')\n",
                "axes[0].set_title('Accuracy per Epoch'); axes[0].legend()\n",
                "\n",
                "syns = [m['synapses'] for m in metrics]\n",
                "segs = [m['segments'] for m in metrics]\n",
                "axes[1].plot(epochs, syns, 'o-', label='synapses/unit')\n",
                "axes[1].plot(epochs, segs, 's-', label='segments/unit')\n",
                "axes[1].set_xlabel('Epoch')\n",
                "axes[1].set_title('Model Growth'); axes[1].legend()\n",
                "\n",
                "plt.tight_layout(); plt.savefig('/data/training_curves.png', dpi=100)\n",
                "plt.show()\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pickle\n",
                "model = pickle.loads(open('/data/model.pkl','rb').read())\n",
                "\n",
                "prompts = ['the capital of', 'science is', 'history of the', 'water is']\n",
                "for p in prompts:\n",
                "    toks = p.split()\n",
                "    gen = model.generate(toks, n=8)\n",
                "    print(' '.join(gen))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "unit = model.units[0]\n",
                "vocab = list(unit.token_sdr.keys())\n",
                "print(f'Vocab size: {len(vocab)}')\n",
                "print('Sample tokens:', vocab[:20])\n",
            ],
        },
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    (VOL_PATH / "analytics.ipynb").write_text(json.dumps(nb, indent=2))


# ---------------------------------------------------------------------------
# Chat web app — ASGI FastAPI with embedded HTML UI
# ---------------------------------------------------------------------------

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CGLM Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 1rem 1.5rem; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.2rem; font-weight: 700; color: #38bdf8; }
  #status-badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; background: #334155; color: #94a3b8; }
  #status-badge.ready { background: #065f46; color: #6ee7b7; }
  main { flex: 1; display: flex; flex-direction: column; max-width: 800px; width: 100%; margin: 0 auto; padding: 1rem; gap: 1rem; overflow: hidden; }
  #messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; padding-right: 4px; }
  .msg { padding: 0.6rem 0.9rem; border-radius: 0.75rem; max-width: 85%; line-height: 1.5; font-size: 0.95rem; }
  .msg.user { align-self: flex-end; background: #1d4ed8; color: #eff6ff; }
  .msg.bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
  .msg.bot .candidates { margin-top: 0.4rem; font-size: 0.8rem; color: #64748b; }
  #controls { display: flex; gap: 0.5rem; }
  #prompt { flex: 1; padding: 0.6rem 0.9rem; background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; color: #e2e8f0; font-size: 0.95rem; outline: none; }
  #prompt:focus { border-color: #38bdf8; }
  button { padding: 0.6rem 1.1rem; border: none; border-radius: 0.75rem; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: opacity 0.15s; }
  button:hover { opacity: 0.85; }
  #send-btn { background: #0ea5e9; color: #fff; }
  #gen-btn  { background: #7c3aed; color: #fff; }
  #metrics-panel { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 0.75rem 1rem; font-size: 0.8rem; color: #94a3b8; }
  #metrics-panel span { color: #38bdf8; font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>CGLM Chat</h1>
  <div id="status-badge">loading&hellip;</div>
</header>
<main>
  <div id="messages"></div>
  <div id="metrics-panel">vocab: <span id="m-vocab">&ndash;</span> &nbsp;|&nbsp; segments/unit: <span id="m-segs">&ndash;</span> &nbsp;|&nbsp; test top-1: <span id="m-top1">&ndash;</span></div>
  <div id="controls">
    <input id="prompt" placeholder="Type a phrase&hellip;" autocomplete="off"/>
    <button id="send-btn">Predict</button>
    <button id="gen-btn">Generate</button>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const messages = $('messages');
const promptEl = $('prompt');

function addMsg(role, text, sub) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  if (sub) {
    const s = document.createElement('div');
    s.className = 'candidates';
    s.textContent = sub;
    el.appendChild(s);
  }
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

async function loadStatus() {
  try {
    const r = await fetch('/stats');
    const d = await r.json();
    if (d.model_loaded) {
      $('status-badge').textContent = 'model ready';
      $('status-badge').className = 'ready';
    } else {
      $('status-badge').textContent = d.status || 'no model';
    }
    $('m-vocab').textContent = d.vocab ?? '–';
    $('m-segs').textContent = d.segments_per_unit ?? '–';
  } catch {}
  try {
    const r = await fetch('/metrics');
    const d = await r.json();
    if (d.length) {
      const last = d[d.length - 1];
      $('m-top1').textContent = last.test_top1 + '%';
    }
  } catch {}
}

$('send-btn').onclick = async () => {
  const text = promptEl.value.trim();
  if (!text) return;
  addMsg('user', text);
  promptEl.value = '';
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tokens: text.toLowerCase().split(/\s+/), topn: 5}),
    });
    const d = await r.json();
    if (d.predictions && d.predictions.length) {
      const top = d.predictions[0][0];
      const rest = d.predictions.slice(1).map(p => p[0]).join(', ');
      addMsg('bot', '→ ' + top, rest ? 'also: ' + rest : '');
    } else {
      addMsg('bot', '(no prediction)');
    }
  } catch(e) { addMsg('bot', 'Error: ' + e.message); }
};

$('gen-btn').onclick = async () => {
  const text = promptEl.value.trim();
  if (!text) return;
  addMsg('user', '[generate] ' + text);
  promptEl.value = '';
  try {
    const r = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tokens: text.toLowerCase().split(/\s+/), n: 15}),
    });
    const d = await r.json();
    addMsg('bot', d.generated ? d.generated.join(' ') : '(no output)');
  } catch(e) { addMsg('bot', 'Error: ' + e.message); }
};

promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('send-btn').click(); }
});

loadStatus();
setInterval(loadStatus, 30000);
</script>
</body>
</html>
"""


@app.cls(
    image=image,
    volumes={_VOL_MOUNT: vol},
    min_containers=1,
)
class WebApp:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        self.model = None
        self.metrics: list = []
        model_path = VOL_PATH / "model.pkl"
        if model_path.exists():
            try:
                self.model = pickle.loads(model_path.read_bytes())
                print("Model loaded.")
            except Exception as e:
                print(f"Could not load model: {e}")
        metrics_path = VOL_PATH / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text())

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse

        api = FastAPI(title="CGLM Chat")

        @api.get("/", response_class=HTMLResponse)
        async def index():
            return CHAT_HTML

        @api.post("/predict")
        async def predict(request: Request):
            body = await request.json()
            tokens = body.get("tokens", [])
            topn = int(body.get("topn", 5))
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            preds = self.model.predict_next(tokens, topn=topn)
            return {"predictions": [[t, round(s, 4)] for t, s in preds]}

        @api.post("/generate")
        async def generate(request: Request):
            body = await request.json()
            tokens = body.get("tokens", [])
            n = int(body.get("n", 10))
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            gen = self.model.generate(tokens, n=n)
            return {"generated": gen}

        @api.get("/stats")
        async def stats():
            if not self.model:
                return {"model_loaded": False, "version": VERSION, "status": "no model — run train first"}
            s = self.model.stats()
            s["model_loaded"] = True
            s["version"] = VERSION
            return s

        @api.get("/metrics")
        async def metrics():
            return self.metrics or []

        @api.get("/health")
        async def health():
            return {"ok": True}

        return api


# ---------------------------------------------------------------------------
# JupyterLab analytics notebook
# ---------------------------------------------------------------------------

@app.function(
    image=nb_image,
    volumes={_VOL_MOUNT: vol},
    timeout=3600,
)
@modal.web_server(8888)
def notebook():
    subprocess.Popen([
        "jupyter", "lab",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=",
        "--NotebookApp.password=",
        "--notebook-dir=/data",
    ])
