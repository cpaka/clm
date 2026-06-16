"""
CGLM Modal deployment — Hierarchical Cortical-Grid Language Model.

This deploys the modular brain-architecture model in ``core/``:
  • hierarchical cortical levels with temporal strides
  • k-WTA lateral inhibition
  • neuromodulatory plasticity
  • hippocampal replay
  • sparse inter-area projections
  • vectorised (GPU-ready) columns
Persistence uses the compressed .npz store in ``persist/`` (model.cglm dir on
the Volume) — ~1000× smaller than raw pickle of the dense segment arrays.

Dataset lifecycle (one-time, shared across all versions):
  make upload-dataset  → downloads AllCombined.txt from Kaggle → /data/datasets/

Per-version workflow:
  make deploy          → modal deploy + bootstrap (fetch_corpus + train + registry)
  make redeploy        → same but forces retrain even if model exists

Versioning:
  Bump VERSION to create a new independent deployment + volume subdir + registry entry.
"""
from __future__ import annotations
import datetime
import json
import subprocess
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Version — bump to create a new independent deployment + isolated volume dir
# ---------------------------------------------------------------------------

VERSION = "v2"

# ---------------------------------------------------------------------------
# Per-version configs
# Corpus: start small so bootstrap finishes quickly; scale up later.
# Model arch: the hierarchical brain architecture. col_dim is the single source
# of truth for SDR width (encoder dim is forced to match).
# ---------------------------------------------------------------------------

CORPUS_CONFIG = {
    "name": "plain-text-wikipedia-simpleenglish (ffatty/plain-text-wikipedia-simpleenglish)",
    "max_chars": 60_000,   # → 400_000 once inference confirmed
}

TRAIN_CONFIG = {
    "epochs": 3,           # → more once inference confirmed
    "max_sequences": 600,  # → 3000 once inference confirmed
    "test_fraction": 0.15,
    "replay_every": 2,     # hippocampal replay cadence (epochs)
}

# Memory note (per column): n_cells × max_segs × syn_per_seg × 4 bytes × 2 arrays
#   n_cells = col_dim × cells_per_col.  Keep max_segs modest on Modal.
MODEL_CONFIG = {
    "n_levels": 2,
    "strides": (1, 4),         # token level + phrase level
    "n_units": 3,              # voting columns per level
    "col_dim": 2048,           # SDR width / mini-column count
    "cells_per_col": 8,        # temporal context depth
    "fp_bits": 21,             # active bits per fingerprint
    "index_bits": 7,           # identity-core bits
    "window": 2,               # semantic co-occurrence window
    "kwta_k": 42,              # lateral inhibition: winners kept
    "replay_cap": 512,         # hippocampal buffer size
    "periods": (7, 11, 13, 17, 19, 23),
    "activation_threshold": 10,
    "syn_per_seg": 24,
    "max_segs": 16,
}

# ---------------------------------------------------------------------------
# Modal plumbing
# ---------------------------------------------------------------------------

app = modal.App(f"cglm-chat-{VERSION}")
vol = modal.Volume.from_name("cglm-data", create_if_missing=True)
_VOL_MOUNT = "/data"                                 # volume mount point (constant)
VOL_PATH = Path(_VOL_MOUNT) / VERSION                # versioned subdirectory
REGISTRY_PATH = Path(_VOL_MOUNT) / "registry.json"   # shared across all versions
DATASET_PATH = Path(_VOL_MOUNT) / "datasets" / "wikisimple-all.txt"  # shared raw dataset
MODEL_DIR_NAME = "model.cglm"                         # persist/ store directory

# The model now lives in the modular `core/` + `persist/` packages, added to
# the image as directories under /root so `import core...` works in-container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "fastapi", "uvicorn[standard]", "requests", "kaggle")
    .add_local_dir("core", "/root/core")
    .add_local_dir("persist", "/root/persist")
    .add_local_dir("benchmarks", "/root/benchmarks")
)

nb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "jupyterlab", "matplotlib", "requests")
    .add_local_dir("core", "/root/core")
    .add_local_dir("persist", "/root/persist")
)

kaggle_secret = modal.Secret.from_name("kaggle-credentials")


# ---------------------------------------------------------------------------
# Registry helpers (run inside Modal containers)
# ---------------------------------------------------------------------------

def _read_registry() -> list[dict]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            return []
    return []


def _write_registry(entries: list[dict]):
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2))


def _upsert_registry(entry: dict):
    entries = _read_registry()
    entries = [e for e in entries if e.get("version") != entry["version"]]
    entries.append(entry)
    _write_registry(entries)


# ---------------------------------------------------------------------------
# Dataset — downloaded ONCE, shared across all model versions
#
# Source : https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish
# File   : AllCombined.txt  (249,396 articles, ~178 MB plain text)
# Storage: /data/datasets/wikisimple-all.txt  (lives outside any version dir)
#
# Run once:  make upload-dataset
# Bootstrap skips this step automatically if the file already exists.
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "ffatty/plain-text-wikipedia-simpleenglish"
KAGGLE_FILE = "AllCombined.txt"


def _kaggle_auth_header(os_env) -> dict:
    """Return Authorization header from available Kaggle env vars."""
    import base64
    api_token = os_env.get("KAGGLE_API_TOKEN") or os_env.get("KAGGLE_KEY") or ""
    username = os_env.get("KAGGLE_USERNAME", "")
    if api_token.startswith("KGAT"):
        return {"Authorization": f"Bearer {api_token}"}
    if username and api_token:
        creds = base64.b64encode(f"{username}:{api_token}".encode()).decode()
        return {"Authorization": f"Basic {creds}"}
    if api_token:
        return {"Authorization": f"Bearer {api_token}"}
    raise RuntimeError(
        f"No Kaggle credentials in environment. "
        f"Vars seen: {[k for k in os_env if 'KAGGLE' in k.upper()]}"
    )


@app.function(image=image, secrets=[kaggle_secret], volumes={_VOL_MOUNT: vol}, timeout=600)
def upload_dataset(force: bool = False) -> dict:
    """
    Download AllCombined.txt from Kaggle once and store in the shared dataset dir.
    Subsequent model versions reuse this file — no re-download needed.
    """
    import io
    import os
    import requests
    import zipfile

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists() and not force:
        size_mb = DATASET_PATH.stat().st_size / 1_048_576
        print(f"Dataset already present: {DATASET_PATH} ({size_mb:.1f} MB) — skipping.")
        return {"status": "already_exists", "path": str(DATASET_PATH), "size_mb": round(size_mb, 1)}

    auth = _kaggle_auth_header(dict(os.environ))
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
    print(f"Downloading {url} ...")

    resp = requests.get(url, headers=auth, stream=True, allow_redirects=True, timeout=300)
    if resp.status_code == 403:
        raise RuntimeError(
            f"403 Forbidden. Accept the dataset license at "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET} then retry."
        )
    resp.raise_for_status()

    downloaded_bytes = 0
    chunks = []
    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        chunks.append(chunk)
        downloaded_bytes += len(chunk)
        if downloaded_bytes % (10 << 20) < (1 << 20):
            print(f"  {downloaded_bytes // (1 << 20)} MB downloaded...")
    print(f"Download complete: {downloaded_bytes // (1 << 20)} MB")

    zip_bytes = io.BytesIO(b"".join(chunks))
    with zipfile.ZipFile(zip_bytes) as zf:
        names = zf.namelist()
        print(f"Zip contents: {names}")
        target = next((n for n in names if n.endswith(KAGGLE_FILE)), None)
        if target is None:
            raise RuntimeError(f"{KAGGLE_FILE} not found in zip. Contents: {names}")
        print(f"Extracting {target} → {DATASET_PATH} ...")
        DATASET_PATH.write_bytes(zf.read(target))

    size_mb = DATASET_PATH.stat().st_size / 1_048_576
    vol.commit()
    print(f"Dataset saved: {DATASET_PATH} ({size_mb:.1f} MB)")
    return {"status": "downloaded", "path": str(DATASET_PATH), "size_mb": round(size_mb, 1)}


# ---------------------------------------------------------------------------
# Corpus sampling — reads from shared dataset, no credentials required.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=120)
def fetch_corpus(max_chars: int = CORPUS_CONFIG["max_chars"]) -> dict:
    """
    Sample a random slice from the shared raw dataset and write to the
    versioned corpus path. No Kaggle credentials needed after upload_dataset.
    """
    import random

    if not DATASET_PATH.exists():
        raise RuntimeError(
            f"Raw dataset not found at {DATASET_PATH}. Run 'make upload-dataset' first."
        )

    file_size = DATASET_PATH.stat().st_size
    rng = random.Random(VERSION)
    read_window = min(max_chars * 6, file_size)  # read 6× to have enough after filtering
    max_start = max(0, file_size - read_window)
    start = rng.randint(0, max_start)

    with open(DATASET_PATH, "rb") as fh:
        fh.seek(start)
        raw = fh.read(read_window).decode("utf-8", errors="ignore")

    articles = [a.strip() for a in raw.split("\n\n") if len(a.strip()) > 80]
    rng.shuffle(articles)

    all_text: list[str] = []
    total_chars = 0
    for article in articles:
        if total_chars >= max_chars:
            break
        all_text.append(article)
        total_chars += len(article)

    VOL_PATH.mkdir(parents=True, exist_ok=True)
    corpus = "\n\n".join(all_text)[:max_chars]
    (VOL_PATH / "corpus.txt").write_text(corpus, encoding="utf-8")
    vol.commit()

    print(f"Corpus: {len(all_text)} articles, {len(corpus):,} chars → {VOL_PATH}/corpus.txt")
    return {"articles": len(all_text), "chars": len(corpus)}


# ---------------------------------------------------------------------------
# Training with per-epoch metrics
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=600, cpu=4)
def train(test_fraction: float = TRAIN_CONFIG["test_fraction"]) -> dict:
    import sys
    import random
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCGLM
    from benchmarks.datasets import _tokenize          # shared tokenizer
    from benchmarks.metrics import accuracy
    from persist.store import save_model

    VOL_PATH.mkdir(parents=True, exist_ok=True)
    corpus_path = VOL_PATH / "corpus.txt"
    if not corpus_path.exists():
        return {"error": "corpus.txt not found — run fetch_corpus first"}

    sequences = [s for s in _tokenize(corpus_path.read_text(encoding="utf-8")) if len(s) >= 2]
    random.shuffle(sequences)
    sequences = sequences[:TRAIN_CONFIG["max_sequences"]]

    split = max(1, int(len(sequences) * test_fraction))
    test_seq, train_seq = sequences[:split], sequences[split:]
    E = TRAIN_CONFIG["epochs"]
    print(f"[train] Train={len(train_seq)} Test={len(test_seq)} Epochs={E}")

    cfg = {**MODEL_CONFIG,
           "strides": tuple(MODEL_CONFIG["strides"]),
           "periods": tuple(MODEL_CONFIG["periods"]),
           "encoder": "semantic"}
    model = HierarchicalCGLM(**cfg)

    epoch_metrics: list[dict] = []

    def cb(epoch, total, burst, segs):
        t_acc = time.time()
        t1, t3 = accuracy(model, test_seq, max_probes=200)
        tr1, _ = accuracy(model, train_seq[:100], max_probes=100)
        rec = {"epoch": epoch, "test_top1": t1, "test_top3": t3,
               "train_top1": tr1, "burst_rate": round(burst, 3),
               "segments": int(segs), "neuromod": model.stats()["neuromod"]}
        epoch_metrics.append(rec)
        print(f"[train] epoch {epoch}/{total}: test_top1={t1}% test_top3={t3}% "
              f"train_top1={tr1}% burst={burst:.2f} segs={segs} "
              f"(eval {time.time()-t_acc:.1f}s)")

    t0 = time.time()
    model.train(train_seq, epochs=E, replay_every=TRAIN_CONFIG["replay_every"],
                progress_cb=cb)
    train_seconds = round(time.time() - t0, 1)

    print(f"[train] saving model → {VOL_PATH/MODEL_DIR_NAME}")
    save_model(model, str(VOL_PATH / MODEL_DIR_NAME))
    (VOL_PATH / "metrics.json").write_text(json.dumps(epoch_metrics, indent=2))
    _write_notebook(epoch_metrics)
    vol.commit()

    stats = model.stats()
    final = epoch_metrics[-1]
    return {
        "test_top1": final["test_top1"],
        "test_top3": final["test_top3"],
        "vocab": stats["vocab"],
        "train_sequences": len(train_seq),
        "total_seconds": train_seconds,
        "epoch_metrics": epoch_metrics,
    }


def _write_notebook(epoch_metrics: list[dict]):
    vol_str = str(VOL_PATH)
    cells = [
        {"cell_type": "markdown", "metadata": {},
         "source": [f"# CGLM Analytics — {VERSION}\n", "Hierarchical brain-architecture model."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "import json, sys\n",
            "sys.path.insert(0, '/root')\n",
            "import matplotlib.pyplot as plt\n\n",
            f"metrics = {json.dumps(epoch_metrics)}\n",
            "epochs = [m['epoch'] for m in metrics]\n",
            "top1 = [m['test_top1'] for m in metrics]\n",
            "top3 = [m['test_top3'] for m in metrics]\n",
            "tr1  = [m['train_top1'] for m in metrics]\n\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "axes[0].plot(epochs, top1, 'o-', label='test top-1')\n",
            "axes[0].plot(epochs, top3, 's-', label='test top-3')\n",
            "axes[0].plot(epochs, tr1, '^--', label='train top-1')\n",
            f"axes[0].set_title('Accuracy — {VERSION}'); axes[0].legend()\n",
            "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('%')\n",
            "burst = [m['burst_rate'] for m in metrics]\n",
            "segs  = [m['segments'] for m in metrics]\n",
            "axes[1].plot(epochs, burst, 'o-', label='burst rate')\n",
            "ax2 = axes[1].twinx(); ax2.plot(epochs, segs, 's-', color='tab:red', label='segments')\n",
            f"axes[1].set_title('Surprise & Growth — {VERSION}'); axes[1].set_xlabel('Epoch')\n",
            "plt.tight_layout()\n",
            f"plt.savefig('{vol_str}/training_curves.png', dpi=100)\n",
            "plt.show()\n",
         ]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
            "import sys; sys.path.insert(0, '/root')\n",
            "from persist.store import load_model\n",
            f"model = load_model('{vol_str}/{MODEL_DIR_NAME}')\n",
            "for p in ['the capital of', 'science is', 'history of the', 'water is']:\n",
            "    print(' '.join(model.generate(p.split(), n=8)))\n",
         ]},
    ]
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11.0"}},
          "cells": cells}
    (VOL_PATH / "analytics.ipynb").write_text(json.dumps(nb, indent=2))


# ---------------------------------------------------------------------------
# Bootstrap — called once after each deploy to fetch + train + record
# ---------------------------------------------------------------------------

@app.function(image=image, secrets=[kaggle_secret], volumes={_VOL_MOUNT: vol}, timeout=900)
def bootstrap(force: bool = False):
    """Idempotent post-deploy pipeline: dataset check → corpus sample → train → registry."""
    import time as _time
    t_boot = _time.time()

    VOL_PATH.mkdir(parents=True, exist_ok=True)

    model_marker = VOL_PATH / MODEL_DIR_NAME / "config.json"
    if model_marker.exists() and not force:
        print(f"[{VERSION}] Model already exists — skipping. Use 'make redeploy' to force retrain.")
        return

    if not DATASET_PATH.exists():
        print(f"[{VERSION}] Step 0/3: raw dataset not found — uploading from Kaggle...")
        t0 = _time.time()
        upload_dataset.local()
        print(f"[{VERSION}] Step 0/3: dataset upload done ({_time.time()-t0:.1f}s)")
    else:
        size_mb = DATASET_PATH.stat().st_size / 1_048_576
        print(f"[{VERSION}] Step 0/3: raw dataset present ({size_mb:.1f} MB) ✓  [{_time.time()-t_boot:.1f}s]")

    print(f"[{VERSION}] Step 1/3: sampling corpus...  [{_time.time()-t_boot:.1f}s]")
    t1 = _time.time()
    corpus_info = fetch_corpus.local(max_chars=CORPUS_CONFIG["max_chars"])
    print(f"[{VERSION}] Step 1/3: corpus ready — {corpus_info['articles']} articles, "
          f"{corpus_info['chars']:,} chars  ({_time.time()-t1:.1f}s)")

    print(f"[{VERSION}] Step 2/3: training...  [{_time.time()-t_boot:.1f}s]")
    t2 = _time.time()
    result = train.local(test_fraction=TRAIN_CONFIG["test_fraction"])
    print(f"[{VERSION}] Step 2/3: training done ({_time.time()-t2:.1f}s)")

    if "error" in result:
        print(f"[{VERSION}] Training failed: {result['error']}")
        return

    entry = {
        "version": VERSION,
        "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "corpus_name": CORPUS_CONFIG["name"],
        "corpus_articles": corpus_info.get("articles"),
        "corpus_chars": corpus_info.get("chars"),
        "epochs": TRAIN_CONFIG["epochs"],
        "max_sequences": TRAIN_CONFIG["max_sequences"],
        "test_top1": result["test_top1"],
        "test_top3": result["test_top3"],
        "vocab": result["vocab"],
        "train_sequences": result["train_sequences"],
        "train_seconds": result["total_seconds"],
    }
    _upsert_registry(entry)
    vol.commit()

    print(f"[{VERSION}] ✓ bootstrap done in {_time.time()-t_boot:.1f}s — "
          f"test_top1={entry['test_top1']}%  test_top3={entry['test_top3']}%  "
          f"vocab={entry['vocab']}  train_seqs={entry['train_sequences']}")


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
  #ver { font-size: 0.75rem; color: #475569; }
  #status-badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; background: #334155; color: #94a3b8; margin-left: auto; }
  #status-badge.ready { background: #065f46; color: #6ee7b7; }
  main { flex: 1; display: flex; flex-direction: column; max-width: 800px; width: 100%; margin: 0 auto; padding: 1rem; gap: 0.75rem; overflow: hidden; }
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
  #metrics-panel { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 0.6rem 1rem; font-size: 0.8rem; color: #94a3b8; display: flex; gap: 1.5rem; flex-wrap: wrap; }
  #metrics-panel span { color: #38bdf8; font-weight: 600; }
  #history { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 0.6rem 1rem; font-size: 0.75rem; color: #64748b; max-height: 80px; overflow-y: auto; }
  #history table { width: 100%; border-collapse: collapse; }
  #history th { color: #475569; font-weight: 600; text-align: left; padding: 1px 8px; }
  #history td { padding: 1px 8px; }
  #history tr.current td { color: #e2e8f0; }
</style>
</head>
<body>
<header>
  <h1>CGLM Chat</h1>
  <span id="ver"></span>
  <div id="status-badge">loading&hellip;</div>
</header>
<main>
  <div id="messages"></div>
  <div id="metrics-panel">
    vocab: <span id="m-vocab">&ndash;</span>
    &nbsp;levels: <span id="m-levels">&ndash;</span>
    &nbsp;segments: <span id="m-segs">&ndash;</span>
    &nbsp;test top-1: <span id="m-top1">&ndash;</span>
    &nbsp;test top-3: <span id="m-top3">&ndash;</span>
  </div>
  <div id="history"><table><thead><tr><th>ver</th><th>corpus</th><th>epochs</th><th>seqs</th><th>top-1</th><th>top-3</th><th>deployed</th></tr></thead><tbody id="reg-body"></tbody></table></div>
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
  if (sub) { const s = document.createElement('div'); s.className = 'candidates'; s.textContent = sub; el.appendChild(s); }
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

async function loadStatus() {
  try {
    const r = await fetch('/stats'); const d = await r.json();
    $('ver').textContent = d.version ? 'ver ' + d.version : '';
    if (d.model_loaded) {
      $('status-badge').textContent = 'model ready'; $('status-badge').className = 'ready';
    } else { $('status-badge').textContent = d.status || 'no model'; }
    $('m-vocab').textContent = d.vocab ?? '–';
    $('m-levels').textContent = d.levels ?? '–';
    $('m-segs').textContent = d.segments_l0 ?? '–';
  } catch {}
  try {
    const r = await fetch('/metrics'); const d = await r.json();
    if (d.length) { const l = d[d.length-1]; $('m-top1').textContent = l.test_top1+'%'; $('m-top3').textContent = l.test_top3+'%'; }
  } catch {}
  try {
    const r = await fetch('/registry'); const rows = await r.json();
    const tbody = $('reg-body'); tbody.innerHTML = '';
    rows.slice().reverse().forEach(row => {
      const tr = document.createElement('tr');
      if (row.version === $('ver').textContent.replace('ver ','')) tr.className = 'current';
      tr.innerHTML = `<td>${row.version}</td><td>${row.corpus_name||'–'}</td><td>${row.epochs||'–'}</td><td>${row.max_sequences||'–'}</td><td>${row.test_top1??'–'}%</td><td>${row.test_top3??'–'}%</td><td>${(row.deployed_at||'').slice(0,10)}</td>`;
      tbody.appendChild(tr);
    });
  } catch {}
}

$('send-btn').onclick = async () => {
  const text = promptEl.value.trim(); if (!text) return;
  addMsg('user', text); promptEl.value = '';
  try {
    const r = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({tokens: text.toLowerCase().split(/\s+/), topn: 5}) });
    const d = await r.json();
    if (d.predictions?.length) { const top = d.predictions[0][0]; const rest = d.predictions.slice(1).map(p=>p[0]).join(', '); addMsg('bot','→ '+top, rest?'also: '+rest:''); }
    else addMsg('bot','(no prediction)');
  } catch(e) { addMsg('bot','Error: '+e.message); }
};

$('gen-btn').onclick = async () => {
  const text = promptEl.value.trim(); if (!text) return;
  addMsg('user','[generate] '+text); promptEl.value = '';
  try {
    const r = await fetch('/generate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({tokens: text.toLowerCase().split(/\s+/), n: 15}) });
    const d = await r.json();
    addMsg('bot', d.generated ? d.generated.join(' ') : '(no output)');
  } catch(e) { addMsg('bot','Error: '+e.message); }
};

promptEl.addEventListener('keydown', e => { if (e.key==='Enter'&&!e.shiftKey) { e.preventDefault(); $('send-btn').click(); } });
loadStatus();
setInterval(loadStatus, 30000);
</script>
</body>
</html>
"""


@app.cls(image=image, volumes={_VOL_MOUNT: vol}, min_containers=1)
class WebApp:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        from persist.store import load_model

        self.model = None
        self.metrics: list = []
        model_dir = VOL_PATH / MODEL_DIR_NAME
        if (model_dir / "config.json").exists():
            try:
                self.model = load_model(str(model_dir))
                print(f"[{VERSION}] Model loaded.")
            except Exception as e:
                print(f"[{VERSION}] Could not load model: {e}")
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
            return {"generated": self.model.generate(tokens, n=n)}

        @api.get("/similar")
        async def similar(word: str, k: int = 6):
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            return {"word": word,
                    "neighbours": [[t, o] for t, o in self.model.similar(word, k=k)]}

        @api.get("/stats")
        async def stats():
            if not self.model:
                return {"model_loaded": False, "version": VERSION,
                        "status": "no model — bootstrap in progress"}
            s = self.model.stats()
            s["model_loaded"] = True
            s["version"] = VERSION
            return s

        @api.get("/metrics")
        async def metrics():
            return self.metrics or []

        @api.get("/registry")
        async def registry():
            try:
                return _read_registry()
            except Exception:
                return []

        @api.get("/health")
        async def health():
            return {"ok": True, "version": VERSION}

        return api


# ---------------------------------------------------------------------------
# JupyterLab analytics notebook
# ---------------------------------------------------------------------------

@app.function(image=nb_image, volumes={_VOL_MOUNT: vol}, timeout=3600)
@modal.web_server(8888)
def notebook():
    subprocess.Popen([
        "jupyter", "lab",
        "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root",
        "--NotebookApp.token=", "--NotebookApp.password=",
        "--notebook-dir=/data",
    ])
