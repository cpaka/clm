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
import time
from pathlib import Path

import modal
from pydantic import BaseModel


# Request body models — defined at MODULE level so FastAPI can resolve their
# (stringified, due to `from __future__ import annotations`) type hints via the
# module globals.  Defining them inside serve() makes them unresolvable, which
# silently demotes body params to query params.
class PredictReq(BaseModel):
    tokens: list[str] = []
    topn: int = 5


class GenerateReq(BaseModel):
    tokens: list[str] = []
    n: int = 10


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

# --- SMALL "smoke-test" preset: ~2 min training on Modal's 4-CPU container. ---
# Scale up (col_dim, n_units, max_sequences, epochs) once predictability is
# confirmed.  The active-column-scoped overlap makes col_dim nearly free for
# training cost, so widen col_dim first when scaling.
CORPUS_CONFIG = {
    "name": "plain-text-wikipedia-simpleenglish (ffatty/plain-text-wikipedia-simpleenglish)",
    "max_chars": 40_000,   # → 400_000 when scaling up
}

TRAIN_CONFIG = {
    "epochs": 2,           # → more when scaling up
    "max_sequences": 150,  # → 3000 when scaling up  (~2 min on 4-CPU)
    "test_fraction": 0.15,
    "replay_every": 2,     # hippocampal replay cadence (epochs)
}

# Memory note (per column): n_cells × max_segs × syn_per_seg × 4 bytes × 2 arrays
#   n_cells = col_dim × cells_per_col.  Keep max_segs modest on Modal.
MODEL_CONFIG = {
    "n_levels": 2,
    "strides": (1, 4),         # token level + phrase level
    "n_units": 3,              # voting columns per level
    "col_dim": 1024,           # SDR width / mini-column count
    "cells_per_col": 8,        # temporal context depth
    "fp_bits": 21,             # active bits per fingerprint
    "index_bits": 7,           # identity-core bits
    "window": 2,               # semantic co-occurrence window
    "kwta_k": 42,              # lateral inhibition: winners kept
    "replay_cap": 128,         # hippocampal buffer size
    "periods": (7, 11, 13, 17, 19, 23),
    "activation_threshold": 8,
    "syn_per_seg": 20,
    "max_segs": 12,
}

# ---------------------------------------------------------------------------
# Modal plumbing
# ---------------------------------------------------------------------------

app = modal.App(f"cglm-chat-{VERSION}")
vol = modal.Volume.from_name("cglm-data", create_if_missing=True)
# Shared live-training status — the train container writes progress here and the
# web container reads it (no Volume commit/reload churn for fast UI polling).
status_dict = modal.Dict.from_name("cglm-train-status", create_if_missing=True)
_VOL_MOUNT = "/data"                                 # volume mount point (constant)
VOL_PATH = Path(_VOL_MOUNT) / VERSION                # versioned subdirectory
REGISTRY_PATH = Path(_VOL_MOUNT) / "registry.json"   # shared across all versions
DATASET_PATH = Path(_VOL_MOUNT) / "datasets" / "wikisimple-all.txt"  # shared raw dataset
MODEL_DIR_NAME = "model.cglm"                         # persist/ store directory


def _set_status(**kw):
    """Merge-update the live training status for this VERSION."""
    cur = {}
    try:
        cur = dict(status_dict.get(VERSION) or {})
    except Exception:
        pass
    cur.update(kw)
    status_dict[VERSION] = cur

# The model now lives in the modular `core/` + `persist/` packages, added to
# the image as directories under /root so `import core...` works in-container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin FastAPI + Pydantic v2 together — unpinned installs produced a
    # version mismatch where request-body models were misread as query params.
    .pip_install("numpy", "fastapi==0.115.6", "pydantic>=2.5,<3",
                 "uvicorn[standard]==0.34.0", "requests", "kaggle")
    .add_local_dir("core", "/root/core")
    .add_local_dir("persist", "/root/persist")
    .add_local_dir("benchmarks", "/root/benchmarks")
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

@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=420, cpu=4)
def train(test_fraction: float = TRAIN_CONFIG["test_fraction"]) -> dict:
    import sys
    import random
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCGLM
    from benchmarks.datasets import _tokenize          # shared tokenizer
    from benchmarks.metrics import accuracy
    from persist.store import save_model

    E = TRAIN_CONFIG["epochs"]
    _set_status(phase="building", message="Building model…", epoch=0,
                total_epochs=E, progress=0.0, metrics=[])

    VOL_PATH.mkdir(parents=True, exist_ok=True)
    corpus_path = VOL_PATH / "corpus.txt"
    if not corpus_path.exists():
        _set_status(phase="error", message="corpus.txt not found — run fetch_corpus first")
        return {"error": "corpus.txt not found — run fetch_corpus first"}

    sequences = [s for s in _tokenize(corpus_path.read_text(encoding="utf-8")) if len(s) >= 2]
    random.shuffle(sequences)
    sequences = sequences[:TRAIN_CONFIG["max_sequences"]]

    split = max(1, int(len(sequences) * test_fraction))
    test_seq, train_seq = sequences[:split], sequences[split:]
    print(f"[train] Train={len(train_seq)} Test={len(test_seq)} Epochs={E}")

    cfg = {**MODEL_CONFIG,
           "strides": tuple(MODEL_CONFIG["strides"]),
           "periods": tuple(MODEL_CONFIG["periods"]),
           "encoder": "semantic"}
    model = HierarchicalCGLM(**cfg)

    _set_status(phase="folding", message="Folding semantic fingerprints…")

    epoch_metrics: list[dict] = []

    def cb(epoch, total, burst, segs):
        t_acc = time.time()
        t1, t3 = accuracy(model, test_seq, max_probes=80)
        tr1, _ = accuracy(model, train_seq[:60], max_probes=60)
        rec = {"epoch": epoch, "test_top1": t1, "test_top3": t3,
               "train_top1": tr1, "burst_rate": round(burst, 3),
               "segments": int(segs), "neuromod": model.stats()["neuromod"]}
        epoch_metrics.append(rec)
        _set_status(phase="learning", epoch=epoch, total_epochs=total,
                    progress=round(epoch / total, 3),
                    message=f"Epoch {epoch}/{total} — test top-1 {t1}%, burst {burst:.2f}",
                    metrics=list(epoch_metrics))
        print(f"[train] epoch {epoch}/{total}: test_top1={t1}% test_top3={t3}% "
              f"train_top1={tr1}% burst={burst:.2f} segs={segs} "
              f"(eval {time.time()-t_acc:.1f}s)")

    t0 = time.time()
    model.train(train_seq, epochs=E, replay_every=TRAIN_CONFIG["replay_every"],
                progress_cb=cb)
    train_seconds = round(time.time() - t0, 1)

    _set_status(phase="saving", message="Saving model…")
    print(f"[train] saving model → {VOL_PATH/MODEL_DIR_NAME}")
    save_model(model, str(VOL_PATH / MODEL_DIR_NAME))
    (VOL_PATH / "metrics.json").write_text(json.dumps(epoch_metrics, indent=2))
    vol.commit()
    _set_status(phase="done", progress=1.0, message="Model ready.",
                metrics=list(epoch_metrics))

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


# ---------------------------------------------------------------------------
# Bootstrap — called once after each deploy to fetch + train + record
# ---------------------------------------------------------------------------

@app.function(image=image, secrets=[kaggle_secret], volumes={_VOL_MOUNT: vol}, timeout=420)
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
<title>CGLM Console</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 0.9rem 1.5rem; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.15rem; font-weight: 700; color: #38bdf8; }
  #ver { font-size: 0.75rem; color: #475569; }
  nav { display: flex; gap: 0.25rem; margin-left: 1rem; }
  nav button { background: transparent; color: #94a3b8; border: none; padding: 0.4rem 0.9rem; border-radius: 0.6rem; cursor: pointer; font-size: 0.85rem; font-weight: 600; }
  nav button.active { background: #0ea5e9; color: #fff; }
  nav button:hover:not(.active) { background: #334155; color: #e2e8f0; }
  #status-badge { font-size: 0.75rem; padding: 2px 10px; border-radius: 9999px; background: #334155; color: #94a3b8; margin-left: auto; }
  #status-badge.ready { background: #065f46; color: #6ee7b7; }
  main { flex: 1; width: 100%; max-width: 980px; margin: 0 auto; padding: 1rem; }
  .view { display: none; }
  .view.active { display: block; }

  /* chat */
  #chat { display: flex; flex-direction: column; gap: 0.75rem; height: calc(100vh - 140px); }
  #messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; }
  .msg { padding: 0.6rem 0.9rem; border-radius: 0.75rem; max-width: 85%; line-height: 1.5; font-size: 0.95rem; }
  .msg.user { align-self: flex-end; background: #1d4ed8; color: #eff6ff; }
  .msg.bot { align-self: flex-start; background: #1e293b; border: 1px solid #334155; }
  .msg.bot .candidates { margin-top: 0.4rem; font-size: 0.8rem; color: #64748b; }
  #controls { display: flex; gap: 0.5rem; }
  input { padding: 0.6rem 0.9rem; background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; color: #e2e8f0; font-size: 0.95rem; outline: none; }
  input:focus { border-color: #38bdf8; }
  #prompt, #wordA, #wordB { flex: 1; }
  button.act { padding: 0.6rem 1.1rem; border: none; border-radius: 0.75rem; cursor: pointer; font-size: 0.9rem; font-weight: 600; }
  button.act:hover { opacity: 0.85; }
  #send-btn { background: #0ea5e9; color: #fff; }
  #gen-btn  { background: #7c3aed; color: #fff; }
  #fp-btn   { background: #0ea5e9; color: #fff; }

  /* cards + charts */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 0.8rem 1rem; }
  .card .k { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
  .card .v { font-size: 1.4rem; font-weight: 700; color: #38bdf8; margin-top: 2px; }
  .chartbox { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem; }
  .chartbox h3 { font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  th { color: #475569; font-weight: 600; text-align: left; padding: 4px 8px; }
  td { padding: 4px 8px; color: #94a3b8; border-top: 1px solid #1e293b; }
  tr.current td { color: #e2e8f0; }

  /* explore / fingerprint */
  .exrow { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  #fpgrid { display: block; background: #0b1220; border: 1px solid #334155; border-radius: 0.6rem; }
  .legend { display: flex; gap: 1.2rem; font-size: 0.8rem; color: #94a3b8; margin: 0.6rem 0; flex-wrap: wrap; }
  .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
  .neigh { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }
  .neigh span { background: #1e293b; border: 1px solid #334155; border-radius: 9999px; padding: 3px 10px; font-size: 0.8rem; cursor: pointer; }
  .neigh span:hover { border-color: #38bdf8; }
  .muted { color: #64748b; font-size: 0.8rem; }
</style>
</head>
<body>
<header>
  <h1>CGLM Console</h1>
  <span id="ver"></span>
  <nav>
    <button data-view="chat" class="active">Chat</button>
    <button data-view="dash">Dashboard</button>
    <button data-view="explore">Explore</button>
  </nav>
  <div id="status-badge">loading&hellip;</div>
</header>
<main>
  <!-- CHAT -->
  <section id="chat" class="view active">
    <div id="messages"></div>
    <div id="controls">
      <input id="prompt" placeholder="Type a phrase&hellip;" autocomplete="off"/>
      <button class="act" id="send-btn">Predict</button>
      <button class="act" id="gen-btn">Generate</button>
    </div>
  </section>

  <!-- DASHBOARD -->
  <section id="dash" class="view">
    <div class="chartbox" id="trainbox">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
        <button class="act" id="train-btn" style="background:#16a34a;color:#fff">Retrain (live)</button>
        <span class="muted" id="train-msg">idle</span>
      </div>
      <div style="background:#0b1220;border-radius:9999px;height:8px;margin-top:0.7rem;overflow:hidden">
        <div id="train-bar" style="height:100%;width:0%;background:#16a34a;transition:width .4s"></div>
      </div>
    </div>
    <div class="cards" id="cards"></div>
    <div class="chartbox"><h3>Accuracy per epoch</h3><canvas id="accChart" height="110"></canvas></div>
    <div class="chartbox"><h3>Surprise (burst rate) &amp; segment growth</h3><canvas id="growthChart" height="110"></canvas></div>
    <div class="chartbox"><h3>Deployment history</h3>
      <table><thead><tr><th>ver</th><th>corpus</th><th>epochs</th><th>seqs</th><th>top-1</th><th>top-3</th><th>deployed</th></tr></thead>
      <tbody id="reg-body"></tbody></table>
    </div>
  </section>

  <!-- EXPLORE -->
  <section id="explore" class="view">
    <p class="muted">Visualise a word's sparse fingerprint (SDR). Add a second word to see shared vs unique bits — that overlap is what drives semantic generalisation.</p>
    <div class="exrow">
      <input id="wordA" placeholder="word (e.g. water)" autocomplete="off"/>
      <input id="wordB" placeholder="compare with (optional)" autocomplete="off"/>
      <button class="act" id="fp-btn">Show</button>
    </div>
    <div class="legend">
      <span><i style="background:#38bdf8"></i><b id="lblA">A</b> only: <b id="cntA">0</b></span>
      <span><i style="background:#f472b6"></i><b id="lblB">B</b> only: <b id="cntB">0</b></span>
      <span><i style="background:#fbbf24"></i>shared: <b id="cntShared">0</b></span>
      <span class="muted" id="fpmeta"></span>
    </div>
    <canvas id="fpgrid" width="640" height="640"></canvas>
    <div class="neigh" id="neigh"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);

/* ---- tab switching ---- */
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  b.classList.add('active');
  $(b.dataset.view).classList.add('active');
  if (b.dataset.view === 'dash') { loadDashboard(); pollStatus(); }
});

/* ---- chat ---- */
const messages = $('messages'), promptEl = $('prompt');
function addMsg(role, text, sub) {
  const el = document.createElement('div'); el.className = 'msg ' + role; el.textContent = text;
  if (sub) { const s = document.createElement('div'); s.className = 'candidates'; s.textContent = sub; el.appendChild(s); }
  messages.appendChild(el); messages.scrollTop = messages.scrollHeight;
}
$('send-btn').onclick = async () => {
  const t = promptEl.value.trim(); if (!t) return; addMsg('user', t); promptEl.value = '';
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tokens: t.toLowerCase().split(/\s+/), topn: 5})});
    const d = await r.json();
    if (d.predictions?.length) { const top = d.predictions[0][0]; const rest = d.predictions.slice(1).map(p=>p[0]).join(', '); addMsg('bot','→ '+top, rest?'also: '+rest:''); }
    else addMsg('bot','(no prediction)');
  } catch(e) { addMsg('bot','Error: '+e.message); }
};
$('gen-btn').onclick = async () => {
  const t = promptEl.value.trim(); if (!t) return; addMsg('user','[generate] '+t); promptEl.value = '';
  try {
    const r = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tokens: t.toLowerCase().split(/\s+/), n: 15})});
    const d = await r.json(); addMsg('bot', d.generated ? d.generated.join(' ') : '(no output)');
  } catch(e) { addMsg('bot','Error: '+e.message); }
};
promptEl.addEventListener('keydown', e => { if (e.key==='Enter') { e.preventDefault(); $('send-btn').click(); } });

/* ---- header status (poll) ---- */
async function loadStatus() {
  try {
    const d = await (await fetch('/stats')).json();
    $('ver').textContent = d.version ? 'ver ' + d.version : '';
    if (d.model_loaded) { $('status-badge').textContent = 'model ready'; $('status-badge').className = 'ready'; }
    else { $('status-badge').textContent = d.status || 'no model'; }
  } catch {}
}

/* ---- dashboard ---- */
let accChart, growthChart;
function card(k, v) { return `<div class="card"><div class="k">${k}</div><div class="v">${v ?? '–'}</div></div>`; }
function renderCharts(m) {
  const ep = m.map(x => x.epoch);
  const grid = { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } };
  if (accChart) accChart.destroy();
  accChart = new Chart($('accChart'), { type:'line', data:{ labels: ep, datasets:[
      {label:'test top-1', data: m.map(x=>x.test_top1), borderColor:'#38bdf8', tension:.3},
      {label:'test top-3', data: m.map(x=>x.test_top3), borderColor:'#34d399', tension:.3},
      {label:'train top-1', data: m.map(x=>x.train_top1), borderColor:'#a78bfa', borderDash:[5,4], tension:.3}
    ]}, options:{ animation:false, plugins:{legend:{labels:{color:'#94a3b8'}}}, scales:{x:grid,y:grid} } });
  if (growthChart) growthChart.destroy();
  growthChart = new Chart($('growthChart'), { type:'line', data:{ labels: ep, datasets:[
      {label:'burst rate', data: m.map(x=>x.burst_rate), borderColor:'#fb7185', yAxisID:'y', tension:.3},
      {label:'segments', data: m.map(x=>x.segments), borderColor:'#fbbf24', yAxisID:'y1', tension:.3}
    ]}, options:{ animation:false, plugins:{legend:{labels:{color:'#94a3b8'}}},
      scales:{ x:grid, y:{...grid, position:'left'}, y1:{...grid, position:'right', grid:{drawOnChartArea:false}} } } });
}
async function loadDashboard() {
  let s = {};
  try { s = await (await fetch('/stats')).json(); } catch {}
  $('cards').innerHTML = [
    card('vocab', s.vocab), card('levels', s.levels), card('units/level', s.units_per_level),
    card('segments', (s.segments_l0||0).toLocaleString()), card('synapses', (s.synapses_l0||0).toLocaleString()),
    card('memory MB', s.memory_mb ? s.memory_mb.toFixed(1) : '–'), card('neuromod', s.neuromod),
    card('replay', s.replay_size)
  ].join('');
  let m = [];
  try { m = await (await fetch('/metrics')).json(); } catch {}
  renderCharts(m);
  try {
    const rows = await (await fetch('/registry')).json();
    const cur = $('ver').textContent.replace('ver ','');
    $('reg-body').innerHTML = rows.slice().reverse().map(r =>
      `<tr class="${r.version===cur?'current':''}"><td>${r.version}</td><td>${r.corpus_name? r.corpus_name.split(' ')[0] : '–'}</td><td>${r.epochs??'–'}</td><td>${r.max_sequences??r.train_sequences??'–'}</td><td>${r.test_top1??'–'}%</td><td>${r.test_top3??'–'}%</td><td>${(r.deployed_at||'').slice(0,10)}</td></tr>`
    ).join('');
  } catch {}
}

/* ---- live training (UI polls every 10s) ---- */
let trainTimer = null;
const ACTIVE = ['building','folding','learning','saving'];
function setBar(p, msg) { $('train-bar').style.width = Math.round((p||0)*100)+'%'; $('train-msg').textContent = msg || ''; }
async function pollStatus() {
  let st = {};
  try { st = await (await fetch('/status')).json(); } catch { return; }
  const active = ACTIVE.includes(st.phase);
  if (active) { setBar(st.progress, st.message || st.phase); $('train-btn').disabled = true; $('train-btn').style.opacity = .5; }
  if (st.metrics && st.metrics.length) renderCharts(st.metrics);
  if (active && !trainTimer) trainTimer = setInterval(pollStatus, 10000);  // poll every ~10s
  if (!active) {
    if (trainTimer) { clearInterval(trainTimer); trainTimer = null; }
    $('train-btn').disabled = false; $('train-btn').style.opacity = 1;
    if (st.phase === 'done') {
      setBar(1, 'Model ready — reloading…');
      try { await fetch('/reload', {method:'POST'}); } catch {}
      await loadStatus(); await loadDashboard(); setBar(0, 'idle');
    } else if (st.phase === 'error') {
      setBar(0, 'Error: ' + (st.message||''));
    }
  }
}
$('train-btn').onclick = async () => {
  const r = await (await fetch('/train', {method:'POST'})).json();
  if (!r.started) { setBar(0, r.reason || 'could not start'); }
  pollStatus();
};

/* ---- explore: fingerprint visualisation ---- */
async function fp(word) {
  const r = await fetch('/fingerprint?word=' + encodeURIComponent(word));
  if (!r.ok) return null;
  return await r.json();
}
function drawGrid(dim, setA, setB) {
  const cv = $('fpgrid'), ctx = cv.getContext('2d');
  const cols = Math.ceil(Math.sqrt(dim)), rows = Math.ceil(dim / cols);
  const cell = Math.floor(cv.width / cols), pad = 1;
  ctx.clearRect(0, 0, cv.width, cv.height);
  cv.height = rows * cell;
  for (let i = 0; i < dim; i++) {
    const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
    const inA = setA.has(i), inB = setB.has(i);
    let col = '#172033';
    if (inA && inB) col = '#fbbf24'; else if (inA) col = '#38bdf8'; else if (inB) col = '#f472b6';
    ctx.fillStyle = col;
    ctx.fillRect(x + pad, y + pad, cell - 2*pad, cell - 2*pad);
  }
}
async function showFingerprint() {
  const a = $('wordA').value.trim().toLowerCase(); if (!a) return;
  const b = $('wordB').value.trim().toLowerCase();
  const fa = await fp(a); if (!fa) { $('fpmeta').textContent = 'model not loaded'; return; }
  const fb = b ? await fp(b) : null;
  const setA = new Set(fa.bits), setB = new Set(fb ? fb.bits : []);
  let shared = 0; setA.forEach(x => { if (setB.has(x)) shared++; });
  $('lblA').textContent = a; $('lblB').textContent = b || 'B';
  $('cntA').textContent = setA.size - shared; $('cntB').textContent = Math.max(0, setB.size - shared);
  $('cntShared').textContent = shared;
  $('fpmeta').textContent = `dim=${fa.dim}, |${a}|=${setA.size}${fa.fitted?'':' (OOV)'}` + (fb ? `, |${b}|=${setB.size}${fb.fitted?'':' (OOV)'}` : '');
  drawGrid(fa.dim, setA, setB);
  // neighbours of A
  try {
    const nb = await (await fetch('/similar?word=' + encodeURIComponent(a) + '&k=10')).json();
    $('neigh').innerHTML = (nb.neighbours||[]).map(([t,o]) => `<span data-w="${t}">${t} · ${o}</span>`).join('');
    document.querySelectorAll('#neigh span').forEach(s => s.onclick = () => { $('wordB').value = s.dataset.w; showFingerprint(); });
  } catch {}
}
$('fp-btn').onclick = showFingerprint;
$('wordA').addEventListener('keydown', e => { if (e.key==='Enter') showFingerprint(); });
$('wordB').addEventListener('keydown', e => { if (e.key==='Enter') showFingerprint(); });

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
        self.model = None
        self.metrics: list = []
        self._load_model()

    def _load_model(self, refresh: bool = False):
        """(Re)load model + metrics from the Volume. `refresh` re-syncs the
        Volume first so a model trained in another container becomes visible."""
        from persist.store import load_model
        if refresh:
            vol.reload()
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
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse

        api = FastAPI(title="CGLM Chat")

        @api.get("/", response_class=HTMLResponse)
        async def index():
            return CHAT_HTML

        @api.post("/predict")
        async def predict(body: PredictReq):
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            preds = self.model.predict_next(body.tokens, topn=body.topn)
            return {"predictions": [[t, round(s, 4)] for t, s in preds]}

        @api.post("/generate")
        async def generate(body: GenerateReq):
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            return {"generated": self.model.generate(body.tokens, n=body.n)}

        @api.get("/similar")
        async def similar(word: str, k: int = 6):
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            return {"word": word,
                    "neighbours": [[t, o] for t, o in self.model.similar(word, k=k)]}

        @api.get("/fingerprint")
        async def fingerprint(word: str):
            if not self.model:
                return JSONResponse({"error": "Model not loaded"}, status_code=503)
            return self.model.fingerprint(word)

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

        # ---- live training -------------------------------------------------

        @api.post("/train")
        async def start_train():
            """Spawn a training run in a separate container (non-blocking).
            Refuses if one is already in progress."""
            cur = {}
            try:
                cur = dict(status_dict.get(VERSION) or {})
            except Exception:
                pass
            if cur.get("phase") in ("building", "folding", "learning", "saving"):
                return {"started": False, "reason": "training already in progress"}
            _set_status(phase="building", message="Starting…", epoch=0,
                        progress=0.0, metrics=[])
            train.spawn()
            return {"started": True}

        @api.get("/status")
        async def status():
            try:
                return status_dict.get(VERSION) or {"phase": "idle"}
            except Exception:
                return {"phase": "idle"}

        @api.post("/reload")
        async def reload():
            """Reload the freshly-trained model from the Volume into this
            web container (call when /status reports phase == done)."""
            self._load_model(refresh=True)
            if not self.model:
                return JSONResponse({"error": "no model on volume"}, status_code=503)
            s = self.model.stats()
            s["model_loaded"] = True
            s["version"] = VERSION
            return s

        return api
