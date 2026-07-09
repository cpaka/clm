"""
CLM Modal deployment — Hierarchical Cortical Language Model.

This deploys the modular brain-architecture model in ``core/``:
  • hierarchical cortical levels with temporal strides
  • k-WTA lateral inhibition
  • neuromodulatory plasticity
  • hippocampal replay
  • sparse inter-area projections
  • vectorised (GPU-ready) columns
Persistence uses the compressed .npz store in ``persist/`` (model.clm dir on
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

VERSION      = "v5.1"
PREV_VERSION = "v5.0"   # weights to inherit until v5.1 is retrained

# ---------------------------------------------------------------------------
# Per-version configs
#
# Corpus scaling plan (Wikipedia, incremental — compare accuracy at each step):
#   V4.0  →  1M chars,  5K seqs  (GPU baseline, same data as v3)
#   V4.1  →  2M chars, 10K seqs  (bump CORPUS_CONFIG + TRAIN_CONFIG, redeploy)
#   V4.2  →  5M chars, 20K seqs  (same)
#
# GPU note: train_unit / train_shard run on T4 GPUs via cupy.  The web-serving
# container and assembly steps (train_parallel / train_shards) run on CPU;
# arrays are saved as numpy on disk via cupy's __array__ protocol.
# ---------------------------------------------------------------------------

CORPUS_CONFIG = {
    "name": "plain-text-wikipedia-simpleenglish (ffatty/plain-text-wikipedia-simpleenglish)",
    "max_chars": 8_000_000,  # V4.0 baseline: ~1M chars → ~5K sentences
    "vocab_size": 2000,      # cap vocabulary; rare words → <UNK> (bigram coverage ~5%)
}

TRAIN_CONFIG = {
    "epochs": 3,            # 3 passes — compression gives ~4× speedup so 3 is sufficient
    "max_sequences": 40_000,  # ~8500 train / ~1500 test
    "test_fraction": 0.15,
    "replay_every": 2,      # hippocampal replay cadence (epochs)
}

INGEST_CONFIG = {
    # Per-night cap on new sequences folded in by the ingest cron.  Bounds the
    # run so it always fits the function timeout; the remaining tail is picked
    # up on subsequent nights (progress tracked via metrics.train_sequences).
    "max_new_sequences": 2_000,
}

# Memory note (per column): n_cells × max_segs × syn_per_seg × 4 bytes × 2 arrays
#   n_cells = col_dim × cells_per_col.  Keep max_segs modest on Modal.
MODEL_CONFIG = {
    "n_levels": 2,
    "strides": (1, 4),         # token level + phrase level
    "n_units": 3,              # voting columns per level
    "col_dim": 2048,           # v5.1: 1024→2048 — 2× representation capacity
    "cells_per_col": 4,        # compressed: 8→4 (2× faster overlap computation)
    "fp_bits": 21,             # active bits per fingerprint
    "index_bits": 7,           # identity-core bits
    "window": 4,               # wider co-occurrence → richer semantic fingerprints
    "kwta_k": 42,              # lateral inhibition: winners kept
    "replay_cap": 256,         # hippocampal buffer size
    "periods": (7, 11, 13, 17, 19, 23),
    "activation_threshold": 5, # compressed: 8→5 (keeps 40% ratio: 5/12 ≈ 8/20)
    "syn_per_seg": 12,         # compressed: 20→12 (1.7× faster inner loop)
    "max_segs": 8,             # compressed: 16→8 (2× faster segment scan)
    "use_spatial_pooler": True,  # learn SDR representations (fit once, frozen, cached)
    # KEEP: A/B on real corpus — punishment ~doubles held-out top-1 (7.3% vs 4.0%)
    # for ~2.4× train time. Do not drop for speed; it prevents predictive bloat.
    "pred_dec": 0.01,
}

# ---------------------------------------------------------------------------
# Modal plumbing
# ---------------------------------------------------------------------------

app = modal.App(f"clm-chat-{VERSION}")
vol = modal.Volume.from_name("cglm-data", create_if_missing=True)
# Shared live-training status — the train container writes progress here and the
# web container reads it (no Volume commit/reload churn for fast UI polling).
status_dict = modal.Dict.from_name("clm-train-status", create_if_missing=True)
_VOL_MOUNT = "/data"                                 # volume mount point (constant)
VOL_PATH = Path(_VOL_MOUNT) / VERSION                # versioned subdirectory
REGISTRY_PATH = Path(_VOL_MOUNT) / "registry.json"   # shared across all versions
DATASET_PATH = Path(_VOL_MOUNT) / "datasets" / "wikisimple-all.txt"  # shared raw dataset
MODEL_DIR_NAME = "model.clm"                         # persist/ store directory


def _set_status(**kw):
    """Merge-update the live training status for this VERSION."""
    cur = {}
    try:
        cur = dict(status_dict.get(VERSION) or {})
    except Exception:
        pass
    cur.update(kw)
    status_dict[VERSION] = cur

# cupy kernel cache — shared across all versions and all training containers.
# cupy JIT-compiles CUDA kernels on first use and caches them under CUPY_CACHE_DIR.
# By pointing that dir at the data Volume, the compiled kernels survive across
# container restarts.  First cold-start: ~15 min compilation.  Warm: seconds.
_CUPY_CACHE_PATH = "/data/cupy-cache"

# The model now lives in the modular `core/` + `persist/` packages, added to
# the image as directories under /root so `import core...` works in-container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin FastAPI + Pydantic v2 together — unpinned installs produced a
    # version mismatch where request-body models were misread as query params.
    # cupy-cuda12x[ctk]: GPU backend + CUDA headers for JIT kernel compilation.
    # Falls back to numpy transparently on CPU-only containers (web serving,
    # assembly steps) because core/xp.py catches the ImportError.
    .pip_install("numpy", "cupy-cuda12x[ctk]",
                 "fastapi==0.115.6", "pydantic>=2.5,<3",
                 "uvicorn[standard]==0.34.0", "requests", "kaggle")
    .env({"CUPY_CACHE_DIR": _CUPY_CACHE_PATH})   # persist compiled kernels
    .add_local_dir("core", "/root/core")
    .add_local_dir("persist", "/root/persist")
    .add_local_dir("benchmarks", "/root/benchmarks")
    .add_local_file("docs_html.py", "/root/docs_html.py")
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
    from core.hierarchy import HierarchicalCLM
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
    model = HierarchicalCLM(**cfg)

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
# Parallel training — NEXT_STEPS #1: fan out the independent voting units
# across containers, then assemble them into one ensemble.
# ---------------------------------------------------------------------------

def _load_corpus_seqs(cap: bool = True) -> list[list[str]]:
    """Load, shuffle (fixed seed), vocab-filter, and cap the corpus sequences.

    Rare words beyond the top CORPUS_CONFIG['vocab_size'] are replaced with
    '<UNK>'.  This keeps bigram coverage at ~5 % for better generalisation
    even on a 1M-char Wikipedia sample.

    ``cap=False`` returns the full filtered corpus (same shuffle order, same
    vocab) — used by ingest to reach sequences beyond the training window."""
    import random
    from collections import Counter
    from benchmarks.datasets import _tokenize
    seqs = [s for s in _tokenize((VOL_PATH / "corpus.txt").read_text(encoding="utf-8"))
            if len(s) >= 2]
    random.Random(0).shuffle(seqs)

    # Vocabulary filter
    vocab_size = CORPUS_CONFIG.get("vocab_size", 0)
    if vocab_size:
        counter = Counter(w for seq in seqs for w in seq)
        top_vocab = {w for w, _ in counter.most_common(vocab_size)}
        seqs = [[w if w in top_vocab else "<UNK>" for w in seq] for seq in seqs]

    return seqs[:TRAIN_CONFIG["max_sequences"]] if cap else seqs


def _load_split(test_fraction: float):
    """Deterministic train/test split shared by every parallel unit."""
    seqs = _load_corpus_seqs()
    split = max(1, int(len(seqs) * test_fraction))
    return seqs[split:], seqs[:split]              # train, test


def _unit_cfg(seed_base: int) -> dict:
    """Single-unit config = MODEL_CONFIG with one flat level, seeded as unit i."""
    base = {k: v for k, v in MODEL_CONFIG.items()
            if k not in ("n_levels", "strides", "n_units")}
    base["periods"] = tuple(MODEL_CONFIG["periods"])
    return dict(n_levels=1, strides=(1,), n_units=1, seed_base=seed_base,
                encoder="semantic", **base)


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=1800, gpu="t4")
def warmup_cupy() -> dict:
    """One-time cupy CUDA kernel compilation.

    Run once with `make warmup` before the first `make train-parallel`.
    Compiled kernels are stored at CUPY_CACHE_DIR (/data/cupy-cache) on the
    persistent Volume, so train_unit / train_shard can start in <5 s thereafter.
    This can take ~15 minutes on a cold T4 container.
    """
    import sys, time as _t
    sys.path.insert(0, "/root")
    t0 = _t.time()
    try:
        import cupy as cp
        print("[warmup] cupy found — running kernel compilation warm-up…")
        n = 8192 * 16 * 20
        # Exercise the gather+reduce pattern used in column.step()
        seg_idx = cp.random.randint(0, 1024, (n,), dtype=cp.int32)
        src = cp.zeros(1024, dtype=cp.bool_)
        src[:21] = True
        _ = src[seg_idx].sum()          # triggers JIT: fancy-index + reduce
        perm = cp.random.rand(n).astype(cp.float32)
        _ = (src[seg_idx] & (perm >= 0.5)).sum()   # connected overlap pattern
        cp.cuda.Stream.null.synchronize()
        elapsed = round(_t.time() - t0, 1)
        print(f"[warmup] kernels compiled in {elapsed}s — committing cache to volume")
        vol.commit()
        return {"status": "ok", "seconds": elapsed, "cache_dir": "/data/cupy-cache"}
    except ImportError:
        return {"status": "cupy_not_available", "seconds": 0}


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=21600, cpu=4)
def train_unit(unit_id: int) -> dict:
    """Train ONE voting unit standalone and save it to unit_<id>.clm.

    CPU training: HTM temporal memory is sequential (one token at a time),
    so GPU kernel-launch overhead exceeds compute benefit for our array sizes.
    4 CPU cores keeps memory bandwidth high for the numpy gather/reduce hot path.
    """
    import sys, time as _t
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCLM
    from persist.store import save_model

    if not (VOL_PATH / "corpus.txt").exists():
        return {"error": "corpus.txt not found"}

    t_load = _t.time()
    train_seq, _ = _load_split(TRAIN_CONFIG["test_fraction"])
    print(f"[unit {unit_id}] corpus loaded in {_t.time()-t_load:.1f}s  ({len(train_seq)} seqs)")

    t_enc = _t.time()
    m = HierarchicalCLM(**_unit_cfg(unit_id))
    m.fit_encoders(train_seq)
    print(f"[unit {unit_id}] encoder fit in {_t.time()-t_enc:.1f}s  vocab={m.stats()['vocab']}")

    t_train = _t.time()
    m.train(train_seq, epochs=TRAIN_CONFIG["epochs"],
            replay_every=TRAIN_CONFIG["replay_every"])
    print(f"[unit {unit_id}] training done in {_t.time()-t_train:.1f}s")

    t_save = _t.time()
    save_model(m, str(VOL_PATH / f"unit_{unit_id}.clm"))
    vol.commit()
    print(f"[unit {unit_id}] saved in {_t.time()-t_save:.1f}s")

    total = round(_t.time() - t_load, 1)
    print(f"[unit {unit_id}] total {total}s")
    return {"unit": unit_id, "seconds": total, "vocab": m.stats()["vocab"]}


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=28800)
def train_parallel() -> dict:
    """Fan out unit training across containers, then assemble + save the ensemble."""
    import sys, time as _t
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCLM
    from persist.store import load_model, save_model
    from benchmarks.metrics import accuracy

    N = MODEL_CONFIG["n_units"]
    _set_status(phase="learning", message=f"Training {N} units in parallel…",
                epoch=0, total_epochs=1, progress=0.1, metrics=[])
    if not (VOL_PATH / "corpus.txt").exists():
        _set_status(phase="error", message="corpus.txt not found — run fetch_corpus")
        return {"error": "no corpus"}

    t0 = _t.time()
    results = list(train_unit.map(range(N)))
    print(f"[parallel] units done: {results}")

    _set_status(phase="saving", message="Assembling ensemble…", progress=0.8)
    vol.reload()
    shell = HierarchicalCLM(n_units=N, seed_base=0, **{
        k: v for k, v in _unit_cfg(0).items() if k not in ("n_units", "seed_base")})
    units = [load_model(str(VOL_PATH / f"unit_{i}.clm")).levels[0].units[0]
             for i in range(N)]
    shell.inject_units(units)

    train_seq, test_seq = _load_split(TRAIN_CONFIG["test_fraction"])
    t1, t3 = accuracy(shell, test_seq, max_probes=200)
    tr1, _ = accuracy(shell, train_seq[:200], max_probes=200)
    stats = shell.stats()
    metrics = [{"epoch": 1, "test_top1": t1, "test_top3": t3, "train_top1": tr1,
                "burst_rate": 0.0, "segments": stats["segments_l0"],
                "neuromod": stats["neuromod"], "saturated": stats["saturated"]}]
    shell.metrics = metrics

    save_model(shell, str(VOL_PATH / MODEL_DIR_NAME))
    (VOL_PATH / "metrics.json").write_text(json.dumps(metrics, indent=2))
    total = round(_t.time() - t0, 1)
    _upsert_registry({
        "version": VERSION,
        "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "corpus_name": CORPUS_CONFIG["name"] + " (parallel)",
        "epochs": TRAIN_CONFIG["epochs"], "max_sequences": TRAIN_CONFIG["max_sequences"],
        "test_top1": t1, "test_top3": t3, "vocab": stats["vocab"],
        "train_sequences": len(train_seq), "train_seconds": total,
    })
    vol.commit()
    _set_status(phase="done", progress=1.0, metrics=metrics,
                message=f"Ensemble ready (parallel) — test top-1 {t1}%")
    print(f"[parallel] assembled {N} units in {total}s — test_top1={t1}% top3={t3}%")
    return {"units": N, "test_top1": t1, "test_top3": t3, "total_seconds": total}


# ---------------------------------------------------------------------------
# Incremental training — corpus split into batches, model queryable after each
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=3600, cpu=4)
def train_unit_batch(unit_id: int, batch_idx: int, n_batches: int) -> dict:
    """Train one unit on one corpus batch.

    Batch 0: fresh model, encoders fit on the full corpus (consistent vocab).
    Batch 1+: warm-start from the previous checkpoint and continue training.
    """
    import sys, time as _t
    sys.path.insert(0, "/root")
    from pathlib import Path as P
    from core.hierarchy import HierarchicalCLM
    from persist.store import load_model, save_model

    vol.reload()
    all_train, _ = _load_split(TRAIN_CONFIG["test_fraction"])

    # Deterministic slice for this batch
    n = len(all_train)
    batch_size = n // n_batches
    start = batch_idx * batch_size
    end = start + batch_size if batch_idx < n_batches - 1 else n
    batch_seqs = all_train[start:end]
    print(f"[unit {unit_id} b{batch_idx}] slice {start}:{end}  ({len(batch_seqs)} seqs)")

    unit_path = str(VOL_PATH / f"unit_{unit_id}.clm")

    t_enc = _t.time()
    if batch_idx == 0 or not (P(unit_path) / "config.json").exists():
        m = HierarchicalCLM(**_unit_cfg(unit_id))
        m.fit_encoders(all_train)   # fit on full corpus → stable vocab across batches
        print(f"[unit {unit_id} b{batch_idx}] encoder fit in {_t.time()-t_enc:.1f}s")
    else:
        m = load_model(unit_path)   # warm-start: column state carries over
        print(f"[unit {unit_id} b{batch_idx}] warm-started in {_t.time()-t_enc:.1f}s  vocab={m.stats()['vocab']}")

    t0 = _t.time()
    m.train(batch_seqs, epochs=TRAIN_CONFIG["epochs"],
            replay_every=TRAIN_CONFIG["replay_every"])
    elapsed = round(_t.time() - t0, 1)
    print(f"[unit {unit_id} b{batch_idx}] trained in {elapsed}s")

    save_model(m, unit_path)
    vol.commit()
    return {"unit": unit_id, "batch": batch_idx, "seconds": elapsed,
            "vocab": m.stats()["vocab"]}


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=3600 * 4)
def train_incremental(n_batches: int = 3) -> list:
    """Split corpus into n_batches and train sequentially.

    After each batch all units are assembled into an ensemble and saved.
    The webapp picks up the updated model on the next /predict or /generate
    request (no redeploy needed) — so the model improves in place.

    Usage:  modal run modal_app.py::train_incremental
            modal run modal_app.py::train_incremental --n-batches 3
    """
    import sys, time as _t
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCLM
    from persist.store import load_model, save_model
    from benchmarks.metrics import accuracy

    N = MODEL_CONFIG["n_units"]

    if not (VOL_PATH / "corpus.txt").exists():
        _set_status(phase="error", message="corpus.txt not found — run fetch_corpus first")
        return [{"error": "no corpus"}]

    train_seq, test_seq = _load_split(TRAIN_CONFIG["test_fraction"])
    results_all = []

    for batch_idx in range(n_batches):
        t0 = _t.time()
        seqs_per_batch = len(train_seq) // n_batches
        msg = f"batch {batch_idx+1}/{n_batches}  (~{seqs_per_batch} seqs/unit)"
        print(f"[incremental] {msg}")
        _set_status(phase="learning", message=f"Training {msg}…",
                    progress=round(batch_idx / n_batches, 2))

        # Fan out: all units train this batch in parallel
        list(train_unit_batch.map(
            range(N),
            kwargs={"batch_idx": batch_idx, "n_batches": n_batches},
        ))

        # Assemble ensemble immediately — model becomes queryable now
        vol.reload()
        shell = HierarchicalCLM(n_units=N, seed_base=0, **{
            k: v for k, v in _unit_cfg(0).items() if k not in ("n_units", "seed_base")
        })
        units = [load_model(str(VOL_PATH / f"unit_{i}.clm")).levels[0].units[0]
                 for i in range(N)]
        shell.inject_units(units)

        t1, t3 = accuracy(shell, test_seq, max_probes=200)
        tr1, _ = accuracy(shell, train_seq[:200], max_probes=200)
        stats = shell.stats()
        elapsed = round(_t.time() - t0, 1)

        metrics = [{"epoch": batch_idx + 1, "test_top1": t1, "test_top3": t3,
                    "train_top1": tr1, "burst_rate": 0.0,
                    "segments": stats["segments_l0"], "neuromod": stats["neuromod"],
                    "saturated": stats["saturated"]}]
        shell.metrics = metrics
        save_model(shell, str(VOL_PATH / MODEL_DIR_NAME))
        (VOL_PATH / "metrics.json").write_text(json.dumps(metrics, indent=2))
        vol.commit()

        phase = "done" if batch_idx == n_batches - 1 else "ready"
        _set_status(phase=phase, progress=round((batch_idx + 1) / n_batches, 2),
                    metrics=metrics,
                    message=f"Batch {batch_idx+1}/{n_batches} — top-1 {t1}%")
        result = {"batch": batch_idx + 1, "n_batches": n_batches,
                  "seconds": elapsed, "test_top1": t1, "test_top3": t3}
        results_all.append(result)
        print(f"[incremental] batch {batch_idx+1}/{n_batches} done in {elapsed}s "
              f"— top1={t1}% top3={t3}% — model queryable")

    _upsert_registry({
        "version": VERSION,
        "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "corpus_name": CORPUS_CONFIG["name"] + " (incremental)",
        "epochs": TRAIN_CONFIG["epochs"],
        "max_sequences": TRAIN_CONFIG["max_sequences"],
        "test_top1": results_all[-1]["test_top1"],
        "test_top3": results_all[-1]["test_top3"],
        "vocab": stats["vocab"],
        "train_sequences": len(train_seq),
        "train_seconds": sum(r["seconds"] for r in results_all),
    })
    vol.commit()
    return results_all


# ---------------------------------------------------------------------------
# #2 NEXT_STEPS: Corpus-shard parallelism + deterministic merge
# ---------------------------------------------------------------------------

def _load_shard(shard_id: int, n_shards: int):
    """Load the vocab-filtered corpus shard assigned to `shard_id`."""
    seqs = _load_corpus_seqs()
    size = max(1, len(seqs) // n_shards)
    start = shard_id * size
    return seqs[start: start + size]


def _merge_columns(base_seg_idx, base_seg_perm, base_n_segs,
                   new_seg_idx, new_seg_perm, new_n_segs, max_segs):
    """Append unique segments from new onto base, up to max_segs per cell.

    Alignment is valid because both models use the same encoder + col_dim,
    so column indices index the same token SDRs in both models."""
    import numpy as np
    n_cells = base_n_segs.shape[0]
    for c in range(n_cells):
        space = max_segs - int(base_n_segs[c])
        if space <= 0:
            continue
        n_new = int(new_n_segs[c])
        if n_new == 0:
            continue
        take = min(space, n_new)
        dst = int(base_n_segs[c])
        base_seg_idx[c, dst: dst + take] = new_seg_idx[c, :take]
        base_seg_perm[c, dst: dst + take] = new_seg_perm[c, :take]
        base_n_segs[c] += take
    return base_seg_idx, base_seg_perm, base_n_segs


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=900, cpu=4)
def train_shard(args: tuple) -> dict:
    """Train a single CLMUnit on one corpus shard, save to shard_<id>.clm."""
    import sys, time as _t
    sys.path.insert(0, "/root")
    shard_id, n_shards = args
    from core.hierarchy import HierarchicalCLM
    from persist.store import save_model

    if not (VOL_PATH / "corpus.txt").exists():
        return {"error": "corpus.txt not found"}
    train_seq = _load_shard(shard_id, n_shards)
    t0 = _t.time()
    m = HierarchicalCLM(**_unit_cfg(shard_id))
    m.fit_encoders(train_seq)
    m.train(train_seq, epochs=TRAIN_CONFIG["epochs"],
            replay_every=TRAIN_CONFIG["replay_every"])
    save_model(m, str(VOL_PATH / f"shard_{shard_id}.clm"))
    vol.commit()   # also persists the cupy kernel cache to the volume
    secs = round(_t.time() - t0, 1)
    print(f"[shard {shard_id}/{n_shards}] trained in {secs}s  vocab={m.stats()['vocab']}")
    return {"shard": shard_id, "seconds": secs, "vocab": m.stats()["vocab"],
            "sequences": len(train_seq)}


@app.function(image=image, volumes={_VOL_MOUNT: vol}, timeout=1200)
def train_shards(n_shards: int = 4) -> dict:
    """Fan out training across N corpus shards, merge segments, save ensemble.

    Each shard trains one CLMUnit on a disjoint slice of the corpus; the merge
    unions their dendritic segments so the ensemble covers the full corpus.
    Total data processed = n_shards × shard_size (vs. one unit seeing all data)."""
    import sys, time as _t
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCLM
    from persist.store import load_model, save_model
    from benchmarks.metrics import accuracy
    import numpy as np

    _set_status(phase="learning", message=f"Shard training: {n_shards} shards…",
                epoch=0, total_epochs=1, progress=0.05, metrics=[])
    if not (VOL_PATH / "corpus.txt").exists():
        _set_status(phase="error", message="corpus.txt not found")
        return {"error": "no corpus"}

    t0 = _t.time()
    args = [(i, n_shards) for i in range(n_shards)]
    results = list(train_shard.map(args))
    print(f"[shards] done: {results}")

    _set_status(phase="saving", message="Merging shard segments…", progress=0.8)
    vol.reload()

    # Load base from shard 0
    base = load_model(str(VOL_PATH / "shard_0.clm"))
    base_unit = base.levels[0].units[0]
    max_segs = base_unit.col.max_segs

    # Merge segments from shards 1..N-1 into shard 0
    for i in range(1, n_shards):
        shard = load_model(str(VOL_PATH / f"shard_{i}.clm"))
        su = shard.levels[0].units[0]
        base_unit.col.seg_idx, base_unit.col.seg_perm, base_unit.col.n_segs = \
            _merge_columns(
                base_unit.col.seg_idx, base_unit.col.seg_perm, base_unit.col.n_segs,
                su.col.seg_idx, su.col.seg_perm, su.col.n_segs, max_segs,
            )

    # Wrap the merged unit in an N-unit ensemble shell so existing serving code works
    N = MODEL_CONFIG["n_units"]
    shell = HierarchicalCLM(n_units=N, seed_base=0, **{
        k: v for k, v in _unit_cfg(0).items() if k not in ("n_units", "seed_base")})
    # Replicate the merged unit N times (each a shallow copy of the best segments)
    import copy
    shell.inject_units([copy.deepcopy(base_unit) for _ in range(N)])

    train_seq, test_seq = _load_split(TRAIN_CONFIG["test_fraction"])
    t1, t3 = accuracy(shell, test_seq, max_probes=200)
    tr1, _ = accuracy(shell, train_seq[:200], max_probes=200)
    stats = shell.stats()
    metrics = [{"epoch": 1, "test_top1": t1, "test_top3": t3, "train_top1": tr1,
                "burst_rate": 0.0, "segments": stats["segments_l0"],
                "neuromod": stats["neuromod"], "saturated": stats["saturated"]}]
    shell.metrics = metrics

    save_model(shell, str(VOL_PATH / MODEL_DIR_NAME))
    (VOL_PATH / "metrics.json").write_text(json.dumps(metrics, indent=2))
    total = round(_t.time() - t0, 1)
    _upsert_registry({
        "version": VERSION,
        "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "corpus_name": CORPUS_CONFIG["name"] + f" ({n_shards} shards)",
        "epochs": TRAIN_CONFIG["epochs"], "max_sequences": TRAIN_CONFIG["max_sequences"],
        "test_top1": t1, "test_top3": t3, "vocab": stats["vocab"],
        "train_sequences": sum(r.get("sequences", 0) for r in results),
        "train_seconds": total,
    })
    vol.commit()
    _set_status(phase="done", progress=1.0, metrics=metrics,
                message=f"Shard ensemble ready — test top-1 {t1}%")
    print(f"[shards] merged {n_shards} shards in {total}s — test_top1={t1}% top3={t3}%")
    return {"n_shards": n_shards, "test_top1": t1, "test_top3": t3, "total_seconds": total}


# ---------------------------------------------------------------------------
# #6 NEXT_STEPS: Scheduled incremental ingest (cron job)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={_VOL_MOUNT: vol},
    timeout=3600,
    schedule=modal.Cron("0 3 * * *"),   # 03:00 UTC daily
)
def ingest() -> dict:
    """Daily incremental training: load model → train on corpus delta → save.

    This implements continual learning: each run folds new corpus content into
    the existing model without retraining from scratch (#6 NEXT_STEPS).
    The corpus holds more sequences than the initial training window
    (TRAIN_CONFIG['max_sequences']); each night up to
    INGEST_CONFIG['max_new_sequences'] unseen sequences beyond that window are
    folded in, using the SAME shuffle order and vocab filter as training so
    the seen-counter, the train/test split, and the encoder vocabulary all
    stay consistent.  Like the web app, it inherits PREV_VERSION weights when
    this version has no checkpoint yet, and always saves to this version's dir."""
    import sys, time as _t
    sys.path.insert(0, "/root")
    from core.hierarchy import HierarchicalCLM
    from persist.store import load_model, save_model
    from benchmarks.metrics import accuracy

    vol.reload()
    if not (VOL_PATH / "corpus.txt").exists():
        print("[ingest] no corpus — skipping")
        return {"skipped": True, "reason": "no corpus"}

    # Resolve weights: this version's checkpoint, else inherit PREV_VERSION's
    model_path = VOL_PATH / MODEL_DIR_NAME
    weights_version = VERSION
    if not (model_path / "config.json").exists() and PREV_VERSION:
        prev_path = Path(_VOL_MOUNT) / PREV_VERSION / MODEL_DIR_NAME
        if (prev_path / "config.json").exists():
            model_path = prev_path
            weights_version = PREV_VERSION
            print(f"[ingest] no {VERSION} checkpoint — inheriting {PREV_VERSION} weights")
    if not (model_path / "config.json").exists():
        print("[ingest] no model found — skipping")
        return {"skipped": True, "reason": "no model"}

    t0 = _t.time()
    model = load_model(str(model_path))

    # How far into the corpus has training progressed?  Ingest entries record
    # train_sequences in metrics; models trained via train_parallel/shards do
    # not, so fall back to the registry entry for the loaded weights.  Floor at
    # the initial training window — everything below it is train or test data.
    seen = max((m.get("train_sequences", 0) for m in model.metrics), default=0)
    if not seen:
        reg = {e.get("version"): e for e in _read_registry()}
        seen = reg.get(weights_version, {}).get("train_sequences", 0)
    seen = max(seen, TRAIN_CONFIG["max_sequences"])

    # Same shuffle + vocab filter as training, but uncapped: the tail beyond
    # the training window is genuinely unseen data (tokenizing raw text here
    # once exploded the vocab by ~68K words and blew the function timeout).
    all_seqs = _load_corpus_seqs(cap=False)

    new_seqs = all_seqs[seen:]
    if not new_seqs:
        print(f"[ingest] no new sequences (seen={seen}, total={len(all_seqs)})")
        return {"skipped": True, "reason": "no new data", "seen": seen}
    cap = INGEST_CONFIG["max_new_sequences"]
    if len(new_seqs) > cap:
        print(f"[ingest] {len(new_seqs)} unseen sequences — capping tonight's run at {cap}")
        new_seqs = new_seqs[:cap]

    # Incremental encoder update (#7) for any new vocab
    for unit in model.levels[0].units:
        from core.encoder import SemanticEncoder
        if isinstance(unit.enc, SemanticEncoder):
            added = unit.enc.update(new_seqs)
            if added:
                print(f"[ingest] +{added} new vocab words")

    model.train(new_seqs, epochs=TRAIN_CONFIG["epochs"],
                replay_every=TRAIN_CONFIG["replay_every"])

    stats = model.stats()
    _, test_seq = _load_split(TRAIN_CONFIG["test_fraction"])
    t1, t3 = accuracy(model, test_seq, max_probes=200)

    # Auto-expand capacity if saturated (#9 + #10)
    if model.is_saturated():
        print("[ingest] model saturated — appending a new unit")
        from core.hierarchy import CLMUnit
        new_unit = CLMUnit(
            col_dim=model.col_dim, w=model.fp_bits,
            cells_per_col=model._col_kwargs["cells_per_col"],
            periods=model._col_kwargs["periods"],
            encoder=model._encoders[0] if model._encoders else None,
            seed=model.seed_base + model.n_units * 100,
            **{k: v for k, v in model._col_kwargs.items()
               if k not in ("cells_per_col", "periods")},
        )
        if new_unit.enc is not None:
            new_unit.enc.update(new_seqs)
        model.append_unit(new_unit)   # unit learns on subsequent ingest cycles

    entry = {
        "epoch": len(model.metrics) + 1,
        "test_top1": t1, "test_top3": t3,
        "train_top1": 0.0, "burst_rate": 0.0,
        "segments": stats["segments_l0"],
        "neuromod": stats["neuromod"],
        "saturated": stats["saturated"],
        "train_sequences": seen + len(new_seqs),
    }
    model.metrics.append(entry)
    # Always save to THIS version's dir (weights may have been inherited from
    # PREV_VERSION) so the web app picks up the version's own checkpoint.
    VOL_PATH.mkdir(parents=True, exist_ok=True)
    save_model(model, str(VOL_PATH / MODEL_DIR_NAME))
    (VOL_PATH / "metrics.json").write_text(json.dumps(model.metrics, indent=2))
    _upsert_registry({
        "version": VERSION,
        "deployed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "corpus_name": CORPUS_CONFIG["name"] + " (incremental)",
        "epochs": TRAIN_CONFIG["epochs"],
        "max_sequences": seen + len(new_seqs),
        "test_top1": t1, "test_top3": t3,
        "vocab": stats["vocab"],
        "train_sequences": seen + len(new_seqs),
        "train_seconds": round(_t.time() - t0, 1),
    })
    vol.commit()
    print(f"[ingest] +{len(new_seqs)} seqs in {round(_t.time()-t0,1)}s — top1={t1}%")
    return {"new_sequences": len(new_seqs), "test_top1": t1, "test_top3": t3}


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
<title>CLM Console</title>
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
  .msg.bot .qa-intent { display:inline-block; background:#1e40af; color:#bfdbfe; border-radius:9999px; padding:1px 9px; font-size:0.73rem; font-weight:700; letter-spacing:.04em; margin-bottom:0.35rem; }
  .msg.bot .qa-answer { font-size:0.95rem; color:#e2e8f0; margin:0.2rem 0 0.3rem; }
  .msg.bot .qa-meta { font-size:0.75rem; color:#475569; margin-top:0.3rem; }
  .msg.bot .qa-related { display:flex; flex-wrap:wrap; gap:0.3rem; margin-top:0.4rem; }
  .msg.bot .qa-related span { background:#0f172a; border:1px solid #334155; border-radius:9999px; padding:2px 9px; font-size:0.78rem; color:#94a3b8; cursor:pointer; }
  .msg.bot .qa-related span:hover { border-color:#38bdf8; color:#e2e8f0; }
  #controls { display: flex; gap: 0.5rem; }
  input { padding: 0.6rem 0.9rem; background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; color: #e2e8f0; font-size: 0.95rem; outline: none; }
  input:focus { border-color: #38bdf8; }
  #prompt, #wordA, #wordB { flex: 1; }
  button.act { padding: 0.6rem 1.1rem; border: none; border-radius: 0.75rem; cursor: pointer; font-size: 0.9rem; font-weight: 600; }
  button.act:hover { opacity: 0.85; }
  #send-btn    { background: #0ea5e9; color: #fff; }
  #predict-btn { background: #7c3aed; color: #fff; }
  #fp-btn      { background: #0ea5e9; color: #fff; }

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

  /* generate */
  #gen-seed { flex: 1; }
  #gen-output { background:#0b1220; border:1px solid #334155; border-radius:0.75rem; padding:1rem; min-height:80px; line-height:1.8; font-size:0.95rem; margin-top:1rem; white-space:pre-wrap; word-break:break-word; }
  .gen-prompt { color:#94a3b8; }
  .gen-word { color:#38bdf8; display:inline-block; padding:1px 4px; border-radius:4px; cursor:default; }
  .gen-word:hover { background:#1e293b; }
  .gen-controls { display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.75rem; }
  .gen-controls label { font-size:0.82rem; color:#64748b; }
  .gen-controls input[type=range] { accent-color:#38bdf8; }
  #gen-btn { background:#0ea5e9; color:#fff; }
  #gen-clear { background:#334155; color:#94a3b8; font-size:0.85rem; }

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
  <h1>CLM Console</h1>
  <span id="ver"></span>
  <nav>
    <button data-view="chat" class="active">Chat</button>
    <button data-view="generate">Generate</button>
    <button data-view="dash">Dashboard</button>
    <button data-view="explore">Explore</button>
    <button data-view="reason">Reason</button>
    <a href="/docs" target="_blank" style="color:#94a3b8;text-decoration:none;padding:0.4rem 0.9rem;border-radius:0.6rem;font-size:0.85rem;font-weight:600;" onmouseover="this.style.background='#334155'" onmouseout="this.style.background=''">Docs ↗</a>
  </nav>
  <div id="status-badge">loading&hellip;</div>
</header>
<main>
  <!-- CHAT -->
  <section id="chat" class="view active">
    <div id="messages"></div>
    <div id="controls">
      <input id="prompt" placeholder="Ask a question or type a phrase&hellip;" autocomplete="off"/>
      <button class="act" id="send-btn">Send</button>
      <button class="act" id="predict-btn">Predict</button>
    </div>
  </section>

  <!-- GENERATE -->
  <section id="generate" class="view">
    <p class="muted">Type a seed phrase — the model continues it token by token. Generated words are shown in blue. Hover a word to see its position.</p>
    <div class="gen-controls">
      <input id="gen-seed" placeholder="Seed phrase (e.g. &quot;a castle is&quot;)" autocomplete="off"/>
      <label>tokens <b id="gen-n-val">20</b><br/><input id="gen-n" type="range" min="5" max="60" value="20" oninput="$('gen-n-val').textContent=this.value"/></label>
      <button class="act" id="gen-btn">Generate</button>
      <button class="act" id="gen-clear">Clear</button>
    </div>
    <div id="gen-output"><span class="muted">Output will appear here…</span></div>
    <div id="gen-stats" class="muted" style="margin-top:0.5rem;font-size:0.8rem;"></div>
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

  <!-- REASON -->
  <section id="reason" class="view">
    <p class="muted">Reasoning over a knowledge space <b>grounded in the model's learned representations</b>. Analogy transfers a displacement (a : b :: c : ?); neighbours finds nearby concepts in that space. Answer quality is bounded by representation quality — a lightly-trained model gives a rough space.</p>
    <div class="exrow">
      <input id="anA" placeholder="a (e.g. king)" autocomplete="off"/>
      <span class="muted" style="align-self:center">:</span>
      <input id="anB" placeholder="b (e.g. queen)" autocomplete="off"/>
      <span class="muted" style="align-self:center;font-weight:700">::</span>
      <input id="anC" placeholder="c (e.g. man)" autocomplete="off"/>
      <button class="act" id="reason-analogy-btn" style="background:#7c3aed;color:#fff">Analogy →</button>
    </div>
    <div class="exrow">
      <input id="reasonWord" placeholder="word (e.g. water)" autocomplete="off"/>
      <button class="act" id="reason-neigh-btn" style="background:#0ea5e9;color:#fff">Neighbours</button>
    </div>
    <div id="reason-out" class="muted" style="margin-top:1rem;line-height:1.9;min-height:60px"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);

/* Robust JSON fetch: never blow up on an HTML/text error page (e.g. a 500
   "Internal Server Error"), which used to surface as
   "Unexpected token 'I', "Internal S"... is not valid JSON". */
async function jget(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.text();
  let data = null;
  try { data = body ? JSON.parse(body) : null; } catch (_) { data = null; }
  if (!r.ok || data === null) {
    const detail = (data && (data.error || data.detail))
      || (body || '').replace(/<[^>]*>/g, '').trim().slice(0, 200)
      || (r.status + ' ' + r.statusText);
    if (r.status === 503 || /not\s*loaded|no model|training/i.test(detail))
      throw new Error('Model is still training or loading — try again shortly.');
    throw new Error(detail);
  }
  return data;
}
const POST_JSON = b => ({method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b)});

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
const Q_RE = /^(what|who|where|when|why|how|which|explain|define|describe)\b|\?/i;

function addMsg(role, text, sub) {
  const el = document.createElement('div'); el.className = 'msg ' + role; el.textContent = text;
  if (sub) { const s = document.createElement('div'); s.className = 'candidates'; s.textContent = sub; el.appendChild(s); }
  messages.appendChild(el); messages.scrollTop = messages.scrollHeight;
}

function addQAMsg(d) {
  const el = document.createElement('div'); el.className = 'msg bot';

  const badge = document.createElement('div'); badge.className = 'qa-intent';
  badge.textContent = d.intent || 'continuation';
  el.appendChild(badge);

  const ans = document.createElement('div'); ans.className = 'qa-answer';
  ans.textContent = d.answer || '(no answer generated)';
  el.appendChild(ans);

  if (d.related?.length) {
    const rel = document.createElement('div'); rel.className = 'qa-related';
    d.related.forEach(w => {
      const s = document.createElement('span'); s.textContent = w;
      s.onclick = () => { promptEl.value = 'what is ' + w + '?'; $('send-btn').click(); };
      rel.appendChild(s);
    });
    el.appendChild(rel);
  }

  const meta = document.createElement('div'); meta.className = 'qa-meta';
  const promptStr = d.prompt ? '"' + d.prompt.join(' ') + '"' : '';
  meta.textContent = [
    d.confidence !== undefined ? 'confidence: ' + d.confidence : '',
    promptStr ? 'prompt: ' + promptStr : ''
  ].filter(Boolean).join(' · ');
  if (meta.textContent) el.appendChild(meta);

  messages.appendChild(el); messages.scrollTop = messages.scrollHeight;
}

/* Send — smart route: question → /qa, else → /generate */
$('send-btn').onclick = async () => {
  const t = promptEl.value.trim(); if (!t) return;
  addMsg('user', t); promptEl.value = '';

  if (Q_RE.test(t)) {
    try {
      const d = await jget('/qa?' + new URLSearchParams({question: t}));
      if (d.error) { addMsg('bot', '⚠ ' + d.error); return; }
      addQAMsg(d);
    } catch(e) { addMsg('bot', '⚠ ' + e.message); }
  } else {
    try {
      const d = await jget('/generate', POST_JSON({tokens: t.toLowerCase().split(/\s+/), n: 20}));
      addMsg('bot', d.generated ? d.generated.join(' ') : '(no output)');
    } catch(e) { addMsg('bot', '⚠ ' + e.message); }
  }
};

/* Predict — raw HTM next-word debug */
$('predict-btn').onclick = async () => {
  const t = promptEl.value.trim(); if (!t) return; addMsg('user', t); promptEl.value = '';
  try {
    const d = await jget('/predict', POST_JSON({tokens: t.toLowerCase().split(/\s+/), topn: 5}));
    if (d.predictions?.length) { const top = d.predictions[0][0]; const rest = d.predictions.slice(1).map(p=>p[0]).join(', '); addMsg('bot','→ '+top, rest?'also: '+rest:''); }
    else addMsg('bot','(no prediction)');
  } catch(e) { addMsg('bot','⚠ '+e.message); }
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

/* ---- sentence generator ---- */
$('gen-btn').onclick = async () => {
  const seed = $('gen-seed').value.trim();
  const n = parseInt($('gen-n').value);
  const out = $('gen-output');
  const stats = $('gen-stats');
  out.innerHTML = '<span class="muted">Generating…</span>';
  stats.textContent = '';
  try {
    const tokens = seed ? seed.toLowerCase().split(/\s+/) : [];
    const t0 = performance.now();
    const d = await jget('/generate', POST_JSON({tokens, n}));
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (d.error) { out.innerHTML = '<span style="color:#f87171">' + d.error + '</span>'; return; }
    const generated = d.generated || [];
    const genWords = generated.slice(tokens.length);
    out.innerHTML = '';
    // Prompt words in grey
    tokens.forEach((w, i) => {
      const sp = document.createElement('span');
      sp.className = 'gen-prompt';
      sp.title = 'prompt[' + i + ']';
      sp.textContent = w + ' ';
      out.appendChild(sp);
    });
    // Generated words in blue, with position on hover
    genWords.forEach((w, i) => {
      const sp = document.createElement('span');
      sp.className = 'gen-word';
      sp.title = 'step ' + (i + 1);
      sp.textContent = w + ' ';
      out.appendChild(sp);
    });
    stats.textContent = genWords.length + ' tokens generated in ' + elapsed + 's';
  } catch(e) { out.innerHTML = '<span style="color:#f87171">Error: ' + e.message + '</span>'; }
};
$('gen-clear').onclick = () => {
  $('gen-output').innerHTML = '<span class="muted">Output will appear here…</span>';
  $('gen-stats').textContent = '';
  $('gen-seed').value = '';
};
$('gen-seed').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); $('gen-btn').click(); } });

/* ---- explore: fingerprint visualisation ---- */
async function fp(word) {
  try { return await jget('/fingerprint?word=' + encodeURIComponent(word)); }
  catch (_) { return null; }
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

/* ---- reason: grounded knowledge-space reasoning ---- */
async function runReason(url) {
  const out = $('reason-out');
  out.innerHTML = '<span class="muted">Reasoning…</span>';
  try {
    const d = await jget(url);
    if (d.mode === 'analogy') {
      out.innerHTML =
        `<div style="font-size:1.05rem"><b>${d.a}</b> : <b>${d.b}</b> &nbsp;::&nbsp; <b>${d.c}</b> : `
        + `<b style="color:#a78bfa">${d.answer ?? '(none)'}</b></div>`
        + `<div class="muted" style="margin-top:.6rem">nearest to the transferred point: `
        + (d.nearest||[]).map(([t,dist]) => `${t} (${dist})`).join(', ') + `</div>`;
    } else if (d.mode === 'neighbors') {
      out.innerHTML = `<div>nearest to <b>${d.word}</b> in the learned space:</div>`
        + `<div style="margin-top:.5rem">`
        + (d.neighbours||[]).map(([t,dist]) =>
            `<span style="display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:9999px;padding:3px 11px;margin:3px;font-size:.85rem;cursor:pointer" data-w="${t}">${t} · ${dist}</span>`).join('')
        + `</div>`;
      document.querySelectorAll('#reason-out span[data-w]').forEach(s =>
        s.onclick = () => { $('reasonWord').value = s.dataset.w; $('reason-neigh-btn').click(); });
    }
  } catch(e) { out.innerHTML = '<span style="color:#f87171">⚠ ' + e.message + '</span>'; }
}
$('reason-analogy-btn').onclick = () => {
  const a=$('anA').value.trim(), b=$('anB').value.trim(), c=$('anC').value.trim();
  if (!a||!b||!c) { $('reason-out').innerHTML='<span class="muted">Fill a, b and c (a : b :: c : ?).</span>'; return; }
  runReason('/reason?' + new URLSearchParams({mode:'analogy', a, b, c}));
};
$('reason-neigh-btn').onclick = () => {
  const w=$('reasonWord').value.trim();
  if (!w) { $('reason-out').innerHTML='<span class="muted">Enter a word.</span>'; return; }
  runReason('/reason?' + new URLSearchParams({mode:'neighbors', word:w, k:8}));
};
['anA','anB','anC'].forEach(id => $(id).addEventListener('keydown', e => { if (e.key==='Enter') $('reason-analogy-btn').click(); }));
$('reasonWord').addEventListener('keydown', e => { if (e.key==='Enter') $('reason-neigh-btn').click(); });

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
        Volume first so a model trained in another container becomes visible.

        Falls back to PREV_VERSION weights when no checkpoint exists for the
        current VERSION yet (e.g. UI-only bumps that don't retrain)."""
        from persist.store import load_model
        if refresh:
            vol.reload()
        model_dir = VOL_PATH / MODEL_DIR_NAME
        self._fallback_active = False
        # Fall back to previous version's checkpoint if this version has none yet
        if not (model_dir / "config.json").exists() and PREV_VERSION:
            prev_dir = Path(_VOL_MOUNT) / PREV_VERSION / MODEL_DIR_NAME
            if (prev_dir / "config.json").exists():
                model_dir = prev_dir
                self._fallback_active = True
                print(f"[{VERSION}] No checkpoint yet — loading from {PREV_VERSION}")
        if (model_dir / "config.json").exists():
            try:
                self.model = load_model(str(model_dir))
                self._reason_space = None      # invalidate grounded space on reload
                print(f"[{VERSION}] Model loaded from {model_dir}")
            except Exception as e:
                print(f"[{VERSION}] Could not load model: {e}")
        metrics_path = VOL_PATH / "metrics.json"
        # Inherit metrics only together with the weights, so the model and its
        # accuracy history always describe the same checkpoint
        if not metrics_path.exists() and self._fallback_active:
            metrics_path = Path(_VOL_MOUNT) / PREV_VERSION / "metrics.json"
        if metrics_path.exists():
            self.metrics = json.loads(metrics_path.read_text())

    def _ensure_reason_space(self):
        """Build (once, cached) a ConceptSpace grounded in the live model's
        learned representations — coordinates discovered from the corpus the
        model was trained on.  Invalidated whenever the model reloads."""
        if getattr(self, "_reason_space", None) is None:
            from core.reasoning import ground_model
            self._reason_space = ground_model(self.model, n_axes=8)
        return self._reason_space

    def _ensure_model(self):
        """Lazily pick up a model trained after this (always-on) container
        started — so a fresh `make deploy` needs no manual /reload."""
        if self.model is None:
            self._load_model(refresh=True)
            return
        # While serving inherited PREV_VERSION weights, keep checking
        # (throttled) for this version's own checkpoint so a headless retrain
        # is picked up without a manual /reload
        if getattr(self, "_fallback_active", False):
            now = time.time()
            if now - getattr(self, "_fallback_checked", 0.0) < 30:
                return
            self._fallback_checked = now
            vol.reload()
            if (VOL_PATH / MODEL_DIR_NAME / "config.json").exists():
                self._load_model()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from docs_html import DOCS_HTML

        api = FastAPI(title="CLM Chat", docs_url=None, redoc_url=None)

        @api.get("/", response_class=HTMLResponse)
        async def index():
            return CHAT_HTML

        @api.get("/docs", response_class=HTMLResponse)
        async def docs():
            return DOCS_HTML

        @api.post("/predict")
        async def predict(body: PredictReq):
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            preds = self.model.predict_next(body.tokens, topn=body.topn)
            return {"predictions": [[t, round(s, 4)] for t, s in preds]}

        @api.post("/generate")
        async def generate(body: GenerateReq):
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            return {"generated": self.model.generate(body.tokens, n=body.n)}

        @api.get("/qa")
        async def qa(question: str):
            """Semantic question answering: multi-level prompt encoding, intent
            detection, SDR-similarity expansion, confidence-scored prompt
            selection, and planned generation with adaptive stopping (the
            response length is decided by the ResponsePlanner, not the caller).
            """
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            from core.qa import SemanticQA
            return SemanticQA(self.model).answer(question)

        @api.get("/similar")
        async def similar(word: str, k: int = 6):
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            return {"word": word,
                    "neighbours": [[t, o] for t, o in self.model.similar(word, k=k)]}

        @api.get("/fingerprint")
        async def fingerprint(word: str):
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            return self.model.fingerprint(word)

        @api.get("/reason")
        async def reason(mode: str = "analogy", a: str = "", b: str = "",
                         c: str = "", word: str = "", k: int = 6):
            """Reasoning over a knowledge space grounded in the model's learned
            (Spatial-Pooler) representations.

              /reason?mode=analogy&a=king&b=queen&c=man   → a:b :: c:?
              /reason?mode=neighbors&word=paris&k=6        → nearest concepts
            """
            import numpy as _np
            self._ensure_model()
            if not self.model:
                return JSONResponse({"error": "Model not ready — training may still be in progress"}, status_code=503)
            try:
                space = self._ensure_reason_space()
            except Exception as e:
                return JSONResponse({"error": f"could not build reasoning space: {e}"}, status_code=500)

            def norm(w):
                return (w or "").strip().lower()

            def nearest(target, exclude, n):
                return sorted(
                    ((name, int(_np.abs(space.coord[name] - target).sum()))
                     for name in space.coord if name not in exclude),
                    key=lambda kv: kv[1])[:n]

            if mode == "analogy":
                a, b, c = norm(a), norm(b), norm(c)
                if not (a and b and c):
                    return JSONResponse({"error": "analogy needs a, b, c (a:b :: c:?)"}, status_code=400)
                missing = [x for x in (a, b, c) if x not in space.coord]
                if missing:
                    return JSONResponse({"error": f"unknown word(s): {', '.join(missing)}"}, status_code=400)
                from core.reasoning import analogy as _analogy
                target = space.coord[b] - space.coord[a] + space.coord[c]
                return {"mode": "analogy", "a": a, "b": b, "c": c,
                        "answer": _analogy(space, a, b, c),
                        "nearest": [[n, d] for n, d in nearest(target, {a, b, c}, k)]}

            if mode == "neighbors":
                word = norm(word)
                if not word:
                    return JSONResponse({"error": "neighbors needs word"}, status_code=400)
                if word not in space.coord:
                    return JSONResponse({"error": f"unknown word: {word}"}, status_code=400)
                return {"mode": "neighbors", "word": word,
                        "neighbours": [[n, d] for n, d in nearest(space.coord[word], {word}, k)]}

            return JSONResponse({"error": f"unknown mode '{mode}' (analogy|neighbors)"}, status_code=400)

        @api.get("/stats")
        async def stats():
            self._ensure_model()
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
