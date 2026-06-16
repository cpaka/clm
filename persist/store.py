"""
persist/store.py — Fast filesystem persistence for HierarchicalCLM.

Format
------
A directory named `<name>.clm/` containing:
  config.json          — hyperparameters and metrics
  enc_<l>_<u>.npz     — SemanticEncoder fingerprints for level l, unit u
  col_<l>_<u>.npz     — CorticalColumn arrays for level l, unit u
  token_sdr_<l>_<u>.npz — token→SDR mapping and inverted index

numpy .npz files are raw binary (ZIP container of .npy blobs); loading
a 100 MB column takes ~50 ms on a fast SSD.  For larger models, the same
arrays can be memory-mapped by passing `mmap_mode="r"` to np.load.

Usage
-----
    save_model(model, "runs/my_model")
    model2 = load_model("runs/my_model", sequences=seqs)  # warm-start ready
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.hierarchy import HierarchicalCLM
from core.encoder import SemanticEncoder


def checkpoint_path(base: str, name: str = "model") -> str:
    """Return <base>/<name>.clm path (does not create it)."""
    return str(Path(base) / f"{name}.clm")


# ── Save ─────────────────────────────────────────────────────────────────────

def save_model(model: HierarchicalCLM, path: str) -> None:
    """
    Persist all model state to `path` (directory created if needed).

    Parameters
    ----------
    model : trained HierarchicalCLM
    path  : directory path (e.g. "runs/experiment1.clm")
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    # Config + metrics
    cfg = {
        "n_levels": model.n_levels,
        "strides": list(model.strides),
        "n_units": model.n_units,
        "col_dim": model.col_dim,
        "fp_bits": model.fp_bits,
        "kwta_k": model.kwta_k,
        "encoder": model.encoder_type,
        "encoder_cfg": model._encoder_cfg,
        "col_kwargs": {
            k: list(v) if isinstance(v, (tuple, np.ndarray)) else v
            for k, v in model._col_kwargs.items()
        },
        "metrics": model.metrics,
    }
    (root / "config.json").write_text(json.dumps(cfg, indent=2))

    if not model.levels:
        return  # nothing trained yet

    for li, level in enumerate(model.levels):
        for ui, unit in enumerate(level.units):
            tag = f"l{li}_u{ui}"

            # Column state
            np.savez_compressed(
                root / f"col_{tag}.npz",
                seg_idx=unit.col.seg_idx,
                seg_perm=unit.col.seg_perm,
                n_segs=unit.col.n_segs,
            )

            # Token SDR map + inverted index
            if unit._token_sdr:
                tokens = list(unit._token_sdr.keys())
                sdrs = np.stack([unit._token_sdr[t] for t in tokens])   # (V, w)
                np.savez_compressed(
                    root / f"token_sdr_{tag}.npz",
                    tokens=np.array(tokens, dtype=object),
                    sdrs=sdrs,
                )

            # SemanticEncoder fingerprints
            if isinstance(unit.enc, SemanticEncoder) and unit.enc.fp:
                vocab = list(unit.enc.fp.keys())
                fps = np.stack([unit.enc.fp[t] for t in vocab])
                np.savez_compressed(
                    root / f"enc_{tag}.npz",
                    vocab=np.array(vocab, dtype=object),
                    fps=fps,
                )


# ── Load ─────────────────────────────────────────────────────────────────────

def load_model(path: str) -> HierarchicalCLM:
    """
    Restore a HierarchicalCLM from a saved directory.

    The returned model is ready for inference and incremental training.
    """
    root = Path(path)
    cfg = json.loads((root / "config.json").read_text())

    col_kwargs = dict(cfg["col_kwargs"])
    col_kwargs["periods"] = tuple(col_kwargs["periods"])

    # encoder_cfg already carries dim/fp_bits/index_bits/window
    model = HierarchicalCLM(
        n_levels=cfg["n_levels"],
        strides=tuple(cfg["strides"]),
        n_units=cfg["n_units"],
        col_dim=cfg["col_dim"],
        kwta_k=cfg["kwta_k"],
        encoder=cfg["encoder"],
        **cfg["encoder_cfg"],
        **col_kwargs,
    )
    model.metrics = cfg.get("metrics", [])

    # Reconstruct semantic encoders BEFORE building so level-0 units bind to
    # SemanticEncoder instances (not the random-encoder fallback). Fingerprints
    # are restored from the enc_*.npz blobs below.
    if cfg["encoder"] == "semantic":
        model._encoders = [
            SemanticEncoder(**cfg["encoder_cfg"], seed=i)
            for i in range(cfg["n_units"])
        ]

    model._build()

    for li, level in enumerate(model.levels):
        for ui, unit in enumerate(level.units):
            tag = f"l{li}_u{ui}"

            # Column
            col_file = root / f"col_{tag}.npz"
            if col_file.exists():
                data = np.load(col_file)
                unit.col.seg_idx = data["seg_idx"]
                unit.col.seg_perm = data["seg_perm"]
                unit.col.n_segs = data["n_segs"]

            # Token SDR map + inverted index
            tsdr_file = root / f"token_sdr_{tag}.npz"
            if tsdr_file.exists():
                data = np.load(tsdr_file, allow_pickle=True)
                unit._inv = defaultdict(list)
                for tok, sdr in zip(data["tokens"].tolist(), data["sdrs"]):
                    unit._token_sdr[tok] = sdr.astype(np.int32)
                    for b in sdr:
                        unit._inv[int(b)].append(tok)

            # Encoder fingerprints
            enc_file = root / f"enc_{tag}.npz"
            if enc_file.exists() and isinstance(unit.enc, SemanticEncoder):
                data = np.load(enc_file, allow_pickle=True)
                for tok, fp in zip(data["vocab"].tolist(), data["fps"]):
                    unit.enc.fp[tok] = fp.astype(np.int32)

    return model
