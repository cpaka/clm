"""
core/sdr.py — Sparse Distributed Representation primitives.

Two representations are used throughout:
  sparse : int32 ndarray of sorted active indices, shape (k,)
  dense  : bool ndarray, shape (..., dim)

All batch ops work on dense form; individual column steps use sparse for
memory efficiency. Replace `np` with `cupy` for transparent GPU execution.
"""
from __future__ import annotations
import numpy as np


# ── Conversion ──────────────────────────────────────────────────────────────

def sparse_to_dense(indices: np.ndarray, dim: int) -> np.ndarray:
    """Sorted active-index array → dense bool vector."""
    v = np.zeros(dim, dtype=np.bool_)
    v[indices] = True
    return v


def dense_to_sparse(v: np.ndarray) -> np.ndarray:
    """Dense bool vector → sorted active indices (int32)."""
    return np.where(v)[0].astype(np.int32)


# ── Single-pair overlap ──────────────────────────────────────────────────────

def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """|a ∩ b| for two sorted sparse SDRs."""
    return int(np.intersect1d(a, b, assume_unique=True).size)


# ── Batch overlap (vectorised, GPU-ready) ───────────────────────────────────

def batch_overlap(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise overlaps between two sets of SDRs.

    queries : (Q, dim) bool
    keys    : (K, dim) bool
    returns : (Q, K) int16 overlap counts

    Equivalent to queries @ keys.T; casting to int16 before matmul keeps
    arithmetic fast on hardware that accelerates narrow integers.
    """
    return (queries.astype(np.int16) @ keys.T.astype(np.int16))


# ── Construction ─────────────────────────────────────────────────────────────

def make_sdr(dim: int, w: int, seed: int) -> np.ndarray:
    """Deterministic random sparse SDR: w active bits from `seed`, sorted int32."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dim, size=w, replace=False)).astype(np.int32)


# ── k-Winners-Take-All ───────────────────────────────────────────────────────

def kwta(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Lateral inhibition: keep only the top-k scoring positions.

    Returns a bool mask with True at the k highest-scoring positions.
    Ties are broken by index (lowest index wins) — deterministic.
    """
    if k <= 0:
        return np.zeros(scores.shape, dtype=np.bool_)
    if k >= scores.size:
        return scores > 0

    threshold = np.partition(scores, -k)[-k]
    mask = scores >= threshold

    # Trim exact ties to enforce exactly k active
    excess = int(mask.sum()) - k
    if excess > 0:
        mask[np.where(mask)[0][:excess]] = False
    return mask
