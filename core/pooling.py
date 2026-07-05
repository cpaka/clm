"""
core/pooling.py — Online temporal-pooling "output layer".

The 2017 cortical-columns theory (Hawkins, Ahmad & Cui, *A Theory of How
Columns in the Neocortex Enable Learning the Structure of the World*) pairs the
fast sequence-memory "input layer" with a slowly-changing "output layer" that
pools activity over a whole sequence into a stable representation of the object
being sensed.  That stable layer feeds back (apically) to bias the input
layer's predictions toward the current object/context.

`ContextPool` is a lightweight, training-free realisation of that output layer
for the language setting: it integrates the mini-columns active across the
context into a decayed accumulator and exposes the top-`pool_w` columns as a
stable "topic" SDR.  Because the encoder's fingerprint space and the column
space share `col_dim`, the overlap between a candidate token's fingerprint and
this pool SDR is a pure-SDR measure of how consistent that token is with the
running context — the apical consistency bias used at decode time.

Cost is O(col_dim) per token (one decayed accumulate + a top-w partition) and
zero at training time, so it adds negligible calculus to either path.
"""
from __future__ import annotations
import numpy as np


class ContextPool:
    """Decayed temporal pooling of active mini-columns into a stable SDR.

    Parameters
    ----------
    col_dim : mini-column count (= encoder fingerprint width)
    pool_w  : active bits kept in the pooled context SDR
    decay   : per-token multiplicative decay of the accumulator (0..1);
              higher = longer context memory / slower-changing pool
    """

    __slots__ = ("col_dim", "pool_w", "decay", "acc")

    def __init__(self, col_dim: int, pool_w: int = 48, decay: float = 0.92):
        self.col_dim = col_dim
        self.pool_w = pool_w
        self.decay = decay
        self.acc = np.zeros(col_dim, dtype=np.float32)

    def reset(self) -> None:
        self.acc[:] = 0.0

    def update(self, active_cols: np.ndarray) -> None:
        """Fold one token's active/winner mini-columns into the pool."""
        self.acc *= self.decay
        if len(active_cols):
            self.acc[active_cols] += 1.0

    def sdr(self) -> np.ndarray:
        """Dense bool vector of the current top-`pool_w` context columns."""
        v = np.zeros(self.col_dim, dtype=bool)
        nz = int((self.acc > 0).sum())
        k = min(self.pool_w, nz)
        if k <= 0:
            return v
        idx = np.argpartition(self.acc, -k)[-k:]
        v[idx[self.acc[idx] > 0]] = True
        return v
