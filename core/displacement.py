"""
core/displacement.py — Grid reference frames and learned relations.

The 2019 grid-cell framework (Hawkins et al.) makes reference frames the basis
of cortical representation.  Here a *knowledge space* is an N-axis grid: every
coordinate maps to a multi-scale SDR (via co-prime ring phases), and a
**relation** is a *displacement* — a constant move in that space.

The payoff (why grid codes, not arbitrary embeddings): displacements **compose
by addition**, so a relation learned from a few local examples generalises to
distances never seen, and chaining relations gives transitive inference for
free:

    D(A→B) then D(B→C)  ==  D(A→C)          # composition = phase addition

This is the mechanism REASONING.md proposes for "reasoning as movement through a
knowledge space".
"""
from __future__ import annotations
import numpy as np

from .grid import GridLocation


class GridSpace:
    """N-axis grid: an integer coordinate → multi-scale sparse SDR.

    Each axis is an independent co-prime ring bank (a `GridLocation`); axis SDRs
    occupy disjoint bit ranges so a coordinate's SDR is their concatenation.
    Uniqueness holds over LCM(periods) positions per axis.
    """

    def __init__(self, n_axes: int = 1, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        self.n_axes = n_axes
        self.periods = tuple(periods)
        self._g = GridLocation(periods)
        self.axis_dim = self._g.dim
        self.dim = n_axes * self.axis_dim

    def sdr(self, coord) -> np.ndarray:
        """Sparse SDR (sorted int32) for an integer coordinate (len n_axes)."""
        coord = np.atleast_1d(coord).astype(np.int64)
        bits = [self._g.at(int(coord[a])) + a * self.axis_dim
                for a in range(self.n_axes)]
        return np.sort(np.concatenate(bits)).astype(np.int32)

    @property
    def active_bits(self) -> int:
        """Number of active bits in any coordinate SDR (for exact-match checks)."""
        return self.n_axes * len(self.periods)


class Relation:
    """A learned displacement in a `GridSpace`: target ≈ source + delta.

    `delta` is an integer move per axis, learned by averaging example
    (source, target) coordinate pairs.  Relations compose by adding deltas."""

    def __init__(self, space: GridSpace, delta: np.ndarray | None = None):
        self.space = space
        self.delta = (np.zeros(space.n_axes, dtype=np.int64)
                      if delta is None else np.asarray(delta, dtype=np.int64))

    def fit(self, pairs: list[tuple]) -> "Relation":
        """Learn the displacement from (source_coord, target_coord) examples."""
        deltas = [np.atleast_1d(t).astype(np.int64) - np.atleast_1d(s).astype(np.int64)
                  for s, t in pairs]
        self.delta = np.round(np.mean(deltas, axis=0)).astype(np.int64)
        return self

    def apply(self, coord, n: int = 1) -> np.ndarray:
        """Move `n` steps of this relation from `coord`."""
        return np.atleast_1d(coord).astype(np.int64) + n * self.delta

    def compose(self, other: "Relation") -> "Relation":
        """Chain two relations (D_self then D_other) — delta addition."""
        return Relation(self.space, self.delta + other.delta)

    def inverse(self) -> "Relation":
        return Relation(self.space, -self.delta)
