"""
core/column.py — Vectorised HTM-style cortical column.

Key design: synaptic state lives in fixed-capacity numpy arrays so that the
critical overlap computation is a single numpy gather + reduce — parallelisable
on GPU (swap `np` for `cupy` with no other changes).

Storage layout
--------------
  seg_idx  : (n_cells, max_segs, syn_per_seg)  int32
      Presynaptic source index per synapse.  Unused slots hold `source_dim`
      (sentinel), which maps to a permanently-False position in the padded
      source vector.
  seg_perm : (n_cells, max_segs, syn_per_seg)  float32
      Permanence value per synapse (0 .. 1).
  n_segs   : (n_cells,)                         int32
      Number of allocated segments per cell.

Source index space
------------------
  [0 .. n_cells-1]               → previous winner cells
  [n_cells .. n_cells+loc_dim-1] → location bits
  source_dim (= n_cells+loc_dim) → sentinel (always inactive)
"""
from __future__ import annotations
import numpy as np


class CorticalColumn:
    """
    HTM temporal memory with vectorised segment overlap.

    Parameters
    ----------
    col_dim              : number of mini-columns (= feature SDR width)
    cells_per_col        : cells per mini-column (high-order context depth)
    loc_dim              : location SDR width (= GridLocation.dim)
    seed                 : RNG seed
    connected            : permanence threshold for a connected synapse
    init_perm            : initial permanence for new synapses
    inc / dec / pred_dec : Hebbian learning rates
    activation_threshold : connected overlaps needed to activate a segment
    min_threshold        : potential overlaps to consider a segment matching
    syn_per_seg          : maximum synapses per segment
    max_segs             : maximum segments per cell
    """

    def __init__(
        self,
        col_dim: int,
        cells_per_col: int,
        loc_dim: int,
        seed: int = 0,
        connected: float = 0.5,
        init_perm: float = 0.21,
        inc: float = 0.10,
        dec: float = 0.02,
        pred_dec: float = 0.01,
        activation_threshold: int = 10,
        min_threshold: int = 5,
        syn_per_seg: int = 24,
        max_segs: int = 32,
    ):
        self.col_dim = col_dim
        self.cpc = cells_per_col
        self.n_cells = col_dim * cells_per_col
        self.loc_dim = loc_dim
        self.source_dim = self.n_cells + loc_dim    # sentinel = source_dim

        self.connected = connected
        self.init_perm = init_perm
        self.inc, self.dec, self.pred_dec = inc, dec, pred_dec
        self.activation_threshold = activation_threshold
        self.min_threshold = min_threshold
        self.syn_per_seg = syn_per_seg
        self.max_segs = max_segs

        # Core state (fixed-size, vectorised)
        self.seg_idx = np.full(
            (self.n_cells, max_segs, syn_per_seg), self.source_dim, dtype=np.int32
        )
        self.seg_perm = np.zeros(
            (self.n_cells, max_segs, syn_per_seg), dtype=np.float32
        )
        self.n_segs = np.zeros(self.n_cells, dtype=np.int32)

        self.rng = np.random.default_rng(seed)
        self.prev_winners = np.zeros(self.n_cells, dtype=np.bool_)
        self.last_winners = np.zeros(self.n_cells, dtype=np.bool_)
        self.last_bursts = 0           # surprised columns in the last step
        self.last_active_cols = 0      # active columns in the last step

    # ── Source vector ────────────────────────────────────────────────────────

    def _build_source(self, loc_sdr: np.ndarray) -> np.ndarray:
        """
        Build the (source_dim+1,) dense bool source vector.
        Position `source_dim` is always False (sentinel for unused synapses).
        """
        src = np.zeros(self.source_dim + 1, dtype=np.bool_)
        src[: self.n_cells] = self.prev_winners
        src[self.n_cells + loc_sdr] = True
        return src  # src[source_dim] stays False

    # ── Vectorised overlap ───────────────────────────────────────────────────

    def _compute_overlaps(self, src: np.ndarray):
        """
        Compute connected and potential overlap for every (cell, segment).

        Parameters
        ----------
        src : (source_dim+1,) bool  — padded source vector

        Returns
        -------
        conn_ov : (n_cells, max_segs) int32
        pot_ov  : (n_cells, max_segs) int32
        valid   : (n_cells, max_segs) bool  — True for allocated segments
        """
        # Gather source activity at every synapse: single numpy fancy-index
        syn_active = src[self.seg_idx]              # (n_cells, max_segs, syn_per_seg)
        syn_connected = self.seg_perm >= self.connected  # bool
        syn_valid = self.seg_idx < self.source_dim       # non-sentinel synapses

        conn_ov = (syn_active & syn_connected & syn_valid).sum(axis=-1).astype(np.int32)
        pot_ov = (syn_active & syn_valid).sum(axis=-1).astype(np.int32)

        seg_range = np.arange(self.max_segs, dtype=np.int32)
        valid = seg_range[None, :] < self.n_segs[:, None]

        return conn_ov, pot_ov, valid

    # ── Learning helpers ─────────────────────────────────────────────────────

    def _best_seg(self, cell: int, pot_ov: np.ndarray) -> tuple[int, int]:
        """Best (segment_index, potential_overlap) for a cell."""
        ns = int(self.n_segs[cell])
        if ns == 0:
            return -1, 0
        si = int(pot_ov[cell, :ns].argmax())
        return si, int(pot_ov[cell, si])

    def _adapt_seg(
        self,
        cell: int,
        si: int,
        src: np.ndarray,
        inc: float,
        dec: float,
        grow_k: int,
    ) -> None:
        """Permanence update + optional synapse growth for one segment."""
        idx = self.seg_idx[cell, si]                    # (syn_per_seg,) int32
        valid = idx < self.source_dim                   # bool mask for real synapses
        active = src[idx[valid]]                        # activity at valid synapses

        delta = np.where(active, inc, -dec).astype(np.float32)
        self.seg_perm[cell, si, valid] = np.clip(
            self.seg_perm[cell, si, valid] + delta, 0.0, 1.0
        )

        if grow_k > 0 and valid.sum() < self.syn_per_seg:
            existing = set(idx[valid].tolist())
            candidates = np.where(src[: self.source_dim])[0]
            candidates = candidates[~np.isin(candidates, list(existing))]
            if candidates.size:
                self.rng.shuffle(candidates)
                new_idx = candidates[:grow_k].astype(np.int32)
                empty = np.where(~valid)[0][: len(new_idx)]
                self.seg_idx[cell, si, empty] = new_idx[: empty.size]
                self.seg_perm[cell, si, empty] = self.init_perm

    def _grow_seg(self, cell: int, src: np.ndarray) -> None:
        """Allocate a new segment on `cell` from current active sources."""
        if self.n_segs[cell] >= self.max_segs:
            return
        si = int(self.n_segs[cell])
        candidates = np.where(src[: self.source_dim])[0]
        self.rng.shuffle(candidates)
        chosen = candidates[: self.syn_per_seg].astype(np.int32)
        self.seg_idx[cell, si, : len(chosen)] = chosen
        self.seg_perm[cell, si, : len(chosen)] = self.init_perm
        self.n_segs[cell] += 1

    # ── Main step ────────────────────────────────────────────────────────────

    def step(
        self,
        feature_sdr: np.ndarray,
        loc_sdr: np.ndarray,
        learn: bool = True,
        modulation: float = 1.0,
    ) -> np.ndarray:
        """
        Process one token.

        Parameters
        ----------
        feature_sdr : sorted int32 — active mini-column indices
        loc_sdr     : sorted int32 — active location bit indices
        modulation  : plasticity scalar from neuromodulatory signal [0..2]

        Returns
        -------
        winners : (n_cells,) bool — active cells this step
        """
        src = self._build_source(loc_sdr)
        conn_ov, pot_ov, valid = self._compute_overlaps(src)

        predictive = ((conn_ov >= self.activation_threshold) & valid).any(axis=1)

        winners = np.zeros(self.n_cells, dtype=np.bool_)
        active_col_set = set(int(c) for c in feature_sdr)
        inc = self.inc * modulation
        dec = self.dec * modulation

        # Burst accounting for the neuromodulatory / replay signals: a column
        # "bursts" (is surprised) when none of its cells were predicted.
        self.last_active_cols = int(len(active_col_set))
        self.last_bursts = 0

        for col in feature_sdr:
            col = int(col)
            start = col * self.cpc
            cells = np.arange(start, start + self.cpc)
            pred_cells = cells[predictive[cells]]

            if pred_cells.size:                         # correct prediction
                winners[pred_cells] = True
                if learn:
                    for c in pred_cells:
                        si, ov = self._best_seg(c, pot_ov)
                        if si >= 0:
                            self._adapt_seg(c, si, src, inc, dec,
                                            grow_k=max(0, self.syn_per_seg - ov))
            else:                                       # surprise → burst
                self.last_bursts += 1
                pots = np.array([
                    int(pot_ov[c, : self.n_segs[c]].max())
                    if self.n_segs[c] > 0 else -1
                    for c in cells
                ])
                winner = int(cells[int(pots.argmax())])
                winners[winner] = True
                if learn:
                    si, ov = self._best_seg(winner, pot_ov)
                    if ov >= self.min_threshold and si >= 0:
                        self._adapt_seg(winner, si, src, inc, dec,
                                        grow_k=self.syn_per_seg - ov)
                    else:
                        self._grow_seg(winner, src)

        # Punish false predictions (cells that predicted but column wasn't active)
        if learn:
            for c in np.where(predictive)[0]:
                if (c // self.cpc) not in active_col_set:
                    for si in range(int(self.n_segs[c])):
                        if conn_ov[c, si] >= self.activation_threshold:
                            idx = self.seg_idx[c, si]
                            v = idx < self.source_dim
                            self.seg_perm[c, si, v] -= np.where(
                                src[idx[v]], self.pred_dec * modulation, 0.0
                            ).astype(np.float32)
                            self.seg_perm[c, si] = np.clip(
                                self.seg_perm[c, si], 0.0, 1.0
                            )

        self.prev_winners = winners
        self.last_winners = winners
        return winners

    def predict_columns(self, loc_next_sdr: np.ndarray) -> np.ndarray:
        """
        Return predicted mini-column indices for the NEXT token,
        given current last_winners and the next location SDR.
        """
        src = np.zeros(self.source_dim + 1, dtype=np.bool_)
        src[: self.n_cells] = self.last_winners
        src[self.n_cells + loc_next_sdr] = True

        conn_ov, _, valid = self._compute_overlaps(src)
        predictive = ((conn_ov >= self.activation_threshold) & valid).any(axis=1)
        return np.unique((np.where(predictive)[0] // self.cpc).astype(np.int32))

    def reset(self) -> None:
        """Clear temporal state (call before each new sequence)."""
        self.prev_winners[:] = False
        self.last_winners[:] = False

    @property
    def n_segments(self) -> int:
        return int(self.n_segs.sum())

    @property
    def n_synapses(self) -> int:
        # Only count non-sentinel synapses
        return int((self.seg_idx < self.source_dim).sum())

    @property
    def memory_mb(self) -> float:
        return (self.seg_idx.nbytes + self.seg_perm.nbytes) / 1e6
