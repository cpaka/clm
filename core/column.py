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
  [n_cells .. n_cells+loc_dim-1] → location bits (optional — pass empty loc_sdr
                                   to decouple temporal memory from position)
  source_dim (= n_cells+loc_dim) → sentinel (always inactive)

#3 NEXT_STEPS: vectorised step() eliminates the Python per-column loop.
  Winners are assigned via numpy fancy-indexing; permanence updates are batched
  across all (cell, segment) pairs at once.  The grow loop runs only for burst
  cells (a shrinking minority as the model learns).
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
    loc_dim              : location SDR width (= GridLocation.dim); pass 0 to
                           disable positional connections entirely
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
        Passing an empty loc_sdr disables positional bits entirely.
        """
        src = np.zeros(self.source_dim + 1, dtype=np.bool_)
        src[: self.n_cells] = self.prev_winners
        if loc_sdr.size:
            src[self.n_cells + loc_sdr] = True
        return src

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

    def _compute_overlaps_scoped(self, src: np.ndarray, cells: np.ndarray):
        """
        Compact overlap for `cells` only — everything stays O(len(cells)),
        never O(n_cells).  This is the key training speedup: only ~2 % of
        columns are active per step.

        Returns compact arrays indexed by position in `cells`:
          conn_c : (A, max_segs) int  — connected overlap
          pot_c  : (A, max_segs) int  — potential overlap
          valid_c: (A, max_segs) bool — allocated-segment mask
        """
        idx = self.seg_idx[cells]                     # (A, max_segs, syn_per_seg)
        syn_active = src[idx]
        syn_valid = idx < self.source_dim
        syn_connected = self.seg_perm[cells] >= self.connected

        conn_c = (syn_active & syn_connected & syn_valid).sum(axis=-1)
        pot_c = (syn_active & syn_valid).sum(axis=-1)

        seg_range = np.arange(self.max_segs, dtype=np.int32)
        valid_c = seg_range[None, :] < self.n_segs[cells][:, None]
        return conn_c, pot_c, valid_c

    @staticmethod
    def _best_seg_compact(pot_row: np.ndarray, n_seg: int) -> tuple[int, int]:
        """Best (segment_index, potential_overlap) from a compact pot row."""
        if n_seg == 0:
            return -1, 0
        si = int(pot_row[:n_seg].argmax())
        return si, int(pot_row[si])

    # ── Batch permanence update ──────────────────────────────────────────────

    def _batch_adapt(
        self,
        cells: np.ndarray,
        segs: np.ndarray,
        src: np.ndarray,
        inc: float,
        dec: float,
    ) -> None:
        """Update permanences for a batch of (cell, seg) pairs in one numpy op."""
        if cells.size == 0:
            return
        idx_b = self.seg_idx[cells, segs]               # (B, syn_per_seg) int32
        valid_b = idx_b < self.source_dim                # (B, syn_per_seg) bool
        safe = np.where(valid_b, idx_b, 0)
        active_b = src[safe] & valid_b                   # (B, syn_per_seg) bool
        delta = np.where(active_b, inc,
                         np.where(valid_b, -dec, 0.0)).astype(np.float32)
        self.seg_perm[cells, segs] = np.clip(
            self.seg_perm[cells, segs] + delta, 0.0, 1.0
        )

    # ── Learning helpers ─────────────────────────────────────────────────────

    def _grow_syn(
        self,
        cell: int,
        si: int,
        active_src: np.ndarray,
        grow_k: int,
    ) -> None:
        """Grow up to `grow_k` new synapses on (cell, si) from active_src."""
        if grow_k <= 0:
            return
        existing = self.seg_idx[cell, si]
        valid_existing = existing[existing < self.source_dim]
        candidates = active_src[~np.isin(active_src, valid_existing)]
        if not candidates.size:
            return
        self.rng.shuffle(candidates)
        new_idx = candidates[:grow_k].astype(np.int32)
        empty = np.where(existing >= self.source_dim)[0][: len(new_idx)]
        self.seg_idx[cell, si, empty] = new_idx[: empty.size]
        self.seg_perm[cell, si, empty] = self.init_perm

    def _grow_seg(self, cell: int, active_src: np.ndarray) -> None:
        """Allocate a new segment on `cell` from the precomputed active sources."""
        if self.n_segs[cell] >= self.max_segs:
            return
        si = int(self.n_segs[cell])
        candidates = active_src.copy()
        self.rng.shuffle(candidates)
        chosen = candidates[: self.syn_per_seg].astype(np.int32)
        self.seg_idx[cell, si, : len(chosen)] = chosen
        self.seg_perm[cell, si, : len(chosen)] = self.init_perm
        self.n_segs[cell] += 1

    # ── Main step (vectorised) ────────────────────────────────────────────────

    def step(
        self,
        feature_sdr: np.ndarray,
        loc_sdr: np.ndarray,
        learn: bool = True,
        modulation: float = 1.0,
    ) -> np.ndarray:
        """
        Process one token.

        Vectorised implementation (#3 NEXT_STEPS): winners are assigned via
        numpy fancy-indexing; permanence updates are batched across all
        (cell, segment) pairs; the grow loop runs only for burst cells.

        Parameters
        ----------
        feature_sdr : sorted int32 — active mini-column indices
        loc_sdr     : sorted int32 — active location bit indices (empty = no-op)
        modulation  : plasticity scalar from neuromodulatory signal [0..2]

        Returns
        -------
        winners : (n_cells,) bool — active cells this step
        """
        src = self._build_source(loc_sdr)

        n_cols = int(feature_sdr.size)
        cell_grid = (
            feature_sdr.astype(np.int64)[:, None] * self.cpc
            + np.arange(self.cpc, dtype=np.int64)[None, :]
        )                                                   # (n_cols, cpc)
        active_cells = cell_grid.ravel()                    # (A = n_cols*cpc,)
        n_segs_c = self.n_segs[active_cells]

        conn_c, pot_c, valid_c = self._compute_overlaps_scoped(src, active_cells)

        predictive_c = ((conn_c >= self.activation_threshold) & valid_c).any(axis=1)
        best_pot_c = np.where(valid_c, pot_c, -1).max(axis=1)              # (A,)

        pred_col = predictive_c.reshape(n_cols, self.cpc)                  # (n_cols, cpc)
        bestpot_col = best_pot_c.reshape(n_cols, self.cpc)

        predicted_any = pred_col.any(axis=1)                               # (n_cols,)
        burst_mask = ~predicted_any

        # ── Winners (no Python per-column loop) ──────────────────────────────
        winners = np.zeros(self.n_cells, dtype=np.bool_)

        # Predicted: mark every cell whose segment fired
        if predicted_any.any():
            pred_col_j, pred_r = np.where(pred_col)                        # (K,) each
            winners[cell_grid[pred_col_j, pred_r]] = True

        # Burst: one winner per burst col — cell with highest potential overlap
        burst_col_j = np.where(burst_mask)[0]                              # (B,)
        r_best = bestpot_col[burst_mask].argmax(axis=1) if burst_col_j.size else np.empty(0, np.intp)
        if burst_col_j.size:
            winners[cell_grid[burst_col_j, r_best]] = True

        self.last_bursts = int(burst_col_j.size)
        self.last_active_cols = n_cols

        # ── Learning ─────────────────────────────────────────────────────────
        if learn:
            inc = self.inc * modulation
            dec = self.dec * modulation
            active_src = np.where(src[: self.source_dim])[0].astype(np.int32)

            # --- Predicted cells: batch-adapt best segment ---
            if predicted_any.any():
                rows = pred_col_j * self.cpc + pred_r               # flat idx into active_cells
                nsegs_row = n_segs_c[rows]
                pot_rows = pot_c[rows]                               # (K, max_segs)

                # Vectorised best_seg per predictive cell
                seg_range = np.arange(self.max_segs, dtype=np.int32)
                valid_seg = seg_range[None, :] < nsegs_row[:, None]
                masked_pot = np.where(valid_seg, pot_rows, -1)
                si_batch = masked_pot.argmax(axis=1)                 # (K,)
                ov_batch = pot_rows[np.arange(len(si_batch)), si_batch]
                has_seg = nsegs_row > 0

                adapt_cells = active_cells[rows[has_seg]].astype(np.int64)
                adapt_segs = si_batch[has_seg]
                grow_k_arr = np.maximum(0, self.syn_per_seg - ov_batch[has_seg])

                self._batch_adapt(adapt_cells, adapt_segs, src, inc, dec)

                # Grow missing synapses (cell-by-cell, minority operation)
                for i in np.where(grow_k_arr > 0)[0]:
                    self._grow_syn(int(adapt_cells[i]), int(adapt_segs[i]),
                                   active_src, int(grow_k_arr[i]))

            # --- Burst cells: adapt existing segment or grow new one ---
            if burst_col_j.size:
                burst_rows = burst_col_j * self.cpc + r_best        # (B,)
                nsegs_burst = n_segs_c[burst_rows]
                pot_burst = pot_c[burst_rows]

                seg_range = np.arange(self.max_segs, dtype=np.int32)
                valid_seg = seg_range[None, :] < nsegs_burst[:, None]
                masked_pot = np.where(valid_seg, pot_burst, -1)
                si_burst = masked_pot.argmax(axis=1)                 # (B,)
                ov_burst = pot_burst[np.arange(len(si_burst)), si_burst]

                adapt_mask = (ov_burst >= self.min_threshold) & (nsegs_burst > 0)
                burst_cells = cell_grid[burst_col_j, r_best]         # (B,) absolute cell ids

                # Batch-adapt burst cells that have a good enough segment
                if adapt_mask.any():
                    self._batch_adapt(burst_cells[adapt_mask], si_burst[adapt_mask],
                                      src, inc, dec)

                # Grow new segments for the rest
                for i in np.where(~adapt_mask)[0]:
                    self._grow_seg(int(burst_cells[i]), active_src)

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
        if loc_next_sdr.size:
            src[self.n_cells + loc_next_sdr] = True

        # Only cells with ≥1 segment can be predictive — gather over those only.
        cells = np.where(self.n_segs > 0)[0]
        if cells.size == 0:
            return np.empty(0, dtype=np.int32)
        conn_c, _, valid_c = self._compute_overlaps_scoped(src, cells)
        predictive = ((conn_c >= self.activation_threshold) & valid_c).any(axis=1)
        return np.unique((cells[predictive] // self.cpc).astype(np.int32))

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
