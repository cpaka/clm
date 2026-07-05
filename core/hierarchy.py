"""
core/hierarchy.py — Hierarchical Cortical Language Model.

Architecture
============

                ┌──────────────────────────────────┐
  tokens ──▶  │ Level 0 (stride=1, token granularity) │
                └──────────┬───────────────────────┘
                           │ winner SDR (every stride[1] steps)
                ┌──────────▼───────────────────────┐
                │ Level 1 (stride=4, phrase-level)  │
                └──────────┬───────────────────────┘
                           │ winner SDR (every stride[2] steps)
                ┌──────────▼───────────────────────┐
                │ Level 2 (stride=16, clause-level) │
                └──────────────────────────────────┘

Each level is a voting ensemble of `n_units` CLM units (Thousand-Brains
consensus). Higher levels fire at coarser time-scales, learning phrase and
clause structure without explicit parse trees.

Between levels a fixed random sparse projection compresses the winner
representation to a manageable feature SDR — this models thalamo-cortical
inter-area axonal projections.

Prediction at inference time collects votes from all levels and decodes by
nearest-neighbour overlap against the token inventory.

Lateral inhibition (k-WTA) is applied to the aggregate column scores before
decoding to sharpen predictions and break ties.

Neuromodulatory plasticity and hippocampal replay are wired in at level 0.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

from .sdr import kwta, dense_to_sparse
from .encoder import TokenEncoder, SemanticEncoder
from .grid import GridLocation
from .column import CorticalColumn
from .modulation import NeuromodSignal
from .replay import HippocampalBuffer
from .pooling import ContextPool
from .spatial_pooler import SpatialPooler
from .xp import asnumpy

# Default co-prime periods for the level-0 location code.  Small and co-prime:
# the LCM (2310) is far larger than any sentence, but each individual ring wraps
# within a few tokens, so location encodes a repeating phrase rhythm rather than
# unbounded absolute position (which would make segments position-specific).
DEFAULT_LOC_PERIODS: tuple[int, ...] = (2, 3, 5, 7, 11)


# ── Projection layer (between levels) ────────────────────────────────────────

class SparseProjection:
    """
    Fixed random sparse projection from one SDR space to another.

    Maps a dense winner vector of `in_dim` bits down to a `out_k`-sparse
    SDR of width `out_dim` via a random binary matrix.  No learning — this
    is the anatomical wiring between cortical areas.

    Parameters
    ----------
    in_dim  : dimension of the input winner vector
    out_dim : mini-column count of the target level
    out_k   : active bits in the projected feature SDR
    seed    : reproducibility
    """

    def __init__(self, in_dim: int, out_dim: int, out_k: int, seed: int = 0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_k = out_k
        # Sparse binary projection matrix (2 % density)
        rng = np.random.default_rng(seed)
        density = max(2, int(0.02 * in_dim))        # ~2 % connections per col
        self._proj = np.zeros((in_dim, out_dim), dtype=np.int8)
        for j in range(out_dim):
            rows = rng.choice(in_dim, size=density, replace=False)
            self._proj[rows, j] = 1

    def project(self, winners: np.ndarray) -> np.ndarray:
        """
        winners : (in_dim,) bool
        returns : sorted int32 sparse SDR of width out_dim, w=out_k
        """
        scores = winners.astype(np.int32) @ self._proj   # (out_dim,)
        return dense_to_sparse(kwta(scores, self.out_k)).astype(np.int32)

    def back_project_to_cells(self, pred_col_indices: np.ndarray) -> np.ndarray:
        """Score in_dim source cells by connectivity to predicted output columns.

        pred_col_indices : (P,) int — predicted level-1 mini-column indices
        returns          : (in_dim,) float32 — connection count per source cell
        """
        if pred_col_indices.size == 0:
            return np.zeros(self.in_dim, dtype=np.float32)
        return self._proj[:, pred_col_indices].sum(axis=1).astype(np.float32)


# ── One CLM unit (encoder + grid + column + inverted index) ─────────────────

class CLMUnit:
    """
    Atomic prediction unit: one encoder, one grid, one column.
    Used as a building block inside each hierarchy level.
    """

    def __init__(
        self,
        col_dim: int,
        w: int,
        cells_per_col: int,
        periods: tuple[int, ...],
        encoder,               # TokenEncoder | SemanticEncoder | None (external feature SDR)
        seed: int = 0,
        use_location: bool = False,
        loc_periods: tuple[int, ...] | None = None,
        use_spatial_pooler: bool = False,
        sp_kwargs: dict | None = None,
        **col_kwargs,
    ):
        self.enc = encoder
        self.use_location = use_location
        # The grid drives the location code (improvement #3).  Small co-prime
        # periods keep it a bounded phrase rhythm.  When location is disabled the
        # grid still sizes the column's (tiny) location space but is never fed.
        self.grid = GridLocation(loc_periods or DEFAULT_LOC_PERIODS)
        self.col = CorticalColumn(
            col_dim, cells_per_col, self.grid.dim, seed=seed, **col_kwargs
        )
        self.w = w                                  # active bits per feature SDR
        self._token_sdr: dict[str, np.ndarray] = {}
        self._inv: dict[int, list[str]] = defaultdict(list)  # bit → token list

        # Spatial Pooler (learned representations) — level-0 only (needs an
        # encoder to pool over).  When present, a token's active mini-columns are
        # the SP's *learned* output SDR rather than the encoder's fixed
        # fingerprint; the temporal memory then learns sequences over learned
        # representations.  Fit once, frozen, and cached (see fit_pooler).
        self.use_sp = bool(use_spatial_pooler and encoder is not None)
        self.sp: SpatialPooler | None = (
            SpatialPooler(input_dim=col_dim, col_dim=col_dim, active_cols=w,
                          seed=seed, **(sp_kwargs or {}))
            if self.use_sp else None
        )

    def _encode(self, tok: str) -> np.ndarray:
        """Raw encoder fingerprint (input to the Spatial Pooler when present)."""
        return self.enc.encode(tok)

    def _sdr(self, tok: str) -> np.ndarray:
        """Active mini-columns for a token: SP-learned SDR if a (frozen) Spatial
        Pooler is present, else the encoder fingerprint.  Cached; also builds the
        bit→token inverted index used by the decoder."""
        s = self._token_sdr.get(tok)
        if s is None:
            s = self.sp.compute(self._encode(tok), learn=False) if self.use_sp \
                else self.enc.encode(tok)
            self._token_sdr[tok] = s
            for b in s:
                self._inv[int(b)].append(tok)
        return s

    def fit_pooler(self, tokens: list[str], passes: int = 3) -> None:
        """Learn the Spatial Pooler on the token inventory, then freeze + cache.

        The SP output depends only on a token's (fixed) encoder fingerprint, so
        it suffices to iterate the unique tokens; boosting spreads the learned
        representation across the whole column population.  After freezing, the
        token→SDR map is deterministic and cached, so the temporal memory trains
        on stable learned representations at no extra per-step cost."""
        if not self.use_sp or self.sp._frozen:
            return
        vocab = list(dict.fromkeys(tokens))          # unique, order-preserving
        rng = np.random.default_rng(0)
        for _ in range(passes):
            rng.shuffle(vocab)
            for tok in vocab:
                self.sp.compute(self._encode(tok), learn=True)
        self.sp.freeze()
        # Reset any cache built before freezing so _sdr rebuilds from frozen SP.
        self._token_sdr.clear()
        self._inv = defaultdict(list)

    # Shared empty loc_sdr for levels/units that run without a location code.
    _NO_LOC = np.empty(0, dtype=np.int32)

    def _loc(self, i: int) -> np.ndarray:
        """Location SDR at path-integration index `i` (empty if disabled).

        Path integration is deterministic (+1 per token), so the location of the
        yet-unseen next token is simply `_loc(len(context))` — forward
        predictable, which is what makes the location code usable for
        prediction rather than only postdiction."""
        return self.grid.at(i) if self.use_location else self._NO_LOC

    def train_sequence(self, tokens: list[str], modulation: float = 1.0) -> float:
        """Train on one sequence. Returns burst rate (burst cols / active cols)."""
        self.col.reset()
        bursts = total = 0
        for i, tok in enumerate(tokens):
            self.col.step(self._sdr(tok), self._loc(i), learn=True,
                          modulation=modulation)
            bursts += self.col.last_bursts
            total += self.col.last_active_cols
        return bursts / total if total else 0.0

    def train_feature_sequence(
        self, feature_sdrs: list[np.ndarray], modulation: float = 1.0
    ) -> float:
        """Train on a pre-projected feature SDR sequence (for level ≥ 1)."""
        self.col.reset()
        bursts = total = 0
        for ftr in feature_sdrs:
            self.col.step(ftr, self._NO_LOC, learn=True, modulation=modulation)
            bursts += self.col.last_bursts
            total += self.col.last_active_cols
        return bursts / total if total else 0.0

    def score_next(self, tokens: list[str]) -> dict[str, float]:
        """Run context and return token → weighted fingerprint-coverage score.

        A token scores by the strength-weighted fraction of ITS OWN fingerprint
        bits that are predicted (∈ [0, 1] after per-step normalisation) — not
        by a per-bit IDF sum, which systematically favoured rare-bit tokens
        regardless of how little of their fingerprint was predicted."""
        self.col.reset()
        for i, tok in enumerate(tokens):
            self.col.step(self._sdr(tok), self._loc(i), learn=False)
        return self._decode_scores(
            *self.col.predict_column_scores(self._loc(len(tokens)))
        )

    # Identity-core bits are unique per token; shell bits are shared between
    # semantically similar tokens.  Weighting cores higher separates the exact
    # predicted token from its semantic neighbours at decode time.
    _CORE_WEIGHT = 3.0
    # Graded decode weights columns by prediction strength; binary treats all
    # predicted columns equally.
    _GRADED = True

    def _decode_scores(
        self, pred_cols: np.ndarray, strengths: np.ndarray
    ) -> dict[str, float]:
        """Decode predicted columns (+ strengths) against the token inventory."""
        if pred_cols.size == 0:
            return {}
        smap = np.zeros(self.col.col_dim, dtype=np.float32)
        smap[pred_cols] = (strengths / strengths.max()) if self._GRADED else 1.0
        candidates: set[str] = set()
        for b in pred_cols:
            candidates.update(self._inv.get(int(b), ()))

        # Identity-core weighting only makes sense when columns ARE the encoder
        # fingerprint; with a Spatial Pooler the columns are learned, so the core
        # bits no longer index them.
        semantic = isinstance(self.enc, SemanticEncoder) and not self.use_sp
        scores: dict[str, float] = {}
        for tok in candidates:
            sc = float(smap[self._token_sdr[tok]].sum())
            denom = float(self.w)
            if semantic:
                core = self.enc.index_vec(tok)
                sc += (self._CORE_WEIGHT - 1.0) * float(smap[core].sum())
                denom += (self._CORE_WEIGHT - 1.0) * core.size
            scores[tok] = sc / denom
        return scores

    def get_winners_after(self, tokens: list[str]) -> np.ndarray:
        """Run context, return last_winners dense bool for upward projection."""
        self.col.reset()
        for i, tok in enumerate(tokens):
            self.col.step(self._sdr(tok), self._loc(i), learn=False)
        return asnumpy(self.col.last_winners)

    def reset(self) -> None:
        self.grid.reset()
        self.col.reset()


# ── One level of the hierarchy ────────────────────────────────────────────────

class HierarchyLevel:
    """
    A voting ensemble of `n_units` CLM units at one temporal scale.

    Level 0 works directly on tokens.
    Level ≥ 1 works on projected feature SDRs produced by the level below.
    """

    def __init__(
        self,
        level: int,
        stride: int,
        n_units: int,
        col_dim: int,
        w: int,
        cells_per_col: int,
        periods: tuple[int, ...],
        encoder_factory,           # callable(seed) → encoder (None for level ≥ 1)
        seed_base: int = 0,
        **col_kwargs,
    ):
        self.level = level
        self.stride = stride
        self.n_units = n_units
        self.col_dim = col_dim
        self.w = w

        self.units: list[CLMUnit] = [
            CLMUnit(
                col_dim=col_dim,
                w=w,
                cells_per_col=cells_per_col,
                periods=periods,
                encoder=encoder_factory(i) if encoder_factory else None,
                seed=seed_base + i + level * 100,
                **col_kwargs,
            )
            for i in range(n_units)
        ]

    @property
    def n_cells(self) -> int:
        return self.units[0].col.n_cells


# ── Hierarchical CLM ─────────────────────────────────────────────────────────

def pick_novel(
    preds: list[tuple[str, float]], out: list[str], window: int = 5
) -> tuple[str, float]:
    """Highest-ranked prediction not seen in the last `window` tokens of `out`.

    Anti-loop heuristic shared by HierarchicalCLM.generate() and core.qa's
    planned generator; falls back to the overall top-1 when every candidate
    is recent (`preds` must be non-empty)."""
    recent = set(out[-window:])
    for t, s in preds:
        if t not in recent:
            return t, s
    return preds[0]


class HierarchicalCLM:
    """
    Multi-level cortical hierarchy with voting, k-WTA, neuromodulation
    and hippocampal replay.

    Parameters
    ----------
    n_levels      : number of hierarchy levels (1 = flat, like original CLM)
    strides       : temporal stride per level (list of len n_levels)
    n_units       : voting columns per level
    col_dim       : mini-column count (= encoder SDR width)
    cells_per_col : cells per mini-column
    periods       : grid module periods
    encoder       : "semantic" | "random"
    dim           : SDR width (for SemanticEncoder)
    fp_bits       : active bits per token fingerprint
    index_bits    : identity-core bits (SemanticEncoder only)
    window        : co-occurrence window (SemanticEncoder only)
    kwta_k        : winners kept after lateral inhibition; None = all
    replay_cap    : hippocampal buffer capacity (0 = disabled)
    replay_thresh : burst rate threshold for episode storage
    """

    # Echo-demotion factors applied to context tokens at decode time (1.0 = off).
    ECHO_DEMOTE_CONTEXT = 0.35
    ECHO_DEMOTE_LAST = 0.1

    # ── Inference-time knobs for improvements #1/#2/#4 (class defaults; copied
    # to instance attributes so they can be swept without persistence churn) ──
    POOL_W = 48              # active bits in the context-pooling SDR (#1)
    POOL_DECAY = 0.92        # per-token decay of the pool accumulator (#1)
    APICAL_WEIGHT = 0.25     # weight of the pool→token consistency bias (#2)
    CONSENSUS_POWER = 0.5    # agreement exponent for cross-unit voting (#4)

    def __init__(
        self,
        n_levels: int = 2,
        strides: tuple[int, ...] = (1, 4),
        n_units: int = 3,
        col_dim: int = 2048,
        cells_per_col: int = 8,
        periods: tuple[int, ...] = (7, 11, 13, 17, 19, 23),
        encoder: str = "semantic",
        dim: int = 2048,
        fp_bits: int = 21,
        index_bits: int = 7,
        window: int = 2,
        kwta_k: int | None = None,
        replay_cap: int = 512,
        replay_thresh: float = 0.3,
        seed_base: int = 0,
        use_location: bool = False,
        loc_periods: tuple[int, ...] | None = None,
        use_spatial_pooler: bool = False,
        sp_kwargs: dict | None = None,
        **col_kwargs,
    ):
        assert len(strides) == n_levels, "strides must have one entry per level"

        self.n_levels = n_levels
        self.strides = strides
        self.n_units = n_units
        self.col_dim = col_dim
        self.fp_bits = fp_bits
        self.kwta_k = kwta_k if kwta_k is not None else fp_bits * 2
        self.encoder_type = encoder
        # Offset applied to every unit/encoder seed — lets a standalone single-unit
        # model reproduce "unit i" of a larger ensemble (used by parallel training).
        self.seed_base = seed_base

        # Neuromodulatory signal (shared across level-0 units)
        self.neuromod = NeuromodSignal()

        # Hippocampal replay buffer
        self.replay = HippocampalBuffer(replay_cap, replay_thresh) if replay_cap else None

        # Semantic encoders — one per unit at level 0 (each sees same corpus).
        # The encoder SDR width MUST equal col_dim because feature bits index
        # mini-columns directly; col_dim is the single source of truth.
        self._encoders: list[SemanticEncoder | TokenEncoder] | None = None
        self._encoder_cfg = dict(
            dim=col_dim, fp_bits=fp_bits, index_bits=index_bits, window=window
        )

        # Projections between levels: proj[l] maps level-l winners to level-(l+1) features
        self._projections: list[SparseProjection] = []

        # Location code (#3): each level-0 unit path-integrates a small co-prime
        # grid and feeds it as a soft distal cue.  OFF by default — on free text
        # it slightly hurts (position dilutes the token-context match) and adds
        # compute, so it earns its cost only on positionally-structured data.
        # loc_periods and use_location ride in _col_kwargs so they thread to every
        # CLMUnit and round-trip through persistence; loc_syn_cap reaches the column.
        self.use_location = use_location
        self.loc_periods = tuple(loc_periods) if loc_periods else DEFAULT_LOC_PERIODS

        # Spatial Pooler (learned SDR representations).  Fit once, then frozen.
        self.use_spatial_pooler = use_spatial_pooler
        self._poolers_fit = False

        # Inference knobs (#1/#2/#4) — instance copies of the class defaults.
        self.pool_w = self.POOL_W
        self.pool_decay = self.POOL_DECAY
        self.apical_weight = self.APICAL_WEIGHT
        self.consensus_power = self.CONSENSUS_POWER

        # Levels are built lazily (after encoders are fit)
        self.levels: list[HierarchyLevel] = []
        # Extra CorticalColumn hyperparameters (activation_threshold, syn_per_seg,
        # max_segs, connected, init_perm, inc, dec, pred_dec, min_threshold,
        # loc_syn_cap) are threaded straight through to every column; use_location
        # and loc_periods are consumed by CLMUnit before the column.
        self._col_kwargs: dict = dict(
            cells_per_col=cells_per_col, periods=periods,
            use_location=use_location, loc_periods=self.loc_periods,
            use_spatial_pooler=use_spatial_pooler, sp_kwargs=sp_kwargs,
            **col_kwargs
        )
        self._n_cells_l0: int = col_dim * cells_per_col
        self._periods = periods

        # Training metrics
        self.metrics: list[dict] = []

    # ── Build (after encoder fitting) ─────────────────────────────────────────

    def _build(self) -> None:
        """Instantiate all levels and inter-level projections."""
        if self.levels:
            return

        enc_list = self._encoders or [
            TokenEncoder(dim=self._encoder_cfg["dim"],
                         w=self.fp_bits, seed=self.seed_base + i)
            for i in range(self.n_units)
        ]

        for lvl in range(self.n_levels):
            if lvl == 0:
                def ef(i, enc_list=enc_list):
                    return enc_list[i]
                w = self.fp_bits
                cdim = self.col_dim
            else:
                # Level ≥ 1: features come from projection of prev level winners
                ef = None          # no per-token encoder; feature SDRs fed externally
                w = self.fp_bits
                cdim = self.col_dim

            self.levels.append(
                HierarchyLevel(
                    level=lvl,
                    stride=self.strides[lvl],
                    n_units=self.n_units,
                    col_dim=cdim,
                    w=w,
                    encoder_factory=ef,
                    seed_base=self.seed_base,
                    **self._col_kwargs,
                )
            )

        # Build inter-level projections
        for lvl in range(self.n_levels - 1):
            n_cells = self.levels[lvl].n_cells
            self._projections.append(
                SparseProjection(
                    in_dim=n_cells,
                    out_dim=self.col_dim,
                    out_k=self.fp_bits,
                    seed=lvl,
                )
            )

    # ── Training ─────────────────────────────────────────────────────────────

    def fit_encoders(self, sequences: list[list[str]]) -> None:
        """
        Fit semantic encoders (SemanticEncoder only).
        Call before train() on first use, or omit for random encoder.
        """
        if self.encoder_type != "semantic":
            return
        self._encoders = [
            SemanticEncoder(**self._encoder_cfg, seed=self.seed_base + i)
            for i in range(self.n_units)
        ]
        for enc in self._encoders:
            enc.fit(sequences)

    def _fit_poolers(self, sequences: list[list[str]]) -> None:
        """Fit + freeze each level-0 unit's Spatial Pooler (once)."""
        if not self.use_spatial_pooler or self._poolers_fit:
            return
        tokens = [t for seq in sequences for t in seq]
        for unit in self.levels[0].units:
            unit.fit_pooler(tokens)
        self._poolers_fit = True

    def inject_units(self, units: list) -> None:
        """Replace level-0 voting units with pre-trained CLMUnit objects.

        Used by parallel training: each unit is trained standalone in its own
        container, then assembled here into one ensemble. Encoders are
        repointed at the injected units for similar()/fingerprint()."""
        self._build()
        assert len(units) == len(self.levels[0].units), "unit count mismatch"
        self.levels[0].units = list(units)
        self._encoders = [u.enc for u in units]

    def train(
        self,
        sequences: list[list[str]],
        epochs: int = 1,
        replay_every: int = 5,
        verbose: bool = False,
        progress_cb=None,
    ) -> None:
        """
        Train on a list of token sequences.

        Parameters
        ----------
        sequences   : list of tokenised sentences/paragraphs
        epochs      : passes over the corpus
        replay_every: replay hippocampal buffer every N epochs
        verbose     : print epoch summaries
        progress_cb : optional callable(epoch, total, acc, segs) for UI polling
        """
        if self.encoder_type == "semantic" and self._encoders is None:
            self.fit_encoders(sequences)
        self._build()
        self._fit_poolers(sequences)

        for epoch in range(epochs):
            burst_total = total_seqs = 0

            for seq in sequences:
                burst_rate = self._train_one(seq)
                burst_total += burst_rate
                total_seqs += 1

                # Store in hippocampal buffer if surprising
                if self.replay:
                    self.replay.record(seq, burst_rate)

            # Hippocampal replay pass
            if self.replay and (epoch + 1) % replay_every == 0 and self.replay.size:
                for rseq in self.replay.sample(n=min(32, self.replay.size)):
                    self._train_one(rseq, modulation=1.5)   # boosted plasticity

            avg_burst = burst_total / max(total_seqs, 1)
            n_seg = self._n_segments()

            m = {"epoch": epoch + 1, "burst_rate": round(avg_burst, 3),
                 "segments": n_seg}
            self.metrics.append(m)

            if verbose:
                print(f"  epoch {epoch+1}/{epochs}  burst={avg_burst:.2f}  segs={n_seg}")
            if progress_cb:
                progress_cb(epoch + 1, epochs, avg_burst, n_seg)

    def _train_one(self, seq: list[str], modulation: float | None = None) -> float:
        """Train all levels on one sequence. Returns level-0 burst rate."""
        if modulation is None:
            modulation = self.neuromod.modulation

        no_loc = CLMUnit._NO_LOC

        # Reset ALL levels — inference resets them too, and phrase-level state
        # must not leak across unrelated sequences.
        for lvl in self.levels:
            for unit in lvl.units:
                unit.col.reset()

        # Last winners per (level, unit): each unit feeds ITS OWN winners
        # upward, matching predict_next()'s per-unit projection at inference.
        last_winners: list[list[np.ndarray | None]] = [
            [None] * self.n_units for _ in range(self.n_levels)
        ]

        bursts_l0 = active_l0 = steps = 0
        for pos, tok in enumerate(seq):
            for uid, unit in enumerate(self.levels[0].units):
                winners = unit.col.step(unit._sdr(tok), unit._loc(pos), learn=True,
                                        modulation=modulation)
                last_winners[0][uid] = winners
                if uid == 0:
                    bursts_l0 += unit.col.last_bursts
                    active_l0 += unit.col.last_active_cols

            steps += 1
            for lvl in range(1, self.n_levels):
                if steps % self.strides[lvl] != 0:
                    continue
                for uid in range(self.n_units):
                    below = last_winners[lvl - 1][uid]
                    if below is None:
                        continue
                    # asnumpy: winners may live on GPU; SparseProjection is numpy
                    ftr = self._projections[lvl - 1].project(asnumpy(below))
                    last_winners[lvl][uid] = self.levels[lvl].units[uid].col.step(
                        ftr, no_loc, learn=True, modulation=modulation
                    )

        burst_rate = bursts_l0 / max(active_l0, 1)
        self.neuromod.update(burst_rate > 0.5)
        return burst_rate

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_next(self, tokens: list[str], topn: int = 3) -> list[tuple[str, float]]:
        """
        Predict the next token given a context.

        Each level-0 unit decodes its predicted columns into token scores by
        fingerprint coverage.  Three cortical mechanisms sharpen the result:

          • Location (#3): a path-integrated grid code disambiguates context by
            phrase position; the next location is known (path integration is
            deterministic), so it conditions the prediction.
          • Apical context bias (#2) from an online pooling "output layer" (#1):
            candidates whose fingerprint overlaps the running context SDR are
            boosted, replacing the old fixed-weight upper-level back-projection.
          • Consensus voting (#4): tokens predicted by more units are weighted
            super-linearly, so the ensemble converges on agreement.

        k-WTA then applies lateral inhibition before ranking.
        Returns list of (token, score) sorted descending.
        """
        self._build()

        # Reset all levels before running context
        for lvl in self.levels:
            for unit in lvl.units:
                unit.col.reset()

        units0 = self.levels[0].units
        pools = [ContextPool(self.col_dim, self.pool_w, self.pool_decay)
                 for _ in units0]

        # Run context through all levels, mirroring the training stride pattern
        last_winners: list[list[np.ndarray | None]] = [
            [None] * self.n_units for _ in range(self.n_levels)
        ]
        step_count = 0
        for pos, tok in enumerate(tokens):
            for uid, unit in enumerate(units0):
                fp = unit._sdr(tok)
                unit.col.step(fp, unit._loc(pos), learn=False)
                last_winners[0][uid] = asnumpy(unit.col.last_winners)
                pools[uid].update(fp)                     # pool the active columns (#1)
            step_count += 1
            for lvl in range(1, self.n_levels):
                if step_count % self.strides[lvl] != 0:
                    continue
                for uid in range(self.n_units):
                    below = last_winners[lvl - 1][uid]
                    if below is None:
                        continue
                    ftr = self._projections[lvl - 1].project(asnumpy(below))
                    unit_l = self.levels[lvl].units[uid]
                    unit_l.col.step(ftr, CLMUnit._NO_LOC, learn=False)
                    last_winners[lvl][uid] = asnumpy(unit_l.col.last_winners)

        # Per-unit decode + apical bias, accumulating agreement for consensus.
        loc_next = len(tokens)
        tok_sum: dict[str, float] = defaultdict(float)
        tok_cnt: dict[str, int] = defaultdict(int)
        for uid, unit in enumerate(units0):
            decoded = unit._decode_scores(
                *unit.col.predict_column_scores(unit._loc(loc_next))
            )
            if not decoded:
                continue
            pool_sdr = pools[uid].sdr() if self.apical_weight else None
            for tok, sc in decoded.items():
                if pool_sdr is not None:
                    fp = unit._token_sdr.get(tok)
                    if fp is not None:
                        sc += self.apical_weight * float(pool_sdr[fp].sum()) / unit.w
                tok_sum[tok] += sc
                tok_cnt[tok] += 1

        if not tok_sum:
            return []

        # Consensus voting (#4): agreement across units weighted super-linearly.
        raw_scores: dict[str, float] = {
            tok: s * (tok_cnt[tok] / self.n_units) ** self.consensus_power
            for tok, s in tok_sum.items()
        }

        # Demote context echoes: the task is to predict the NEXT token, but
        # fingerprint self-overlap makes just-seen tokens systematic false
        # winners.  The immediate last token is demoted hardest.
        if self.ECHO_DEMOTE_CONTEXT < 1.0:
            for t in set(tokens):
                if t in raw_scores:
                    raw_scores[t] *= self.ECHO_DEMOTE_CONTEXT
        if tokens and tokens[-1] in raw_scores:
            raw_scores[tokens[-1]] *= self.ECHO_DEMOTE_LAST

        # k-WTA on aggregate scores for lateral inhibition
        all_tokens = list(raw_scores)
        scores_arr = np.array([raw_scores[t] for t in all_tokens])
        mask = kwta(scores_arr, self.kwta_k)
        filtered = {t: float(s) for t, s, m in zip(all_tokens, scores_arr, mask) if m}

        ranked = sorted(filtered.items(), key=lambda kv: -kv[1])
        return ranked[:topn]

    def generate(self, prompt: list[str], n: int = 8) -> list[str]:
        """Greedy continuation for `n` tokens with recency-exclusion to prevent loops."""
        out = list(prompt)
        for _ in range(n):
            # Extra candidates so pick_novel can skip recently used tokens
            preds = self.predict_next(out, topn=6)
            if not preds:
                break
            out.append(pick_novel(preds, out)[0])
        return out

    def similar(self, word: str, k: int = 6) -> list[tuple[str, int]]:
        """Nearest neighbours by SDR fingerprint overlap (semantic only)."""
        self._build()
        if not isinstance(self._encoders[0], SemanticEncoder):
            return []
        enc = self._encoders[0]
        w = word.lower()
        target = enc.encode(w)
        scored = [
            (tok, int(np.intersect1d(target, enc.encode(tok), assume_unique=True).size))
            for tok in enc.vocab if tok != w
        ]
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]

    def fingerprint(self, word: str) -> dict:
        """Active SDR bits for `word` (for UI visualisation).

        Returns {word, bits, dim, fitted} where `bits` is the sorted active-bit
        list and `fitted` is True if the word was seen during encoder fitting
        (vs an OOV fingerprint)."""
        self._build()
        enc = self._encoders[0] if self._encoders else self.levels[0].units[0].enc
        w = word.lower()
        fitted = bool(getattr(enc, "fp", {}).get(w) is not None) if hasattr(enc, "fp") else True
        bits = enc.encode(w)
        return {"word": w, "bits": [int(b) for b in bits],
                "dim": int(enc.dim), "fitted": fitted}

    # ── Utilities ─────────────────────────────────────────────────────────────

    # ── Capacity growth (#9 NEXT_STEPS) ──────────────────────────────────────

    def append_unit(self, unit: "CLMUnit") -> None:
        """Add a pre-trained CLMUnit to level-0 and the encoder list.

        The new unit must have the same col_dim / fp_bits as the ensemble.
        It immediately participates in predict_next() voting.  Additive and
        non-disruptive to existing units."""
        self._build()
        assert unit.col.col_dim == self.col_dim, "col_dim mismatch"
        self.levels[0].units.append(unit)
        self.n_units = len(self.levels[0].units)
        if self._encoders is not None:
            self._encoders.append(unit.enc)

    def append_level(
        self,
        stride: int,
        seed_base: int | None = None,
    ) -> None:
        """Add a new hierarchy level on top of the existing ones.

        The new level trains via the inter-level projection mechanism the next
        time train() is called.  Call after additional training data arrives."""
        self._build()
        new_lvl = self.n_levels
        if seed_base is None:
            seed_base = self.seed_base + new_lvl * 100

        self.n_levels += 1
        self.strides = self.strides + (stride,)

        self.levels.append(
            HierarchyLevel(
                level=new_lvl,
                stride=stride,
                n_units=self.n_units,
                col_dim=self.col_dim,
                w=self.fp_bits,
                encoder_factory=None,
                seed_base=seed_base,
                **self._col_kwargs,
            )
        )
        # New projection from the last existing level up to the new one
        n_cells_below = self.levels[new_lvl - 1].n_cells
        self._projections.append(
            SparseProjection(
                in_dim=n_cells_below,
                out_dim=self.col_dim,
                out_k=self.fp_bits,
                seed=new_lvl,
            )
        )

    # ── Saturation (#10 NEXT_STEPS) ───────────────────────────────────────────

    def is_saturated(self, window: int = 3, tol: float = 0.02) -> bool:
        """Return True when burst_rate has plateaued over the last `window` epochs.

        A plateau means capacity is saturated → time to call append_unit()."""
        if len(self.metrics) < window:
            return False
        rates = [m.get("burst_rate", 1.0) for m in self.metrics[-window:]]
        return (max(rates) - min(rates)) < tol

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _n_segments(self) -> int:
        return sum(u.col.n_segments for lvl in self.levels for u in lvl.units)

    def stats(self) -> dict:
        self._build()
        u0 = self.levels[0].units[0]
        return {
            "levels": self.n_levels,
            "strides": list(self.strides),
            "units_per_level": self.n_units,
            "encoder": self.encoder_type,
            # Encoder vocab, not the lazily-built SDR cache (which is empty
            # until tokens are first encoded and misreported vocab=0 in logs).
            "vocab": len(u0.enc.vocab) if u0.enc is not None else len(u0._token_sdr),
            "col_dim": self.col_dim,
            "fp_bits": self.fp_bits,
            "loc_dim": u0.grid.dim,
            "cells_per_col": u0.col.cpc,
            "segments_l0": sum(u.col.n_segments for u in self.levels[0].units),
            "synapses_l0": sum(u.col.n_synapses for u in self.levels[0].units),
            "memory_mb": sum(
                u.col.memory_mb for lvl in self.levels for u in lvl.units
            ),
            "replay_size": self.replay.size if self.replay else 0,
            "neuromod": round(self.neuromod.modulation, 3),
            "saturated": self.is_saturated(),
        }
