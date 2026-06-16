"""
Cortical-Grid Language Model — numpy-optimised implementation.

Optimisations vs reference:
  1. Synapses stored as numpy int32/float32 arrays (Segment class) instead of
     Python dicts → vectorised overlap with a reusable boolean source buffer.
  2. Per-Column active-source buffer (n_sources bools) set once per step and
     sliced via fancy-indexing; no Python loop over permanences.
  3. Connected-synapse frozenset cache for O(cells) inference (no perm scan).
  4. Parallel column units via ThreadPoolExecutor (training + inference).
"""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
import random
import numpy as np


# ---------------------------------------------------------------------------
# SDR utilities
# ---------------------------------------------------------------------------

def sdr_overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.intersect1d(a, b, assume_unique=True).size)


def make_sdr(dim: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dim, size=w, replace=False)).astype(np.int64)


_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r"[.!?\n]+")


def tokenize(text: str) -> list[list[str]]:
    seqs = []
    for chunk in _SENT.split(text or ""):
        toks = _WORD.findall(chunk.lower())
        if len(toks) >= 2:
            seqs.append(toks)
    return seqs


# ---------------------------------------------------------------------------
# Token encoder
# ---------------------------------------------------------------------------

class TokenEncoder:
    def __init__(self, dim: int = 1024, w: int = 16, seed: int = 0):
        self.dim, self.w, self.seed = dim, w, seed
        self._cache: dict[str, np.ndarray] = {}

    def _token_seed(self, token: str) -> int:
        h = hashlib.md5(f"{self.seed}:{token}".encode()).digest()
        return int.from_bytes(h[:8], "little")

    def encode(self, token: str) -> np.ndarray:
        s = self._cache.get(token)
        if s is None:
            s = make_sdr(self.dim, self.w, self._token_seed(token))
            self._cache[token] = s
        return s


# ---------------------------------------------------------------------------
# Grid location
# ---------------------------------------------------------------------------

class GridLocation:
    def __init__(self, periods=(7, 11, 13, 17, 19, 23)):
        self.periods = list(periods)
        self.offsets = np.concatenate([[0], np.cumsum(self.periods)[:-1]]).astype(np.int64)
        self.dim = int(sum(self.periods))
        self.reset()

    def reset(self):
        self.phase = np.zeros(len(self.periods), dtype=np.int64)

    def advance(self, k: int = 1):
        self.phase = (self.phase + k) % np.array(self.periods)

    def sdr(self, steps_ahead: int = 0) -> np.ndarray:
        ph = (self.phase + steps_ahead) % np.array(self.periods)
        return np.sort(self.offsets + ph).astype(np.int64)


# ---------------------------------------------------------------------------
# Numpy-backed synapse segment
# ---------------------------------------------------------------------------

class Segment:
    """Dendritic segment: parallel int32 source indices + float32 permanences."""
    __slots__ = ("sources", "perms")

    def __init__(self, source_list: list[int], init_perm: float):
        self.sources = np.array(source_list, dtype=np.int32)
        self.perms = np.full(len(source_list), init_perm, dtype=np.float32)


# ---------------------------------------------------------------------------
# Column — numpy-vectorised temporal memory
# ---------------------------------------------------------------------------

class Column:
    def __init__(self, col_dim, cells_per_col, loc_dim, seed=0,
                 connected=0.5, init_perm=0.21, inc=0.10, dec=0.02, pred_dec=0.01,
                 activation_threshold=10, min_threshold=5, new_synapses=20,
                 max_segs_per_cell=5):
        self.cpc = cells_per_col
        self.total_cells = col_dim * cells_per_col
        self.loc_dim = loc_dim
        self.n_sources = self.total_cells + loc_dim
        self.segments: dict[int, list[Segment]] = defaultdict(list)
        self.connected = np.float32(connected)
        self.init_perm = init_perm
        self.inc = np.float32(inc)
        self.dec = np.float32(dec)
        self.pred_dec = np.float32(pred_dec)
        self.activation_threshold = activation_threshold
        self.min_threshold = min_threshold
        self.new_synapses = new_synapses
        self.max_segs_per_cell = max_segs_per_cell
        self.rng = random.Random(seed)
        # Inference cache: cell -> list of frozenset(connected source ids)
        self._conn_cache: dict[int, list[frozenset]] | None = None
        # Reusable boolean buffer (avoids per-step allocation)
        self._active_buf = np.zeros(self.n_sources, dtype=bool)
        self.reset()

    def reset(self):
        self.prev_winners: set[int] = set()
        self.last_winners: set[int] = set()

    def invalidate_cache(self):
        self._conn_cache = None

    def rebuild_cache(self):
        cache: dict[int, list[frozenset]] = {}
        for cell, segs in self.segments.items():
            conn_segs = [
                frozenset(int(s) for s, p in zip(seg.sources, seg.perms)
                          if p >= self.connected)
                for seg in segs
            ]
            conn_segs = [s for s in conn_segs if s]
            if conn_segs:
                cache[cell] = conn_segs
        self._conn_cache = cache

    def _set_active(self, winners: set[int], loc_sdr) -> np.ndarray:
        buf = self._active_buf
        buf[:] = False
        for w in winners:
            if w < self.n_sources:
                buf[w] = True
        for b in loc_sdr:
            idx = self.total_cells + int(b)
            if idx < self.n_sources:
                buf[idx] = True
        return buf

    def _segment_overlaps(self, active: np.ndarray, hint_cols=None):
        """Vectorised: boolean-index each segment's sources against active buffer.

        hint_cols: if provided, only evaluate segments for cells in those columns.
        This makes training O(active_cols × cells_per_col × max_segs) instead of
        O(total_segments), eliminating the quadratic slowdown as segments accumulate.
        """
        predictive: set[int] = set()
        pot: dict[tuple, int] = {}
        conn: dict[tuple, int] = {}
        thresh = self.activation_threshold
        connected = self.connected

        if hint_cols is not None:
            for col in hint_cols:
                for ci in range(self.cpc):
                    cell = col * self.cpc + ci
                    segs = self.segments.get(cell)
                    if not segs:
                        continue
                    for si, seg in enumerate(segs):
                        m = active[seg.sources]
                        po = int(m.sum())
                        co = int((m & (seg.perms >= connected)).sum())
                        pot[(cell, si)] = po
                        conn[(cell, si)] = co
                        if co >= thresh:
                            predictive.add(cell)
        else:
            for cell, segs in self.segments.items():
                for si, seg in enumerate(segs):
                    m = active[seg.sources]
                    po = int(m.sum())
                    co = int((m & (seg.perms >= connected)).sum())
                    pot[(cell, si)] = po
                    conn[(cell, si)] = co
                    if co >= thresh:
                        predictive.add(cell)

        return predictive, pot, conn

    def _predictive_fast(self, source_set: frozenset) -> set[int]:
        if self._conn_cache is None:
            self.rebuild_cache()
        predictive: set[int] = set()
        thresh = self.activation_threshold
        for cell, conn_segs in self._conn_cache.items():
            for seg_sources in conn_segs:
                if len(seg_sources & source_set) >= thresh:
                    predictive.add(cell)
                    break
        return predictive

    def _best_segment(self, cell: int, pot: dict) -> tuple[int, int]:
        best_si, best_ov = -1, -1
        for si in range(len(self.segments.get(cell, ()))):
            ov = pot.get((cell, si), 0)
            if ov > best_ov:
                best_si, best_ov = si, ov
        return best_si, best_ov

    def _adapt(self, seg: Segment, active: np.ndarray, grow: int):
        m = active[seg.sources]
        seg.perms[m] = np.minimum(np.float32(1.0), seg.perms[m] + self.inc)
        seg.perms[~m] = np.maximum(np.float32(0.0), seg.perms[~m] - self.dec)
        if grow > 0:
            existing = set(int(s) for s in seg.sources)
            candidates = [int(i) for i in np.where(active)[0] if i not in existing]
            self.rng.shuffle(candidates)
            new_srcs = candidates[:grow]
            if new_srcs:
                seg.sources = np.concatenate([seg.sources,
                                               np.array(new_srcs, dtype=np.int32)])
                seg.perms = np.concatenate([seg.perms,
                                             np.full(len(new_srcs), self.init_perm,
                                                     dtype=np.float32)])

    def _grow_segment(self, cell: int, active: np.ndarray):
        candidates = [int(i) for i in np.where(active)[0]]
        self.rng.shuffle(candidates)
        if candidates:
            self.segments[cell].append(
                Segment(candidates[:self.new_synapses], self.init_perm)
            )

    def step(self, feature_sdr, loc_sdr, learn: bool = True):
        active_cols = [int(c) for c in feature_sdr]
        active = self._set_active(self.prev_winners, loc_sdr)
        predictive, pot, conn = self._segment_overlaps(active, hint_cols=active_cols)

        active_col_set = set(active_cols)
        winners: set[int] = set()

        for col in active_cols:
            cells = range(col * self.cpc, (col + 1) * self.cpc)
            predicted = [c for c in cells if c in predictive]
            if predicted:
                for c in predicted:
                    winners.add(c)
                    if learn:
                        si, ov = self._best_segment(c, pot)
                        if si >= 0:
                            self._adapt(self.segments[c][si], active,
                                        grow=max(0, self.new_synapses - ov))
            else:
                si_best, ov_best, cell_best = -1, -1, None
                for c in cells:
                    si, ov = self._best_segment(c, pot)
                    if ov > ov_best:
                        si_best, ov_best, cell_best = si, ov, c
                if cell_best is None:
                    cell_best = min(cells, key=lambda c: len(self.segments.get(c, ())))
                winners.add(cell_best)
                if learn:
                    if ov_best >= self.min_threshold and si_best >= 0:
                        self._adapt(self.segments[cell_best][si_best], active,
                                    grow=self.new_synapses - ov_best)
                    elif len(self.segments.get(cell_best, ())) < self.max_segs_per_cell:
                        self._grow_segment(cell_best, active)

        if learn:
            for cell in predictive:
                if (cell // self.cpc) not in active_col_set:
                    for seg in self.segments[cell]:
                        m = active[seg.sources]
                        seg.perms[m] = np.maximum(np.float32(0.0),
                                                   seg.perms[m] - self.pred_dec)
            self.invalidate_cache()

        self.prev_winners = winners
        self.last_winners = winners
        return winners

    def predict_columns(self, loc_next_sdr) -> set[int]:
        source_set = frozenset(self.last_winners) | frozenset(
            self.total_cells + int(b) for b in loc_next_sdr
        )
        return {c // self.cpc for c in self._predictive_fast(source_set)}

    def finalize(self):
        self.rebuild_cache()

    def stats(self) -> dict:
        n_seg = sum(len(v) for v in self.segments.values())
        n_syn = sum(s.sources.size for v in self.segments.values() for s in v)
        return {"segments": n_seg, "synapses": n_syn}


# ---------------------------------------------------------------------------
# Single unit: encoder + grid + column + inverted index
# ---------------------------------------------------------------------------

class CGLMUnit:
    def __init__(self, col_dim=1024, w=16, cells_per_col=8,
                 periods=(7, 11, 13, 17, 19, 23), seed=0,
                 activation_threshold=10, new_synapses=20, max_segs_per_cell=5):
        self.enc = TokenEncoder(col_dim, w, seed)
        self.grid = GridLocation(periods)
        self.col = Column(col_dim, cells_per_col, self.grid.dim, seed=seed,
                          activation_threshold=activation_threshold,
                          new_synapses=new_synapses,
                          max_segs_per_cell=max_segs_per_cell)
        self.w = w
        self.token_sdr: dict[str, np.ndarray] = {}
        self.inv: dict[int, list[str]] = defaultdict(list)

    def _sdr(self, tok: str) -> np.ndarray:
        s = self.token_sdr.get(tok)
        if s is None:
            s = self.enc.encode(tok)
            self.token_sdr[tok] = s
            for b in s:
                self.inv[int(b)].append(tok)
        return s

    def train_sequence(self, tokens: list[str]):
        self.grid.reset()
        self.col.reset()
        for tok in tokens:
            self.col.step(self._sdr(tok), self.grid.sdr(), learn=True)
            self.grid.advance(1)

    def _run_context(self, tokens: list[str]):
        self.grid.reset()
        self.col.reset()
        for tok in tokens:
            self.col.step(self._sdr(tok), self.grid.sdr(), learn=False)
            self.grid.advance(1)

    def score_next(self, tokens: list[str]) -> dict[str, float]:
        self._run_context(tokens)
        pred_cols = self.col.predict_columns(self.grid.sdr())
        scores: dict[str, float] = defaultdict(float)
        for b in pred_cols:
            for tok in self.inv.get(int(b), ()):
                scores[tok] += 1.0
        return {t: v / self.w for t, v in scores.items()}

    def finalize(self):
        self.col.finalize()


# ---------------------------------------------------------------------------
# Parallel ensemble
# ---------------------------------------------------------------------------

class CorticalGridLM:
    def __init__(self, n_columns: int = 3, **unit_kwargs):
        self.units = [CGLMUnit(seed=i, **unit_kwargs) for i in range(n_columns)]

    def _train_unit(self, unit: CGLMUnit, sequences: list, epochs: int):
        for _ in range(epochs):
            for seq in sequences:
                unit.train_sequence(seq)
        unit.finalize()

    def train(self, sequences: list, epochs: int = 1, verbose: bool = False):
        with ThreadPoolExecutor(max_workers=len(self.units)) as ex:
            futures = {
                ex.submit(self._train_unit, u, sequences, epochs): i
                for i, u in enumerate(self.units)
            }
            for f in as_completed(futures):
                if verbose:
                    print(f"  unit {futures[f]} done")
                f.result()

    def train_one_epoch(self, sequences: list):
        """Train all units for one epoch in parallel (for per-epoch metrics)."""
        def _one(unit):
            for seq in sequences:
                unit.train_sequence(seq)

        with ThreadPoolExecutor(max_workers=len(self.units)) as ex:
            list(ex.map(_one, self.units))

    def predict_next(self, tokens: list[str], topn: int = 5) -> list[tuple[str, float]]:
        votes: dict[str, float] = defaultdict(float)
        with ThreadPoolExecutor(max_workers=len(self.units)) as ex:
            for scores in ex.map(lambda u: u.score_next(tokens), self.units):
                for tok, sc in scores.items():
                    votes[tok] += sc
        return sorted(votes.items(), key=lambda kv: -kv[1])[:topn]

    def generate(self, prompt: list[str], n: int = 10) -> list[str]:
        out = list(prompt)
        for _ in range(n):
            pred = self.predict_next(out, topn=1)
            if not pred:
                break
            out.append(pred[0][0])
        return out

    def stats(self) -> dict:
        u = self.units[0]
        col_stats = u.col.stats()
        return {
            "units": len(self.units),
            "vocab": len(u.token_sdr),
            "feature_sdr_dim": u.enc.dim,
            "feature_active_bits": u.w,
            "segments_per_unit": col_stats["segments"],
            "synapses_per_unit": col_stats["synapses"],
        }


# ---------------------------------------------------------------------------
# Semantic encoder + units
# ---------------------------------------------------------------------------

class SemanticEncoder:
    def __init__(self, dim=2048, fp_bits=21, index_bits=7, window=2, seed=0):
        self.dim = dim
        self.fp_bits = fp_bits
        self.index_bits = index_bits
        self.window = window
        self.seed = seed
        self._index_cache: dict[str, np.ndarray] = {}
        self.fp: dict[str, np.ndarray] = {}

    def index_vec(self, token: str) -> np.ndarray:
        v = self._index_cache.get(token)
        if v is None:
            h = hashlib.md5(f"{self.seed}:idx:{token}".encode()).digest()
            v = make_sdr(self.dim, self.index_bits, int.from_bytes(h[:8], "little"))
            self._index_cache[token] = v
        return v

    def fit(self, sequences: list[list[str]]):
        import math
        df: dict[str, int] = defaultdict(int)
        for seq in sequences:
            for tok in seq:
                df[tok] += 1
        total = sum(df.values())
        weight = {t: math.log(1.0 + total / c) for t, c in df.items()}

        acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.dim, np.float64))
        vocab: set[str] = set()
        for seq in sequences:
            n = len(seq)
            for i, tok in enumerate(seq):
                vocab.add(tok)
                a = acc[tok]
                for j in range(max(0, i - self.window), min(n, i + self.window + 1)):
                    if j != i:
                        a[self.index_vec(seq[j])] += weight[seq[j]]

        shell_quota = self.fp_bits - self.index_bits
        for tok in vocab:
            core = self.index_vec(tok)
            core_set = set(int(b) for b in core)
            a = acc[tok]
            order = np.argsort(-a)
            shell: list[int] = []
            for b in order:
                if a[b] <= 0 or len(shell) == shell_quota:
                    break
                if int(b) not in core_set:
                    shell.append(int(b))
            bits = sorted(core_set | set(shell))
            if len(bits) < self.fp_bits:
                extra = make_sdr(self.dim, self.fp_bits, self.seed * 7919 + len(bits))
                for b in extra:
                    if int(b) not in bits:
                        bits.append(int(b))
                    if len(bits) == self.fp_bits:
                        break
            self.fp[tok] = np.sort(np.array(bits[:self.fp_bits], dtype=np.int64))

    def encode(self, token: str) -> np.ndarray:
        f = self.fp.get(token)
        if f is None:
            f = self.index_vec(token)
            self.fp[token] = f
        return f


class SemanticUnit(CGLMUnit):
    def __init__(self, encoder: SemanticEncoder, cells_per_col=8,
                 periods=(7, 11, 13, 17, 19, 23), seed=0,
                 activation_threshold=10, new_synapses=20, max_segs_per_cell=5):
        self.enc = encoder
        self.grid = GridLocation(periods)
        self.col = Column(encoder.dim, cells_per_col, self.grid.dim, seed=seed,
                          activation_threshold=activation_threshold,
                          new_synapses=new_synapses,
                          max_segs_per_cell=max_segs_per_cell)
        self.w = encoder.fp_bits
        self.token_sdr: dict[str, np.ndarray] = {}
        self.inv: dict[int, list[str]] = defaultdict(list)


class SemanticCorticalGridLM(CorticalGridLM):
    def __init__(self, n_columns=3, dim=2048, fp_bits=21, index_bits=7,
                 window=2, cells_per_col=8, periods=(7, 11, 13, 17, 19, 23),
                 activation_threshold=10, new_synapses=20, max_segs_per_cell=5):
        self.encoders = [
            SemanticEncoder(dim, fp_bits, index_bits, window, seed=i)
            for i in range(n_columns)
        ]
        self._cfg = dict(
            cells_per_col=cells_per_col,
            periods=periods,
            activation_threshold=activation_threshold,
            new_synapses=new_synapses,
            max_segs_per_cell=max_segs_per_cell,
        )
        self.units: list[SemanticUnit] = []

    def _ensure_units(self, sequences: list[list[str]]):
        if not self.units:
            with ThreadPoolExecutor(max_workers=len(self.encoders)) as ex:
                list(ex.map(lambda e: e.fit(sequences), self.encoders))
            self.units = [
                SemanticUnit(self.encoders[i], seed=i, **self._cfg)
                for i in range(len(self.encoders))
            ]

    def train(self, sequences: list, epochs: int = 1, verbose: bool = False):
        self._ensure_units(sequences)
        super().train(sequences, epochs=epochs, verbose=verbose)

    def train_one_epoch(self, sequences: list):
        self._ensure_units(sequences)
        super().train_one_epoch(sequences)


def accuracy(model: CorticalGridLM, sequences: list,
             max_probes: int = 200) -> tuple[float, float]:
    probes = []
    for seq in sequences:
        for i in range(1, len(seq)):
            probes.append((seq[:i], seq[i]))
    if not probes:
        return 0.0, 0.0
    step = max(1, len(probes) // max_probes)
    probes = probes[::step][:max_probes]
    top1 = top3 = 0
    for prefix, target in probes:
        preds = model.predict_next(prefix, topn=3)
        names = [t for t, _ in preds]
        top1 += bool(names) and names[0] == target
        top3 += target in names
    n = len(probes)
    return round(100.0 * top1 / n, 1), round(100.0 * top3 / n, 1)
