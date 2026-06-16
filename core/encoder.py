"""
core/encoder.py — Token-to-SDR encoding.

Two encoders are provided:

TokenEncoder (random)
    Each token maps to a stable, deterministic random SDR.  Near-orthogonal
    by construction; no corpus required.  Fast, works for small data.

SemanticEncoder (distributional)
    Each token gets an identity core (random, unique) plus a semantic shell
    grown by IDF-weighted context folding (random indexing over a window).
    Similar words share shell bits → prediction generalises across paraphrases.
    Requires a corpus to call .fit() before use.
"""
from __future__ import annotations
import hashlib
import math
from collections import defaultdict

import numpy as np

from .sdr import make_sdr


# ── Random encoder ────────────────────────────────────────────────────────────

class TokenEncoder:
    """
    Deterministic random SDR per token (hash-seeded).

    Parameters
    ----------
    dim  : SDR width
    w    : number of active bits (~2% of dim)
    seed : global salt for reproducibility
    """

    def __init__(self, dim: int = 2048, w: int = 21, seed: int = 0):
        self.dim, self.w, self.seed = dim, w, seed
        self._cache: dict[str, np.ndarray] = {}

    def _token_seed(self, token: str) -> int:
        h = hashlib.md5(f"{self.seed}:{token}".encode()).digest()
        return int.from_bytes(h[:8], "little")

    def encode(self, token: str) -> np.ndarray:
        """Return sorted sparse SDR for `token` (int32)."""
        s = self._cache.get(token)
        if s is None:
            s = make_sdr(self.dim, self.w, self._token_seed(token))
            self._cache[token] = s
        return s

    @property
    def vocab(self) -> list[str]:
        return list(self._cache)


# ── Semantic encoder ──────────────────────────────────────────────────────────

class SemanticEncoder:
    """
    Distributional fingerprint encoder (IDF-weighted random indexing).

    Each token's fingerprint = identity core (index SDR) ∪ semantic shell
    (top context bits from co-occurrence accumulation).

    Parameters
    ----------
    dim        : SDR width
    fp_bits    : total active bits per fingerprint
    index_bits : identity-core size (subset of fp_bits)
    window     : co-occurrence window radius
    seed       : global salt
    """

    def __init__(
        self,
        dim: int = 2048,
        fp_bits: int = 21,
        index_bits: int = 7,
        window: int = 2,
        seed: int = 0,
    ):
        self.dim = dim
        self.fp_bits = fp_bits
        self.index_bits = index_bits
        self.window = window
        self.seed = seed
        self._index_cache: dict[str, np.ndarray] = {}
        self._perm_cache: dict[str, np.ndarray] = {}
        self.fp: dict[str, np.ndarray] = {}

    # ── Index (identity) SDRs ─────────────────────────────────────────────

    def index_vec(self, token: str) -> np.ndarray:
        """Deterministic sparse index SDR for `token` (int32)."""
        v = self._index_cache.get(token)
        if v is None:
            h = hashlib.md5(f"{self.seed}:idx:{token}".encode()).digest()
            v = make_sdr(self.dim, self.index_bits, int.from_bytes(h[:8], "little"))
            self._index_cache[token] = v
        return v

    # ── Corpus fitting ────────────────────────────────────────────────────

    def fit(self, sequences: list[list[str]]) -> None:
        """Build semantic fingerprints from a tokenised corpus."""
        # IDF weights — rare context carries more signal
        df: dict[str, int] = defaultdict(int)
        for seq in sequences:
            for tok in seq:
                df[tok] += 1
        total = sum(df.values())
        weight = {t: math.log(1.0 + total / c) for t, c in df.items()}

        # Accumulate context into a float accumulator per token
        acc: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.dim, np.float64)
        )
        vocab: set[str] = set()
        for seq in sequences:
            n = len(seq)
            for i, tok in enumerate(seq):
                vocab.add(tok)
                a = acc[tok]
                lo = max(0, i - self.window)
                hi = min(n, i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        iv = self.index_vec(seq[j])
                        a[iv] += weight[seq[j]]

        # Build fingerprints
        shell_quota = self.fp_bits - self.index_bits
        for tok in vocab:
            core = self.index_vec(tok)
            core_set = set(core.tolist())
            a = acc[tok]
            order = np.argsort(-a)

            shell: list[int] = []
            for b in order:
                if a[b] <= 0 or len(shell) == shell_quota:
                    break
                if int(b) not in core_set:
                    shell.append(int(b))

            bits = sorted(core_set | set(shell))

            # Rare tokens: pad with random bits
            if len(bits) < self.fp_bits:
                pad = make_sdr(self.dim, self.fp_bits, self.seed * 7919 + len(bits))
                for b in pad:
                    if int(b) not in set(bits):
                        bits.append(int(b))
                    if len(bits) == self.fp_bits:
                        break

            self.fp[tok] = np.sort(np.array(bits[: self.fp_bits], dtype=np.int32))

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode(self, token: str) -> np.ndarray:
        """Return sorted sparse fingerprint of exactly `fp_bits` active bits.

        For OOV tokens (not seen during fit) we build a deterministic full-width
        fingerprint from the identity core padded with hashed bits — same width
        as fitted words so feature SDRs stay uniform (required for stacking,
        scoring, and persistence)."""
        f = self.fp.get(token)
        if f is None:
            bits = set(int(b) for b in self.index_vec(token))     # identity core
            h = hashlib.md5(f"{self.seed}:oov:{token}".encode()).digest()
            pad = make_sdr(self.dim, self.fp_bits, int.from_bytes(h[:8], "little"))
            for b in pad:
                if len(bits) >= self.fp_bits:
                    break
                bits.add(int(b))
            f = np.sort(np.array(sorted(bits)[: self.fp_bits], dtype=np.int32))
            self.fp[token] = f
        return f

    # ── Structural encodings (collocations, role binding) ─────────────────

    def encode_collocation(self, w1: str, w2: str) -> np.ndarray:
        """Order-insensitive 2-word fingerprint (union, re-sparsified)."""
        a, b = self.encode(w1), self.encode(w2)
        score: dict[int, int] = defaultdict(int)
        for s in a:
            score[int(s)] += 1
        for s in b:
            score[int(s)] += 1
        ranked = sorted(score, key=lambda k: (-score[k], k))
        return np.sort(np.array(ranked[: self.fp_bits], dtype=np.int32))

    def _perm(self, role: str) -> np.ndarray:
        p = self._perm_cache.get(role)
        if p is None:
            h = hashlib.md5(f"{self.seed}:perm:{role}".encode()).digest()
            p = np.random.default_rng(int.from_bytes(h[:8], "little")).permutation(self.dim)
            self._perm_cache[role] = p
        return p

    def bind(self, role: str, word: str) -> np.ndarray:
        """Role-filler binding via deterministic permutation (VSA)."""
        return np.sort(self._perm(role)[self.encode(word)])

    @property
    def vocab(self) -> list[str]:
        return list(self.fp)
