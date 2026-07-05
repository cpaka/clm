"""
core/vsa.py — Vector Symbolic Architecture operations over SDRs.

SDR is the *representation* (a sparse binary vector: sorted int32 active-bit
indices in a space of width ``dim``, ~``w`` active).  VSA is the *algebra* run
ON those SDRs so they can carry **structure** — roles, order, trees — instead of
just sets.  The two are complementary layers, not alternatives: SDR is the noun,
VSA is the verbs.

Why it is needed: with raw SDRs you can only measure similarity (overlap) and
represent a *set* (a bag of bits).  A set loses structure — ``union(cat,on,mat)``
equals ``union(mat,on,cat)``, so "cat on mat" and "mat on cat" collide (the
binding problem).  VSA fixes this:

    sentence = bundle( bind(SUBJECT, cat), bind(RELATION, on), bind(OBJECT, mat) )
    unbind(SUBJECT, sentence)  ->  cat        # structure is recoverable

Operations
----------
  bind    — attach a filler to a role (permutation: sparsity-preserving, reversible)
  unbind  — recover the filler bound to a role (inverse permutation)
  bundle  — superpose several SDRs into one (union, optionally k-WTA re-sparsified)
  permute — positional / sequence encoding (repeated permutation)
  similarity — overlap

Binding uses **permutation**, not dense elementwise multiplication, precisely
because permutation keeps the result ~``w``-sparse.  This is the sparse-friendly
VSA that fits HTM SDRs; the temporal-memory encoder already binds this way
(``SemanticEncoder.bind``).
"""
from __future__ import annotations
import hashlib

import numpy as np

from .sdr import overlap, kwta


# ── Permutations (roles / positions) ────────────────────────────────────────

def role_perm(dim: int, role: str, seed: int = 0) -> np.ndarray:
    """Deterministic permutation of [0, dim) for a named role."""
    h = hashlib.md5(f"{seed}:{role}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.permutation(dim)


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# ── Core operations ─────────────────────────────────────────────────────────

def bind(sdr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Bind an SDR to a role/position given by `perm` (sparsity-preserving)."""
    return np.sort(perm[sdr]).astype(np.int32)


def unbind(sdr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Invert `bind`: recover the filler bound under `perm`."""
    return np.sort(inverse_perm(perm)[sdr]).astype(np.int32)


def bundle(sdrs: list[np.ndarray], dim: int, k: int | None = None) -> np.ndarray:
    """Superpose SDRs (set union).  With `k`, keep the top-k most-agreed bits
    (k-WTA re-sparsification); with `k=None` keep every active bit."""
    acc = np.zeros(dim, dtype=np.int32)
    for s in sdrs:
        acc[s] += 1
    if k is None:
        return np.where(acc > 0)[0].astype(np.int32)
    return np.where(kwta(acc.astype(np.float64), k))[0].astype(np.int32)


def permute(sdr: np.ndarray, perm: np.ndarray, power: int = 1) -> np.ndarray:
    """Apply `perm` `power` times — encodes sequence position (ρⁿ)."""
    out = np.asarray(sdr)
    for _ in range(power):
        out = perm[out]
    return np.sort(out).astype(np.int32)


def similarity(a: np.ndarray, b: np.ndarray) -> int:
    """Overlap (sparse intersection size) — the VSA similarity measure."""
    return overlap(a, b)


# ── Named-role binder ───────────────────────────────────────────────────────

class RoleBook:
    """Cache of role permutations + inverses for symbolic (role, filler) binding."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self._fwd: dict[str, np.ndarray] = {}
        self._inv: dict[str, np.ndarray] = {}

    def _perm(self, role: str) -> np.ndarray:
        if role not in self._fwd:
            p = role_perm(self.dim, role, self.seed)
            self._fwd[role] = p
            self._inv[role] = inverse_perm(p)
        return self._fwd[role]

    def bind(self, role: str, filler: np.ndarray) -> np.ndarray:
        return bind(filler, self._perm(role))

    def unbind(self, role: str, bound: np.ndarray) -> np.ndarray:
        self._perm(role)
        return np.sort(self._inv[role][bound]).astype(np.int32)


# ── Cleanup memory (associative recall) ─────────────────────────────────────

class Codebook:
    """Associative cleanup memory: snap a noisy SDR back to the nearest stored
    symbol by overlap.  For toy scale a linear scan is fine; at scale, index by
    active bit (as CLMUnit._inv does) for sub-linear recall."""

    def __init__(self):
        self._store: dict[str, np.ndarray] = {}

    def add(self, name: str, sdr: np.ndarray) -> None:
        self._store[name] = np.asarray(sdr, dtype=np.int32)

    def get(self, name: str) -> np.ndarray:
        return self._store[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def cleanup(self, noisy: np.ndarray, topn: int = 1) -> list[tuple[str, int]]:
        """Return the `topn` nearest stored symbols as (name, overlap)."""
        noisy = np.asarray(noisy, dtype=np.int32)
        scored = sorted(
            ((name, overlap(noisy, s)) for name, s in self._store.items()),
            key=lambda kv: -kv[1],
        )
        return scored[:topn]
