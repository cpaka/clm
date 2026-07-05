"""
core/reasoning.py — Reasoning as movement through an SDR knowledge space.

Implements the roadmap in REASONING.md on top of the VSA algebra
(``core/vsa.py``) and grid reference frames (``core/displacement.py``):

  • ConceptSpace  — concepts placed at grid coordinates + feature SDRs, with an
                    associative cleanup from a coordinate SDR back to a concept.
  • walk / infer  — reasoning as a *path*: apply a relation (displacement)
                    repeatedly, cleaning up to a concept at each step, until a
                    goal is reached (transitive inference).
  • analogy       — "A is to B as C is to ?" via displacement transfer + cleanup.
  • SentenceSpace — structured language: bind words to roles, bundle into ONE
                    SDR, recover any role by unbinding.  Demonstrates why VSA is
                    needed over raw SDR union (order/roles survive).

None of this touches the cortical core; it is a thin reasoning layer that reuses
the grid, SDR, and VSA primitives.
"""
from __future__ import annotations
import numpy as np

from .vsa import RoleBook, Codebook, bundle, similarity
from .displacement import GridSpace, Relation


# ── Concept space (grid-located concepts) ────────────────────────────────────

class ConceptSpace:
    """Concepts placed at integer grid coordinates in an N-axis `GridSpace`.

    Each concept has (a) a location SDR from its coordinate and (b) a random
    feature SDR (its identity).  `at()` is the associative cleanup: a coordinate
    SDR → the concept exactly located there (or None if the coordinate is empty).
    """

    def __init__(self, n_axes: int = 1, periods: tuple[int, ...] = (2, 3, 5, 7, 11),
                 dim: int = 1024, w: int = 21, seed: int = 0):
        self.grid = GridSpace(n_axes, periods)
        self.dim = dim
        self.w = w
        self.rng = np.random.default_rng(seed)
        self.coord: dict[str, np.ndarray] = {}
        self._feat: dict[str, np.ndarray] = {}
        self._loc = Codebook()                    # location SDR → concept

    def place(self, concept: str, coord) -> None:
        self.coord[concept] = np.atleast_1d(coord).astype(np.int64)
        self._loc.add(concept, self.grid.sdr(coord))

    def feature(self, concept: str) -> np.ndarray:
        if concept not in self._feat:
            self._feat[concept] = np.sort(
                self.rng.choice(self.dim, self.w, replace=False)
            ).astype(np.int32)
        return self._feat[concept]

    def at(self, coord, exact: bool = True) -> str | None:
        """Concept located at `coord`, via SDR cleanup.  With `exact`, requires a
        full-overlap match (the coordinate is actually occupied)."""
        sdr = self.grid.sdr(coord)
        hit = self._loc.cleanup(sdr, topn=1)
        if not hit:
            return None
        name, score = hit[0]
        if exact and score < self.grid.active_bits:
            return None
        return name

    def learn_relation(self, example_pairs: list[tuple[str, str]]) -> Relation:
        """Learn a displacement relation from (source, target) concept pairs."""
        pairs = [(self.coord[a], self.coord[b]) for a, b in example_pairs]
        return Relation(self.grid).fit(pairs)


# ── Reasoning as movement (transitive inference) ─────────────────────────────

def walk(space: ConceptSpace, start: str, relation: Relation,
         max_steps: int = 16, goal: str | None = None) -> list[str]:
    """Reasoning path: repeatedly apply `relation` from `start`, cleaning up to a
    concept at each step, until `goal` is reached or the path leaves the space."""
    coord = space.coord[start].copy()
    path = [start]
    for _ in range(max_steps):
        coord = relation.apply(coord)
        name = space.at(coord)
        if name is None:
            break
        path.append(name)
        if goal is not None and name == goal:
            break
    return path


def infer_relation(space: ConceptSpace, a: str, b: str, relation: Relation,
                   max_steps: int = 16) -> int | None:
    """Transitive query: how many `relation` steps take `a` to `b` (None if not
    reachable within `max_steps`)?  `a`→`b` may never have been seen directly —
    the answer emerges from composing the learned local displacement."""
    path = walk(space, a, relation, max_steps, goal=b)
    return (len(path) - 1) if path and path[-1] == b else None


# ── Goal-steered reasoning (planning as movement) ────────────────────────────

class RelationSet:
    """A named set of relations available as reasoning moves."""

    def __init__(self):
        self._r: dict[str, Relation] = {}

    def add(self, name: str, relation: Relation) -> "RelationSet":
        self._r[name] = relation
        return self

    def items(self):
        return self._r.items()

    def names(self) -> list[str]:
        return list(self._r)


def plan(space: ConceptSpace, start: str, goal: str, relations: RelationSet,
         beam: int = 16, max_depth: int = 12,
         avoid: set[str] | None = None,
         policy: "ReasoningPolicy | None" = None) -> dict | None:
    """Goal-directed beam search: find a chain of relations from `start` to
    `goal`, combining DIFFERENT relations across steps (multi-step inference).

    Steered by the reference frame's own metric — states are expanded
    best-first by Manhattan distance to the goal in grid-coordinate space (grid
    cells encode distance, so this is reading the frame, not cheating).  `avoid`
    is the survival loop: never enter those concepts (predators).  An optional
    `policy` orders which relations to try first (learned value steering).

    Returns {'chain': [relation names], 'path': [concepts]} or None.
    """
    goal_c = np.atleast_1d(space.coord[goal]).astype(np.int64)
    avoid = set(avoid or ())

    def dist(coord) -> int:
        return int(np.abs(np.atleast_1d(coord).astype(np.int64) - goal_c).sum())

    rel_items = (policy.ordered(relations) if policy else list(relations.items()))

    start_c = space.coord[start].copy()
    frontier = [(start_c, [], [start])]
    seen = {tuple(int(x) for x in start_c)}

    for _ in range(max_depth):
        cand: list[tuple] = []
        for coord, chain, path in frontier:
            for rname, rel in rel_items:
                nc = rel.apply(coord)
                key = tuple(int(x) for x in nc)
                if key in seen:
                    continue
                name = space.at(nc)
                if name is None or name in avoid:
                    continue
                seen.add(key)
                npath, nchain = path + [name], chain + [rname]
                if name == goal:
                    return {"chain": nchain, "path": npath}
                cand.append((dist(nc), nc, nchain, npath))
        if not cand:
            break
        cand.sort(key=lambda x: x[0])                     # nearest to goal first
        frontier = [(c, ch, p) for _, c, ch, p in cand[:beam]]
    return None


def explain(result: dict) -> str:
    """Render a plan as a human-readable reasoning trace."""
    if not result:
        return "(no path found)"
    parts = [result["path"][0]]
    for rel, concept in zip(result["chain"], result["path"][1:]):
        parts += [f"--{rel}-->", concept]
    return " ".join(parts)


class ReasoningPolicy:
    """Value over relations, reinforced by successful plans — the RL/survival
    loop: moves that reach reward are preferred next time.  Steers which
    relations the planner expands first (it never changes what is *reachable*,
    only the search order / efficiency)."""

    def __init__(self, relations: RelationSet, lr: float = 0.5):
        self.lr = lr
        self.value = {name: 1.0 for name in relations.names()}

    def reinforce(self, result: dict, reward: float = 1.0) -> None:
        for name in set(result.get("chain", ())):
            self.value[name] += self.lr * reward

    def ordered(self, relations: RelationSet) -> list[tuple]:
        return sorted(relations.items(),
                      key=lambda kv: -self.value.get(kv[0], 1.0))


# ── Analogy (displacement transfer) ──────────────────────────────────────────

def analogy(space: ConceptSpace, a: str, b: str, c: str) -> str | None:
    """"a : b :: c : ?" — apply the displacement a→b to c, then clean up.

    This is the ``king - man + woman = queen`` operation, done with structured
    grid coordinates instead of dense embeddings."""
    delta = space.coord[b] - space.coord[a]
    return space.at(space.coord[c] + delta)


# ── Structured language (why VSA is needed over raw SDR) ─────────────────────

class SentenceSpace:
    """Encode role-structured sentences as a SINGLE SDR and query them by role.

    ``encode(AGENT='cat', ACTION='chased', PATIENT='mouse')`` binds each word to
    its role and bundles them; ``query(sentence, 'AGENT')`` unbinds and cleans up
    to recover ``cat``.  ``bag()`` is the raw-SDR contrast (union, no roles):
    order and roles are lost, so ``cat chased mouse`` and ``mouse chased cat``
    become identical — which is exactly the structure VSA restores.
    """

    def __init__(self, dim: int = 1024, w: int = 21, seed: int = 0):
        self.dim = dim
        self.w = w
        self.rng = np.random.default_rng(seed)
        self.roles = RoleBook(dim, seed)
        self.words = Codebook()
        self._sdr: dict[str, np.ndarray] = {}

    def word(self, w: str) -> np.ndarray:
        if w not in self._sdr:
            s = np.sort(self.rng.choice(self.dim, self.w, replace=False)).astype(np.int32)
            self._sdr[w] = s
            self.words.add(w, s)
        return self._sdr[w]

    def encode(self, **role_word: str) -> np.ndarray:
        """Structured (VSA) sentence SDR: bundle of (role⊗word) bindings."""
        bound = [self.roles.bind(role, self.word(word))
                 for role, word in role_word.items()]
        return bundle(bound, self.dim)

    def query(self, sentence_sdr: np.ndarray, role: str, topn: int = 1):
        """Recover the filler bound to `role` (unbind → cleanup)."""
        noisy = self.roles.unbind(role, sentence_sdr)
        return self.words.cleanup(noisy, topn=topn)

    def bag(self, **role_word: str) -> np.ndarray:
        """Raw-SDR union (no roles) — the structure-blind contrast."""
        return bundle([self.word(word) for word in role_word.values()], self.dim)
