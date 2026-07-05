"""
tests/test_reasoning.py — Falsifiable milestones for the reasoning layer.

These assert the three claims in REASONING.md §6:
  1. Transitive inference generalises to unseen distances (grid displacement).
  2. Analogy works by displacement transfer (king:queen :: man:?).
  3. VSA recovers structure that raw SDR union destroys (SVO order/roles).

Run:  python -m tests.test_reasoning
"""
from __future__ import annotations

import numpy as np

from core.vsa import RoleBook, Codebook, bind, unbind, bundle, permute, similarity
from core.displacement import GridSpace, Relation
from core.reasoning import (
    ConceptSpace, walk, infer_relation, analogy, SentenceSpace,
    RelationSet, plan, explain, ReasoningPolicy, sp_representations,
)
from core.displacement import Relation
from core.spatial_pooler import SpatialPooler


# ── Structured-SDR fixture: two orthogonal latent factors ────────────────────

def _factored_sdrs(dim=600, seed=0):
    """Concepts built from two latent factors (gender, royalty) on disjoint bit
    blocks, plus a unique identity block — a controlled space with known linear
    structure to test that grounding recovers it.  The factors have different
    sizes (gender > royalty) so the structure is non-degenerate, as any real
    learned representation is."""
    rng = np.random.default_rng(seed)
    G0, G1 = np.arange(0, 45), np.arange(45, 90)        # gender: male / female
    R0, R1 = np.arange(90, 110), np.arange(110, 130)    # royalty: commoner / royal
    def ident():
        return rng.choice(np.arange(130, dim), 8, replace=False)
    def make(g, r):
        return np.sort(np.concatenate([g, r, ident()])).astype(np.int32)
    return {
        "man":   make(G0, R0), "woman": make(G1, R0),
        "king":  make(G0, R1), "queen": make(G1, R1),
    }


# ── VSA primitives ───────────────────────────────────────────────────────────

def test_bind_unbind_roundtrip():
    dim, w = 512, 21
    rb = RoleBook(dim, seed=1)
    filler = np.sort(np.random.default_rng(0).choice(dim, w, replace=False)).astype(np.int32)
    bound = rb.bind("SUBJECT", filler)
    recovered = rb.unbind("SUBJECT", bound)
    assert np.array_equal(recovered, filler), "unbind must exactly invert bind"
    # A bound SDR is dissimilar to its filler (binding hides identity)
    assert similarity(bound, filler) < w // 2


def test_bind_preserves_sparsity():
    dim, w = 512, 21
    rb = RoleBook(dim, seed=2)
    s = np.sort(np.random.default_rng(3).choice(dim, w, replace=False)).astype(np.int32)
    assert rb.bind("R", s).size == w, "permutation binding must preserve w"


def test_permute_encodes_position():
    dim, w = 512, 21
    perm = np.random.default_rng(4).permutation(dim)
    s = np.sort(np.random.default_rng(5).choice(dim, w, replace=False)).astype(np.int32)
    p1, p2 = permute(s, perm, 1), permute(s, perm, 2)
    assert similarity(p1, p2) < w // 2, "different positions must differ"
    assert p1.size == w and p2.size == w


# ── Milestone 1: transitive inference ────────────────────────────────────────

def test_transitive_inference_generalises():
    """Learn a 'successor' relation from ADJACENT pairs only; infer the order of
    FAR pairs never seen together — the grid-composition payoff."""
    items = ["a", "b", "c", "d", "e", "f", "g"]
    space = ConceptSpace(n_axes=1, dim=1024, w=21, seed=0)
    for i, it in enumerate(items):
        space.place(it, i)

    # Train only on a subset of adjacent pairs
    succ = space.learn_relation([("a", "b"), ("c", "d"), ("e", "f")])

    # Held-out far pairs: distances the relation was never shown
    assert infer_relation(space, "b", "e", succ) == 3      # b<c<d<e, never seen
    assert infer_relation(space, "a", "g", succ) == 6
    assert infer_relation(space, "c", "f", succ) == 3
    # Backwards is NOT reachable by the (forward) successor relation
    assert infer_relation(space, "f", "c", succ) is None


def test_relation_composition():
    space = ConceptSpace(n_axes=1, dim=512, w=21, seed=1)
    for i, it in enumerate("abcde"):
        space.place(it, i)
    succ = space.learn_relation([("a", "b")])
    grandsucc = succ.compose(succ)                 # +2
    assert space.at(grandsucc.apply(space.coord["a"])) == "c"


# ── Milestone 2: analogy ─────────────────────────────────────────────────────

def _royalty_space():
    # 2-axis grid: axis0 = gender (0 male / 1 female), axis1 = status (0 / 1 royal)
    space = ConceptSpace(n_axes=2, dim=1024, w=21, seed=0)
    space.place("man", (0, 0))
    space.place("woman", (1, 0))
    space.place("king", (0, 1))
    space.place("queen", (1, 1))
    return space


def test_analogy_king_queen():
    space = _royalty_space()
    assert analogy(space, "man", "king", "woman") == "queen"     # +royal
    assert analogy(space, "man", "woman", "king") == "queen"     # +female
    assert analogy(space, "king", "man", "queen") == "woman"     # -royal


# ── Milestone 2b: goal-steered planning (multi-step, multi-relation) ─────────

def _royalty_relations(space):
    rels = RelationSet()
    rels.add("make_royal", space.learn_relation([("man", "king")]))    # (0,+1)
    rels.add("make_female", space.learn_relation([("man", "woman")]))  # (+1,0)
    return rels


def test_plan_combines_relations():
    """Reach a goal that needs TWO different relations chained — real multi-step
    inference, not repetition of one move."""
    space = _royalty_space()
    rels = _royalty_relations(space)
    res = plan(space, "man", "queen", rels)
    assert res is not None and res["path"][-1] == "queen"
    assert len(res["chain"]) == 2
    assert set(res["chain"]) == {"make_royal", "make_female"}


def test_plan_transitive_chain():
    space = ConceptSpace(n_axes=1, dim=1024, w=21, seed=0)
    for i, it in enumerate("abcdef"):
        space.place(it, i)
    rels = RelationSet().add("succ", space.learn_relation([("a", "b")]))
    res = plan(space, "a", "f", rels)
    assert res["path"] == ["a", "b", "c", "d", "e", "f"]
    assert res["chain"] == ["succ"] * 5


def test_plan_avoids_predator():
    """Survival loop: route from prey-hunter start to prey, avoiding a predator
    sitting on the direct lattice path."""
    space = ConceptSpace(n_axes=2, dim=1024, w=21, seed=0)
    grid = {}
    for x in range(3):
        for y in range(3):
            name = f"c{x}{y}"
            space.place(name, (x, y))
            grid[(x, y)] = name
    rels = RelationSet()
    rels.add("east", Relation(space.grid, [1, 0]))
    rels.add("north", Relation(space.grid, [0, 1]))
    predator = grid[(1, 1)]
    res = plan(space, "c00", "c22", rels, avoid={predator})
    assert res is not None and res["path"][-1] == "c22"
    assert predator not in res["path"], "path must avoid the predator"


def test_policy_reinforcement_steers():
    space = _royalty_space()
    rels = _royalty_relations(space)
    policy = ReasoningPolicy(rels)
    before = dict(policy.value)
    res = plan(space, "man", "queen", rels, policy=policy)
    policy.reinforce(res, reward=1.0)
    # Relations that were used gain value; the loop learns which moves pay off.
    for name in res["chain"]:
        assert policy.value[name] > before[name]


# ── Milestone 4: coordinates GROUNDED in learned representations ─────────────

def _coord_dist(space, a, b):
    return int(np.abs(space.coord[a] - space.coord[b]).sum())


def test_ground_recovers_latent_structure():
    """Coordinates discovered from structured SDRs (not hand-placed) recover the
    two latent factors, so displacement analogy works on learned coordinates."""
    sdrs = _factored_sdrs()
    space = ConceptSpace(n_axes=2, dim=600, seed=0).ground(sdrs)
    # queen shares no factor with man → it must be the farthest concept
    dists = {c: _coord_dist(space, "man", c) for c in ("woman", "king", "queen")}
    assert dists["queen"] == max(dists.values())
    # analogy on the LEARNED coordinates
    assert analogy(space, "man", "king", "woman") == "queen"


def test_ground_from_spatial_pooler_preserves_similarity():
    """Full pipeline: structured inputs → Spatial Pooler learned SDRs → grounded
    coordinates.  Grounding preserves the learned *nearest neighbour*: whatever
    the SP decides is most similar to a concept ends up closest in coordinate
    space.  (Which factors the SP captures is its own business — the point is
    that the coordinates reflect the *learned* representation, not hand-placed
    structure.)"""
    from core.sdr import overlap
    inputs = _factored_sdrs(dim=600, seed=1)
    sp = SpatialPooler(input_dim=600, col_dim=512, active_cols=21, seed=0)
    for _ in range(20):                                   # fit
        for s in inputs.values():
            sp.compute(s, learn=True)
    sp.freeze()

    reps = sp_representations(sp, inputs)
    assert all(r.size == 21 for r in reps.values())       # learned SDRs
    space = ConceptSpace(n_axes=2, dim=512, seed=0).ground(reps)

    for anchor in inputs:
        others = [c for c in inputs if c != anchor]
        sp_nearest = max(others, key=lambda c: overlap(reps[anchor], reps[c]))
        grounded_nearest = min(others, key=lambda c: _coord_dist(space, anchor, c))
        assert grounded_nearest == sp_nearest, (
            f"{anchor}: SP-nearest {sp_nearest} but grounded-nearest {grounded_nearest}")


# ── Milestone 3: VSA recovers structure raw SDR loses ────────────────────────

def test_svo_role_recovery():
    ss = SentenceSpace(dim=2048, w=21, seed=0)
    sent = ss.encode(AGENT="cat", ACTION="chased", PATIENT="mouse")
    assert ss.query(sent, "AGENT")[0][0] == "cat"
    assert ss.query(sent, "ACTION")[0][0] == "chased"
    assert ss.query(sent, "PATIENT")[0][0] == "mouse"


def test_vsa_beats_raw_union_on_word_order():
    """The killer demo: 'cat chased mouse' vs 'mouse chased cat'.
    Raw SDR union makes them identical; VSA keeps them distinct."""
    ss = SentenceSpace(dim=2048, w=21, seed=0)
    s1 = ss.encode(AGENT="cat", ACTION="chased", PATIENT="mouse")
    s2 = ss.encode(AGENT="mouse", ACTION="chased", PATIENT="cat")
    b1 = ss.bag(AGENT="cat", ACTION="chased", PATIENT="mouse")
    b2 = ss.bag(AGENT="mouse", ACTION="chased", PATIENT="cat")

    raw = similarity(b1, b2) / b1.size                 # raw union: ~identical
    vsa = similarity(s1, s2) / s1.size                 # VSA: only ACTION shared
    assert raw > 0.95, f"raw union should collapse order, got {raw:.2f}"
    assert vsa < 0.6, f"VSA should separate order, got {vsa:.2f}"
    assert vsa < raw - 0.3


# ── Manual runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
