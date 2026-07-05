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
    RelationSet, plan, explain, ReasoningPolicy,
)
from core.displacement import Relation


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
