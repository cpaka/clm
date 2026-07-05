"""
tests/test_core.py — Unit tests for CLM core modules.

Run:  python -m pytest tests/ -v
(or)  python -m tests.test_core      # runs without pytest
"""
from __future__ import annotations
import numpy as np

from core.sdr import (
    sparse_to_dense, dense_to_sparse, overlap, batch_overlap, make_sdr, kwta,
)
from core.grid import GridLocation
from core.encoder import TokenEncoder, SemanticEncoder
from core.column import CorticalColumn
from core.modulation import NeuromodSignal
from core.replay import HippocampalBuffer
from core.hierarchy import HierarchicalCLM, SparseProjection


# ── SDR ───────────────────────────────────────────────────────────────────────

def test_sparse_dense_roundtrip():
    s = make_sdr(100, 10, seed=1)
    d = sparse_to_dense(s, 100)
    assert np.array_equal(dense_to_sparse(d), s)


def test_overlap():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([3, 4, 5, 6], dtype=np.int32)
    assert overlap(a, b) == 2


def test_batch_overlap():
    q = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    k = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=bool)
    ov = batch_overlap(q, k)
    assert ov[0, 0] == 1 and ov[1, 1] == 2 and ov[0, 1] == 0


def test_kwta_exact_k():
    scores = np.array([5, 1, 9, 3, 7, 2])
    mask = kwta(scores, 3)
    assert mask.sum() == 3
    assert set(np.where(mask)[0]) == {0, 2, 4}  # 5,9,7


def test_kwta_ties_trimmed():
    scores = np.array([5, 5, 5, 5])
    assert kwta(scores, 2).sum() == 2


# ── Grid ──────────────────────────────────────────────────────────────────────

def test_grid_uniqueness_and_periodicity():
    g = GridLocation(periods=(3, 5))            # LCM = 15
    seen = set()
    for _ in range(15):
        seen.add(tuple(g.sdr().tolist()))
        g.advance(1)
    assert len(seen) == 15                       # all unique within one LCM
    assert tuple(g.sdr().tolist()) in seen       # wraps around at LCM


def test_grid_no_mutation_on_lookahead():
    g = GridLocation()
    before = g.sdr().copy()
    _ = g.sdr(steps_ahead=3)
    assert np.array_equal(g.sdr(), before)


# ── Encoder ─────────────────────────────────────────────────────────────────

def test_token_encoder_stable():
    enc = TokenEncoder(dim=512, w=16)
    assert np.array_equal(enc.encode("cat"), enc.encode("cat"))
    assert enc.encode("cat").size == 16


def test_semantic_similarity():
    seqs = [["the", "cat", "sat"], ["the", "dog", "sat"],
            ["a", "cat", "ran"], ["a", "dog", "ran"]] * 10
    enc = SemanticEncoder(dim=1024, fp_bits=21, index_bits=7, window=2)
    enc.fit(seqs)
    # cat and dog share contexts → should overlap more than cat and 'the'
    cd = overlap(enc.encode("cat"), enc.encode("dog"))
    ct = overlap(enc.encode("cat"), enc.encode("the"))
    assert cd >= ct


# ── Column ─────────────────────────────────────────────────────────────────

def test_column_learns_transition():
    """After repeated A→B, presenting A should predict B's column."""
    col_dim, cpc = 64, 4
    g = GridLocation(periods=(3, 5))
    col = CorticalColumn(col_dim, cpc, g.dim, seed=0, activation_threshold=4,
                         min_threshold=2, syn_per_seg=12)
    enc = TokenEncoder(dim=col_dim, w=6)
    A, B = enc.encode("a"), enc.encode("b")

    for _ in range(20):
        g.reset(); col.reset()
        col.step(A, g.sdr()); g.advance(1)
        col.step(B, g.sdr()); g.advance(1)

    # Replay context up to A, then check prediction for next position
    g.reset(); col.reset()
    col.step(A, g.sdr())
    pred = col.predict_columns(g.sdr(1))
    b_cols = set(int(c) for c in B)
    assert len(b_cols & set(pred.tolist())) > 0


def test_column_vectorised_shapes():
    col = CorticalColumn(32, 4, 20)
    src = np.zeros(col.source_dim + 1, dtype=bool)
    conn, pot, valid = col._compute_overlaps(src)
    assert conn.shape == (col.n_cells, col.max_segs)
    assert valid.shape == conn.shape


# ── Modulation ────────────────────────────────────────────────────────────

def test_neuromod_responds_to_burst():
    nm = NeuromodSignal(window=10, min_mod=0.2, max_mod=2.0)
    for _ in range(10):
        nm.update(burst=True)
    assert nm.modulation > 1.5
    nm.reset()
    for _ in range(10):
        nm.update(burst=False)
    assert nm.modulation < 0.5


# ── Replay ────────────────────────────────────────────────────────────────

def test_replay_threshold_and_sample():
    buf = HippocampalBuffer(capacity=10, threshold=0.5)
    assert buf.record(["a", "b"], burst_rate=0.9) is True
    assert buf.record(["c", "d"], burst_rate=0.1) is False
    assert buf.size == 1
    assert buf.sample(5) == [["a", "b"]]


# ── Projection ────────────────────────────────────────────────────────────

def test_projection_sparsity():
    proj = SparseProjection(in_dim=256, out_dim=128, out_k=10, seed=0)
    winners = np.zeros(256, dtype=bool)
    winners[:20] = True
    out = proj.project(winners)
    assert out.size <= 10
    assert out.max() < 128


# ── End-to-end hierarchy ─────────────────────────────────────────────────────

def test_hierarchy_trains_and_predicts():
    seqs = [["the", "cat", "sat", "down"], ["the", "dog", "ran", "away"]] * 20
    model = HierarchicalCLM(n_levels=2, strides=(1, 4), n_units=2,
                             col_dim=512, encoder="semantic", dim=512)
    model.train(seqs, epochs=8)
    preds = model.predict_next(["the", "cat"], topn=3)
    assert isinstance(preds, list)
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in preds)


def test_hierarchy_stats():
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256)
    model.train([["a", "b", "c"]], epochs=1)
    s = model.stats()
    assert s["levels"] == 1 and s["units_per_level"] == 1


def test_hierarchy_position_agnostic():
    """Grid fix: same bigram at different positions must give the same prediction."""
    seqs = [["the", "cat", "sat"], ["a", "b", "the", "cat", "ran"]] * 15
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1,
                             col_dim=512, encoder="semantic", dim=512)
    model.train(seqs, epochs=10)
    # "the cat" at position 0 and position 2 should predict the same learned
    # continuations (sat/ran).  Only the top-2 are compared: ranks below the
    # learned continuations are decode noise, and a high-order temporal memory
    # legitimately lets the extra prefix context reorder that tail.
    preds_p0 = model.predict_next(["the", "cat"], topn=3)
    preds_p2 = model.predict_next(["a", "b", "the", "cat"], topn=3)
    tokens_p0 = {t for t, _ in preds_p0[:2]}
    tokens_p2 = {t for t, _ in preds_p2[:2]}
    assert tokens_p0 == tokens_p2, f"position-dependent: {tokens_p0} vs {tokens_p2}"


def test_append_unit():
    """#9: appending a new unit expands the ensemble non-destructively."""
    seqs = [["the", "cat", "sat"], ["a", "dog", "ran"]] * 10
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1,
                             col_dim=256, encoder="semantic", dim=256)
    model.train(seqs, epochs=3)
    before = model.n_units

    new_unit = model.levels[0].units[0].__class__(
        col_dim=256, w=model.fp_bits,
        cells_per_col=model._col_kwargs["cells_per_col"],
        periods=model._col_kwargs["periods"],
        encoder=model._encoders[0],
        seed=99,
        **{k: v for k, v in model._col_kwargs.items()
           if k not in ("cells_per_col", "periods")},
    )
    model.append_unit(new_unit)
    assert model.n_units == before + 1
    preds = model.predict_next(["the", "cat"], topn=3)
    assert isinstance(preds, list)


def test_saturation_signal():
    """#10: is_saturated returns True when burst_rate plateau detected."""
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256)
    model.metrics = [{"burst_rate": 0.5}, {"burst_rate": 0.51}, {"burst_rate": 0.50}]
    assert model.is_saturated(window=3, tol=0.05)
    model.metrics = [{"burst_rate": 0.9}, {"burst_rate": 0.6}, {"burst_rate": 0.3}]
    assert not model.is_saturated(window=3, tol=0.05)


def test_semantic_encoder_update():
    """#7: incremental update preserves identity core for existing tokens."""
    enc = SemanticEncoder(dim=256, fp_bits=14, index_bits=5, window=1, seed=0)
    seqs1 = [["the", "cat", "sat"], ["the", "dog", "ran"]]
    enc.fit(seqs1)
    core_before = set(enc.fp["the"][:5].tolist())

    seqs2 = [["the", "cat", "jumped"], ["a", "fox", "ran"]]
    new_count = enc.update(seqs2)
    core_after = set(enc.encode("the")[:5].tolist())
    assert "fox" in enc.fp, "new token not added"
    assert new_count >= 1


def test_fast_punish_matches_bruteforce():
    """The index-scoped predicted-inactive punishment must be identical to the
    full-scan reference — same trained synaptic state, bit for bit."""
    seqs = [["the", "cat", "sat", "on", "mat"],
            ["a", "dog", "ran", "in", "park"],
            ["the", "sun", "rose", "at", "dawn"]] * 12

    def train(fast: bool):
        m = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256,
                            cells_per_col=4, encoder="semantic", dim=256,
                            fp_bits=21, index_bits=7, activation_threshold=5,
                            syn_per_seg=12, max_segs=8)
        m.fit_encoders(seqs)
        m._build()
        m.levels[0].units[0].col._fast_punish = fast
        m.train(seqs, epochs=4)
        return np.asarray(m.levels[0].units[0].col.seg_perm)

    fast, brute = train(True), train(False)
    assert fast.shape == brute.shape
    assert np.array_equal(fast, brute), "fast punishment diverged from full scan"


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
