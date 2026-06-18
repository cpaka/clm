"""
tests/test_persist.py — Round-trip persistence tests.
"""
from __future__ import annotations
import tempfile

import numpy as np

from core.hierarchy import HierarchicalCLM
from persist.store import save_model, load_model


def test_save_load_roundtrip():
    seqs = [["the", "cat", "sat"], ["the", "dog", "ran"]] * 15
    model = HierarchicalCLM(n_levels=2, strides=(1, 4), n_units=2,
                             col_dim=512, encoder="semantic", dim=512)
    model.train(seqs, epochs=5)
    before = model.predict_next(["the", "cat"], topn=3)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/model.clm"
        save_model(model, path)
        restored = load_model(path)

    after = restored.predict_next(["the", "cat"], topn=3)
    # Predictions must be identical after reload
    assert [t for t, _ in before] == [t for t, _ in after]


def test_replay_buffer_persisted():
    """#8: replay episodes survive save/load."""
    seqs = [["the", "cat", "sat"], ["a", "dog", "ran"]] * 20
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256,
                             encoder="semantic", dim=256, replay_cap=64, replay_thresh=0.0)
    model.train(seqs, epochs=3)
    size_before = model.replay.size

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.clm"
        save_model(model, path)
        restored = load_model(path)

    assert restored.replay is not None
    assert restored.replay.size == size_before


def test_encoder_accumulator_persisted():
    """#7: encoder accumulator survives save/load, enabling update()."""
    seqs = [["the", "cat", "sat"], ["a", "dog", "ran"]] * 15
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256,
                             encoder="semantic", dim=256)
    model.train(seqs, epochs=3)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.clm"
        save_model(model, path)
        restored = load_model(path)

    enc = restored.levels[0].units[0].enc
    assert hasattr(enc, "_acc"), "accumulator not restored"
    new_count = enc.update([["the", "cat", "jumped", "high"]])
    assert "jumped" in enc.fp


def test_warm_start_training():
    """Loaded model can continue training (incremental learning)."""
    seqs = [["a", "b", "c"], ["a", "b", "d"]] * 10
    model = HierarchicalCLM(n_levels=1, strides=(1,), n_units=1, col_dim=256,
                             encoder="random")
    model.train(seqs, epochs=3)
    segs_before = model.stats()["segments_l0"]

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.clm"
        save_model(model, path)
        restored = load_model(path)
        restored.train([["a", "b", "e"]] * 5, epochs=3)

    assert restored.stats()["segments_l0"] >= segs_before


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception:
                print(f"  FAIL  {name}")
                traceback.print_exc()
