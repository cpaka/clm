"""
tests/test_persist.py — Round-trip persistence tests.
"""
from __future__ import annotations
import tempfile

import numpy as np

from core.hierarchy import HierarchicalCGLM
from persist.store import save_model, load_model


def test_save_load_roundtrip():
    seqs = [["the", "cat", "sat"], ["the", "dog", "ran"]] * 15
    model = HierarchicalCGLM(n_levels=2, strides=(1, 4), n_units=2,
                             col_dim=512, encoder="semantic", dim=512)
    model.train(seqs, epochs=5)
    before = model.predict_next(["the", "cat"], topn=3)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/model.cglm"
        save_model(model, path)
        restored = load_model(path)

    after = restored.predict_next(["the", "cat"], topn=3)
    # Predictions must be identical after reload
    assert [t for t, _ in before] == [t for t, _ in after]


def test_warm_start_training():
    """Loaded model can continue training (incremental learning)."""
    seqs = [["a", "b", "c"], ["a", "b", "d"]] * 10
    model = HierarchicalCGLM(n_levels=1, strides=(1,), n_units=1, col_dim=256,
                             encoder="random")
    model.train(seqs, epochs=3)
    segs_before = model.stats()["segments_l0"]

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/m.cglm"
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
