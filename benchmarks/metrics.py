"""
benchmarks/metrics.py — Evaluation metrics for next-token prediction.
"""
from __future__ import annotations
import math


def evaluate(model, test_seqs: list[list[str]], topn: int = 3) -> dict:
    """
    Compute top-1 / top-3 accuracy and mean reciprocal rank over all
    next-token transitions in the test set.

    Returns dict with: transitions, top1, top3, mrr.
    """
    top1 = top3 = total = 0
    rr_sum = 0.0
    for seq in test_seqs:
        for i in range(1, len(seq)):
            preds = model.predict_next(seq[:i], topn=max(topn, 10))
            names = [t for t, _ in preds]
            target = seq[i]
            total += 1
            if names and names[0] == target:
                top1 += 1
            if target in names[:topn]:
                top3 += 1
            if target in names:
                rr_sum += 1.0 / (names.index(target) + 1)
    if total == 0:
        return {"transitions": 0, "top1": 0.0, "top3": 0.0, "mrr": 0.0}
    return {
        "transitions": total,
        "top1": round(100.0 * top1 / total, 2),
        "top3": round(100.0 * top3 / total, 2),
        "mrr": round(rr_sum / total, 4),
    }


def accuracy(model, sequences: list[list[str]], max_probes: int = 200) -> tuple[float, float]:
    """
    Probe-capped top-1 / top-3 accuracy for fast in-the-loop evaluation
    (used by the Modal training job).  Subsamples transitions to `max_probes`.
    """
    probes = [
        (seq[:i], seq[i])
        for seq in sequences
        for i in range(1, len(seq))
    ]
    if not probes:
        return 0.0, 0.0
    step = max(1, len(probes) // max_probes)
    probes = probes[::step][:max_probes]
    top1 = top3 = 0
    for prefix, target in probes:
        names = [t for t, _ in model.predict_next(prefix, topn=3)]
        top1 += bool(names) and names[0] == target
        top3 += target in names
    n = len(probes)
    return round(100.0 * top1 / n, 1), round(100.0 * top3 / n, 1)


def perplexity_proxy(model, test_seqs: list[list[str]]) -> float:
    """
    A perplexity-style proxy from reciprocal-rank scores (lower is better).
    Not a true probabilistic perplexity — the model is not normalised — but
    monotonic with predictive quality, useful for relative comparisons.
    """
    log_sum = 0.0
    total = 0
    for seq in test_seqs:
        for i in range(1, len(seq)):
            preds = model.predict_next(seq[:i], topn=50)
            names = [t for t, _ in preds]
            rank = names.index(seq[i]) + 1 if seq[i] in names else len(names) + 1
            log_sum += math.log(rank)
            total += 1
    return round(math.exp(log_sum / total), 2) if total else float("inf")
