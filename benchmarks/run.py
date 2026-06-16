"""
benchmarks/run.py — Benchmark runner / harness.

Run from the project root:

    python -m benchmarks.run                 # all datasets, default config
    python -m benchmarks.run --dataset motifs --epochs 15
    python -m benchmarks.run --scaling        # units-vs-accuracy scaling study

Outputs a table of accuracy, MRR, training time, and model size.
"""
from __future__ import annotations
import argparse
import time

from core.hierarchy import HierarchicalCGLM
from benchmarks.datasets import REGISTRY
from benchmarks.metrics import evaluate, perplexity_proxy


def build(n_units=3, n_levels=2, strides=(1, 4), encoder="semantic",
          col_dim=2048, **kw) -> HierarchicalCGLM:
    return HierarchicalCGLM(
        n_levels=n_levels, strides=strides, n_units=n_units,
        col_dim=col_dim, encoder=encoder, **kw
    )


def run_one(name, loader, epochs, n_units, encoder, verbose=True) -> dict:
    train, test = loader()
    model = build(n_units=n_units, encoder=encoder)

    t0 = time.time()
    model.train(train, epochs=epochs)
    train_time = time.time() - t0

    t0 = time.time()
    m = evaluate(model, test)
    eval_time = time.time() - t0

    stats = model.stats()
    row = {
        "dataset": name,
        "train_seqs": len(train),
        "test_seqs": len(test),
        "epochs": epochs,
        "units": n_units,
        "top1": m["top1"],
        "top3": m["top3"],
        "mrr": m["mrr"],
        "ppx": perplexity_proxy(model, test),
        "train_s": round(train_time, 2),
        "eval_s": round(eval_time, 2),
        "mem_mb": round(stats["memory_mb"], 1),
        "segments": stats["segments_l0"],
    }
    if verbose:
        _print_row(row)
    return row


def _print_header():
    cols = ["dataset", "units", "top1", "top3", "mrr", "ppx",
            "train_s", "mem_mb", "segments"]
    print("  ".join(f"{c:>9}" for c in cols))
    print("-" * (11 * len(cols)))


def _print_row(r):
    cols = ["dataset", "units", "top1", "top3", "mrr", "ppx",
            "train_s", "mem_mb", "segments"]
    print("  ".join(f"{str(r[c]):>9}" for c in cols))


def main():
    ap = argparse.ArgumentParser(description="CGLM benchmark harness")
    ap.add_argument("--dataset", default="all", choices=["all", *REGISTRY])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--encoder", default="semantic", choices=["semantic", "random"])
    ap.add_argument("--scaling", action="store_true",
                    help="run a units-vs-accuracy scaling study on 'motifs'")
    args = ap.parse_args()

    print(f"\nCGLM benchmark — encoder={args.encoder} epochs={args.epochs}\n")
    _print_header()

    if args.scaling:
        for u in (1, 2, 4, 8):
            run_one("motifs", REGISTRY["motifs"], args.epochs, u, args.encoder)
        return

    datasets = REGISTRY if args.dataset == "all" else {args.dataset: REGISTRY[args.dataset]}
    rows = [run_one(n, loader, args.epochs, args.units, args.encoder)
            for n, loader in datasets.items()]

    print()
    avg_top1 = sum(r["top1"] for r in rows) / len(rows)
    print(f"mean top-1 across datasets: {avg_top1:.2f}%")


if __name__ == "__main__":
    main()
