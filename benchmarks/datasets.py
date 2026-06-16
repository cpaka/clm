"""
benchmarks/datasets.py — Reproducible benchmark corpora.

Provides small, self-contained datasets so benchmarks run without external
downloads.  Each loader returns (train_sequences, test_sequences) where a
sequence is a list[str] of lowercase tokens.
"""
from __future__ import annotations
import random
import re

_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r"[.!?\n]+")


def _tokenize(text: str) -> list[list[str]]:
    return [
        toks
        for chunk in _SENT.split(text)
        if len(toks := _WORD.findall(chunk.lower())) >= 2
    ]


def _split(seqs: list[list[str]], frac: float = 0.8, seed: int = 0):
    rng = random.Random(seed)
    seqs = list(seqs)
    rng.shuffle(seqs)
    n = int(len(seqs) * frac)
    return seqs[:n], seqs[n:]


# ── 1. Deterministic grammar (perfect-learnability ceiling) ──────────────────

def grammar(n: int = 400, seed: int = 0):
    """
    A tiny context-free grammar.  A well-functioning sequence learner should
    approach 100 % top-1 because every transition is deterministic given
    enough context.
    """
    rng = random.Random(seed)
    subjects = ["the cat", "a dog", "the bird", "my friend", "the robot"]
    verbs = ["chased", "watched", "followed", "found", "ignored"]
    objects = ["the ball", "a mouse", "the car", "some food", "the light"]
    ends = ["quickly", "at night", "in the park", "without a sound", "again"]

    sents = []
    for _ in range(n):
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(ends)}."
        sents.append(s)
    return _split(_tokenize(" ".join(sents)), seed=seed)


# ── 2. Repeated motifs (memorisation + generalisation) ───────────────────────

def motifs(n: int = 300, seed: int = 0):
    """
    Fixed phrase motifs interleaved with filler.  Tests whether the model
    memorises long high-order sequences and generalises shared prefixes.
    """
    rng = random.Random(seed)
    motifs = [
        "machine learning models need large amounts of data",
        "the quick brown fox jumps over the lazy dog",
        "sparse distributed representations are robust to noise",
        "cortical columns vote on the next prediction",
    ]
    fillers = ["meanwhile", "in addition", "however", "for example", "notably"]
    sents = []
    for _ in range(n):
        m = rng.choice(motifs)
        f = rng.choice(fillers)
        sents.append(f"{f} {m}.")
    return _split(_tokenize(" ".join(sents)), seed=seed)


# ── 3. Natural text (from bundled corpus if present) ──────────────────────────

def natural(path: str = "general_training_1000_lines.txt", limit: int = 2000):
    """
    Load a natural-language corpus from disk (the bundled training file).
    Falls back to the motifs dataset if the file is missing.
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return motifs()
    seqs = _tokenize(text)[:limit]
    return _split(seqs)


REGISTRY = {
    "grammar": grammar,
    "motifs": motifs,
    "natural": natural,
}
