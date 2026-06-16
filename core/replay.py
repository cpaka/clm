"""
core/replay.py — Hippocampal episodic replay buffer.

Captures surprising sequences (high burst rate) and replays them during
offline consolidation, analogous to hippocampal sharp-wave ripple replay
in biological sleep.

Usage
-----
    buf = HippocampalBuffer(capacity=256)

    # During training: record burst rate per sequence
    buf.record(seq, burst_rate=0.7)

    # Periodically replay surprising episodes back through training
    for seq in buf.sample(n=16):
        model.train_sequence(seq)
"""
from __future__ import annotations
import random
from collections import deque


class HippocampalBuffer:
    """
    Circular buffer of high-surprise sequences.

    Only sequences whose burst rate exceeds `threshold` are retained.
    When the buffer is full the least-recent entry is evicted (FIFO).

    Parameters
    ----------
    capacity  : maximum number of stored sequences
    threshold : minimum burst rate to qualify for storage
    """

    def __init__(self, capacity: int = 512, threshold: float = 0.3):
        self.capacity = capacity
        self.threshold = threshold
        self._buf: deque[tuple[list[str], float]] = deque(maxlen=capacity)

    def record(self, sequence: list[str], burst_rate: float) -> bool:
        """
        Optionally store a sequence if it was sufficiently surprising.

        Returns True if the sequence was stored.
        """
        if burst_rate >= self.threshold:
            self._buf.append((list(sequence), burst_rate))
            return True
        return False

    def sample(self, n: int, prioritise: bool = True) -> list[list[str]]:
        """
        Return up to `n` sequences for replay.

        Parameters
        ----------
        prioritise : if True, bias sampling toward higher burst-rate episodes
                     (proportional to burst_rate weight); otherwise uniform.
        """
        if not self._buf:
            return []
        sequences, weights = zip(*self._buf)
        k = min(n, len(sequences))
        if prioritise:
            total = sum(weights)
            probs = [w / total for w in weights]
            idxs = random.choices(range(len(sequences)), weights=probs, k=k)
        else:
            idxs = random.sample(range(len(sequences)), k)
        return [list(sequences[i]) for i in idxs]

    @property
    def size(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()
