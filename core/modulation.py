"""
core/modulation.py — Neuromodulatory signal.

Models the effect of neuromodulators (dopamine / norepinephrine) on synaptic
plasticity:

  • High surprise (burst rate)  → raise plasticity (learn fast from error).
  • High confidence (prediction rate) → lower plasticity (stable representations).

The returned `modulation` scalar scales the `inc` and `dec` learning rates
inside CorticalColumn.step().  Value of 1.0 = baseline; range ~[0.2, 2.0].
"""
from __future__ import annotations
import numpy as np


class NeuromodSignal:
    """
    Running estimate of prediction quality → plasticity scalar.

    Parameters
    ----------
    window      : number of recent steps to average
    base        : baseline modulation (returned when no history)
    min_mod     : floor on modulation (prevents complete learning shutdown)
    max_mod     : ceiling on modulation
    """

    def __init__(
        self,
        window: int = 64,
        base: float = 1.0,
        min_mod: float = 0.2,
        max_mod: float = 2.0,
    ):
        self.window = window
        self.base = base
        self.min_mod = min_mod
        self.max_mod = max_mod
        self._history: list[float] = []   # 1 = burst (surprised), 0 = predicted

    def update(self, burst: bool) -> float:
        """
        Record outcome of one column step and return current modulation scalar.

        Parameters
        ----------
        burst : True if the column bursted (surprise), False if predicted correctly.

        Returns
        -------
        modulation : float in [min_mod, max_mod]
        """
        self._history.append(float(burst))
        if len(self._history) > self.window:
            self._history.pop(0)
        return self.modulation

    @property
    def modulation(self) -> float:
        """Current plasticity scalar based on recent burst rate."""
        if not self._history:
            return self.base
        burst_rate = float(np.mean(self._history))
        # Linear interpolation: burst_rate=1 → max_mod, burst_rate=0 → min_mod
        m = self.min_mod + (self.max_mod - self.min_mod) * burst_rate
        return float(np.clip(m, self.min_mod, self.max_mod))

    def reset(self) -> None:
        self._history.clear()
