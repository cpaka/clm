"""
core/grid.py — Multi-scale grid-cell path integrator.

A bank of ring oscillators with co-prime periods.  Advancing by 1 shifts
every ring by 1 (path integration).  The union of active phases is a sparse
location SDR that is unique over LCM(periods) positions — the same
multi-scale uniqueness trick used by entorhinal grid cells.

The `dim` of the location SDR equals sum(periods).
"""
from __future__ import annotations
import numpy as np


class GridLocation:
    """
    Multi-scale location coder via co-prime ring modules.

    Parameters
    ----------
    periods : tuple[int, ...]
        Ring sizes; should be mutually co-prime.  Their LCM is the number
        of uniquely representable positions.

    Example
    -------
    >>> g = GridLocation()
    >>> g.advance(3)
    >>> g.sdr()          # sorted int32 array of 6 active bits
    """

    def __init__(self, periods: tuple[int, ...] = (7, 11, 13, 17, 19, 23)):
        self.periods = np.array(periods, dtype=np.int32)
        self.offsets = np.concatenate(
            [[0], np.cumsum(self.periods[:-1])]
        ).astype(np.int32)
        self.dim: int = int(self.periods.sum())
        self._phase = np.zeros(len(periods), dtype=np.int32)

    # ── State ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Return to position 0 on all rings."""
        self._phase[:] = 0

    def advance(self, k: int = 1) -> None:
        """Path-integrate k steps forward."""
        self._phase = (self._phase + k) % self.periods

    # ── Encoding ─────────────────────────────────────────────────────────

    def sdr(self, steps_ahead: int = 0) -> np.ndarray:
        """Sparse location SDR (sorted int32) at current + steps_ahead.
        Does NOT mutate state."""
        ph = (self._phase + steps_ahead) % self.periods
        return np.sort(self.offsets + ph).astype(np.int32)

    def dense(self, steps_ahead: int = 0) -> np.ndarray:
        """Dense bool vector at current + steps_ahead."""
        v = np.zeros(self.dim, dtype=np.bool_)
        v[self.sdr(steps_ahead)] = True
        return v
