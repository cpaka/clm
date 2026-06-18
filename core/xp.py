"""numpy / cupy backend shim.

Import `xp` here instead of numpy in compute-heavy modules.  When cupy is
available (Modal GPU worker) all array ops execute on the GPU.  When it is not
(local CPU, CI) the shim is a transparent no-op and `xp is numpy`.
"""
from __future__ import annotations
import numpy as _np

try:
    import cupy as _cp          # type: ignore[import-untyped]
    _cp.cuda.runtime.getDeviceCount()   # raises CUDARuntimeError if no GPU driver
    xp = _cp
    _GPU = True

    def asnumpy(a):
        """Device-to-host: cupy → numpy (or numpy → numpy no-op)."""
        return xp.asnumpy(a) if isinstance(a, xp.ndarray) else _np.asarray(a)

    def asxp(a):
        """Host-to-device: numpy → cupy (or cupy → cupy no-op)."""
        return a if isinstance(a, xp.ndarray) else xp.asarray(a)

except Exception:
    xp = _np                   # type: ignore[assignment]
    _GPU = False

    def asnumpy(a):            # type: ignore[misc]
        return _np.asarray(a)

    def asxp(a):               # type: ignore[misc]
        return _np.asarray(a)

__all__ = ["xp", "_GPU", "asnumpy", "asxp"]
