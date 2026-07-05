"""
core/spatial_pooler.py — HTM Spatial Pooler (learned SDR representations).

The missing first half of HTM.  The temporal memory in ``core/column.py``
learns *sequences*; the Spatial Pooler learns *representations* — a stable,
distributed, semantically-organised mapping from an input pattern to a sparse
set of active mini-columns.  Where the ``SemanticEncoder`` produces a *fixed*
fingerprint from corpus co-occurrence counts, the Spatial Pooler *learns* its
output SDRs by competitive Hebbian learning:

  • Proximal synapses — each mini-column owns a fixed random potential pool of
    input bits, each with a learned permanence.  A synapse is connected when
    permanence ≥ ``connected``.
  • Overlap — a column's score is the number of connected synapses onto active
    input bits.
  • Boosting — a homeostatic term lifts columns that have been active too
    rarely, so representation spreads across the whole population and no column
    dies (this is what makes the code *distributed*).
  • Inhibition (k-WTA) — the top ``active_cols`` boosted columns fire; the rest
    stay silent, yielding a ~2 %-sparse output SDR.
  • Learning — connected synapses onto active input strengthen, onto inactive
    input weaken (the same permanence rule the temporal memory uses).

Similar inputs come to share output columns (overlap → similarity is preserved
and *sharpened* by learning), so the output is a genuine learned distributed
representation stored in an SDR.

Design note — cost: the pooler is fit ONCE as preprocessing and then frozen, so
its per-token output is a deterministic function of the (fixed) encoder
fingerprint and is cached by the caller.  It therefore adds nothing to the
temporal-memory training loop or to inference after fitting.  During fitting the
hot path is a single gather+reduce over the potential pools — the same pattern
the temporal memory already uses — sized ``col_dim × potential_syn``.
"""
from __future__ import annotations
import numpy as np

from .sdr import kwta


class SpatialPooler:
    """Competitive-Hebbian input→SDR encoder (HTM Spatial Pooler).

    Parameters
    ----------
    input_dim     : width of the input SDR (encoder fingerprint dimension)
    col_dim       : number of mini-columns (= output SDR width)
    active_cols   : active bits in the output SDR (k for k-WTA); ~2 % of col_dim
    potential_pct : fraction of input bits in each column's potential pool
    connected     : permanence threshold for a connected synapse
    inc / dec     : Hebbian potentiation / depression per step
    boost_strength: homeostatic boosting gain (0 disables boosting)
    duty_period   : time constant (in steps) of the activity moving average
    stimulus_threshold : minimum raw overlap for a column to be eligible
    seed          : RNG seed for the (fixed) potential pools and init perms
    """

    def __init__(
        self,
        input_dim: int,
        col_dim: int,
        active_cols: int,
        potential_pct: float = 0.5,
        connected: float = 0.5,
        inc: float = 0.05,
        dec: float = 0.02,
        boost_strength: float = 2.0,
        duty_period: int = 1000,
        stimulus_threshold: int = 1,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.col_dim = col_dim
        self.active_cols = active_cols
        self.connected = connected
        self.inc = inc
        self.dec = dec
        self.boost_strength = boost_strength
        self.duty_period = duty_period
        self.stimulus_threshold = stimulus_threshold

        rng = np.random.default_rng(seed)
        potential_syn = max(active_cols, int(potential_pct * input_dim))
        self.potential_syn = potential_syn

        # Fixed potential pool: which input bits each column may connect to.
        self.syn_idx = np.empty((col_dim, potential_syn), dtype=np.int32)
        for c in range(col_dim):
            self.syn_idx[c] = rng.choice(input_dim, size=potential_syn, replace=False)

        # Permanences initialised around the connected threshold so ~half start
        # connected — learning then sculpts each column toward its inputs.
        self.syn_perm = (
            connected + 0.1 * (rng.random((col_dim, potential_syn)) - 0.5)
        ).astype(np.float32)

        # Homeostatic activity moving average (fraction of steps each col fired).
        self.active_duty = np.full(col_dim, active_cols / col_dim, dtype=np.float32)
        self._frozen = False

    # ── Core ──────────────────────────────────────────────────────────────────

    def _overlap(self, inp_dense: np.ndarray) -> np.ndarray:
        """Connected overlap of every column with the dense input vector."""
        active_syn = inp_dense[self.syn_idx]                 # (col_dim, potential_syn)
        connected = self.syn_perm >= self.connected
        return (active_syn & connected).sum(axis=1).astype(np.int32)

    def compute(self, input_sparse: np.ndarray, learn: bool = True) -> np.ndarray:
        """Map an input SDR to the learned active-column SDR.

        Parameters
        ----------
        input_sparse : sorted int32 active input-bit indices
        learn        : adapt permanences + duty cycles for the active columns

        Returns
        -------
        sorted int32 active mini-column indices (the learned output SDR)
        """
        inp = np.zeros(self.input_dim, dtype=bool)
        inp[input_sparse] = True

        overlap = self._overlap(inp)
        overlap[overlap < self.stimulus_threshold] = 0

        # Homeostatic boost: rarely-active columns get a multiplicative lift.
        target = self.active_cols / self.col_dim
        boost = np.exp(self.boost_strength * (target - self.active_duty)).astype(np.float32)
        boosted = overlap.astype(np.float32) * boost

        mask = kwta(boosted, self.active_cols)
        active = np.where(mask & (overlap > 0))[0].astype(np.int32)

        if learn and not self._frozen and active.size:
            idx = self.syn_idx[active]                       # (A, potential_syn)
            act = inp[idx]                                   # (A, potential_syn) bool
            delta = np.where(act, self.inc, -self.dec).astype(np.float32)
            self.syn_perm[active] = np.clip(
                self.syn_perm[active] + delta, 0.0, 1.0
            )

        if learn and not self._frozen:
            # Exponential moving average of per-column activity.
            self.active_duty *= (1.0 - 1.0 / self.duty_period)
            if active.size:
                self.active_duty[active] += 1.0 / self.duty_period

        return np.sort(active)

    def freeze(self) -> None:
        """Stop learning — output becomes a deterministic function of the input,
        so callers can safely cache token → output SDR."""
        self._frozen = True

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def n_connected(self) -> int:
        return int((self.syn_perm >= self.connected).sum())

    def state(self) -> dict:
        """Serialisable arrays for persistence."""
        return {
            "syn_idx": self.syn_idx,
            "syn_perm": self.syn_perm,
            "active_duty": self.active_duty,
        }

    def load_state(self, syn_idx, syn_perm, active_duty) -> None:
        self.syn_idx = np.asarray(syn_idx, dtype=np.int32)
        self.syn_perm = np.asarray(syn_perm, dtype=np.float32)
        self.active_duty = np.asarray(active_duty, dtype=np.float32)
        self._frozen = True
