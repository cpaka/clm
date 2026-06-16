# CGLM — Next Steps

Roadmap for scaling training (faster / parallel) and continual learning
(absorb more data without retraining from scratch). Each item builds on the
existing code (`core/`, `persist/`, `modal_app.py`) — none requires a rewrite.

## Status (where we left off)

The brain-architecture CGLM is **live on Modal** as `cglm-chat-v2`:
- URL: https://christophe-paka--cglm-chat-v2-webapp-serve.modal.run
- Endpoints verified: `/predict`, `/generate`, `/similar`, `/stats`, `/health`.

Three deploy fixes are committed and pushed to GitHub (`main`), matching the
live deployment:
1. **Compact-space training speedup** — `CorticalColumn.step` works in active-cell
   space; `predict_columns` scoped to cells with segments. Training dropped from a
   900s timeout to ~59s for the smoke-test config.
2. **Full-width OOV encoding** — `SemanticEncoder.encode` always returns
   `fp_bits`-width SDRs (was 7-bit for OOV), fixing the ragged-stack save crash.
3. **Module-level FastAPI body models + pinned deps** — request models moved to
   module scope so FastAPI resolves their hints under
   `from __future__ import annotations`; `fastapi`/`pydantic` pinned.

Current config (`modal_app.py`): `col_dim=1024`, `n_units=3`, 2 levels,
`max_sequences=150`, `epochs=2`. Bootstrap ~59s, vocab ~1028.

> Predictability note: on diverse Simple-English Wikipedia at this small scale,
> top-1 is low (~0–6%). This architecture is much stronger on structured/repetitive
> text (bundled `grammar`/`motifs` benchmarks hit ~33% top-1). Scale data + capacity
> to improve; use the steps below.

---

## A. Train faster / in parallel

### 1. Fan out the voting units across containers (free n_units× speedup)
Each `CGLMUnit` is fully independent (own encoder seed, grid, column); units only
interact at vote time in `predict_next`. Train the units in separate Modal
containers with `train.map(...)` / `.spawn()`, then assemble into one
`HierarchicalCGLM` and save. No accuracy change, ~3× wall-clock.

- Touch: `modal_app.py` (new `train_unit` function + assembler), `core/hierarchy.py`
  (helper to build a model from pre-trained units).
- Risk: low.

### 2. Corpus-shard parallelism with a deterministic merge (the big one)
Encoders are content-addressable hashes → the same token maps to the same SDR and
same active mini-columns in **every** model. So shard-trained models are
structurally aligned and **mergeable**: for each cell, append/union shard-B's
dendritic segments onto shard-A's (up to `max_segs`). Train K full models on K
shards in parallel, merge `seg_idx`/`seg_perm` once. Scales data throughput
linearly with containers.

- Touch: new `merge_models()` in `persist/` or `core/`, `modal_app.py` fan-out.
- Risk: medium (segment-merge logic + `max_segs` capacity handling).

### 3. Vectorize the inner per-column loop (biggest single-container win)
Remaining bottleneck is the Python `for j in range(n_cols)` loop in
`CorticalColumn.step`. Per-active-column work (best-segment, adapt, grow) is
independent across columns → replace with batched fancy-indexing / `np.add.at`
scatter ops. Likely 5–20× per step, no algorithm change.

- Touch: `core/column.py::step` (+ `_adapt_seg`/`_grow_seg` vectorized variants).
- Risk: medium (correctness of vectorized growth vs current tests).

### 4. GPU via cupy
The hot path (`_compute_overlaps_scoped` gather+reduce) is already GPU-shaped.
`import cupy as np` in `core/` + a Modal `gpu="T4"` container runs it unchanged.
Best combined with #3 (a tight Python loop won't benefit much alone).

- Touch: `core/` import shim (numpy/cupy switch), `modal_app.py` image + `gpu=`.
- Risk: medium (array-type edge cases at numpy/cupy boundary, persistence).

### 5. Numba @njit on the step loop
Lighter-weight alternative to #3 if you'd rather not rewrite in pure numpy.

- Touch: `core/column.py`, add `numba` to image.
- Risk: low–medium (njit constraints on the loop body).

---

## B. Absorb more data gradually (no retrain from scratch)

**Core capability already exists:** `load_model()` → `model.train(new_seqs)`
warm-starts and keeps growing synapses (covered by `test_warm_start_training`).
Make continual learning robust with:

### 6. A streaming ingest job
Scheduled Modal function: load `model.cglm` → train on the new corpus delta →
save. Use a `modal` cron schedule. The `.cglm` store is tiny (~0.1 MB compressed),
so checkpoint-per-batch is cheap.

- Touch: `modal_app.py` (new `ingest` function + schedule).
- Risk: low.

### 7. Incremental semantics for new words
New tokens currently get an OOV fingerprint with **no learned semantic shell**.
Persist the `SemanticEncoder`'s co-occurrence accumulator (random indexing is
itself incremental) and periodically refresh fingerprints, so new vocabulary
gains real semantic overlap instead of staying orthogonal.

- Touch: `core/encoder.py` (persist + incremental update of the accumulator),
  `persist/store.py`.
- Risk: medium (fingerprint refresh must not invalidate learned columns — keep
  identity-core bits stable, only update shell).

### 8. Fight forgetting with the replay buffer (already built)
`HippocampalBuffer` exists but `stats` shows `replay_size:0` because it is **not
persisted** across sessions. Save/restore it in `persist.store`, and interleave
replayed old episodes with new data each session — the consolidation mechanism
that retains old knowledge while absorbing new.

- Touch: `persist/store.py` (save/load buffer), `core/hierarchy.py` (use on warm
  start).
- Risk: low.

### 9. Grow capacity by adding units/levels, not resizing
`col_dim`, `cells_per_col`, `max_segs` are fixed array dimensions — can't enlarge
in place without migration. The clean "add brain capacity" move is to **append new
`CGLMUnit`s** (train fresh on recent data, add to the vote) or a new hierarchy
level. Additive, non-disruptive to existing columns.

- Touch: `core/hierarchy.py` (append-unit / append-level APIs), `persist/store.py`.
- Risk: medium (decode/vote weighting across heterogeneous units).

### 10. Use burst_rate as a saturation signal
`burst_rate` already trends down as the model learns. When it stops dropping on
new data, current capacity is saturated → trigger #9.

- Touch: `modal_app.py` metrics + a threshold check.
- Risk: low.

---

## Suggested order

1. **#1 (fan-out units)** + **#8 (persist replay)** + **#6 (cron ingest)** —
   parallel speed + true continual learning, little code. *Start here.*
2. **#3 (vectorize loop)** — per-container throughput.
3. **#2 (shard + merge)** — scale data hard.
4. **#4 (GPU)** / **#7 (incremental semantics)** / **#9 (capacity growth)** —
   as needed once the above are in.

## How to resume

```bash
cd cglm_modal
git pull
python -m tests.test_core && python -m tests.test_persist   # confirm green baseline
python -m benchmarks.run --dataset motifs --epochs 15        # sanity on structured data

# pick a step above, implement, then redeploy:
make deploy        # or: make redeploy  (force retrain)
```

Pickup prompt for next session:
> "Continue CGLM improvements per NEXT_STEPS.md — implement #1, #6, and #8
> (parallel unit fan-out, scheduled incremental ingest, persisted replay buffer)."
