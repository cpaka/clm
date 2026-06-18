# CLM — Next Steps

Status roadmap. Items marked **DONE** are committed and deployed.

## Completed (as of 2026-06-18)

### ✅ DONE — #1 Fan-out voting units (parallel containers)
`train_unit` + `train_parallel` + `inject_units()` + `seed_base` offset.
`make train-parallel` trains N units in N containers simultaneously.

### ✅ DONE — Grid fix (accuracy unlock, architectural)
`CLMUnit` passes `_NO_LOC = np.empty(0)` to every `column.step()` call.
Temporal-memory segments now connect only to previous winner cells — NOT
to absolute sequence positions — so the same bigram predicts the same
continuation regardless of where in the sequence it appears.
Local controlled test: 0% → 37.5% top-1 on structured corpus.

### ✅ DONE — #3 Vectorise inner per-column loop
`column.step()` has no Python `for j in range(n_cols)` loop:
- Winners assigned via `np.where` + fancy-index
- Permanence updates batched via `_batch_adapt()` across all (cell, seg) pairs
- Grow loop runs only for burst cells (minority after training)

### ✅ DONE — #7 Incremental semantics
- `SemanticEncoder.fit()` stores raw accumulator (`_acc`, `_df`)
- `SemanticEncoder.update(new_seqs)`: fold new text without discarding
  old co-occurrences; identity-core bits stay fixed, only shell refreshed
- `persist/store.py` saves/restores accumulator so `update()` continues correctly

### ✅ DONE — #8 Persist replay buffer
`replay.npz` saved beside model; `load_model` restores episodes into
`HippocampalBuffer` so cross-session replay prevents catastrophic forgetting.

### ✅ DONE — #2 Corpus-shard parallelism + merge
`train_shard` trains on a disjoint slice; `train_shards` fans out N shards,
merges `seg_idx`/`seg_perm` via `_merge_columns()`, saves assembled model.
`make train-shards` target added.

### ✅ DONE — #9 Capacity growth (append units / levels)
`HierarchicalCLM.append_unit(unit)` and `append_level(stride)` — additive,
non-disruptive to existing columns.

### ✅ DONE — #10 Saturation signal
`is_saturated(window, tol)` detects burst_rate plateau; `stats()` includes
`"saturated"` flag; `ingest()` auto-appends a unit on saturation.

### ✅ DONE — #6 Scheduled incremental ingest
`@modal.Cron("0 3 * * *")` daily job: load model → train on corpus delta
→ call `enc.update()` for new vocab → save.  `make ingest` for manual run.

### ✅ DONE — Vocabulary filtering
`CORPUS_CONFIG["vocab_size"]=2000` replaces rare words with `<UNK>`,
reducing effective vocabulary from 10510 to ~2000 and bigram coverage from
0.05% to ~1.3% on the 1M-char Wikipedia sample.

---

## Current accuracy situation

The model architecture is correct (verified locally: 37.5% top-1 on
structured 4-pattern corpus, 4.5% on random 42-word corpus).

Wikipedia evaluation is still near-floor (~0.5% top-1) because:
- 4250 training sequences × 12 tokens ≈ 51K unique bigrams seen
- 2000-word vocab → 4M possible bigrams → only 1.3% coverage
- `accuracy()` probes RANDOM positions → mostly rare/unseen bigrams

**The model correctly predicts common patterns** (e.g. "the United" → "States").
The evaluation metric (random probe) penalises rare bigrams equally.

To reach meaningful accuracy (5–15% top-1 on random probes):

---

## A. Remaining high-impact items

### 4. GPU via cupy
The hot path (`_compute_overlaps_scoped`) is already GPU-shaped.
`import cupy as np` in `core/` + `gpu="T4"` in Modal.

- Touch: `core/` import shim (numpy/cupy switch), `modal_app.py` image + `gpu=`.
- Risk: medium (array-type edge cases, persistence).

### Scale up corpus (most impactful for accuracy)
Increase `max_chars` from 1M to 5–10M and `max_sequences` from 5000 to 20K+.
With 200K training sequences bigram coverage reaches ~30% → 5–15% top-1
expected on random probes. Use `make train-shards` to spread the load.

```python
CORPUS_CONFIG["max_chars"] = 5_000_000   # 5M chars → ~50K sentences
TRAIN_CONFIG["max_sequences"] = 20_000
```

### Domain-specific corpus (quick win)
Replace Wikipedia with a repetitive, structured corpus (news, Wikipedia
intro paragraphs only, legal text) to increase bigram repetition rate.
The model already excels on repetitive text (37.5% top-1 locally).

### Evaluation metric: common-bigram probe
Replace random probe with a probe set that samples only bigrams appearing
≥ 5× in training, measuring the "known-pattern recall" rather than overall
language modelling.

---

## How to resume

```bash
cd cglm_modal
git pull
python3 -m tests.test_core && python3 -m tests.test_persist   # confirm 24/24 green

# Scale up and retrain:
make train-parallel        # parallel unit training (current default)
make train-shards          # shard+merge for wider corpus coverage

# Scheduled ingest (normally automatic at 03:00 UTC):
make ingest
```
