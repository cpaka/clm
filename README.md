# CLM — Hierarchical Cortical Language Model

A biologically-inspired language model built on **Sparse Distributed
Representations (SDRs)** — no backprop, no float matmuls at inference — deployed
serverlessly on [Modal.com](https://modal.com).

This version is a ground-up modular refactor that adds six brain-architecture
mechanisms on top of the original flat model:

| Mechanism | Brain analogue | Module |
|---|---|---|
| **Hierarchical levels** with temporal strides | cortical hierarchy / temporal abstraction | `core/hierarchy.py` |
| **k-WTA lateral inhibition** | inhibitory interneurons enforcing sparsity | `core/sdr.py::kwta` |
| **Neuromodulatory plasticity** | dopamine / norepinephrine gating of learning | `core/modulation.py` |
| **Hippocampal replay** | sharp-wave-ripple consolidation during sleep | `core/replay.py` |
| **Sparse inter-area projections** | thalamo-cortical axonal wiring | `core/hierarchy.py::SparseProjection` |
| **Vectorised columns** | dense binary algebra, GPU-parallelisable | `core/column.py` |

## Why it stays fast (and is GPU-ready)

Every signal is a binary SDR (~2 % active bits). The hot path — segment overlap
— is a single numpy gather + reduce over fixed-shape arrays:

```python
syn_active = src[self.seg_idx]                          # fancy-index gather
conn_ov    = (syn_active & connected & valid).sum(-1)   # reduce
```

Because all column state lives in regular `(n_cells, max_segs, syn_per_seg)`
arrays, swapping `import numpy as np` for `import cupy as np` in `core/` runs the
same code on a GPU with no algorithmic change.

---

> **Live interactive docs:** [`/docs`](https://christophe-paka--clm-chat-v4-webapp-serve.modal.run/docs) on the deployed app — rendered with diagrams and colour-coded examples.

---

## Architecture

```
                       tokens
                         │
              ┌──────────▼───────────┐   stride 1   ← token granularity
   Level 0   │  N voting columns     │   (semantic encoder)
              └──────────┬───────────┘
                         │  winners → SparseProjection
              ┌──────────▼───────────┐   stride 4   ← phrase granularity
   Level 1   │  N voting columns     │
              └──────────────────────┘

  prediction = vote(level-0 units) → k-WTA → decode by SDR overlap
  plasticity = base × neuromod(recent burst rate)
  consolidation = replay surprising episodes from hippocampal buffer
```

It **improves along two axes** with no code change: more **columns** (`n_units`)
sharpen the vote; more **data** (online Hebbian learning + warm-start from a
checkpoint) grows the synaptic memory.

---

## Architecture deep dive

### 1. Sparse Distributed Representations (SDRs)

Every signal in CLM is a **sparse binary vector** — only ~2 % of bits are active at
any time, mirroring neocortical population codes.

```
Dense (one-hot, 1024 bits) — no overlap, no similarity:
  "cat" → [0 0 0 0 0 1 0 0 0 0 … 0 0 0]   overlap("cat","dog") = 0

SDR (1024 bits, 21 active) — overlap encodes semantic distance:
  "cat" → bits {12, 87, 143, 201, 305 … 21 total}
  "dog" → bits {12, 87, 201, 310, 502 … 21 total}   overlap = 14
                ↑    ↑   ↑
              shared bits ← semantic similarity as bit overlap
```

Key properties:
- **Overlap = similarity** — words with similar contexts share more bits
- **Huge capacity** — C(1024,21) ≈ 10⁴² patterns; effectively no collisions
- **Noise-robust** — 17/21 bits still recognised as the same word
- **2 % sparsity** makes accidental overlap probability ~10⁻³⁰

---

### 2. Semantic Encoder — words as fingerprints

Every word is assigned a 21-bit fingerprint (SDR) in a 1024-bit space. The
fingerprint has two parts:

```
┌─────────────────────────────────────────────────────────────┐
│  1024-bit fingerprint for "cat"                             │
│                                                             │
│  ┌────────────────┐  ┌────────────────────────────────────┐ │
│  │ Identity core  │  │       Context shell                │ │
│  │   7 bits       │  │         14 bits                    │ │
│  │   (FIXED)      │  │   (learned from co-occurrence)     │ │
│  │  hash("cat")   │  │   which words appear near "cat"    │ │
│  │  → stable ID   │  │   in training text                 │ │
│  └────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**How fingerprints are built:**
1. **Co-occurrence window** — count which words appear within ±4 positions
2. **Random projection** — co-occurrence vector → 1024-bit space
3. **Identity core** — 7 bits fixed from `hash(word)`, never change
4. **Context shell** — 14 bits re-selected each time co-occurrence data is updated

**Semantic similarity emerges automatically:** "cat" and "dog" appear in similar
contexts → similar co-occurrence vectors → ~10/21 shared bits.
"cat" and "democracy" share ~0 bits.

---

### 3. Cortical Column — learning sequences

Each CLM unit contains one `CorticalColumn`: 1024 mini-columns × 8 cells = 8192
cells, each with up to 16 dendritic segments of 20 synapses.

```
CorticalColumn (col_dim=1024 × cells_per_col=8 = 8192 cells)

Mini-column 0          Mini-column 1         …  Mini-column 1023
┌─────────────────┐   ┌─────────────────┐        ┌─────────────────┐
│  cell 0  [segs] │   │  cell 0  [segs] │        │  cell 0  [segs] │
│  cell 1  [segs] │   │  cell 1  [segs] │        │  …              │
│  …              │   │  …              │        │  cell 7  [segs] │
│  cell 7  [segs] │   │  cell 7  [segs] │        └─────────────────┘
└─────────────────┘   └─────────────────┘
      ↑                     ↑
  SDR bit 0             SDR bit 1          ← active columns = the current token
```

**Predictive vs burst state:**
```
PREDICTIVE (sequence already learned):
  Token "the" arrives → columns for "the" activate
  Cell 87·3 has a segment connected to prev "the" winners
  → Only cell 87·3 fires   ✓ Correct prediction, single winner

BURST (first time seeing this transition):
  Token "the" arrives → columns activate
  No cell is predictive for this context
  → ALL 8 cells in each column fire   ⚡ Surprise! → learning triggered
```

**Synaptic permanence (Hebbian learning):**
```
Each synapse has permanence ∈ [0, 1]; connected if ≥ 0.5

If presynaptic cell was active:      permanence += 0.10  (strengthen)
If presynaptic cell was NOT active:  permanence -= 0.02  (weaken)

After ~10 repetitions of "the → cat":
  Segment in cat-columns → prev "the" winners: all 20 synapses at ≥ 0.5
  Connected overlap ≥ 8 threshold → cell fires PREDICTIVELY
```

---

### 4. How sequences are learned step-by-step

Tracing "the cat sat" across training epochs:

```
EPOCH 1 — First exposure

  t=0: "the" → fingerprint bits {3,87,143…} → columns burst (no context)
       winners w₀ = {3·2, 87·5, 143·1 …}   ← one cell per column chosen

  t=1: "cat" → fingerprint bits {12,87,201…} → columns burst (no prior segment)
       winners w₁ = {12·0, 87·3, 201·6 …}
       LEARN: grow new segment in each w₁ cell, connect synapses → w₀

  t=2: "sat" → burst, winners w₂, learn segment → w₁

EPOCH 10 — After repetition

  t=0: "the"  → burst (sentence start, no prior context)   w₀ fires

  t=1: "cat"  → segments in cat-columns overlap w₀, score ≥ 8
       → cells PREDICT before token arrives, only w₁ fires  ✓

  t=2: "sat"  → segments in sat-columns overlap w₁
       → cells PREDICT, only w₂ fires   ✓
```

The column doesn't learn "after word X comes word Y" — it learns which *specific
winner cells* should fire given the *exact prior context*, enabling
context-sensitive disambiguation.

---

### 5. Neuromodulation & Hippocampal Replay

**Neuromodulatory signal (dopamine analogue):**
```
burst_rate high → modulation = 2.0   "Pay attention, this is new"
burst_rate low  → modulation = 0.2   "Familiar, consolidate"
```

`NeuromodSignal` tracks the recent burst rate and scales learning rates
accordingly. High surprise → high plasticity → faster encoding of novel patterns.

**Hippocampal replay:** When `burst_rate > 0.3`, the sequence is stored in
`HippocampalBuffer` (capacity 256). Every 2 epochs, stored sequences are replayed
with boosted plasticity (modulation=1.5), strengthening synaptic weights for
surprising episodes — mirroring hippocampal sharp-wave-ripple consolidation.

---

### 6. Hierarchical Architecture

```
Input tokens:  the   cat   sat   on   a   mat

Level 0  (stride=1, token granularity)
  Unit 0 ──┬──┬──┬──┬──┬──   processes every token
  Unit 1 ──┘  │  │  │  │
  Unit 2 ─────┘  │  │  │
                 │  │  │
  every 4 tokens ──► SparseProjection ──► Level 1 feature SDR (21 bits)

Level 1  (stride=4, phrase granularity)
  Unit 0 ─────┬─────   processes every 4th token → phrase rhythm
  Unit 1 ─────┘         (learns "the cat sat" as one temporal unit)
  Unit 2
```

**SparseProjection** is a fixed random sparse matrix that projects 8192-dimensional
Level-0 winner bits down to a 21-bit SDR for Level 1 — analogous to thalamo-cortical
wiring that compresses fine-grained activity into coarser phrase-level signals.

---

### 7. Voting Ensemble (Thousand Brains)

```
Unit 0 → score_next("the cat") → {sat: 0.8, ran: 0.3}
Unit 1 → score_next("the cat") → {sat: 0.6, ran: 0.5}
Unit 2 → score_next("the cat") → {sat: 0.7, on:  0.4}
                          ↓
              Aggregate: {sat: 2.1, ran: 0.8, on: 0.4}
                          ↓
              k-WTA (k=42): keep top candidates
                          ↓
              Top-1: "sat"
```

Each unit trains with a different random seed. Voting averages over diverse
representations, reducing variance on ambiguous inputs. This mirrors the
**Thousand Brains Theory** — each cortical column holds a complete world model;
consensus emerges from cross-column votes.

In `make train-parallel`, units train simultaneously in separate Modal containers,
cutting wall-clock time from N × per-unit time to max(per-unit time).

---

### 8. How the model improves

**More data → more bigram coverage:**
```
1M chars,  5K seqs  → ~1.3% bigram coverage   (V4.0 baseline)
2M chars, 10K seqs  → ~2.5%                   (V4.1 target)
5M chars, 20K seqs  → ~6%                     (V4.2 target)
```

**Incremental learning:** `enc.update(new_seqs)` folds new co-occurrence data into
the semantic encoder without discarding old knowledge. Identity-core bits are fixed,
so existing column segments stay valid.

**Saturation & capacity growth:** When burst rate plateaus (model saturated), the
daily ingest cron appends a new `CLMUnit` to the ensemble — fresh capacity without
retraining.

---

### 9. GPU Acceleration (V4)

The column hot path — gather + reduce over synaptic weight arrays — maps directly
onto GPU SIMD:

```
CPU (numpy):   gather + reduce 107K elements → ~80–110 μs per token
GPU (cupy T4): same ops → ~10–20 μs per token   (~5–10× speedup)
```

`core/xp.py` is a numpy/cupy shim: `import xp` resolves to cupy on GPU workers
and numpy elsewhere — zero code duplication. Grow operations (rare; only for burst
cells without segments) run on CPU to avoid complex GPU RNG management.

Compiled cupy CUDA kernels are cached to the Modal Volume at `/data/cupy-cache`,
so the ~15 min first-run JIT penalty is paid only once.

---

### Key hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `col_dim` | 1024 | SDR width; also encoder fingerprint dimension |
| `cells_per_col` | 8 | Context depth; more cells = more disambiguation capacity |
| `fp_bits` | 21 | Active bits per fingerprint (~2 % sparsity) |
| `index_bits` | 7 | Fixed identity-core bits per fingerprint |
| `window` | 4 | Co-occurrence window for semantic encoder |
| `max_segs` | 16 | Max segments per cell; bounds memory |
| `syn_per_seg` | 20 | Synapses per segment; ≥8 needed to fire |
| `activation_threshold` | 8 | Connected synapses required for segment to activate |
| `vocab_size` | 2000 | Rare words → `<UNK>`; improves bigram coverage |
| `n_units` | 3 | Voting ensemble size |
| `n_levels` | 2 | Hierarchy depth (token + phrase) |
| `replay_cap` | 256 | Hippocampal buffer capacity |

> **Memory per unit:** `8192 cells × 16 segs × 20 synapses × 8 bytes = 20 MB`.
> Three units + two levels ≈ 120 MB total.

---

### Module map

```
cglm_modal/
├── core/
│   ├── sdr.py          SDR primitives: convert, overlap, batch_overlap, make_sdr, kwta
│   ├── grid.py         GridLocation — multi-scale path-integration location code
│   ├── encoder.py      TokenEncoder (random) + SemanticEncoder (distributional)
│   ├── column.py       CorticalColumn — vectorised HTM temporal memory
│   ├── modulation.py   NeuromodSignal — plasticity gating from prediction quality
│   ├── replay.py       HippocampalBuffer — surprise-prioritised episodic replay
│   └── hierarchy.py    HierarchicalCLM — levels, projections, voting, inference
├── persist/
│   └── store.py        save_model / load_model — compressed .npz filesystem store
├── benchmarks/
│   ├── datasets.py     reproducible corpora (grammar / motifs / natural)
│   ├── metrics.py      top-1, top-3, MRR, perplexity proxy, accuracy
│   └── run.py          benchmark harness + scaling study
├── tests/
│   ├── test_core.py    unit tests for every module
│   └── test_persist.py round-trip + warm-start tests
├── modal_app.py        Modal deployment: dataset → corpus → train → serve (chat + dashboard + explore)
├── model_core.py       legacy flat model (kept as fallback; no longer wired)
├── Makefile
└── requirements.txt
```

## Persistence (fast filesystem store)

The dense segment arrays are large in memory but ~99.9 % empty, so the model is
saved with **compressed `.npz`** (`persist/store.py`), giving ~1000× smaller
files than raw pickle:

```python
from persist.store import save_model, load_model
save_model(model, "model.clm")     # directory of compressed .npz blobs
model = load_model("model.clm")     # ready for inference + warm-start
model.train(more_corpus, epochs=5)   # incremental learning continues
```

On Modal, the checkpoint is written to the `cglm-data` Volume at
`/data/<VERSION>/model.clm` and loaded by the web container on cold start.

## Quickstart (Modal)

### Prerequisites
- [Modal account](https://modal.com) + `modal` CLI, Python 3.11+
- Kaggle credentials in a Modal secret named `kaggle-credentials` (for the dataset)

```bash
pip install modal
modal setup
```

### One-time dataset upload
```bash
make upload-dataset       # AllCombined.txt (Simple English Wikipedia) → Volume
```

### Deploy + train (idempotent per VERSION)
```bash
make deploy               # modal deploy + bootstrap (fetch_corpus → train → registry)
make redeploy             # force retrain even if a checkpoint exists
make logs                 # tail the running web app
```

`bootstrap` samples a corpus slice, trains `HierarchicalCLM`, writes
`model.clm` and `metrics.json` to the Volume, and records a registry entry.
Bump `VERSION` in `modal_app.py` for an isolated new deployment.

## REST API (served by `WebApp.serve`)

| Method | Path | Body | Response |
|---|---|---|---|
| `GET`  | `/` | — | Console UI (Chat / Dashboard / Explore) |
| `GET`  | `/docs` | — | Architecture documentation (SDRs, columns, sequence learning…) |
| `POST` | `/predict` | `{"tokens":["the","cat"],"topn":5}` | `{"predictions":[["sat",0.8],...]}` |
| `POST` | `/generate` | `{"tokens":["the","cat"],"n":10}` | `{"generated":[...]}` |
| `GET`  | `/similar?word=cat&k=6` | — | nearest words by fingerprint overlap |
| `GET`  | `/fingerprint?word=cat` | — | active SDR bits + dim (for the Explore viz) |
| `GET`  | `/stats` | — | model stats (levels, vocab, segments, neuromod…) |
| `GET`  | `/metrics` | — | per-epoch training metrics |
| `GET`  | `/registry` | — | all deployed versions |
| `POST` | `/train` | — | spawn a live training run (non-blocking) |
| `GET`  | `/status` | — | live training phase / progress / metrics |
| `POST` | `/reload` | — | reload the freshly-trained model into the web container |
| `GET`  | `/health` | — | `{"ok": true}` |

## Run locally (no Modal)

```bash
pip install numpy

# benchmarks
python -m benchmarks.run                    # all datasets
python -m benchmarks.run --dataset motifs --epochs 15
python -m benchmarks.run --scaling          # units ∈ {1,2,4,8} vs accuracy

# tests
python -m tests.test_core
python -m tests.test_persist
```

Python API:

```python
from core.hierarchy import HierarchicalCLM
model = HierarchicalCLM(n_levels=2, strides=(1,4), n_units=4,
                         col_dim=2048, encoder="semantic",
                         kwta_k=42, replay_cap=512)
model.train(corpus_sequences, epochs=10)
model.predict_next(["the","cat"], topn=3)
model.generate(["the","cat"], n=5)
model.similar("cat", k=5)
```

## Configuration (`modal_app.py`)

`MODEL_CONFIG` controls the architecture; `col_dim` is the single source of truth
for SDR width (encoder dim is forced to match).

| Parameter | Default | Notes |
|---|---|---|
| `n_levels` / `strides` | `2` / `(1,4)` | hierarchy depth & temporal scales |
| `n_units` | `3` | voting columns per level |
| `col_dim` | `2048` | SDR width / mini-column count |
| `cells_per_col` | `8` | temporal context depth |
| `fp_bits` / `index_bits` | `21` / `7` | fingerprint & identity-core bits |
| `window` | `2` | semantic co-occurrence window |
| `kwta_k` | `42` | lateral inhibition: winners kept |
| `replay_cap` | `512` | hippocampal buffer size |
| `max_segs` / `syn_per_seg` | `16` / `24` | per-cell capacity (drives memory) |

> **Memory per column** ≈ `col_dim × cells_per_col × max_segs × syn_per_seg × 8 bytes`.
> Keep `max_segs` modest on Modal; raise the container `memory` for larger models.

## Benchmarks

`benchmarks/run.py` reports **top-1 / top-3 / MRR / perplexity-proxy**, training
time, model size, and segment count across three bundled corpora:

| Dataset | Tests |
|---|---|
| `grammar` | deterministic transitions (learnability ceiling) |
| `motifs`  | memorisation + generalisation of repeated phrases |
| `natural` | real text from a bundled corpus file |

## How it compares to LLMs

| | CLM | Transformer LLM |
|---|---|---|
| Learning | Online Hebbian + neuromodulation + replay | Offline gradient descent |
| Compute | Binary SDR algebra; CPU or GPU | GPU, hours–days |
| Memory | Sparse SDRs; explicit synapses | Dense weight matrices |
| Interpretability | High (traceable predictions) | Low |
| Accuracy on general text | Low–moderate | State of the art |

CLM is not competing on perplexity — it explores whether sparse,
biologically-inspired memory can produce useful predictions with a fraction of
the compute, and whether brain-architecture mechanisms (hierarchy, inhibition,
neuromodulation, replay) improve a backprop-free learner.

## License

MIT
