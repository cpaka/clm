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
