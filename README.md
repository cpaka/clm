# CGLM — Cortical-Grid Language Model

A biologically-inspired language model built on Hierarchical Temporal Memory (HTM) principles, deployed serverlessly on [Modal.com](https://modal.com).

## What it is

CGLM uses **Sparse Distributed Representations (SDRs)** and a grid-cell location signal to learn temporal sequences from text, rather than gradient descent. Each token is encoded as a sparse binary vector; the model learns via Hebbian synapse growth and decay. It is designed to be:

- **Fast at inference** — frozenset-based connected-synapse cache, O(columns) lookup
- **Computationally lightweight** — no GPU required, runs on Modal CPU containers
- **Interpretable** — synapse permanences are explicit, predictions are traceable

### Architecture

```
Input text
   │
   ▼
SemanticEncoder (IDF-weighted co-occurrence → sparse fingerprint, dim=2048, 21 active bits)
   │
   ▼
GridLocation (multi-period phase code: periods 7,11,13,17,19,23 → 90-bit location SDR)
   │
   ▼
Column (numpy-backed temporal memory — Segment arrays, vectorised overlap)
   │  × 3 parallel units (ThreadPoolExecutor)
   ▼
Voting ensemble → next-token predictions
```

**Key optimisation:** Synapse segments use `numpy.int32` source arrays + `numpy.float32` permanence arrays. Overlap computation uses a reusable boolean active-buffer sliced via fancy indexing, avoiding per-step Python allocation.

## Quickstart

### Prerequisites

- [Modal account](https://modal.com) + `modal` CLI installed
- Python 3.11+

```bash
pip install modal
modal setup   # authenticate
```

### Deploy

```bash
cd cglm_modal
modal deploy modal_app.py
```

This deploys:
| Endpoint | Description |
|---|---|
| `WebApp.serve` | Chat UI + REST API |
| `notebook` | JupyterLab analytics (port 8888) |

### Train a model

```bash
# 1. Download Wikipedia corpus (~400k chars from Simple English Wikipedia)
modal run modal_app.py::fetch_corpus

# 2. Train (5 epochs, per-epoch metrics saved to volume)
modal run modal_app.py::train
```

Training saves `model.pkl`, `metrics.json`, and `analytics.ipynb` to the `cglm-data` Modal Volume.

### Use the chat UI

After deploying, open the `WebApp.serve` URL from `modal deploy` output. You'll see:

- **Predict** — enter a phrase, get the top-5 predicted next tokens
- **Generate** — auto-complete a phrase for 15 tokens
- Live metrics: vocab size, segments/unit, test top-1 accuracy

## REST API

All endpoints are served by `WebApp.serve`:

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/` | — | Chat HTML UI |
| `POST` | `/predict` | `{"tokens": ["the", "cat"], "topn": 5}` | `{"predictions": [["sat", 0.8], ...]}` |
| `POST` | `/generate` | `{"tokens": ["the", "cat"], "n": 10}` | `{"generated": ["the", "cat", "sat", ...]}` |
| `GET` | `/stats` | — | Model stats (vocab, segments, etc.) |
| `GET` | `/metrics` | — | Per-epoch training metrics JSON |
| `GET` | `/health` | — | `{"ok": true}` |

## Analytics

The JupyterLab notebook (`notebook` function) serves a pre-generated `analytics.ipynb` with:

- Training accuracy curves (test top-1, test top-3, train top-1 per epoch)
- Model growth (synapses and segments per unit per epoch)
- Completion demos with sample prompts
- Vocabulary browser

Run it:
```bash
modal run modal_app.py::notebook
```

## Configuration

Hyperparameters are fixed in `BEST_CONFIG` in `modal_app.py`:

| Parameter | Value | Notes |
|---|---|---|
| `n_columns` | 3 | Parallel ensemble units |
| `dim` | 2048 | SDR dimension |
| `fp_bits` | 21 | Active bits per token fingerprint |
| `index_bits` | 7 | Identity bits within fingerprint |
| `window` | 2 | Co-occurrence context window |
| `cells_per_col` | 8 | HTM cells per mini-column |
| `periods` | (7,11,13,17,19,23) | Grid-cell phase periods |
| `activation_threshold` | 10 | Min connected synapses to predict |
| `new_synapses` | 24 | Synapses grown per learning step |
| `epochs` | 5 | Training passes over corpus |
| `max_sequences` | 3000 | Corpus sentence cap |

## Project structure

```
cglm_modal/
├── model_core.py    # CGLM implementation (numpy-optimised)
├── modal_app.py     # Modal functions: fetch_corpus, train, WebApp, notebook
├── requirements.txt
└── README.md
```

## How it compares to LLMs

| | CGLM | Transformer LLM |
|---|---|---|
| Learning | Online Hebbian (synapse permanences) | Offline gradient descent |
| Compute | CPU-only, seconds/epoch | GPU, hours–days |
| Memory | Sparse (explicit synapse lists) | Dense weight matrices |
| Interpretability | High (traceable predictions) | Low |
| Accuracy on general text | Low–moderate | State of the art |

CGLM is not competing on perplexity — it explores whether sparse, biologically-inspired memory can produce useful predictions with a fraction of the compute.

## License

MIT
