# CLM — Cortical Language Model

A language model that learns the way a **brain** does instead of the way
ChatGPT does. It has **no backpropagation, no giant matrix multiplications, and
no GPU requirement** — every piece of information is a small pattern of "on"
switches (like neurons firing), and the model learns by strengthening the
connections between patterns that occur together. It is deployed serverlessly on
[Modal.com](https://modal.com) and ships with a live web console, an
architecture-docs page, and a knowledge-space *reasoning* layer.

> **New here? Read the next two sections ("The big idea" and "How it differs
> from ChatGPT").** They explain the whole system in plain language, with no
> neuroscience or machine-learning background assumed. Then jump straight to
> **[Quickstart (deploy in ~10 minutes)](#quickstart-deploy-in-10-minutes)** —
> you do not need to understand the internals to run it.

---

## The big idea (in plain language)

Your neocortex — the wrinkled outer sheet of your brain — is made of millions of
nearly identical little circuits called **cortical columns**. The neuroscientist
Jeff Hawkins and colleagues (Numenta / the Thousand Brains Project) spent two
decades working out, in mathematical detail, *what one column computes* and *how
many columns vote together to understand the world*. CLM is a language model
built directly from that theory. Four of their papers are the blueprint (see
[Papers](#the-four-papers-this-is-built-on)); this repo is a working, deployable
implementation of the ideas in them.

Here are the four ideas you need, each in one sentence:

1. **Everything is a sparse pattern.** Instead of representing a word as one
   number or a dense list of numbers, CLM represents it as a wide row of bits
   (2048 of them) where only ~21 are switched on — a **Sparse Distributed
   Representation (SDR)**, exactly how a small population of real neurons signals
   a concept. Two words are "similar" when their on-bits overlap.

2. **A column learns sequences by prediction.** Feed a column a stream of
   words. Each new word makes some cells fire. The column constantly tries to
   **predict the next pattern**; when it is wrong (a surprise), it grows new
   connections so it won't be surprised next time. That is the entire learning
   rule — no gradients, no training runs measured in GPU-days.

3. **Learning is gated by surprise, and consolidated during "sleep."** A
   chemical-signal analogue (think dopamine) turns learning *up* when the model
   is surprised and *down* when things are familiar, and surprising episodes are
   **replayed** later to burn them in — the same trick your hippocampus uses
   overnight.

4. **Many columns vote.** No single column is trusted. CLM runs an ensemble of
   columns, each having seen the data slightly differently, and they **vote** on
   the next word. This is the "Thousand Brains" idea: intelligence is a consensus
   of many small models, not one big one.

On top of these, a **reasoning layer** treats concepts as *places* in a
knowledge space and treats relationships (like "is bigger than" or "capital
of") as *movements* between places — so the model can chain facts together and
answer analogies. More on that below.

---

## How it differs from ChatGPT

| | **CLM (this project)** | **Transformer LLM (ChatGPT, etc.)** |
|---|---|---|
| **How it represents words** | Sparse on/off bit patterns (~1% active) | Dense lists of thousands of decimal numbers |
| **How it learns** | Strengthens connections between co-occurring patterns, one pass, online | Backpropagation / gradient descent over the whole dataset, many passes |
| **Hardware** | Runs on a plain CPU; optionally 5–10× faster on GPU | Needs many GPUs for hours to days |
| **Can it learn a new fact tonight?** | Yes — a nightly job folds in new text without forgetting old text | No — requires expensive retraining or fine-tuning |
| **Can you see *why* it predicted a word?** | Yes — you can trace the exact cells and connections | Mostly no (a "black box") |
| **Raw accuracy on open-ended text** | Low-to-moderate (this is research, not a product) | State of the art |

**CLM is not trying to beat GPT on quality.** It is an honest exploration of a
different question: *can a brain-style, backprop-free memory produce useful
predictions with a tiny fraction of the compute, and do the brain's own tricks
(hierarchy, inhibition, neuromodulation, replay, voting) actually help a learner
that doesn't use gradients?* The code is written to be **read and traced**, not
just run.

---

## The four papers this is built on

Each mechanism in the code maps to a specific published theory. You don't need
to read them to use CLM, but here is what each contributes, in plain terms.

| Paper | Plain-language contribution | Where it lives in the code |
|---|---|---|
| **Hawkins & Ahmad, 2016** — *Why Neurons Have Thousands of Synapses: A Theory of Sequence Memory in Neocortex* (`papers/fncir-10-00023.pdf`) | How one column learns and predicts sequences using cells with many dendritic "segments." This is the heart of CLM. | `core/column.py` (temporal memory) |
| **Hawkins, Ahmad & Cui, 2017** — *A Theory of How Columns in the Neocortex Enable Learning the Structure of the World* (`papers/fncir-11-00081.pdf`) | A column has a fast "input layer" that tracks the sequence **plus** a slow "output layer" that pools it into a stable sense of *what object/topic am I looking at*, which feeds back to sharpen predictions. | `core/pooling.py` (`ContextPool`) |
| **Hawkins et al., 2019** — *A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex* (`papers/fncir-12-00121.pdf`) | The brain represents everything — even abstract concepts — using **grid-cell reference frames**, the same machinery it uses for physical location. Relationships become *displacements* you can compose. This is the basis of CLM's reasoning layer. | `core/grid.py`, `core/displacement.py`, `core/reasoning.py` |
| **Leadholm, Clay et al., 2025** — *Thousand Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference* (`papers/2507.04494v1.pdf`) | Many independent column-models sensing and **voting** produces fast, robust learning. | ensemble voting in `core/hierarchy.py` |

Two classic HTM ideas round out the picture: the **Spatial Pooler**
(`core/spatial_pooler.py`) learns the *representations* that the temporal memory
then sequences over, and **Vector Symbolic Architecture** (`core/vsa.py`) is the
algebra that lets sparse patterns carry *structure* (roles, order) instead of
just being a bag of bits.

---

## A guided tour of the pieces

The rest of this section walks through the architecture from the bottom up. Each
part is first explained in plain terms, then in technical detail.

### 1. Sparse Distributed Representations (SDRs) — the "alphabet"

**Plain version:** every concept is a wide row of switches, almost all off. The
*pattern* of which switches are on *is* the meaning. Similar things share
switches.

```
Dense (one-hot) — no notion of similarity:
  "cat" → [0 0 0 0 0 1 0 0 … 0]      overlap("cat","dog") = 0

SDR (2048 bits, 21 on) — overlap = similarity:
  "cat" → on-bits {12, 87, 143, 201, 305 … 21 total}
  "dog" → on-bits {12, 87, 201, 310, 502 … 21 total}   shared = 14 → similar
  "democracy" → {…}                                     shared = 0  → unrelated
```

Why this is powerful:
- **Overlap encodes similarity** for free — no distance metric to tune.
- **Enormous capacity** — the number of distinct 21-of-2048 patterns is
  astronomically large, so accidental collisions essentially never happen.
- **Noise-robust** — flip a few bits and it's still recognisably the same
  concept.

`core/sdr.py` holds the primitives: `make_sdr`, `overlap`, `kwta` (keep the top
*k* bits — a stand-in for the brain's inhibitory neurons that enforce sparsity).

### 2. The Spatial Pooler — *learning* good representations

**Plain version:** the encoder (next section) gives every word a first-draft
fingerprint. The Spatial Pooler *improves* those fingerprints so that words used
in similar ways come to share even more bits — it **learns** the representations
rather than fixing them by hand.

**Technical version (`core/spatial_pooler.py`):** each output mini-column owns a
random pool of input bits with learnable "permanences." It scores how many of
its connected inputs are active, a **boosting** term lifts columns that fire too
rarely (so representation spreads across the whole population and no column goes
dead), and a k-WTA step keeps only the strongest ~1% as the output SDR.
Connections onto active inputs strengthen; onto inactive inputs, weaken. It is
**fit once as preprocessing and then frozen and cached**, so it costs nothing at
inference time. This is the "input representation" half of HTM that pairs with
the sequence-memory half in the next sections. (`use_spatial_pooler: True` in
`MODEL_CONFIG`.)

### 3. The Semantic Encoder — words as fingerprints

**Plain version:** before anything can be learned, each word needs a starting
fingerprint. Half of it is a fixed ID (so "cat" is always recognisably "cat"),
and half is shaped by *which words show up near it* in the training text (so
"cat" and "dog" end up looking alike).

```
┌──────────────────────────────────────────────────────────┐
│  fingerprint for "cat"  (2048-bit space, 21 bits on)      │
│  ┌───────────────┐   ┌──────────────────────────────────┐ │
│  │ Identity core │   │        Context shell             │ │
│  │  7 bits FIXED │   │        14 bits LEARNED           │ │
│  │  hash("cat")  │   │  from words seen within ±4 of     │ │
│  │  never moves  │   │  "cat" in the corpus              │ │
│  └───────────────┘   └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**Technical version (`core/encoder.py`):** count co-occurrences in a ±`window`
band, project that count vector into the SDR space, keep `index_bits` fixed from
`hash(word)` and re-select the remaining `fp_bits - index_bits` from
co-occurrence. Rare words beyond `vocab_size` collapse to `<UNK>`, which keeps
bigram coverage high. Because identity bits never move, old learned connections
stay valid when the encoder is later updated with new text (this is what makes
nightly incremental learning safe).

### 4. The Cortical Column — learning sequences by prediction

**Plain version:** this is the engine. Picture a grid of 2048 mini-columns, each
holding a few cells. A word switches on its mini-columns. Within each active
mini-column, if some cell *predicted* this word (based on the word before), only
that cell fires — a clean, confident prediction. If nothing predicted it, **all**
the cells fire at once ("bursting") — that's the model's way of shouting
*surprise!*, and surprise is what triggers learning.

```
CorticalColumn = col_dim (2048) mini-columns × cells_per_col (4) cells

PREDICTIVE (sequence already known):
  "the" → "cat" arrives; a cell in each cat-column was expecting it
  → just that one cell fires        ✓ confident, single winner

BURST (never seen this transition):
  "the" → "xylophone" arrives; nobody predicted it
  → ALL cells in each column fire   ⚡ surprise → grow new connections
```

**Technical version (`core/column.py`, from Hawkins & Ahmad 2016):** each cell
has up to `max_segs` dendritic **segments**, each a set of `syn_per_seg`
synapses onto other cells, with a permanence in [0,1] (connected at ≥ 0.5). A
segment "activates" when at least `activation_threshold` of its connected
synapses are onto currently-active cells. On a correct prediction the
responsible synapses strengthen (+); a small `pred_dec` punishes segments that
predicted wrongly (this roughly doubled held-out accuracy in A/B tests — see the
comment on `pred_dec` in `MODEL_CONFIG`). Crucially, the column doesn't learn
"word X → word Y"; it learns *which specific cell should fire given the exact
prior context*, which is what lets it disambiguate ("bank" of a river vs. a
money bank).

### 5. Neuromodulation & hippocampal replay — surprise-gated "sleep"

**Plain version:** when the model is being surprised a lot, it turns its learning
rate *up* ("this is new, pay attention"); when things are familiar, it turns it
*down* ("I know this, just reinforce"). Separately, surprising sentences are
saved and **replayed** later with extra learning strength — the brain's
overnight-consolidation trick.

**Technical version:** `core/modulation.py` (`NeuromodSignal`) tracks the recent
burst (surprise) rate and scales plasticity — high surprise → ~2× learning, low
→ ~0.2×. `core/replay.py` (`HippocampalBuffer`, capacity `replay_cap`) stores
high-surprise sequences and, every couple of epochs, replays them with boosted
plasticity, mirroring hippocampal sharp-wave-ripple consolidation. The **same**
dopamine-like signal that gates column learning also gates the reasoning layer
(§9), so perception and reasoning share one learning loop.

### 6. Hierarchy — tokens on one level, phrases on the next

**Plain version:** the bottom level sees every word; the level above only gets a
summary every few words, so it naturally learns *phrase-level* rhythm rather than
word-level detail. Higher = slower and more abstract, just like real cortex.

```
tokens:   the   cat   sat   on   a   mat
Level 0  (stride 1): sees every token          ← word granularity
   every 4 tokens ─► SparseProjection ─► compressed 21-bit summary
Level 1  (stride 4): sees the summary          ← phrase granularity
```

**Technical version (`core/hierarchy.py`):** `SparseProjection` is a fixed random
sparse matrix that compresses level-0 winners into a 21-bit SDR for level 1 —
analogous to thalamo-cortical wiring. Depth is `n_levels`, temporal scales are
`strides`.

### 7. The context "output layer" — a stable sense of topic

**Plain version:** while the fast sequence memory tracks word-to-word, a slow
accumulator keeps a fuzzy running picture of *what this passage is about*, and
gently biases predictions toward words that fit the topic.

**Technical version (`core/pooling.py`, from Hawkins/Ahmad/Cui 2017):**
`ContextPool` is a decayed accumulator of recently-active mini-columns; its
top-`pool_w` bits form a slowly-changing "topic" SDR. At decode time, the overlap
between a candidate word's fingerprint and this pool is a pure-SDR "does this word
fit the context?" bias (the apical feedback of the theory). It costs O(col_dim)
per token and nothing at training time.

### 8. The voting ensemble — Thousand Brains

**Plain version:** run several columns that each saw the data a bit differently,
and let them vote on the next word. Disagreement averages out; the consensus is
more reliable than any single column.

```
Unit 0 → {sat: 0.8, ran: 0.3}
Unit 1 → {sat: 0.6, ran: 0.5}     ─►  sum ─► k-WTA ─► decode ─► "sat"
Unit 2 → {sat: 0.7, on:  0.4}
```

**Technical version (`core/hierarchy.py`, from Leadholm et al. 2025):** `n_units`
columns per level, each seeded differently, produce candidate scores that are
summed, sparsified with k-WTA (`kwta_k`), and decoded back to words by SDR
overlap. In `make train-parallel`, units train simultaneously in separate Modal
containers, so wall-clock time is `max(per-unit)` instead of `sum(per-unit)`.

### 9. The reasoning layer — concepts as places, relations as moves

**Plain version:** picture concepts laid out on a map. "Paris" and "France" sit
at two spots; the relation "capital-of" is the *arrow* between them. If the same
arrow also takes "Tokyo" to "Japan," the model has learned a general rule it can
apply to spots it was never explicitly taught — and it can chain arrows to reach
conclusions several steps away. That is reasoning as **movement through a
knowledge space**.

**Why you need VSA for this (`core/vsa.py`):** a plain SDR can only say *how
similar* two things are; it can't say *cat chased mouse* differently from *mouse
chased cat*, because a bag of bits loses word order. **Vector Symbolic
Architecture** fixes this with a tiny, sparsity-preserving algebra:
- `bind` — attach a filler to a role (e.g. SUBJECT ↔ cat), reversibly;
- `unbind` — recover the filler for a role;
- `bundle` — superpose several bound pairs into ONE pattern;
- `permute` — encode order/position.

So a whole sentence becomes one SDR you can still take apart:
`sentence = bundle(bind(SUBJECT,cat), bind(RELATION,chased), bind(OBJECT,mouse))`,
and `unbind(SUBJECT, sentence) → cat`.

**Grid frames make relations composable (`core/displacement.py`, `core/grid.py`,
from Hawkins et al. 2019):** a knowledge space is an N-axis grid where each
coordinate maps to a multi-scale SDR (co-prime ring phases, the grid-cell trick).
A relation is a **displacement** — a constant move — and displacements **add**:
`D(A→B) then D(B→C) == D(A→C)`. That single property gives transitive inference
and analogy almost for free.

**What's actually built (`core/reasoning.py`, tested in
`tests/test_reasoning.py`):** `ConceptSpace` places concepts on the grid; `walk`
/ `infer_relation` follow a relation repeatedly (transitive inference that
generalises to **unseen** distances); `analogy` solves *man:king :: woman:?* by
transferring a displacement; `plan` / `ReasoningPolicy` chain *different*
relations toward a goal and route around obstacles (e.g. avoid a predator).
`ground_model` learns the whole space **from a corpus-trained CLM's own learned
SDRs** rather than hand-placed coordinates, and `active_infer` closes the loop by
feeding a plan's success/failure back into the same `NeuromodSignal` that gates
perception. An honest finding baked into the tests: **reasoning quality is
bounded by representation quality** — a lightly-trained pooler yields a
lightly-structured space.

### 10. Question answering (`core/qa.py`)

The web console routes anything that looks like a question to a small QA
pipeline: it encodes the prompt into a word-level and an intent-level SDR,
classifies the intent ("what is…", "who…"), plans a response length, picks the
phrasing the model is *most confident* about, and generates token-by-token —
using the model's own certainty, not caller-supplied settings, to decide when to
stop.

---

## Quickstart (deploy in ~10 minutes)

You do **not** need to understand any of the above to run CLM. Training and
serving happen entirely on [Modal](https://modal.com) (serverless, free tier is
enough); the only thing you run on your own machine is the `modal` CLI. You'll
need two free accounts:

- a **Modal** account — runs the training + web app,
- a **Kaggle** account — hosts the Wikipedia training text (downloaded once).

### Step 1 — Get the code

```bash
git clone https://github.com/cpaka/clm.git
cd clm
```

### Step 2 — Install the Modal CLI and log in

The only local dependency is Python 3.11+ and the `modal` package. You do **not**
need numpy, a GPU, or any of the project requirements locally — those live in the
Modal container image.

```bash
pip install modal          # the CLI that submits jobs to Modal
modal setup                # opens a browser to log in / link your Modal account
```

### Step 3 — Give Modal your Kaggle credentials (one time)

The corpus lives on Kaggle, so Modal needs a Kaggle API key to fetch it once.

1. Create a Kaggle account, then go to **kaggle.com → your avatar → Settings →
   API → "Create New Token"**. This downloads a `kaggle.json` file containing
   your `username` and `key`.
2. Accept this dataset's license (click **Download** once while logged in):
   <https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish>
   (otherwise the upload step returns `403 Forbidden`).
3. Store those two values as a Modal secret named **`kaggle-credentials`** — the
   env-var names must be exactly `KAGGLE_USERNAME` and `KAGGLE_KEY`:

```bash
modal secret create kaggle-credentials \
    KAGGLE_USERNAME=<your-kaggle-username> \
    KAGGLE_KEY=<your-kaggle-api-key>
```

(You can also create it from the Modal dashboard: **Secrets → Create → Custom**,
with the same two keys.)

### Step 4 — Upload the dataset (one time, shared across all versions)

```bash
make upload-dataset        # Simple-English Wikipedia (~178 MB) → shared Modal Volume
```

This downloads the corpus into the shared `cglm-data` Volume once; every model
version reuses it, so you never repeat this step.

### Step 5 — Deploy + train

```bash
make deploy                # modal deploy + bootstrap (fetch corpus slice → train → register)
make logs                  # tail the running web app; prints the public URL
```

`bootstrap` samples a corpus slice, trains the model, writes `model.clm` +
`metrics.json` to the shared `cglm-data` Volume, and records a registry entry.
When it finishes, `make logs` prints the public URL — open it for the **Chat /
Dashboard / Explore** console, and append `/docs` for the interactive
architecture page.

> **Versioning.** Every `VERSION` bump in `modal_app.py` creates an independent
> app `clm-chat-<VERSION>`; all versions share one Volume (per-version subdirs +
> a shared `registry.json`). A new version can inherit the previous version's
> weights until it is retrained. Push code without touching training state with a
> plain `modal deploy modal_app.py`; use `make redeploy` to force a retrain.

---

## REST API (served by `WebApp.serve`)

| Method | Path | Body / query | Response |
|---|---|---|---|
| `GET`  | `/` | — | Console UI (Chat / Dashboard / Explore) |
| `GET`  | `/docs` | — | Interactive architecture documentation |
| `POST` | `/predict` | `{"tokens":["the","cat"],"topn":5}` | ranked next-word predictions |
| `POST` | `/generate` | `{"tokens":["the","cat"],"n":10}` | continued text |
| `GET`  | `/qa?question=…` | — | intent-routed question answer |
| `GET`  | `/reason?…` | — | knowledge-space reasoning (walk / analogy / plan) |
| `GET`  | `/similar?word=cat&k=6` | — | nearest words by fingerprint overlap |
| `GET`  | `/fingerprint?word=cat` | — | active SDR bits (for the Explore viz) |
| `GET`  | `/stats` | — | levels, vocab, segments, neuromod state |
| `GET`  | `/metrics` | — | per-epoch training metrics |
| `GET`  | `/registry` | — | all deployed versions |
| `POST` | `/train` | — | spawn a live training run (non-blocking) |
| `GET`  | `/status` | — | live training phase / progress |
| `POST` | `/reload` | — | hot-swap the freshly-trained model into the web container |
| `GET`  | `/health` | — | `{"ok": true}` |

---

## Run locally (no Modal, no GPU)

Optional — this path skips Modal entirely and runs the benchmarks, tests, and
Python API on your own machine. The only dependency is numpy (clone the repo
first, per Step 1 above):

```bash
pip install numpy

# benchmarks
python -m benchmarks.run                      # all bundled datasets
python -m benchmarks.run --dataset motifs --epochs 15
python -m benchmarks.run --scaling            # accuracy vs. ensemble size

# tests (each module is independently tested)
python -m tests.test_core
python -m tests.test_persist
python -m tests.test_reasoning                # VSA, grid, transitive inference, analogy
```

Python API:

```python
from core.hierarchy import HierarchicalCLM
model = HierarchicalCLM(n_levels=2, strides=(1, 4), n_units=4,
                        col_dim=2048, encoder="semantic",
                        kwta_k=42, replay_cap=512)
model.train(corpus_sequences, epochs=10)
model.predict_next(["the", "cat"], topn=3)
model.generate(["the", "cat"], n=5)
model.similar("cat", k=5)
```

---

## Configuration (`MODEL_CONFIG` in `modal_app.py`)

`col_dim` is the single source of truth for SDR width (the encoder is forced to
match it). Current v5.1 defaults:

| Parameter | Value | What it controls |
|---|---|---|
| `n_levels` / `strides` | `2` / `(1,4)` | hierarchy depth & temporal scales (token, phrase) |
| `n_units` | `3` | voting columns per level |
| `col_dim` | `2048` | SDR width / mini-column count (representation capacity) |
| `cells_per_col` | `4` | context depth per mini-column (disambiguation capacity) |
| `fp_bits` / `index_bits` | `21` / `7` | fingerprint bits / fixed identity-core bits |
| `window` | `4` | semantic co-occurrence window |
| `kwta_k` | `42` | lateral inhibition: winners kept |
| `activation_threshold` | `5` | connected synapses needed to fire a segment |
| `syn_per_seg` / `max_segs` | `12` / `8` | per-cell synaptic capacity (drives memory) |
| `replay_cap` | `256` | hippocampal replay buffer size |
| `use_spatial_pooler` | `True` | learn SDR representations (fit once, frozen, cached) |
| `pred_dec` | `0.01` | punishment for wrong predictions (nearly 2× held-out top-1) |

> **Memory per column** ≈ `col_dim × cells_per_col × max_segs × syn_per_seg ×
> 8 bytes`. Keep `max_segs` modest on Modal; raise the container `memory` for
> larger models.

---

## Why it stays fast (and is GPU-ready)

Every signal is a binary SDR (~1% active). The hot path — scoring a dendritic
segment — is a single numpy gather + reduce over fixed-shape arrays:

```python
syn_active = src[self.seg_idx]                          # fancy-index gather
conn_ov    = (syn_active & connected & valid).sum(-1)   # reduce
```

Because all column state lives in regular `(n_cells, max_segs, syn_per_seg)`
arrays, `core/xp.py` swaps `numpy` for `cupy` on GPU workers with **no
algorithmic change** — the same code runs ~5–10× faster on a T4. Compiled cupy
kernels are cached to the Volume so the first-run JIT cost is paid only once.

---

## Persistence

Segment arrays are large but ~99.9% empty, so models are saved as **compressed
`.npz`** (`persist/store.py`) — roughly 1000× smaller than raw pickle.

```python
from persist.store import save_model, load_model
save_model(model, "model.clm")       # directory of compressed .npz blobs
model = load_model("model.clm")       # ready for inference + warm-start
model.train(more_corpus, epochs=5)    # incremental learning continues
```

On Modal the checkpoint lives on the `cglm-data` Volume at
`/data/<VERSION>/model.clm`, loaded on web-container cold start. A **nightly
ingest cron** folds a capped batch of new corpus sequences into the model without
retraining, and appends fresh capacity (a new voting unit) when the model
saturates.

---

## Module map

```
clm/
├── core/
│   ├── sdr.py            SDR primitives: overlap, kwta, make_sdr, convert
│   ├── encoder.py        TokenEncoder + SemanticEncoder (distributional fingerprints)
│   ├── spatial_pooler.py SpatialPooler — learned SDR representations (HTM input layer)
│   ├── column.py         CorticalColumn — vectorised HTM temporal memory (seq learning)
│   ├── pooling.py        ContextPool — slow "output layer" topic bias (Hawkins 2017)
│   ├── modulation.py     NeuromodSignal — surprise-gated plasticity
│   ├── replay.py         HippocampalBuffer — surprise-prioritised replay
│   ├── hierarchy.py      HierarchicalCLM — levels, projections, voting, inference
│   ├── grid.py           GridLocation — multi-scale path-integration code
│   ├── displacement.py   GridSpace + Relation — reference frames, composable relations
│   ├── vsa.py            Vector Symbolic Architecture: bind/unbind/bundle/permute
│   ├── reasoning.py      ConceptSpace, walk/analogy/plan, ground_model, active_infer
│   ├── qa.py             SemanticQA — intent-routed question answering
│   └── xp.py             numpy/cupy shim (CPU ↔ GPU, zero code duplication)
├── persist/store.py      save_model / load_model — compressed .npz store
├── benchmarks/           reproducible corpora, metrics, scaling study
├── tests/                per-module unit tests (core, persist, reasoning)
├── papers/               the four source papers (see above)
├── modal_app.py          Modal deployment: dataset → train → serve → nightly ingest
├── docs_html.py          the interactive /docs architecture page
├── REASONING.md          design roadmap for the reasoning layer
└── Makefile
```

---

## Benchmarks

`benchmarks/run.py` reports **top-1 / top-3 / MRR / perplexity-proxy**, training
time, model size, and segment count across three bundled corpora:

| Dataset | What it tests |
|---|---|
| `grammar` | deterministic transitions (a learnability ceiling) |
| `motifs`  | memorisation + generalisation of repeated phrases |
| `natural` | real text from a bundled corpus file |

---

## License

MIT
