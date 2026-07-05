# Reasoning as Movement Through a Knowledge Space — Design Roadmap

> Working document. Captures the hypothesis, the SDR/VSA machinery to realize it,
> what already exists in this repo vs. what's missing, and a falsifiable first
> experiment. Intended as the starting point for building the reasoning layer of
> the CLM on top of the cortical-column core.

## Status: first slice IMPLEMENTED ✅

The VSA algebra, grid reference frames, and the first reasoning tasks are built
and passing their falsifiable milestone tests (§6):

| Module | Contents |
|---|---|
| `core/vsa.py` | bind / unbind / bundle / permute / similarity, `RoleBook`, `Codebook` (cleanup) |
| `core/displacement.py` | `GridSpace` (N-axis grid), `Relation` (learned displacement, compose) |
| `core/reasoning.py` | `ConceptSpace` (+ `ground`), `walk` / `infer_relation` (transitive), `analogy`, `SentenceSpace` (SVO), `plan` / `RelationSet` / `ReasoningPolicy` (goal-steered walk), `sp_representations` |
| `tests/test_reasoning.py` | 14 tests: bind/unbind, transitive generalisation, analogy, VSA-beats-raw-union, multi-relation planning, predator avoidance, policy steering, grounding recovers structure, SP→grounding preserves similarity |

Run: `python -m tests.test_reasoning` (12/12 passing). Verified milestones:
transitive inference generalises to **unseen** distances from adjacent-only
training; `man:king :: woman:queen` by displacement transfer; VSA keeps
"cat chased mouse" ≠ "mouse chased cat" where raw SDR union collapses them; and
goal-steered `plan()` chains **different** relations to reach a goal
(`man --make_royal--> king --make_female--> queen`) and routes **around a
predator** (the survival/avoidance loop) via best-first search over the grid
frame's own distance metric.

**Grounding in learned representations (DONE):** `ConceptSpace.ground()` now
discovers grid coordinates directly from learned SDRs (e.g. Spatial Pooler
outputs, via `sp_representations`) by embedding their similarity structure onto
principal axes and discretising to the grid — so the space, its axes, and its
metric are *learned from data*, not hand-placed.  Verified: grounding recovers
latent factors well enough for analogy on learned coordinates, and the full
input→SpatialPooler→grounding pipeline preserves the learned nearest neighbour.
A useful honest finding: reasoning quality is bounded by representation quality —
a lightly-trained SP captured the dominant factor and collapsed a weaker one, so
the grounded space reflected exactly what the SP had learned (no more, no less).

**Next (not yet built):** scale from toy concept spaces + SVO triples toward
dependency-parsed language; drive grounding from the *corpus-trained* Spatial
Pooler (deployed in v5.0) rather than synthetic inputs; and couple the reward in
`ReasoningPolicy` to the temporal memory's `NeuromodSignal` for end-to-end
active inference.

---

## 1. The hypothesis

**Reasoning is an evolution of the grid-cell reference-frame mechanism.**

Archaic evolutionary pressure: an animal survives by (a) locating itself, its
prey, and its predators, (b) predicting the movement of other entities, (c)
recognizing pattern/position, and (d) triggering the right motor sequence.
Success = a better internal model of the environment (where food/threats are)
plus better action sequences to reach food and avoid threats. Layered on top:
a **semantic** mechanism that categorizes and tags environment elements, and a
**reinforcement/avoidance** mechanism that assigns value.

Grid cells give animals a *reference frame* for physical space (Numenta / Hawkins
2019 makes reference frames the basis of all cortical representation).

**Claim:** the same machinery was reused for abstract thought. Knowledge becomes
a *space* — with topology, categories, relations, "colors", "fluids", species,
mineral/vegetal/animal, etc. **Attention moves through that space** the way an
animal moves through an environment, and the objective is the same: **improve
prediction and reduce error** (toward a goal / reward).

This is not fringe — it is a supported research frontier (see §7).

---

## 2. The reframe: a reference frame = 4 SDR ingredients + 1 loop

A knowledge space you can *reason in* decomposes into things each representable
as an SDR:

| Ingredient | Meaning | SDR realization | In repo today |
|---|---|---|---|
| **Element** | a concept / object / feature | learned fingerprint SDR | ✅ `core/encoder.py` (SemanticEncoder) + `core/spatial_pooler.py` (learned) |
| **Location** | a point in the space | multi-scale grid SDR (co-prime rings) | ✅ `core/grid.py` (`GridLocation`, path integration) |
| **Binding** | "feature *at* location" | reversible bind operator | ✅ `encoder.bind()` (permutation) — currently unused |
| **Displacement** | a *relation* / a move | phase shift on the grid code | ⚠️ partial (`grid.advance` / `grid.at`) — not learned as relations |

**The loop that makes it reasoning:**

> **Reasoning = a path of displacements through the space that reduces
> prediction error toward a goal.**

```
start at query concept's location
repeat:
    apply a displacement (a candidate relation / "move")
    read the SDR now at that location   (predict / unbind)
    score it against the goal / expectation  (error, value)
    keep the best moves (beam)          (attention)
until goal reached or error minimized
```

This is **active inference** (Friston) expressed on grid-coded SDRs, and it is
the geometric form of the "multi-step inference" the CLM currently lacks.

---

## 3. The operation set — Vector Symbolic Architecture (VSA)

Everything above reduces to a small, mathematically grounded algebra of SDR
operations (VSA / Hyperdimensional Computing — Kanerva; Plate's HRR). Mapping to
this repo:

| Operation | Meaning | SDR implementation | Have it? |
|---|---|---|---|
| **BIND** `a⊗r` | attach feature to a role/location | permutation of `a` by role `r` | ✅ `encoder.bind()` |
| **UNBIND** `b⊗r⁻¹` | "what feature is at this location?" | inverse permutation | ⚠️ add inverse of `_perm` |
| **BUNDLE** `a⊕b⊕…` | superpose a *set* (object = many feature@location) | OR, then k-WTA to re-sparsify | ✅ `sdr.kwta` |
| **SIMILARITY** | how related are two SDRs | sparse overlap | ✅ `sdr.overlap`, `sdr.batch_overlap` |
| **DISPLACE** | move / apply a relation | grid phase offset (learned per relation) | ⚠️ `grid.advance`; make it a learned per-relation transform |
| **CLEANUP** | snap a noisy SDR back to the nearest stored concept | associative retrieval | ✅ `_inv` inverted index (in `CLMUnit`) |

### Why this is the payoff: reasoning becomes arithmetic

- **An object** is a *structured* representation, not a bag of words:
  ```
  object = BUNDLE( BIND(loc1, feat1), BIND(loc2, feat2), ... )
  ```
- **A relation** ("capital-of", "is-a", "next-in-sequence") is a learned
  **displacement** `D`. Because grid codes are periodic, displacements
  **compose by phase addition**:
  ```
  D(A→B) then D(B→C)  ==  D(A→C)          # transitive inference for free
  ```
- **Analogy** ("king : queen :: man : ?"):
  ```
  D = UNBIND(king, queen)      # extract the relation as a displacement
  guess = DISPLACE(man, D)     # apply it elsewhere
  answer = CLEANUP(guess)      # -> woman
  ```
  This is the `king - man + woman = queen` trick, but with **sparse, binding-
  based, learnable, structured** SDRs instead of dense embeddings.

Value / reward (the predator-prey RL intuition) steers which displacement to
apply next. The existing `core/modulation.py` (dopamine-like `NeuromodSignal`)
is the natural hook for that value signal.

---

## 4. What's missing in the repo (small, concrete)

Already present: elements (encoder + Spatial Pooler), locations (grid), binding
(permutation), similarity, cleanup (inverted index), a value signal (neuromod).
The three gaps:

1. **Bind feature⊗location into the temporal memory** — the 2019 theory's
   "feature at location." Location was disabled as a *positional tag*; here it
   returns as a *binding* (structure), not a position.
2. **Learn displacements as reusable relations** — a `Displacement` / relation
   object learned as the phase-transform mapping one concept's grid location to
   another's. This is what turns "movement" into "reasoning operators."
3. **The inference walk** — a beam search over displacements: from a query,
   apply candidate relations, read + cleanup the result, score by goal/error,
   iterate.

---

## 5. Proposed implementation (first slice)

Keep the core untouched; add a thin reasoning layer.

```
core/vsa.py            # ~100 lines: bind, unbind, bundle, cleanup, similarity
core/displacement.py   # learn/store per-relation grid displacements
core/reasoning.py      # the active-inference walk (beam over displacements)
```

Sketch:

```python
# core/vsa.py
def bind(a_sdr, role_perm): ...          # permutation binding (reuse encoder._perm)
def unbind(b_sdr, role_perm): ...        # inverse permutation
def bundle(*sdrs, k): ...                # OR + kwta re-sparsify
def similarity(a, b): ...                # sparse overlap
def cleanup(noisy_sdr, inverted_index): ...  # nearest stored concept

# core/displacement.py
class Relation:
    """A learned displacement D such that loc(B) ≈ apply(loc(A), D)."""
    def fit(self, pairs): ...            # average phase offset over (A,B) examples
    def apply(self, loc_sdr): ...        # grid phase add
    def compose(self, other): ...        # D1 ∘ D2  (phase addition -> transitivity)

# core/reasoning.py
def walk(query, goal, relations, space, beam=8, depth=4):
    """Beam search of displacements minimizing error toward goal."""
```

The Spatial Pooler (`core/spatial_pooler.py`, added this session) is a
prerequisite: reasoning needs *learned* representations, not fixed co-occurrence
fingerprints, so concepts sit in a semantically organized space.

---

## 6. First experiment (falsifiable, before language)

**Do not start with free-text language.** Start with the standard proving ground
for reference-frame reasoning — small enough to verify with hard numbers, the
same discipline used for the bug fixes / four improvements / Spatial Pooler.

Toy "concept space" — e.g. a family tree, or a 2-D grid of related concepts:

1. Encode each concept as an SDR element; assign grid locations so related
   concepts are near.
2. Learn a few **displacement-relations** (e.g. `parent_of`, `sibling_of`; or
   grid moves `north/east`).
3. Test held-out:
   - **Transitive inference** — "A is grandparent of C?" via `D(parent)∘D(parent)`.
   - **Analogy** — "A : B :: C : ?" via unbind-displacement-cleanup.

**Success criterion:** displacement composition yields correct transitive /
analogical answers on held-out pairs → the grid-cell-as-reasoning hypothesis
holds in SDR form → scale up. **If it fails, the hypothesis is cheaply
falsified.**

---

## 7. Grounding — this is a real research program (not fringe)

- **Constantinescu, O'Reilly & Behrens (2016), *Science*** — humans show a
  grid-like hexagonal code in entorhinal cortex when navigating an *abstract*
  concept space. Direct evidence.
- **Behrens et al. (2018), *Neuron*, "What is a cognitive map?"** — the
  hippocampal-entorhinal system as a general engine for relational reasoning.
- **Whittington et al. (2020), the Tolman-Eichenbaum Machine (TEM)** — a working
  model unifying spatial navigation and relational/transitive reasoning with
  grid-cell-like codes. Closest existing thing to this design.
- **Hawkins, Lewis, Klukas, Purdy & Ahmad (2019)** — *A Framework for Intelligence
  and Cortical Function Based on Grid Cells* (in `papers/fncir-12-00121.pdf`).
  Reference frames for everything, including concepts.
- **Kanerva (2009), Plate (1995)** — Vector Symbolic Architectures / Holographic
  Reduced Representations: the bind/bundle/cleanup algebra used above.

### Honest maturity note

The pieces (VSA algebra, grid codes) are solid and the neuroscience supports the
hypothesis, but **no one has assembled them into a general reasoning engine that
rivals LLMs.** TEM works on *toy* relational tasks and is trained by gradient
descent, not pure HTM. So this is a genuine multi-year research bet — but unlike
"scale the CLM to GPT-4," it has **buildable, falsifiable milestones** (§6),
and each milestone reuses machinery this repo already has.

---

## 8. Repo assets to reuse

| Need | Existing asset |
|---|---|
| Element SDRs | `core/encoder.py` (SemanticEncoder), `core/spatial_pooler.py` (learned) |
| Locations + path integration | `core/grid.py` (`GridLocation.at`, `.advance`) |
| Binding (permutation) | `core/encoder.py` `bind()`, `_perm()` |
| Bundle / re-sparsify | `core/sdr.py` `kwta` |
| Similarity | `core/sdr.py` `overlap`, `batch_overlap` |
| Cleanup memory | `CLMUnit._inv` inverted index |
| Sequence/temporal prediction | `core/column.py` (CorticalColumn) |
| Value / reward steering | `core/modulation.py` (NeuromodSignal) |
| Context "output layer" | `core/pooling.py` (ContextPool, added this session) |

Start with `core/vsa.py` + the toy transitive-inference/analogy experiment.
