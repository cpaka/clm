"""docs_html.py — Architecture documentation page for the CLM webapp."""

DOCS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>CLM — Architecture</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    --green: #34d399; --yellow: #fbbf24; --pink: #f472b6;
    --orange: #fb923c;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         font-size: 15px; line-height: 1.7; }
  .page { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
  h1 { font-size: 2rem; color: var(--accent); margin-bottom: 0.3rem; }
  h2 { font-size: 1.4rem; color: var(--yellow); margin: 2.5rem 0 0.8rem;
       border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
  h3 { font-size: 1.1rem; color: var(--green); margin: 1.5rem 0 0.5rem; }
  h4 { font-size: 0.95rem; color: var(--orange); margin: 1.2rem 0 0.4rem; }
  p  { margin-bottom: 0.9rem; }
  ul, ol { margin: 0.5rem 0 0.9rem 1.4rem; }
  li { margin-bottom: 0.3rem; }
  code { background: var(--surface); color: var(--pink); padding: 0.1em 0.4em;
         border-radius: 3px; font-size: 0.88em; font-family: 'Consolas', monospace; }
  pre  { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
         padding: 1rem 1.2rem; overflow-x: auto; font-size: 0.82em; line-height: 1.5;
         font-family: 'Consolas', monospace; color: var(--text); margin: 0.8rem 0 1rem; }
  pre .c  { color: var(--muted); }         /* comment */
  pre .h  { color: var(--accent); }        /* highlight */
  pre .g  { color: var(--green); }         /* green */
  pre .y  { color: var(--yellow); }        /* yellow */
  pre .p  { color: var(--pink); }          /* pink */
  pre .o  { color: var(--orange); }        /* orange */
  .badge { display: inline-block; background: var(--surface); border: 1px solid var(--border);
           border-radius: 4px; padding: 0.1em 0.5em; font-size: 0.8em; color: var(--muted);
           margin-right: 0.4em; }
  .callout { background: var(--surface); border-left: 3px solid var(--accent); padding: 0.8rem 1rem;
             border-radius: 0 6px 6px 0; margin: 1rem 0; }
  .callout.warn { border-color: var(--yellow); }
  .callout.green { border-color: var(--green); }
  nav { background: var(--surface); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 2rem;
        border: 1px solid var(--border); }
  nav a { color: var(--accent); text-decoration: none; display: block; padding: 0.2rem 0;
          font-size: 0.9em; }
  nav a:hover { text-decoration: underline; }
  .back { color: var(--muted); font-size: 0.88em; margin-bottom: 1rem; display: block; }
  .back a { color: var(--accent); }
  table { width: 100%; border-collapse: collapse; margin: 0.8rem 0 1rem; font-size: 0.88em; }
  th { background: var(--surface); color: var(--yellow); text-align: left; padding: 0.5rem 0.8rem;
       border: 1px solid var(--border); }
  td { padding: 0.4rem 0.8rem; border: 1px solid var(--border); vertical-align: top; }
  tr:nth-child(even) td { background: #162032; }
</style>
</head>
<body>
<div class="page">

<a href="/" class="back">← back to playground</a>

<h1>Cortical Language Model — Architecture</h1>
<p style="color:var(--muted)">A brain-inspired language model using Sparse Distributed Representations,
HTM temporal memory, and hierarchical cortical columns.</p>

<nav>
  <strong style="color:var(--text)">Contents</strong>
  <a href="#sdr">1. Sparse Distributed Representations</a>
  <a href="#encoder">2. Semantic Encoder — words as fingerprints</a>
  <a href="#column">3. Cortical Column — learning sequences</a>
  <a href="#sequence">4. How sequences are learned step-by-step</a>
  <a href="#neuromod">5. Neuromodulation &amp; replay</a>
  <a href="#hierarchy">6. Hierarchical architecture</a>
  <a href="#ensemble">7. Voting ensemble</a>
  <a href="#scaleup">8. How the model improves</a>
  <a href="#gpu">9. GPU acceleration (V4)</a>
  <a href="#params">10. Key hyperparameters</a>
</nav>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="sdr">1. Sparse Distributed Representations (SDRs)</h2>

<p>The brain represents information as patterns of activity across large neural populations.
Most neurons are silent at any moment — only ~2 % fire.  This <em>sparse distributed</em>
representation has powerful mathematical properties that CLM exploits.</p>

<h3>Dense vs sparse encoding</h3>

<pre>
<span class="c">Dense (one-hot, 1024 bits):  only 1 bit active, no overlap possible</span>
"cat" → [0 0 0 0 0 1 0 0 0 0 … 0 0 0]   (1/1024 bits active)
"dog" → [0 0 0 0 0 0 0 1 0 0 … 0 0 0]   overlap = 0  (no similarity)

<span class="c">Sparse distributed (SDR, 1024 bits, w=21 active):</span>
"cat" → bits { <span class="g"> 12  87 143 201 305 … 21 active bits</span> }   21/1024 = 2%
"dog" → bits { <span class="g"> 12  87 201 310 502 … 21 active bits</span> }   overlap = 14
<span class="c">           ↑   ↑   ↑                                 ↑</span>
<span class="c">       shared bits → semantic similarity encoded as overlap count</span>
</pre>

<h3>Key properties of SDRs</h3>
<ul>
  <li><strong>Overlap = similarity</strong> — words with similar contexts share more bits</li>
  <li><strong>High capacity</strong> — C(1024,21) ≈ 10⁴² possible patterns; no two words collide</li>
  <li><strong>Robust to noise</strong> — a pattern with 17/21 bits active is still recognised</li>
  <li><strong>Composability</strong> — a set of active columns encodes a sentence-level context</li>
</ul>

<div class="callout">
  <strong>Why 2 % sparsity?</strong> At 2 % active bits, the probability of accidental overlap
  between two random SDRs is astronomically small (~10⁻³⁰), so patterns don't interfere.
  Yet the overlap between semantically related words is large enough to drive prediction.
</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="encoder">2. Semantic Encoder — words as fingerprints</h2>

<p>Every word gets a unique <em>fingerprint</em>: a 21-bit SDR within the 1024-bit space.
The fingerprint has two parts:</p>

<pre>
┌─────────────────────────────────────────────────────────────┐
│  1024-bit fingerprint for "cat"                             │
│                                                             │
│  ┌────────────────┐  ┌────────────────────────────────────┐ │
│  │ Identity core  │  │       Context shell                │ │
│  │   7 bits       │  │         14 bits                    │ │
│  │   (FIXED)      │  │   (learned from co-occurrence)     │ │
│  │                │  │                                    │ │
│  │  hash("cat")   │  │   neighbours of "cat" in training  │ │
│  │   → stable     │  │   text determine which 14 positions│ │
│  │   identity     │  │   are active                       │ │
│  └────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
</pre>

<h3>How fingerprints are built</h3>

<ol>
  <li><strong>Co-occurrence window</strong> — for each word, count which other words appear
      nearby in the training text (window ±4)</li>
  <li><strong>Fingerprint matrix</strong> — each word gets a random projection of its
      co-occurrence vector into the 1024-bit space</li>
  <li><strong>Identity core</strong> — 7 bits are reserved and fixed from a hash of the word
      string; they never change even as new co-occurrence data arrives</li>
  <li><strong>Context shell</strong> — the remaining 14 active bits are re-selected whenever
      the co-occurrence accumulator is updated (<em>incremental learning</em>)</li>
</ol>

<div class="callout green">
  <strong>Semantic similarity emerges automatically:</strong>
  "cat" and "dog" appear in similar contexts ("the ___ sat", "my ___ runs").
  Their co-occurrence vectors are similar → their fingerprints share ~10/21 bits.
  "cat" and "democracy" appear in completely different contexts → overlap ≈ 0.
</div>

<h3>Out-of-vocabulary (OOV) words</h3>
<p>Words unseen during training get a random 21-bit fingerprint with no context shell.
They are predicted with lower confidence but don't crash the model.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="column">3. Cortical Column — learning sequences</h2>

<p>Each CLM unit contains one <strong>CorticalColumn</strong> — an HTM temporal memory
implemented as a fixed-size array of synaptic weights.
This is where sequence learning happens.</p>

<h3>Structure: mini-columns and cells</h3>

<pre>
CorticalColumn  (col_dim=1024 mini-columns × cells_per_col=4 cells = 4096 cells)

Mini-column 0        Mini-column 1        …  Mini-column 1023
┌──────────────┐    ┌──────────────┐           ┌──────────────┐
│  cell 0  [seg] [seg] …           │           │  cell 0      │
│  cell 1  [seg] [seg] …           │           │  cell 1      │
│  cell 2                          │           │  cell 2      │
│  cell 3                          │           │  cell 3      │
└──────────────┘    └──────────────┘           └──────────────┘
      ↑                   ↑
  SDR bit 0           SDR bit 1
 "cat" activates   "cat" activates
  this column       this column

Each cell has up to max_segs=8 dendritic segments.  When a cell is full, the
weakest segment (lowest total permanence) is recycled — learning never stops.
Each segment has syn_per_seg=12 synapses connecting to previous winner cells.
</pre>

<h3>Predictive vs burst state</h3>

<pre>
<span class="c">Context: "the ___" was learned previously</span>

PREDICTIVE step (column already learned "the → cat"):
  Token "the" arrives → column 87 is active (part of "the" fingerprint)
  Cell 87·3 has a segment with connected synapses to prev winners from "the"
  Cell 87·3 fires in PREDICTIVE mode → winner = just cell 87·3
  <span class="g">✓ Prediction was correct → only 1 cell per column fires</span>

BURST step (first time seeing "the cat"):
  Token "the" arrives → column 87 activates
  No cell in column 87 is predictive (no segment connects to prev "the" winners)
  ALL 4 cells in column 87 fire → BURST
  <span class="y">⚡ Surprise! Model didn't predict this → ALL cells fire → learning triggered</span>

Burst winner selection (which cell learns the new context):
  1. A cell with a MATCHING segment (potential overlap ≥ min_threshold)
     wins on potential — the same cell keeps learning the same transition
  2. Otherwise the LEAST-USED cell (fewest segments) wins — novel contexts
     spread across the column's cells instead of merging onto one cell
  3. At sequence start (no context at all) cell 0 is chosen deterministically
     so chains are reproducible across epochs and at inference
</pre>

<h3>Permanence: the synaptic learning signal</h3>

<pre>
Each synapse has a permanence value ∈ [0, 1].
Synapse is "connected" if permanence ≥ connected_threshold = 0.5.

During learning (Hebbian rule):
  If presynaptic cell was active AND postsynaptic cell is winner:
    permanence += inc = 0.10   <span class="g">← strengthen</span>
  If presynaptic cell was NOT active AND postsynaptic cell is winner:
    permanence -= dec = 0.02   <span class="y">← weaken</span>
  If a segment PREDICTED a column that did NOT activate:
    permanence -= pred_dec = 0.01   <span class="y">← punish false prediction</span>

Punishment keeps the predictive set sharp: without it, spurious segments
accumulate until a third of all columns are "predicted" at every step and
decoding becomes impossible.

No learning happens on the first token of a sequence — there is no previous
context to connect to, so adapting would only decay permanences and grow
empty (zero-synapse) segments.

After enough repetitions of "the → cat":
  Segment in "cat" column → prev "the" winners:
  permanences → [0.8, 0.9, 0.7, …]  (12 synapses all ≥ 0.5)
  Connected overlap = 12 ≥ activation_threshold = 5
  → Cell fires PREDICTIVELY next time "the" is seen
</pre>

<h3>Vectorised implementation</h3>
<p>Instead of looping over columns in Python, the column step is a single GPU/numpy
gather-and-reduce:</p>
<ol>
  <li><strong>Gather</strong> — index <code>seg_idx[active_cells]</code> to get all synapse sources for active cells</li>
  <li><strong>Reduce</strong> — count connected active synapses per (cell, segment) pair</li>
  <li><strong>Select winners</strong> — predicted cells win; burst cells pick the best-matching
      segment's cell, or the least-used cell for novel contexts</li>
  <li><strong>Batch adapt</strong> — update permanences for all winning (cell, segment) pairs at once;
      punish predictive segments whose column did not activate (<code>pred_dec</code>)</li>
</ol>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="sequence">4. How sequences are learned step-by-step</h2>

<p>Let's trace "the cat sat" through 20 training repetitions:</p>

<pre>
<span class="c">EPOCH 1 — First time seeing "the cat sat"</span>

t=0: token="the"  fingerprint bits={3, 87, 143, 201, …}
  → cols 3, 87, 143, 201 … are active
  → no prev winners → all columns BURST (surprise=100%)
  → winner anchored on cell 0 per column: w₀ = {3·0, 87·0, 143·0, …}
  → <span class="c">NO learning (no previous context to connect to)</span>

t=1: token="cat"  fingerprint bits={12, 87, 201, 310, …}
  → cols 12, 87, 201, 310 … are active
  → check segments: no segment connects to w₀ → BURST
  → winners (least-used cells): w₁ = {12·0, 87·3, 201·2, …}
  → <span class="y">LEARN: grow new segment in each winner cell, connect to w₀</span>

t=2: token="sat"  fingerprint bits={5, 88, 150, 300, …}
  → cols 5, 88, 150, 300 … are active
  → check segments: no segment connects to w₁ → BURST
  → winners: w₂ = {5·4, 88·0, 150·2, …}
  → <span class="y">LEARN: grow segment, connect to w₁</span>

<span class="c">EPOCH 10 — After many repetitions</span>

t=0: token="the"  → cols burst (no prior context)  winners=w₀ (same cells)

t=1: token="cat"  → segments in cat-cols have synapses connecting to w₀
  connected overlap ≥ 5 → cells in cat-cols PREDICT before "cat" arrives
  → only w₁ cells fire (narrow, single-cell-per-column)
  <span class="g">✓ "cat" was predicted given "the"!</span>

t=2: token="sat"  → segments in sat-cols have synapses connecting to w₁
  → cells PREDICT, only w₂ fires
  <span class="g">✓ "sat" was predicted given "the cat"!</span>

<span class="c">Prediction accuracy = fraction of steps where prediction was correct</span>
<span class="c">Burst rate = fraction of columns that fire all cells (surprise)</span>
</pre>

<h3>What the column actually learns</h3>
<p>The column does NOT learn "after word X comes word Y". It learns
"after <em>these specific winner cells were active</em>, <em>these other cells</em> should fire
predictively". Because each word's winner cells are unique (the same fingerprint bits
are active, but which specific cells were chosen depends on context), the column
implicitly encodes <em>context × word</em> transitions.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="neuromod">5. Neuromodulation &amp; Hippocampal Replay</h2>

<h3>Neuromodulatory signal (dopamine-like)</h3>
<p>The <code>NeuromodSignal</code> tracks the recent burst rate (fraction of surprised columns).
It acts like a dopamine signal: high surprise → high plasticity, low surprise → consolidation.</p>

<pre>
burst_rate high (model surprised):
  modulation → 2.0  ← inc/dec learning rates scaled UP
  "Pay more attention, this is new"

burst_rate low (model predicting well):
  modulation → 0.2  ← learning rates scaled DOWN
  "Things are familiar, consolidate"
</pre>

<h3>Hippocampal replay</h3>
<p>When the model is surprised (burst_rate > 0.3), the sequence is stored in a
<code>HippocampalBuffer</code> (capacity 256). Every <code>replay_every=2</code> epochs, stored sequences
are replayed with boosted plasticity (modulation=1.5). This mirrors hippocampal
consolidation in biological learning — surprising episodes are rehearsed to
strengthen long-term synaptic weights.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="hierarchy">6. Hierarchical Architecture</h2>

<pre>
Input tokens:  the   cat   sat   on   a   mat

Level 0 (stride=1, TOKEN granularity)
  Unit 0  ──┬──┬──┬──┬──┬──  (processes every token)
  Unit 1  ──┘  │  │  │  │
  Unit 2  ─────┘  │  │  │
                  │  │  │
     every 4 tokens → SparseProjection → Level 1 feature SDR
                  │
Level 1 (stride=4, PHRASE granularity)
  Unit 0  ─────┬─────  (processes every 4th token → phrase rhythm)
  Unit 1  ─────┘
  Unit 2
  (learns phrase-level patterns: "the cat sat" as one unit)
</pre>

<h3>SparseProjection between levels</h3>
<p>Level-0 winner cells (4096-dimensional dense bool) are projected through a fixed
random sparse matrix to produce a 21-bit feature SDR for Level 1. This is analogous
to thalamo-cortical projections — a fixed anatomical wiring that compresses
fine-grained temporal activity into a coarser phrase-level signal.
Each unit projects <em>its own</em> winners upward, and the upper-level columns are
reset per sequence — training and inference run the identical per-unit pathway.</p>

<h3>Grid location cells (currently disabled)</h3>
<p>The architecture includes a <code>GridLocation</code> module that generates position-sensitive
SDRs from co-prime periods (7, 11, 13, …, 23). However, these are NOT fed into
temporal memory segments — passing <code>loc_sdr=[]</code> to every column step.
If enabled, each segment would become position-specific ("bigram at position 3"),
preventing generalisation. Disabled = position-agnostic predictions.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="ensemble">7. Voting Ensemble (Thousand Brains Theory)</h2>

<pre>
┌─────────────────────────────────────────────────────────────┐
│                   Voting ensemble (n_units=3)               │
│                                                             │
│  Unit 0 ──▶ score_next("the cat") ──▶ {sat:0.8, ran:0.3}  │
│  Unit 1 ──▶ score_next("the cat") ──▶ {sat:0.6, ran:0.5}  │
│  Unit 2 ──▶ score_next("the cat") ──▶ {sat:0.7, on:0.4}   │
│                           ↓                                 │
│               Aggregate raw scores                          │
│               {sat:2.1, ran:0.8, on:0.4}                   │
│                           ↓                                 │
│               k-WTA (k=42): keep top-42 tokens              │
│               Lateral inhibition sharpens prediction        │
│                           ↓                                 │
│               Top-1: "sat" (score=2.1)                      │
└─────────────────────────────────────────────────────────────┘
</pre>

<p>Each unit trains on the same corpus with a different random seed. The ensemble
vote averages over different learned representations, reducing variance and
improving robustness to ambiguous inputs. This mirrors the Thousand Brains Theory:
each cortical column maintains a complete model of the world; consensus emerges
from voting across columns.</p>

<h3>Decoding predictions into words</h3>
<p>Each unit's temporal memory outputs <em>predicted mini-columns</em> with a graded
strength (the connected overlap of the segments that predict them). A candidate
word's score is the strength-weighted fraction of <strong>its own fingerprint bits</strong>
that are predicted:</p>

<pre>
score(word) = Σ strength(bit)  over  bit ∈ fingerprint(word)   ÷   fp_bits
              (strengths normalised to the strongest predicted column)

  • Identity-core bits are weighted 3× — cores are unique per word, so
    they separate the exact predicted word from its semantic neighbours
    (shell bits are shared between similar words by design).
  • Context echoes are demoted: tokens already in the context ×0.35,
    the immediately preceding token ×0.1 — the model must predict the
    NEXT word, not re-emit the current one.
</pre>

<div class="callout">
  <strong>Why coverage, not per-bit rarity:</strong> an earlier decoder summed an
  inverse-popularity (IDF) weight per predicted bit. With ~41 words sharing each
  bit (2 001 words × 21 bits in 1 024 positions), a fully-predicted frequent word
  scored <em>less</em> than a wrong word owning one rare bit — top-1 accuracy pinned
  at exactly 0 %. Scoring each word by how much of <em>its own</em> fingerprint is
  predicted removes that bias.
</div>

<h3>Parallel training</h3>
<p>In <code>make train-parallel</code>, each unit trains in its own Modal container simultaneously.
After all units finish, they are assembled into the ensemble and the merged model is
saved. This cuts wall-clock training time from <em>N × per-unit time</em> to
<em>max(per-unit time)</em>.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="scaleup">8. How the model improves</h2>

<h3>More data = more segments = wider vocabulary coverage</h3>
<pre>
Bigram coverage = fraction of (word_A, word_B) pairs seen in training:

  1M chars,  5K seqs,  2K vocab → ~1.3% bigram coverage
  2M chars, 10K seqs,  2K vocab → ~2.5% (V4.1 target)
  5M chars, 20K seqs,  2K vocab → ~6%   (V4.2 target)

Accuracy on random probe ≈ bigram coverage × recall_per_seen_pair
</pre>

<h3>Incremental learning (#7)</h3>
<p>The semantic encoder stores its raw co-occurrence accumulator on disk.
When new text arrives via <code>make ingest</code>, <code>enc.update(new_seqs)</code> folds it in
without discarding old co-occurrences. Identity-core bits are fixed, so the
column's existing segments remain valid — the new context shell bits are
added incrementally.</p>

<h3>Saturation &amp; capacity growth (#9, #10)</h3>
<p>When the burst rate plateaus over 3 epochs (is_saturated=True), the model has
learned as much as its current capacity allows. The daily ingest cron job
automatically appends a new CLMUnit to the ensemble, adding fresh capacity.
Within a full cell, new contexts recycle the weakest segment (lowest total
permanence) instead of being silently dropped, so recent data always lands.</p>

<h3>Corpus scaling plan (V4)</h3>
<table>
  <tr><th>Version</th><th>Corpus</th><th>Seqs</th><th>Expected top-1</th><th>Training</th></tr>
  <tr><td>V4.0</td><td>1M chars Wikipedia</td><td>5K</td><td>~0.5%</td><td>GPU baseline</td></tr>
  <tr><td>V4.1</td><td>2M chars Wikipedia</td><td>10K</td><td>~1–2%</td><td>bump + redeploy</td></tr>
  <tr><td>V4.2</td><td>5M chars Wikipedia</td><td>20K</td><td>~3–6%</td><td>bump + redeploy</td></tr>
</table>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="gpu">9. GPU Acceleration (V4)</h2>

<p>The column hot path is a single gather-and-reduce over the synaptic weight arrays.
This maps perfectly onto GPU SIMD execution:</p>

<pre>
CPU step (per token, n_active_cols≈42):
  gather seg_idx[active_cells]     →  107K int32 lookups in seg_idx
  reduce (active & connected)      →  sum along syn_per_seg axis
  argmax for predictive/burst      →  small array
  batch permanence update          →  scatter add
  Total: ~70–110 μs on CPU

GPU step (same operations, cupy):
  Transfer feature_sdr (42 ints)   →  CPU→GPU, ~2 μs
  Gather + reduce (107K elements)  →  T4 tensor cores, ~3 μs
  Scatter update                   →  ~2 μs
  Total: ~10–20 μs on GPU
  Speedup: ~5–10× over CPU numpy
</pre>

<h3>cupy / numpy shim (core/xp.py)</h3>
<p>All array operations in <code>core/column.py</code> import from <code>core/xp.py</code>:</p>
<pre>
import xp              # cupy on GPU workers, numpy on CPU workers
                       # Falls back silently — no code change needed
</pre>

<p>Grow operations (rare: only for burst cells without segments) run on CPU to avoid
complex GPU random-number management. Device-to-host transfers are isolated
to the small candidate arrays, not the full weight matrices.</p>

<!-- ═══════════════════════════════════════════════════════════════════ -->
<h2 id="params">10. Key Hyperparameters</h2>

<table>
  <tr><th>Parameter</th><th>Value</th><th>Effect</th></tr>
  <tr><td><code>col_dim</code></td><td>1024</td><td>SDR width; also the encoder fingerprint dimension</td></tr>
  <tr><td><code>cells_per_col</code></td><td>4</td><td>Context depth; more cells = more disambiguation capacity</td></tr>
  <tr><td><code>fp_bits</code></td><td>21</td><td>Active bits per fingerprint; controls sparsity (~2 %)</td></tr>
  <tr><td><code>index_bits</code></td><td>7</td><td>Fixed identity core bits in each fingerprint (3× weight at decode)</td></tr>
  <tr><td><code>window</code></td><td>4</td><td>Co-occurrence window for semantic encoder</td></tr>
  <tr><td><code>max_segs</code></td><td>8</td><td>Max segments per cell; weakest segment recycled when full</td></tr>
  <tr><td><code>syn_per_seg</code></td><td>12</td><td>Synapses per segment; must overlap ≥ 5 to activate</td></tr>
  <tr><td><code>activation_threshold</code></td><td>5</td><td>Connected synapses needed for a segment to fire</td></tr>
  <tr><td><code>pred_dec</code></td><td>0.01</td><td>Punishment for segments that predicted a column that stayed inactive</td></tr>
  <tr><td><code>vocab_size</code></td><td>2000</td><td>Rare words → &lt;UNK&gt;; improves bigram coverage</td></tr>
  <tr><td><code>n_units</code></td><td>3</td><td>Voting ensemble size</td></tr>
  <tr><td><code>n_levels</code></td><td>2</td><td>Hierarchy depth (token + phrase level)</td></tr>
  <tr><td><code>replay_cap</code></td><td>256</td><td>Hippocampal buffer capacity (surprising sequences)</td></tr>
</table>

<div class="callout">
  <strong>Memory footprint per unit:</strong>
  <code>n_cells × max_segs × syn_per_seg × 8 bytes</code> (idx + perm) =
  <code>4096 × 8 × 12 × 8</code> = <strong>~3 MB per unit</strong>.
  Three units + two levels = ~19 MB total.
</div>

</div><!-- /page -->
</body>
</html>
"""
