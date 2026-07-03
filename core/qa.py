"""
core/qa.py — Multi-level semantic question answering.

Architecture
============

                   ┌─────────────────────────────────┐
  raw question ──▶ │   PromptEncoder                 │
                   │  ┌───────────────────────────┐  │
                   │  │ L1: word SDRs             │  │  ← fingerprint per token
                   │  │    "castle" → [4,23,67…]  │  │    (+ fitted/OOV flag)
                   │  ├───────────────────────────┤  │
                   │  │ L2: global intent SDR     │  │  ← k-WTA bundle of all
                   │  │    bundle + k-WTA → 21b   │  │    word fingerprints;
                   │  └───────────────────────────┘  │    encodes WHAT topic
                   └──────────────┬──────────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │   IntentClassifier              │
                   │  structure:  "what is" → def    │
                   └──────────────┬──────────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │   ResponsePlanner               │
                   │  intent → max_tokens            │
                   │         → min_tokens            │  ← decided by the model,
                   │         → stop_on_punct         │    not by the caller
                   │         → confidence threshold  │
                   └──────────────┬──────────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │   PromptSelector                │
                   │  build candidate prompts        │
                   │  score each with predict_next   │  ← picks the phrasing the
                   │  return highest-confidence one  │    model is most certain of
                   └──────────────┬──────────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │   PlannedGenerator              │
                   │  generate token by token        │
                   │  stop when:                     │
                   │    • confidence collapses        │  ← adaptive stopping
                   │    • relative drop > 70%        │
                   │    • sentence boundary (intent) │
                   │    • min_tokens reached          │
                   └──────────────┬──────────────────┘
                                  │
                              answer dict
                         (question, intent, subject,
                          global_sdr, related, prompt,
                          answer, confidence, plan,
                          all_prompts)
"""
from __future__ import annotations
import re
from collections import defaultdict
from .hierarchy import pick_novel


# ── Constants ────────────────────────────────────────────────────────────────

_STOP = frozenset({
    "what", "who", "where", "when", "how", "why", "which",
    "is", "are", "was", "were", "does", "do", "did", "be",
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "that", "this", "it", "i", "you",
    "define", "explain", "describe", "tell", "me", "about",
})

_Q_INTENT: dict[str, str] = {
    "what":     "definition",
    "define":   "definition",
    "explain":  "definition",
    "describe": "definition",
    "who":      "person",
    "where":    "location",
    "when":     "time",
    "how":      "process",
    "why":      "reason",
    "which":    "selection",
}

_INTENT_TEMPLATES: dict[str, list[str]] = {
    "definition": ["{s} is", "a {s} is", "the {s} is", "{s} are", "{s} refers"],
    "person":     ["{s} was", "{s} is", "{s} was a", "{s} was born"],
    "location":   ["{s} is", "{s} is located", "{s} is a country", "{s} lies"],
    "process":    ["{s} works by", "{s} involves", "{s} is done by", "to {s}"],
    "reason":     ["{s} because", "{s} causes", "{s} results from"],
    "time":       ["{s} was", "{s} occurred", "{s} happened"],
    "selection":  ["{s} is", "the best {s}", "{s} depends"],
    "continuation": ["{s}", "{s} is", "{s} can"],
}

# Sentence-terminating tokens.  NOTE: the training tokenizer
# (benchmarks/datasets.py) currently strips all punctuation, so none of these
# can be predicted — the stop rule only becomes live if the corpus pipeline
# ever preserves sentence punctuation as tokens.
_PUNCT_STOP = frozenset({".", "!", "?", "...", ";", ":"})


def _tokenise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


# ── PromptEncoder ─────────────────────────────────────────────────────────────

class PromptEncoder:
    """
    Encodes a prompt at two complementary levels.

    L1 — word SDRs
        The semantic fingerprint of each individual token, plus whether the
        word was seen during encoder fitting (vs a hashed OOV fingerprint).

    L2 — global intent SDR
        k-WTA of the bit-frequency bundle across all word fingerprints.
        Bits that appear in many of the prompt's words dominate → the
        semantic centroid of the whole prompt.  Captures WHAT topic the
        prompt is about, independent of word order.
    """

    def encode(self, model, tokens: list[str]) -> dict:
        word_sdrs: dict[str, list[int]] = {}
        fitted: dict[str, bool] = {}
        bit_freq: dict[int, int] = defaultdict(int)

        for tok in tokens:
            if tok not in word_sdrs:
                # fingerprint() reports `fitted` before caching the OOV
                # fingerprint, so the flag is accurate on first encounter
                fp = model.fingerprint(tok)
                word_sdrs[tok] = fp["bits"]
                fitted[tok] = fp["fitted"]
            for b in word_sdrs[tok]:
                bit_freq[int(b)] += 1

        # L2: global intent SDR — top fp_bits by frequency (ties → lowest bit)
        ranked = sorted(bit_freq, key=lambda b: (-bit_freq[b], b))
        global_sdr = sorted(ranked[: model.fp_bits])

        return {
            "word_sdrs": word_sdrs,
            "fitted": fitted,               # word → seen during encoder fit?
            "global_sdr": global_sdr,       # topic centroid (bag-of-bits)
        }


# ── IntentClassifier ──────────────────────────────────────────────────────────

class IntentClassifier:
    """Classifies intent from structural cues (question word + copula)."""

    def classify(self, tokens: list[str]) -> tuple[str, list[str]]:
        """Return (intent, subject_tokens)."""
        if not tokens:
            return "continuation", []

        # Structural: question-word + copula prefix
        intent = _Q_INTENT.get(tokens[0], "continuation")
        skip_n = 0
        if tokens[0] in _Q_INTENT:
            skip_n = 1
            if len(tokens) > 1 and tokens[1] in {
                "is", "are", "was", "were", "does", "do", "did"
            }:
                skip_n = 2

        subject = [t for t in tokens[skip_n:] if t not in _STOP] or tokens[skip_n:]
        return intent, subject or tokens


# ── ResponsePlanner ───────────────────────────────────────────────────────────

class ResponsePlanner:
    """
    Decides response shape from intent + confidence level.

    The model "chooses" the response length — not the caller.
    Low confidence → shorter, more cautious answer.
    High confidence → fuller explanation up to the intent's natural maximum.
    """

    # (max_tokens, min_tokens, stop_on_punct)
    _PLANS: dict[str, tuple[int, int, bool]] = {
        "definition":   (30, 8,  True),
        "person":       (25, 5,  True),
        "location":     (20, 4,  True),
        "process":      (40, 10, False),   # process explanations need room
        "reason":       (25, 5,  True),
        "time":         (15, 4,  True),
        "selection":    (20, 5,  True),
        "continuation": (18, 3,  False),
    }

    def plan(self, intent: str, confidence: float) -> dict:
        max_t, min_t, stop_punct = self._PLANS.get(
            intent, self._PLANS["continuation"]
        )
        # Scale down if the model is uncertain about this topic
        if confidence < 0.005:
            max_t = min(max_t, 12)
            min_t = min(min_t, 4)
        elif confidence < 0.02:
            max_t = min(max_t, 20)

        return {
            "max_tokens": max_t,
            "min_tokens": min_t,
            "stop_on_punct": stop_punct,
        }


# ── PromptSelector ────────────────────────────────────────────────────────────

def _build_prompts(intent: str, subject: list[str]) -> list[list[str]]:
    s = " ".join(subject)
    prompts = []
    for tmpl in _INTENT_TEMPLATES.get(intent, _INTENT_TEMPLATES["continuation"]):
        toks = _tokenise(tmpl.replace("{s}", s))
        if toks:
            prompts.append(toks)
    return prompts


def _score_prompt(model, prompt: list[str]) -> float:
    preds = model.predict_next(prompt, topn=1)
    return preds[0][1] if preds else 0.0


def _select_prompt(
    model, intent: str, subject: list[str], related: list[str],
    subject_oov: bool = False,
) -> tuple[list[str], float, list[tuple]]:
    """Score all candidate prompts; return (best_prompt, score, all_scored)."""
    candidates = _build_prompts(intent, subject)

    # Fallback: if the subject is OOV, also try the most SDR-similar known word
    if subject_oov and related:
        candidates += _build_prompts(intent, [related[0]])

    if not candidates:
        candidates = [subject or ["the"]]

    scored = [(p, _score_prompt(model, p)) for p in candidates]
    scored.sort(key=lambda x: -x[1])
    best, best_score = scored[0]
    return best, best_score, scored


# ── PlannedGenerator ──────────────────────────────────────────────────────────

def _generate_planned(model, prompt: list[str], plan: dict) -> list[str]:
    """
    Generate tokens with intent-aware adaptive stopping.

    Stops when any of these fire (after min_tokens):
      • model produces no predictions
      • top-1 confidence falls below absolute floor (0.001)
      • confidence drops >70% relative to previous step → sudden uncertainty
      • sentence-boundary token with stop_on_punct=True
    """
    out = list(prompt)
    prev_score = 1.0
    generated = []

    for i in range(plan["max_tokens"]):
        preds = model.predict_next(out, topn=6)
        if not preds:
            break

        top_word, _ = pick_novel(preds, out)
        # Confidence stops track the model's true top-1 score, not the
        # recency-filtered pick, so the anti-loop filter can't fake a collapse
        model_top = preds[0][1]
        past_min = i >= plan["min_tokens"]

        # Absolute confidence floor
        if past_min and model_top < 0.001:
            break

        # Relative confidence collapse
        if past_min and prev_score > 0 and model_top / prev_score < 0.30:
            break

        # Sentence boundary (dead until punctuation survives tokenisation —
        # see _PUNCT_STOP note above)
        if past_min and plan["stop_on_punct"] and top_word in _PUNCT_STOP:
            generated.append(top_word)
            break

        out.append(top_word)
        generated.append(top_word)
        prev_score = model_top

    return generated


# ── SemanticQA (public API) ───────────────────────────────────────────────────

class SemanticQA:
    """
    Multi-level semantic Q&A over a trained HierarchicalCLM.

    The pipeline:
      1. PromptEncoder  — word SDRs (+ fitted flags) + global intent SDR
      2. IntentClassifier — what kind of question is this?
      3. SDR expansion   — find semantically similar known words
      4. PromptSelector  — score candidate phrasings; pick highest confidence
      5. ResponsePlanner — decide max/min tokens and stopping rules
      6. PlannedGenerator — generate with adaptive stopping

    Usage::

        qa = SemanticQA(model)
        result = qa.answer("what is a castle?")
        # result keys: question, intent, subject, word_sdrs, global_sdr,
        #              related, prompt, answer, confidence, plan, all_prompts
    """

    def __init__(self, model):
        self.model = model
        self._encoder = PromptEncoder()
        self._classifier = IntentClassifier()
        self._planner = ResponsePlanner()

    def answer(self, question: str) -> dict:
        tokens = _tokenise(question)

        # ── 1. Multi-level encoding ──────────────────────────────────────────
        encoded = self._encoder.encode(self.model, tokens)

        # ── 2. Intent + subject ──────────────────────────────────────────────
        intent, subject = self._classifier.classify(tokens)

        # ── 3. SDR-similarity expansion ──────────────────────────────────────
        seen: dict[str, int] = {}
        for w in subject:
            for rw, sc in self.model.similar(w, k=5):
                if rw not in seen:
                    seen[rw] = sc
        related = sorted(seen, key=lambda w: -seen[w])

        # ── 4. Prompt selection ───────────────────────────────────────────────
        subject_oov = bool(subject) and not encoded["fitted"].get(subject[0], True)
        best_prompt, confidence, all_prompts = _select_prompt(
            self.model, intent, subject, related, subject_oov=subject_oov
        )

        # ── 5. Response planning ──────────────────────────────────────────────
        plan = self._planner.plan(intent, confidence)

        # ── 6. Planned generation ─────────────────────────────────────────────
        generated = _generate_planned(self.model, best_prompt, plan)

        return {
            "question": question,
            "intent": intent,
            "subject": subject,
            # SDR introspection
            "word_sdrs": {
                w: encoded["word_sdrs"].get(w, [])[:6]   # first 6 bits shown
                for w in subject
            },
            "global_sdr": encoded["global_sdr"][:12],    # 12 of 21 bits shown
            # Semantic expansion
            "related": related[:6],
            # Generation
            "prompt": best_prompt,
            "answer": " ".join(generated),
            "confidence": round(confidence, 4),
            "plan": plan,
            "all_prompts": [(p, round(s, 4)) for p, s in all_prompts[:4]],
        }
