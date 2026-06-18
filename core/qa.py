"""
core/qa.py — Semantic question answering using SDR fingerprints.

Rather than surface pattern matching, this pipeline:

  1. Tokenises the question and fingerprints every word via the model's
     SemanticEncoder — so "castle" activates the bits it shares with
     "fortress", "medieval", "tower", etc.
  2. Classifies intent (definition / person / process / reason / location)
     from the question-word role ("what is" → definition, "who is" → person,
     "how does" → process …).
  3. Extracts the content-bearing subject words by filtering out
     question-words, copulas, and determiners.
  4. Expands the subject via SDR similarity (model.similar()) — if the model
     doesn't know "chateau" it finds "castle"/"fortress" which it does know.
  5. Builds several candidate completion prompts for the detected intent
     (e.g. "castle is", "a castle is", "castles are" …).
  6. Scores each candidate by asking the model how confident it is about the
     next token — picks the prompt with the highest top-prediction score.
  7. Generates the answer with model.generate() from the winning prompt.

Limitations:
  - Quality is proportional to training data; rare subjects get weak answers.
  - Generation is left-to-right continuation, not structured extraction.
  - Intent detection is structural, not neural; exotic phrasings may
    be mis-classified.
"""
from __future__ import annotations
import re
from collections import defaultdict


# Words that carry no subject content
_STOP = frozenset({
    "what", "who", "where", "when", "how", "why", "which",
    "is", "are", "was", "were", "does", "do", "did", "be",
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "that", "this", "it", "i", "you",
    "define", "explain", "describe", "tell", "me", "about",
})

# Question-word → intent label
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

# Intent → ordered list of prompt templates.
# {s} = subject tokens joined as string; {sw} = subject token list.
# Templates are tried in order; the one with the highest top-1 score wins.
_INTENT_TEMPLATES: dict[str, list[str]] = {
    "definition": ["{s} is", "a {s} is", "the {s} is", "{s} are", "{s} refers"],
    "person":     ["{s} was", "{s} is", "{s} was a", "{s} was born"],
    "location":   ["{s} is", "{s} is located", "{s} lies", "{s} is a"],
    "process":    ["{s} works by", "{s} is done by", "{s} involves", "to {s} you"],
    "reason":     ["{s} because", "{s} is caused", "{s} results from"],
    "time":       ["{s} was", "{s} occurred", "{s} happened"],
    "selection":  ["{s} is", "the best {s} is", "{s} depends"],
    "continuation": ["{s}", "{s} is", "{s} can"],
}


def _tokenise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def _detect_intent(tokens: list[str]) -> tuple[str, list[str]]:
    """Return (intent, subject_tokens)."""
    if not tokens:
        return "continuation", []

    intent = _Q_INTENT.get(tokens[0], "continuation")

    # Skip question-word + copula prefix ("what is", "who was", "how does" …)
    skip_n = 0
    if tokens and tokens[0] in _Q_INTENT:
        skip_n = 1
        if len(tokens) > 1 and tokens[1] in {"is", "are", "was", "were", "does", "do", "did"}:
            skip_n = 2

    subject = [t for t in tokens[skip_n:] if t not in _STOP]
    if not subject:
        subject = [t for t in tokens[skip_n:] if t]   # at least keep something

    return intent, subject


def _expand_subject(model, subject: list[str], k: int = 4) -> list[str]:
    """Find semantically similar words via SDR overlap. Returns ranked list."""
    seen: dict[str, int] = {}
    for word in subject:
        for w, score in model.similar(word, k=k):
            if w not in seen:
                seen[w] = score
    return sorted(seen, key=lambda w: -seen[w])


def _build_prompts(intent: str, subject: list[str]) -> list[list[str]]:
    """Expand intent templates → list of token-list prompts."""
    s = " ".join(subject)
    prompts: list[list[str]] = []
    for tmpl in _INTENT_TEMPLATES.get(intent, _INTENT_TEMPLATES["continuation"]):
        filled = tmpl.replace("{s}", s)
        toks = _tokenise(filled)
        if toks:
            prompts.append(toks)
    return prompts


def _score_prompt(model, prompt: list[str]) -> float:
    """Return top-1 prediction score for this prompt (0 if no predictions)."""
    preds = model.predict_next(prompt, topn=1)
    return preds[0][1] if preds else 0.0


class SemanticQA:
    """
    Semantic question answering over a trained HierarchicalCLM.

    Usage::

        qa = SemanticQA(model)
        result = qa.answer("what is a castle?")
        print(result["answer"])
    """

    def __init__(self, model):
        self.model = model

    def answer(self, question: str, n: int = 20) -> dict:
        """
        Process a natural-language question and return a dict with:
          - intent       : detected question type
          - subject      : content words extracted from the question
          - fingerprints : {word: [active_bits]} for each subject word
          - related      : semantically similar words found via SDR overlap
          - prompt       : token list fed to generate()
          - answer       : generated continuation
          - confidence   : top-1 score for the chosen prompt
        """
        tokens = _tokenise(question)
        intent, subject = _detect_intent(tokens)

        # Fingerprint each subject word — shows what the model "associates" with it
        fingerprints = {}
        for w in subject:
            fp = self.model.fingerprint(w)
            fingerprints[w] = {"bits": fp["bits"][:8], "fitted": fp["fitted"]}

        # Expand via SDR similarity to find words the model knows well
        related = _expand_subject(self.model, subject, k=5)

        # Build all candidate prompts for this intent
        candidates = _build_prompts(intent, subject)

        # Also try prompts built from the most similar known word if subject is OOV
        primary_fp = self.model.fingerprint(subject[0]) if subject else {}
        if subject and not primary_fp.get("fitted", True) and related:
            candidates += _build_prompts(intent, [related[0]])

        if not candidates:
            candidates = [tokens]

        # Score all candidates; pick the one the model is most confident about
        scored = [(prompt, _score_prompt(self.model, prompt)) for prompt in candidates]
        scored.sort(key=lambda x: -x[1])
        best_prompt, confidence = scored[0]

        # Generate answer from winning prompt
        generated = self.model.generate(best_prompt, n=n)

        return {
            "question": question,
            "intent": intent,
            "subject": subject,
            "fingerprints": fingerprints,
            "related": related[:6],
            "prompt": best_prompt,
            "answer": " ".join(generated),
            "confidence": round(confidence, 4),
            "all_prompts": [(p, round(s, 4)) for p, s in scored[:4]],
        }
