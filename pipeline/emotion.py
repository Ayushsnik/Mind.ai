"""Emotion detection.

Primary: j-hartmann/emotion-english-distilroberta-base (DistilRoBERTa fine-tuned on
6 emotion datasets). Labels: anger, disgust, fear, joy, neutral, sadness, surprise.

Long entries are chunked; per-chunk probabilities are averaged, weighted by
chunk length, so a single dramatic sentence doesn't dominate a long entry.

Fallback: a small keyword lexicon so the app still runs (and tests pass) on
machines where torch/transformers aren't installed. The result always reports
which backend produced it, so evaluation numbers are never mixed up.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

from .preprocess import chunk_text

log = logging.getLogger(__name__)

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


@dataclass
class EmotionResult:
    primary: str
    confidence: float
    scores: dict[str, float]
    backend: str
    secondary: str | None = None   # second emotion if it's close (mixed feelings)
    chunks: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary": self.primary, "confidence": round(self.confidence, 4),
            "secondary": self.secondary,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "backend": self.backend, "chunks": self.chunks, "notes": self.notes,
        }


def _finalize(scores: dict[str, float], backend: str, chunks: int, notes=None) -> EmotionResult:
    total = sum(scores.values()) or 1.0
    scores = {e: scores.get(e, 0.0) / total for e in EMOTIONS}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, p1), (second, p2) = ranked[0], ranked[1]
    # Flag mixed emotion when the runner-up is a real emotion and within 15 points.
    secondary = second if (p1 - p2) < 0.15 and second != "neutral" and p2 > 0.2 else None
    return EmotionResult(top, p1, scores, backend, secondary, chunks, notes or [])


class RobertaEmotionModel:
    """Loads the tokenizer and model directly instead of through transformers.pipeline().

    pipeline() imports extra libraries (SciPy among them) that locked-down Windows
    machines can block ("An Application Control policy has blocked this file").
    Tokenizer + model + softmax needs only torch and transformers.
    """
    name = "roberta"

    def __init__(self, model_name: str, chunk_words: int = 250, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.model_name = model_name
        self.chunk_words = chunk_words
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        cfg = self.model.config
        self.labels = [cfg.id2label[i].lower() for i in range(cfg.num_labels)]

    def _forward(self, texts: list[str]) -> list[list[float]]:
        """Class probabilities for each text (one row per text, columns in self.labels order)."""
        rows = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=512,
                                 return_tensors="pt")
            with self.torch.no_grad():
                rows += self.model(**enc).logits.softmax(dim=-1).tolist()
        return rows

    def predict(self, text: str) -> EmotionResult:
        chunks = chunk_text(text, self.chunk_words) or [text]
        agg = {e: 0.0 for e in EMOTIONS}
        total_weight = 0
        for chunk, probs in zip(chunks, self._forward(chunks)):
            weight = max(len(chunk.split()), 1)
            total_weight += weight
            for label, p in zip(self.labels, probs):
                agg[label] = agg.get(label, 0.0) + p * weight
        agg = {k: v / total_weight for k, v in agg.items()}
        return _finalize(agg, f"roberta:{self.model_name}", len(chunks))

    def predict_many(self, texts: list[str]) -> list[EmotionResult]:
        """One result per short text (sentences), batched."""
        return [_finalize(dict(zip(self.labels, probs)), f"roberta:{self.model_name}", 1)
                for probs in self._forward(texts)]


class LexiconEmotionModel:
    """Tiny keyword model. Only a fallback — never report its numbers as results."""
    name = "lexicon"

    LEXICON = {
        "joy": "happy glad excited great amazing love loved proud grateful thankful fun enjoyed relieved "
               "awesome wonderful calm peaceful smile laughed celebrate won",
        "sadness": "sad lonely miss missed cried cry upset down depressed hurt disappointed tired empty "
                   "lost heartbroken alone regret",
        "anger": "angry mad furious annoyed irritated hate frustrated unfair rude yelled argument fight pissed",
        "fear": "scared afraid anxious nervous worried worry panic stress stressed tense exam deadline "
                "overwhelmed dread terrified",
        "surprise": "surprised shocked unexpected suddenly wow unbelievable amazed",
        "disgust": "disgusted gross disgusting sick awful nasty revolting",
    }
    NEGATORS = {"not", "no", "never", "n't", "hardly", "isn't", "wasn't", "don't", "didn't"}

    def __init__(self):
        self.words = {w: e for e, ws in self.LEXICON.items() for w in ws.split()}

    def predict(self, text: str) -> EmotionResult:
        tokens = re.findall(r"[a-z']+", text.lower())
        scores = {e: 0.0 for e in EMOTIONS}
        for i, tok in enumerate(tokens):
            emo = self.words.get(tok)
            if not emo:
                continue
            negated = any(t in self.NEGATORS for t in tokens[max(0, i - 3):i])
            if negated and emo == "joy":
                scores["sadness"] += 0.6
            elif not negated:
                scores[emo] += 1.0
        scores["neutral"] = 0.8  # baseline so emotion-free text lands on neutral
        return _finalize(scores, "lexicon-fallback", 1,
                         ["Using keyword fallback; install torch + transformers for RoBERTa."])

    def predict_many(self, texts: list[str]) -> list[EmotionResult]:
        return [self.predict(t) for t in texts]


class EmotionService:
    """Loads the configured backend once (thread-safe) and exposes predict()."""

    def __init__(self, backend="auto",
                 model_name="j-hartmann/emotion-english-distilroberta-base", chunk_words=250):
        self.backend_pref = backend
        self.model_name = model_name
        self.chunk_words = chunk_words
        self._model = None
        self._lock = threading.Lock()
        self.load_error: str | None = None

    def _load(self):
        if self.backend_pref in ("auto", "roberta"):
            try:
                return RobertaEmotionModel(self.model_name, self.chunk_words)
            except Exception as exc:  # missing package, no internet on first run, low RAM...
                self.load_error = f"{type(exc).__name__}: {exc}"
                if self.backend_pref == "roberta":
                    raise
                log.warning("RoBERTa unavailable (%s); using lexicon fallback.", self.load_error)
        return LexiconEmotionModel()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def predict(self, text: str) -> EmotionResult:
        if not text.strip():
            return _finalize({"neutral": 1.0}, self.model.name, 0, ["Empty text."])
        return self.model.predict(text)

    def predict_many(self, texts: list[str]) -> list[EmotionResult]:
        if not texts:
            return []
        return self.model.predict_many([t if t.strip() else "." for t in texts])

    def status(self) -> dict:
        loaded = self._model is not None
        return {
            "preferred": self.backend_pref,
            "active": self._model.name if loaded else "not loaded yet",
            "model": self.model_name,
            "load_error": self.load_error,
        }