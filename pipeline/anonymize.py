"""Remove personal details before anything is sent to an online model (Gemini).

Three layers, because no single one is enough (spaCy's small model misses many
Indian names and calls phone numbers dates):

1. Patterns    emails, links, phone numbers (Indian and international), ID-like
               numbers (Aadhaar-style, roll numbers, PIN codes).
2. NER         spaCy entities: PERSON, places (GPE/LOC/FAC), ORG, NORP (groups).
3. Names       any capitalised proper noun the tagger finds, plus a user list of
               names to always hide (Settings).

Each distinct value gets a stable placeholder ([PERSON_1], [PLACE_2]...), the
mapping never leaves the device, and the model's reply is restored locally so
the user sees the real names again.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from .event_taxonomy import EVENTS, GENERIC_NOUNS

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("LINK", re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    ("ID", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")),                         # Aadhaar-style
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:[6-9]\d{4}[\s-]?\d{5}|\d{3,5}[\s-]\d{3,4}[\s-]\d{3,4})(?!\w)")),
    ("ID", re.compile(r"\b(?=[A-Z0-9-]*\d)[A-Z]{1,5}[-/]?\d{3,}[A-Z0-9-]*\b")),  # roll no., PAN-like, ticket IDs
    ("ID", re.compile(r"\b\d{6,}\b")),                                           # PIN codes, long numbers
]
_NER_KIND = {"PERSON": "PERSON", "GPE": "PLACE", "LOC": "PLACE", "FAC": "PLACE", "ORG": "ORG", "NORP": "GROUP"}
_PLACEHOLDER = re.compile(r"\[?\b(PERSON|PLACE|ORG|GROUP|NAME|PHONE|EMAIL|LINK|ID)_(\d+)\b\]?")

_SAFE_WORDS = {w.lower() for w in """
I Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July
August September October November December Today Tomorrow Yesterday Mom Mum Dad Papa Mummy Aai Baba God
Mr Mrs Ms Dr Sir Maam Ma'am Bhai Didi Dada OK Okay Hi Hello Thanks English Hindi Marathi
""".split()}


@dataclass
class Redaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)     # placeholder -> original (stays local)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ph in self.mapping:
            kind = ph.strip("[]").rsplit("_", 1)[0]
            out[kind] = out.get(kind, 0) + 1
        return out


class Anonymizer:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        self._nlp = None
        self._lock = threading.Lock()
        self.ner_available: bool | None = None
        self._safe = set(_SAFE_WORDS) | GENERIC_NOUNS
        for spec in EVENTS.values():
            for trig in spec["triggers"]:
                self._safe.update(trig.partition(":")[0].lower().split())

    @property
    def nlp(self):
        if self.ner_available is None:
            with self._lock:
                if self.ner_available is None:
                    try:
                        import spacy
                        self._nlp = spacy.load(self.spacy_model)
                        self.ner_available = True
                    except Exception:
                        self.ner_available = False
        return self._nlp

    def _spans(self, text: str, hidden_names: list[str]) -> list[tuple[int, int, str]]:
        spans = []
        for kind, rx in _PATTERNS:
            spans += [(m.start(), m.end(), kind) for m in rx.finditer(text)]
        for name in hidden_names:
            name = name.strip()
            if len(name) >= 2:
                spans += [(m.start(), m.end(), "PERSON")
                          for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.I)]
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                kind = _NER_KIND.get(ent.label_)
                if kind and ent.text.lower() not in self._safe:
                    # The small model often tags lone first names as FAC ("Rohan" -> facility);
                    # a neutral NAME placeholder avoids telling the model a person is a place.
                    if ent.label_ == "FAC" and len(ent) == 1:
                        kind = "NAME"
                    spans.append((ent.start_char, ent.end_char, kind))
            for tok in doc:
                # Proper nouns the NER missed (common for Indian names): capitalised, not an acronym,
                # not a weekday/month/known situation word.
                if (tok.pos_ == "PROPN" and tok.text[:1].isupper() and not tok.text.isupper()
                        and tok.is_alpha and tok.text.lower() not in self._safe):
                    spans.append((tok.idx, tok.idx + len(tok.text), "NAME"))
                # Sentence-initial subjects that look like names ("Sneha said..."): the tagger often
                # calls them NOUN. Known situation words, generic nouns and plurals are left alone.
                elif (tok.is_sent_start and tok.text[:1].isupper() and not tok.text.isupper() and tok.is_alpha
                      and tok.pos_ in ("NOUN", "PROPN") and tok.dep_ in ("nsubj", "nsubjpass")
                      and tok.tag_ not in ("NNS", "NNPS") and tok.text.lower() not in self._safe
                      and tok.lemma_.lower() not in self._safe):
                    spans.append((tok.idx, tok.idx + len(tok.text), "NAME"))
        return spans

    def anonymize(self, text: str, hidden_names: list[str] | None = None) -> Redaction:
        spans = self._spans(text, hidden_names or [])
        # Resolve overlaps: earliest start wins, then the longest span.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        chosen, last_end = [], -1
        for s in spans:
            if s[0] >= last_end:
                chosen.append(s)
                last_end = s[1]
        by_value: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}
        mapping: dict[str, str] = {}
        out, pos = [], 0
        for start, end, kind in chosen:
            value = text[start:end]
            key = (kind if kind != "NAME" else "PERSON", value.lower())
            ph = by_value.get(key)
            if ph is None:
                label = "NAME" if kind == "NAME" else kind
                counters[label] = counters.get(label, 0) + 1
                ph = f"[{label}_{counters[label]}]"
                by_value[key] = ph
                mapping[ph] = value
            out.append(text[pos:start])
            out.append(ph)
            pos = end
        out.append(text[pos:])
        return Redaction("".join(out), mapping)

    @staticmethod
    def restore(text: str, mapping: dict[str, str]) -> str:
        """Put the real values back into a model's reply (tolerates missing brackets)."""
        def sub(m):
            return mapping.get(f"[{m.group(1)}_{m.group(2)}]", m.group(0))
        return _PLACEHOLDER.sub(sub, text)
