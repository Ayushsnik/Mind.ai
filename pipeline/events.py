"""Event extraction: what happened in an entry, and when.

For every sentence we find life situations from the taxonomy (and recurring
personal topics), skip negated ones ("didn't go to the gym"), decide whether
the event already happened or is coming up ("exam tomorrow" -> upcoming, with
a date), and remember which sentence it came from.

The sentence index matters: Phase 3 attributes each event to the emotion of
ITS OWN sentence, not the whole entry, so "the exam was awful but the gym after
was great" links the exam to fear and the gym to joy.

Uses spaCy (lemmas + dependency parse). Without spaCy a simpler keyword matcher
runs instead, so the app never breaks; results record which extractor ran.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from .event_taxonomy import EVENTS, GENERIC_NOUNS
from .preprocess import split_sentences

log = logging.getLogger(__name__)

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_SHORT = {d[:3]: i for i, d in enumerate(WEEKDAYS)}
NEGATORS = {"not", "no", "never", "without", "cancel", "skip", "n't", "nobody", "none"}
NEG_WINDOW = 3
# Verbs whose negation means the event did not take place.
ATTEND_VERBS = {"go", "attend", "have", "do", "make", "get", "take", "visit", "call", "meet", "play", "see",
                "join", "reach", "write", "give", "finish", "submit", "hit", "happen", "come"}


@dataclass
class EventMention:
    event: str                 # taxonomy key, or "topic:<noun>"
    category: str
    sentence_idx: int
    timing: str                # "past" (includes present) or "upcoming"
    target_date: str | None    # ISO date for upcoming events when we can tell
    trigger: str               # the words that matched (for debugging / the report)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Extraction:
    sentences: list[str]
    mentions: list[EventMention]
    extractor: str


# ---------------------------------------------------------------------------
# Date words -> date (shared by both extractors)
# ---------------------------------------------------------------------------
_IN_N_DAYS = re.compile(r"\bin (\d{1,2}|a|one|two|three|four|five|six|seven) days?\b")
_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_FUTURE_PHRASES = ("tomorrow", "tonight", "next week", "this weekend", "upcoming", "coming up", "later today",
                   "this evening", "in a few days", "next month", "day after tomorrow", "soon")


def resolve_date(text: str, today: date) -> tuple[bool, date | None]:
    """Find future time words in a clause. Returns (is_future_marker, date or None)."""
    t = text.lower()
    if "day after tomorrow" in t:
        return True, today + timedelta(days=2)
    if "tomorrow" in t:
        return True, today + timedelta(days=1)
    if any(p in t for p in ("tonight", "later today", "this evening")):
        return True, today
    m = _IN_N_DAYS.search(t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
        return True, today + timedelta(days=n)
    for i, day in enumerate(WEEKDAYS):
        if re.search(rf"\b(?:on |this |next |coming )?{day}\b", t):
            delta = (i - today.weekday()) % 7
            if delta == 0:
                delta = 7 if re.search(rf"\bnext {day}\b", t) else 0
            return True, today + timedelta(days=delta)
    if "this weekend" in t:
        return True, today + timedelta(days=(5 - today.weekday()) % 7 or 7)
    if "next week" in t:
        return True, today + timedelta(days=7)
    if any(p in t for p in _FUTURE_PHRASES):
        return True, None
    return False, None


# ---------------------------------------------------------------------------
# spaCy extractor
# ---------------------------------------------------------------------------
def _register_sentence_component():
    """Sentence boundaries from punctuation and line breaks, set before the parser runs.

    The small parser sometimes splits "The meeting was this morning and I felt nervous" into
    two sentences, which would separate an event from its feeling. Journal text and Whisper
    transcripts are punctuated, so punctuation is the more reliable signal here.
    """
    from spacy.language import Language

    if "punct_sentences" in Language.factories:
        return

    @Language.component("punct_sentences")
    def punct_sentences(doc):
        for i, tok in enumerate(doc):
            if i == 0:
                tok.is_sent_start = True
                continue
            prev = doc[i - 1]
            boundary = (prev.text[-1:] in ".!?" and not tok.is_punct) or "\n" in prev.text or "\n" in prev.whitespace_
            tok.is_sent_start = boundary
        return doc


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class SpacyExtractor:
    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm", topics: bool = True, max_topics: int = 5):
        import spacy
        from spacy.matcher import Matcher
        _register_sentence_component()
        from spacy.tokens import Token
        self.nlp = spacy.load(model, disable=["ner"])
        if "punct_sentences" not in self.nlp.pipe_names:
            self.nlp.add_pipe("punct_sentences", before="parser")
        self.topics = topics
        self.max_topics = max_topics
        if not Token.has_extension("lemma_lc"):
            Token.set_extension("lemma_lc", getter=lambda t: t.lemma_.lower())
        # Triggers are written in base form, so each trigger token is compared with the text's
        # lower-cased lemma ("exams" -> "exam"). Lemmatising the trigger words on their own is
        # unreliable (spaCy turns a lone "outing" into "out").
        self.matcher = Matcher(self.nlp.vocab)
        self.trigger_lemmas: set[str] = set()
        for key, spec in EVENTS.items():
            patterns = []
            for trig in spec["triggers"]:
                word, _, pos = trig.partition(":")
                toks = [t.text.lower() for t in self.nlp.tokenizer(word)]
                # Two variants: lemma match ("exams" -> exam) and exact word match, because spaCy
                # sometimes lemmatises nouns like "meeting" as the verb "meet".
                for variant in ({"_": {"lemma_lc": None}}, {"LOWER": None}):
                    pattern = []
                    for t in toks:
                        tok = {"_": {"lemma_lc": t}} if "_" in variant else {"LOWER": t}
                        if pos and len(toks) == 1:
                            tok["POS"] = pos
                        pattern.append(tok)
                    patterns.append(pattern)
                self.trigger_lemmas.update(toks)
            self.matcher.add(key, patterns)

    @staticmethod
    def _governor(token):
        """Nearest verb/auxiliary above the token: the clause the event belongs to."""
        t = token
        # Compare indices, not objects: spaCy returns a new Token object on every .head access.
        while t.pos_ not in ("VERB", "AUX") and t.head.i != t.i:
            t = t.head
        return t

    @staticmethod
    def _clause_tokens(gov):
        """The governor's subtree without coordinated/subordinate clauses (they have their own tense)."""
        out, stack = [], [gov]
        while stack:
            t = stack.pop()
            out.append(t)
            for child in t.children:
                if child.dep_ in ("conj", "advcl", "ccomp", "parataxis", "relcl") and child.pos_ in ("VERB", "AUX"):
                    continue
                stack.append(child)
        return sorted(out, key=lambda t: t.i)

    def _negated(self, span, gov) -> bool:
        window = list(span.doc[max(span.sent.start, span.start - NEG_WINDOW):span.start])
        for i, t in enumerate(window):
            if t.lemma_.lower() in NEGATORS:
                # "no exam", "skipped the gym" -> negated. But in "didn't LIKE the movie" the negation
                # belongs to another verb, so a non-attendance verb in between blocks it.
                between = window[i + 1:]
                if not any(b.pos_ in ("VERB", "AUX") and b.lemma_.lower() not in ATTEND_VERBS for b in between):
                    return True
        # "didn't go to the gym" cancels the event; "didn't like the movie" doesn't (the movie happened).
        return (gov.lemma_.lower() in ATTEND_VERBS and gov.i < span.start
                and any(c.dep_ == "neg" for c in gov.children))

    def _timing(self, span, gov, today):
        clause = self._clause_tokens(gov)
        text = " ".join(t.text for t in clause)
        future, when = resolve_date(text, today)
        will = any(c.lemma_.lower() in ("will", "shall") and c.dep_ == "aux" for c in gov.children)
        going_to = gov.lemma_ == "go" and any(c.dep_ == "xcomp" for c in gov.children) and gov.tag_ == "VBG"
        needs = gov.lemma_ in ("have", "need") and any(c.dep_ == "xcomp" for c in gov.children)
        past = "Past" in gov.morph.get("Tense") and not will
        if past and not any(w in text.lower() for w in ("tomorrow", "next", "tonight")):
            return "past", None
        if future or will or going_to or needs:
            return "upcoming", when.isoformat() if when else None
        return "past", None

    def extract(self, text: str, today: date) -> Extraction:
        doc = self.nlp(text)
        sents = list(doc.sents)
        sent_index = {s.start: i for i, s in enumerate(sents)}
        from spacy.tokens import Span
        from spacy.util import filter_spans
        spans = [Span(doc, start, end, label=match_id) for match_id, start, end in self.matcher(doc)]
        mentions, seen, covered = [], set(), set()
        for span in filter_spans(spans):              # keep the longest match where triggers overlap
            key = span.label_
            spec = EVENTS[key]
            gov = self._governor(span.root)
            if not spec.get("negation_ok") and self._negated(span, gov):
                continue
            timing, when = self._timing(span, gov, today)
            idx = sent_index[span.sent.start]
            covered.update(range(span.start, span.end))
            if (key, idx, timing) in seen:
                continue
            seen.add((key, idx, timing))
            mentions.append(EventMention(key, spec["category"], idx, timing, when, span.text))

        if self.topics:
            mentions.extend(self._topics(doc, sent_index, covered, today))
        return Extraction([s.text.strip() for s in sents], mentions, self.name)

    def _topics(self, doc, sent_index, covered, today):
        """Recurring nouns outside the taxonomy, e.g. "thesis" or "dissertation". Stopwords removed here."""
        counts, first = Counter(), {}
        for t in doc:
            lemma = t.lemma_.lower()
            # Capitalised words are skipped: names often get tagged NOUN, and topics are stored
            # unencrypted for pattern mining, so a person's name must never become one.
            if (t.pos_ != "NOUN" or t.is_stop or not t.is_alpha or len(lemma) < 3 or t.i in covered
                    or t.text[:1].isupper()
                    or lemma in GENERIC_NOUNS or lemma in self.trigger_lemmas):
                continue
            gov = self._governor(t)
            if self._negated(doc[t.i:t.i + 1], gov):
                continue
            counts[lemma] += 1
            first.setdefault(lemma, t)
        out = []
        for lemma, _ in counts.most_common(self.max_topics):
            t = first[lemma]
            gov = self._governor(t)
            timing, when = self._timing(doc[t.i:t.i + 1], gov, today)
            out.append(EventMention(f"topic:{lemma}", "topic", sent_index[t.sent.start], timing, when, t.text))
        return out


# ---------------------------------------------------------------------------
# Keyword fallback (no spaCy)
# ---------------------------------------------------------------------------
def _naive_lemma(w: str) -> str:
    w = w.lower().strip("'")
    for suf, rep, min_stem in (("ies", "y", 3), ("ing", "", 4), ("ed", "", 4), ("es", "", 3), ("s", "", 3)):
        if w.endswith(suf) and len(w) - len(suf) >= min_stem and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


def _expand_contractions(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bwon't\b", "will not", s)
    return re.sub(r"\b([a-z]+)n't\b", r"\1 not", s)


class KeywordExtractor:
    name = "keyword-fallback"

    def __init__(self):
        self.patterns = []   # (lemma tuple, event key)
        for key, spec in EVENTS.items():
            for trig in spec["triggers"]:
                word = trig.partition(":")[0]
                self.patterns.append((tuple(_naive_lemma(w) for w in re.findall(r"[a-z]+", _expand_contractions(word))), key))
        self.patterns.sort(key=lambda p: -len(p[0]))

    def extract(self, text: str, today: date) -> Extraction:
        sents = split_sentences(text)
        mentions, seen = [], set()
        for idx, sent in enumerate(sents):
            words = re.findall(r"[a-z]+", _expand_contractions(sent))
            lemmas = [_naive_lemma(w) for w in words]
            used = set()
            future, when = resolve_date(sent, today)
            future = future or bool(re.search(r"\b(will|going to|have to|need to)\b", sent.lower()))
            for pat, key in self.patterns:
                n = len(pat)
                for i in range(len(lemmas) - n + 1):
                    if tuple(lemmas[i:i + n]) != pat or used & set(range(i, i + n)):
                        continue
                    if not EVENTS[key].get("negation_ok") and set(lemmas[max(0, i - NEG_WINDOW - 1):i]) & NEGATORS:
                        continue
                    used.update(range(i, i + n))
                    timing = "upcoming" if future else "past"
                    if (key, idx, timing) not in seen:
                        seen.add((key, idx, timing))
                        mentions.append(EventMention(key, EVENTS[key]["category"], idx, timing,
                                                     when.isoformat() if (future and when) else None,
                                                     " ".join(words[i:i + n])))
        return Extraction(sents, mentions, self.name)


def load_extractor(model: str = "en_core_web_sm", topics: bool = True):
    try:
        return SpacyExtractor(model, topics=topics)
    except Exception as exc:   # spaCy or the model not installed
        log.warning("spaCy unavailable (%s); using keyword event extractor.", exc)
        return KeywordExtractor()
