"""Event-emotion pattern mining.

Three kinds of pattern, each answering a question the user can act on:

1. Event patterns: "How do I usually feel when X happens?"
   Uses SENTENCE-LEVEL emotion: each event takes the emotion of the sentence it
   was mentioned in. For each event we compute
     confidence  P(negative | event), or P(positive | event)
     lift        confidence / P(negative) over all events  (>1 = this event is worse than usual)
     strength    Wilson lower bound of the confidence (penalises tiny samples:
                 3/3 = 100% but strength 0.53; 9/10 = 90% with strength 0.65)
     established strength > the user's baseline rate: a one-sided significance test (5% level).
                 Only established patterns are asserted or used for alerts; weaker ones are
                 shown as "emerging".
   and a trend (getting easier / harder) once there are enough mentions.

2. Combination rules: "Which situations are worse TOGETHER?"
   Apriori (mlxtend) over entry-level transactions {events..., weekday} ->
   {entry emotion, entry mood}. Only rules with 2+ antecedents are kept, and only
   when the combination is clearly worse/better than each part alone
   (its lower confidence bound beats the best single event), e.g. {exam, poor sleep} -> fear.

3. Day patterns: "Are some weekdays harder?"
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .event_taxonomy import EVENTS

VALENCE = {"joy": "positive", "sadness": "negative", "anger": "negative", "fear": "negative",
           "disgust": "negative", "surprise": "neutral", "neutral": "neutral"}
FEELING_WORD = {"fear": "anxious", "sadness": "low", "anger": "angry", "disgust": "put off", "joy": "happy",
                "surprise": "surprised", "neutral": "flat"}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def event_label(key: str) -> str:
    if key.startswith("topic:"):
        return key[6:].capitalize()
    return EVENTS[key]["label"] if key in EVENTS else key


def event_phrase(key: str) -> str:
    if key.startswith("topic:"):
        return f"something about your {key[6:]}"
    return EVENTS[key]["phrase"] if key in EVENTS else key


def wilson_lower(successes: int, n: int, z: float = 1.645) -> float:
    """Lower bound of the Wilson score interval for a proportion.

    z = 1.645 makes "lower bound > baseline" a ONE-SIDED test at the 5% level, which is the
    question we ask: is this situation significantly worse (or better) than the user's usual?
    """
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


@dataclass
class EventPattern:
    event: str
    label: str
    category: str
    n: int                          # entries mentioning the event
    emotions: dict[str, int]        # emotion -> number of those entries
    negative_rate: float
    positive_rate: float
    kind: str                       # "drains" | "lifts" | "mixed"
    top_emotion: str                # most common emotion within the dominant direction
    top_count: int
    lift: float
    strength: float                 # Wilson lower bound of the dominant rate
    baseline: float                 # how often ANY event goes that way for this user
    established: bool               # strength > baseline: significantly worse/better than usual
    trend: str | None = None        # "easier" | "harder" | None
    recent_entries: list[int] = field(default_factory=list)

    def summary(self) -> str:
        word = FEELING_WORD.get(self.top_emotion, self.top_emotion)
        return f"You felt {word} in {self.top_count} of {self.n} entries that mention it."


@dataclass
class ComboRule:
    events: list[str]
    labels: list[str]
    consequent: str                 # e.g. "feels:fear" or "mood:negative"
    count: int
    confidence: float
    lift: float
    best_single_confidence: float

    @property
    def outcome(self) -> str:
        kind, _, value = self.consequent.partition(":")
        if kind == "feels":
            return FEELING_WORD.get(value, value)
        return {"negative": "a hard day", "positive": "a good day"}.get(value, value)


@dataclass
class DayPattern:
    weekday: int
    name: str
    n: int
    negative_rate: float
    positive_rate: float
    kind: str                       # "hard" | "good" | "usual"
    lift: float


@dataclass
class Insights:
    total_entries: int
    analysed_entries: int
    enough_data: bool
    min_entries: int
    events: list[EventPattern]
    combos: list[ComboRule]
    days: list[DayPattern]
    baseline_negative: float

    @property
    def by_event(self) -> dict[str, EventPattern]:
        return {p.event: p for p in self.events}

    def drains(self, established_only: bool = True) -> list[EventPattern]:
        return [p for p in self.events if p.kind == "drains" and (p.established or not established_only)]

    def lifts(self, established_only: bool = True) -> list[EventPattern]:
        return [p for p in self.events if p.kind == "lifts" and (p.established or not established_only)]

    def emerging(self) -> list[EventPattern]:
        return [p for p in self.events if p.kind != "mixed" and not p.established]

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries, "analysed_entries": self.analysed_entries,
            "enough_data": self.enough_data, "min_entries": self.min_entries,
            "baseline_negative": round(self.baseline_negative, 3),
            "events": [asdict(p) | {"summary": p.summary()} for p in self.events],
            "combos": [asdict(c) | {"outcome": c.outcome} for c in self.combos],
            "days": [asdict(d) for d in self.days],
        }


# ---------------------------------------------------------------------------
def _entry_event_emotions(mentions):
    """(entry, event) -> one emotion: the majority over that event's sentences in the entry."""
    grouped = defaultdict(list)
    for m in mentions:
        grouped[(m["entry_id"], m["event"])].append(m)
    out = {}
    for key, ms in grouped.items():
        votes = Counter()
        for m in ms:
            votes[m["emotion"]] += 1 + m["emotion_conf"] / 10       # count first, confidence breaks ties
        out[key] = votes.most_common(1)[0][0]
    return out


def event_patterns(mentions, entry_dates, min_occurrences=3, min_rate=0.6, min_lift=1.15,
                   trend_min=6) -> tuple[list[EventPattern], float]:
    labels = _entry_event_emotions(mentions)
    if not labels:
        return [], 0.0
    all_vals = [VALENCE[e] for e in labels.values()]
    base_neg = all_vals.count("negative") / len(all_vals)
    base_pos = all_vals.count("positive") / len(all_vals)
    category = {m["event"]: m["category"] for m in mentions}

    per_event = defaultdict(list)       # event -> [(entry_date, entry_id, emotion)]
    for (entry_id, event), emo in labels.items():
        per_event[event].append((entry_dates.get(entry_id, ""), entry_id, emo))

    patterns = []
    for event, rows in per_event.items():
        n = len(rows)
        if n < min_occurrences:
            continue
        rows.sort()
        emos = Counter(e for _, _, e in rows)
        neg = sum(c for e, c in emos.items() if VALENCE[e] == "negative")
        pos = emos.get("joy", 0)
        neg_rate, pos_rate = neg / n, pos / n
        neg_lift = neg_rate / base_neg if base_neg else float("inf")
        pos_lift = pos_rate / base_pos if base_pos else float("inf")
        if neg_rate >= min_rate and neg_lift >= min_lift:
            kind, hits, lift, base = "drains", neg, neg_lift, base_neg
            top, top_n = max(((e, c) for e, c in emos.items() if VALENCE[e] == "negative"), key=lambda x: x[1])
        elif pos_rate >= min_rate and pos_lift >= min_lift:
            kind, hits, lift, top, top_n, base = "lifts", pos, pos_lift, "joy", pos, base_pos
        else:
            kind = "mixed"
            top, top_n = emos.most_common(1)[0]
            hits, base = top_n, 1.0
            lift = (top_n / n) / max(1e-9, sum(1 for e in labels.values() if e == top) / len(labels))
        strength = wilson_lower(hits, n)

        trend = None
        if n >= trend_min and kind != "mixed":
            half = n // 2
            bad = (lambda e: VALENCE[e] == "negative") if kind == "drains" else (lambda e: e == "joy")
            early = sum(bad(e) for _, _, e in rows[:half]) / half
            late = sum(bad(e) for _, _, e in rows[-half:]) / half
            if late - early <= -0.34:
                trend = "easier" if kind == "drains" else "fading"
            elif late - early >= 0.34:
                trend = "harder" if kind == "drains" else "growing"

        patterns.append(EventPattern(
            event=event, label=event_label(event), category=category.get(event, "topic"), n=n,
            emotions=dict(emos), negative_rate=round(neg_rate, 3), positive_rate=round(pos_rate, 3),
            kind=kind, top_emotion=top, top_count=top_n, lift=round(min(lift, 99.0), 2),
            strength=round(strength, 3), baseline=round(base if kind != "mixed" else 0.0, 3),
            established=kind != "mixed" and strength > base, trend=trend,
            recent_entries=[eid for _, eid, _ in rows[-3:]][::-1],
        ))
    order = {"drains": 0, "lifts": 1, "mixed": 2}
    patterns.sort(key=lambda p: (order[p.kind], not p.established, -p.strength, -p.n))
    return patterns, base_neg


def combo_rules(entries, mentions, min_occurrences=4, min_conf=0.7, min_lift=1.2, min_gain=0.15,
                max_rules=8) -> list[ComboRule]:
    """Apriori over entry transactions; keep non-redundant rules with 2+ events."""
    events_by_entry = defaultdict(set)
    for m in mentions:
        events_by_entry[m["entry_id"]].add(m["event"])
    transactions = []
    for e in entries:
        items = set(events_by_entry.get(e["id"], ()))
        if not items:
            continue
        items |= {f"feels:{e['emotion']}", f"mood:{VALENCE[e['emotion']]}"}
        transactions.append(sorted(items))
    n = len(transactions)
    if n < min_occurrences * 2:
        return []

    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    frequent = apriori(df, min_support=min_occurrences / n, use_colnames=True, max_len=3)
    if frequent.empty:
        return []
    import numpy as np
    with np.errstate(divide="ignore", invalid="ignore"):   # mlxtend divides by zero in metrics we don't use
        try:
            rules = association_rules(frequent, metric="confidence", min_threshold=min_conf, num_itemsets=n)
        except TypeError:       # mlxtend < 0.23 has no num_itemsets
            rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    support = {frozenset(s): v for s, v in zip(frequent["itemsets"], frequent["support"])}

    is_outcome = lambda item: item.startswith(("feels:", "mood:"))
    best: dict[frozenset, ComboRule] = {}
    for _, r in rules.iterrows():
        ante, cons = frozenset(r["antecedents"]), frozenset(r["consequents"])
        if len(cons) != 1 or len(ante) < 2 or any(is_outcome(i) for i in ante):
            continue
        (c,) = cons
        if not is_outcome(c) or r["lift"] < min_lift:
            continue
        # The claim is "worse (or better) TOGETHER", so the combination's lower confidence bound must
        # beat its strongest single part, not just the baseline. Without this, pairing any strongly
        # negative event with a random one produces "rules" by chance.
        singles = [support.get(frozenset([a, c]), 0) / support[frozenset([a])] for a in ante]
        ante_n = round(support[ante] * n)
        hits = round(r["support"] * n)
        if r["confidence"] < max(singles) + min_gain or wilson_lower(hits, ante_n) <= max(singles):
            continue
        rule = ComboRule(events=sorted(ante), labels=[event_label(a) for a in sorted(ante)], consequent=c,
                         count=round(r["support"] * n), confidence=round(float(r["confidence"]), 3),
                         lift=round(float(r["lift"]), 2), best_single_confidence=round(max(singles), 3))
        prev = best.get(ante)
        # Prefer a specific emotion over a general mood when it is just as reliable.
        if (prev is None or (c.startswith("feels:") and not prev.consequent.startswith("feels:")
                             and rule.confidence >= min_conf) or rule.confidence > prev.confidence + 0.05):
            best[ante] = rule
    out = sorted(best.values(), key=lambda r: (-r.confidence * math.log1p(r.count), -r.count))
    return out[:max_rules]


def day_patterns(entries, min_occurrences=3, min_rate=0.6, min_lift=1.2) -> list[DayPattern]:
    if not entries:
        return []
    overall_neg = sum(VALENCE[e["emotion"]] == "negative" for e in entries) / len(entries)
    overall_pos = sum(e["emotion"] == "joy" for e in entries) / len(entries)
    by_day = defaultdict(list)
    for e in entries:
        by_day[datetime.fromisoformat(e["created_at"]).weekday()].append(e["emotion"])
    out = []
    for wd in range(7):
        emos = by_day.get(wd, [])
        n = len(emos)
        neg = sum(VALENCE[x] == "negative" for x in emos) / n if n else 0.0
        pos = sum(x == "joy" for x in emos) / n if n else 0.0
        kind, lift = "usual", 1.0
        if n >= min_occurrences:
            if neg >= min_rate and overall_neg and neg / overall_neg >= min_lift:
                kind, lift = "hard", neg / overall_neg
            elif pos >= min_rate and overall_pos and pos / overall_pos >= min_lift:
                kind, lift = "good", pos / overall_pos
        out.append(DayPattern(wd, DAY_NAMES[wd], n, round(neg, 3), round(pos, 3), kind, round(lift, 2)))
    return out


def analyze(entries: list[dict], mentions: list[dict], *, total_entries: int | None = None,
            min_entries: int = 8, min_occurrences: int = 3, min_rate: float = 0.6, min_lift: float = 1.15,
            combo_conf: float = 0.7) -> Insights:
    """entries: [{id, created_at, emotion}] (analysed ones); mentions: rows from entry_events."""
    dates = {e["id"]: e["created_at"] for e in entries}
    events, base_neg = event_patterns(mentions, dates, min_occurrences, min_rate, min_lift)
    enough = len(entries) >= min_entries
    return Insights(
        total_entries=total_entries if total_entries is not None else len(entries),
        analysed_entries=len(entries), enough_data=enough, min_entries=min_entries,
        events=events if enough else [],
        combos=combo_rules(entries, mentions, max(min_occurrences, 4), combo_conf, max(min_lift, 1.2)) if enough else [],
        days=day_patterns(entries, min_occurrences, min_rate, max(min_lift, 1.2)),
        baseline_negative=base_neg,
    )
