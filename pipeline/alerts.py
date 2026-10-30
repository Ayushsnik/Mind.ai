"""Predictive alerts: early warnings before situations that have weighed on you.

An alert is raised when
  * an entry mentions an UPCOMING event ("viva tomorrow", "meeting on Monday"), and
  * that event is a "drains" pattern in your history (see patterns.py), or
  * today/tomorrow is a weekday that has repeatedly been hard for you.

Each alert carries the evidence ("anxious in 6 of 7 entries") and, from your own
journal, the things that tend to lift your mood. Phase 4 turns that into a
personalised coping suggestion with the language model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .patterns import FEELING_WORD, Insights, event_phrase

UNDATED_DAYS = 3            # "exam coming up" without a date stays relevant this long


@dataclass
class AlertDraft:
    key: str                # unique: same situation + date never alerts twice
    kind: str               # "event" | "day"
    event: str | None
    target_date: str | None
    expires_on: str
    message: str
    suggestion: str | None
    evidence: dict
    source_entry_id: int | None


def when_phrase(target: date | None, today: date) -> str:
    if target is None:
        return "coming up"
    delta = (target - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if 1 < delta < 7:
        return f"on {target.strftime('%A')}"
    return f"on {target.strftime('%d %b').lstrip('0')}"


def what_helps(insights: Insights, exclude: str | None = None, limit: int = 2) -> str | None:
    """From the user's own journal: situations that most reliably lift their mood."""
    helpers = [p for p in insights.lifts() if p.event != exclude and not p.event.startswith("topic:")][:limit]
    if not helpers:
        return None
    names = [h.label.lower() for h in helpers]
    joined = names[0] if len(names) == 1 else f"{names[0]} and {names[1]}"
    best = helpers[0]
    return (f"What has helped before: {joined}. You felt happy in {best.top_count} of {best.n} "
            f"entries about {best.label.lower()}.")


def build_alerts(insights: Insights, upcoming: list[dict], today: date) -> list[AlertDraft]:
    """upcoming: entry_events rows with timing='upcoming' (+ entry_created). Returns drafts to upsert."""
    drafts: dict[str, AlertDraft] = {}
    patterns = insights.by_event

    for m in upcoming:
        p = patterns.get(m["event"])
        if not p or p.kind != "drains" or not p.established:   # only warn on significant patterns
            continue
        written = date.fromisoformat(m["entry_created"][:10])
        if m["target_date"]:
            target = date.fromisoformat(m["target_date"])
            if target < today:
                continue
            expires, key_date = target, target.isoformat()
        else:
            target = None
            expires = written + timedelta(days=UNDATED_DAYS)
            if expires < today:
                continue
            key_date = f"entry{m['entry_id']}"
        key = f"event:{m['event']}:{key_date}"
        feeling = FEELING_WORD.get(p.top_emotion, "low")
        message = (f"You mentioned {event_phrase(m['event'])} {when_phrase(target, today)}. "
                   f"Entries about {p.label.lower()} have usually been hard: you felt {feeling} "
                   f"in {p.top_count} of {p.n}.")
        if p.trend == "easier":
            message += " They have been getting easier lately."
        drafts[key] = AlertDraft(
            key=key, kind="event", event=m["event"], target_date=target.isoformat() if target else None,
            expires_on=expires.isoformat(), message=message, suggestion=what_helps(insights, m["event"]),
            evidence={"n": p.n, "negative_rate": p.negative_rate, "top_emotion": p.top_emotion,
                      "top_count": p.top_count, "strength": p.strength, "lift": p.lift},
            source_entry_id=m["entry_id"],
        )

    if insights.enough_data:
        for d in (today, today + timedelta(days=1)):
            day = insights.days[d.weekday()]
            if day.kind != "hard":
                continue
            hard = round(day.negative_rate * day.n)
            drafts[f"day:{d.isoformat()}"] = AlertDraft(
                key=f"day:{d.isoformat()}", kind="day", event=None, target_date=d.isoformat(),
                expires_on=d.isoformat(),
                message=(f"{day.name}s have been tough: {hard} of your {day.n} {day.name} entries were "
                         f"negative. {'Today' if d == today else 'Tomorrow'} is {day.name}."),
                suggestion=what_helps(insights),
                evidence={"n": day.n, "negative_rate": day.negative_rate, "lift": day.lift},
                source_entry_id=None,
            )
    return list(drafts.values())
