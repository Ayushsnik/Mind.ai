"""Mind.ai's written response to an entry, and preparation plans for heads-ups.

Order of operations for every entry:

  1. Safety check  -> crisis language never reaches a model; fixed message + helplines.
  2. Pick backend  -> Gemini only if the user opted in; otherwise the local model;
                      otherwise the template responder (works with no AI at all).
  3. Build context -> emotion, what happened, and what this user's own history shows
                      (Phase 3 patterns), so suggestions are grounded, not generic.
  4. Privacy       -> local model: full entry, stays on the device.
                      Gemini: only an ANONYMISED SUMMARY (key sentences with names,
                      places, numbers replaced) - the user can preview exactly what is sent.
  5. Validate      -> JSON shape, lengths, and a filter that rejects diagnoses or
                      medication advice. Any failure falls through to the next backend.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import asdict, dataclass

from .anonymize import Anonymizer
from .llm import LLMError, parse_json_reply  # noqa: F401  (re-exported for tests)
from .patterns import FEELING_WORD, VALENCE, event_label, event_phrase
from .safety import check as crisis_check, crisis_response

REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string"},
        "affirmation": {"type": "string"},
        "suggestions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["reflection", "affirmation", "suggestions"],
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"intro": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["intro", "steps"],
}

SYSTEM_PROMPT = """You are the reflection companion inside Mind.ai, a private journaling app.
The user has just written a journal entry. Reply like a warm, perceptive friend who has read it carefully.

Rules:
- "reflection": 2-3 sentences, in second person. Refer to specific things they wrote. Name the feeling gently. No clichés.
- "affirmation": one short, sincere sentence of encouragement that fits this entry.
- "suggestions": 2 or 3 small, concrete actions for the next day or two. Prefer things the user's own history shows
  help them (given below). Each under 30 words. No generic "practise self-care".
- Never diagnose, never mention medication, never claim to be a therapist. If the entry shows serious or lasting
  distress, one suggestion should be to talk to someone they trust or a counsellor.
- Keep placeholders such as [PERSON_1] or [PLACE_2] exactly as written; never guess who they are.
- Reply with JSON only: {"reflection": "...", "affirmation": "...", "suggestions": ["...", "..."]}"""

PLAN_PROMPT = """You are the planning helper inside Mind.ai, a private journaling app.
The user has an upcoming situation that, according to their own journal, has usually been hard for them.
Give a short, practical plan to prepare.

Rules:
- "intro": one warm sentence that acknowledges this tends to be hard for them, without drama.
- "steps": exactly 3 concrete steps (each under 30 words): one for before, one for during, one for after.
  Use what has helped them before where it fits.
- No diagnoses, no medication, no therapy claims.
- Reply with JSON only: {"intro": "...", "steps": ["...", "...", "..."]}"""

_UNSAFE = re.compile(r"\b(medication|medicine|pills?|tablets?|dosage|\d+\s?mg|antidepressants?|prescri\w*|"
                     r"you (?:have|suffer from|are suffering from) (?:depression|an anxiety disorder|bipolar|ptsd|adhd|ocd))\b",
                     re.I)


@dataclass
class Reply:
    kind: str                     # "reflection" | "crisis"
    reflection: str
    affirmation: str
    suggestions: list[str]
    backend: str                  # "local" | "gemini" | "template" | "safety"
    model: str
    privacy: str                  # "on-device" | "anonymised-summary" | "none"
    sent: str | None = None       # exactly what left the device (Gemini only)
    latency_ms: int = 0
    note: str | None = None       # e.g. why a fallback was used

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Context from Phase 3
# ---------------------------------------------------------------------------
def build_context(entry: dict, events: list[dict], insights) -> dict:
    by_event = insights.by_event if insights and insights.enough_data else {}
    happened, upcoming, history = [], [], []
    seen = set()
    for ev in events:
        label = event_label(ev["event"])
        if ev["timing"] == "upcoming":
            upcoming.append(label)
        else:
            happened.append(f"{label} ({FEELING_WORD.get(ev['emotion'], ev['emotion'])})")
        p = by_event.get(ev["event"])
        if p and p.established and ev["event"] not in seen:
            seen.add(ev["event"])
            feel = FEELING_WORD.get(p.top_emotion, p.top_emotion)
            line = f"{p.label}: felt {feel} in {p.top_count} of {p.n} entries"
            if p.trend:
                line += f" (getting {p.trend} lately)" if p.trend in ("easier", "harder") else ""
            history.append(line)
    helpers = [p for p in (insights.lifts() if insights and insights.enough_data else [])
               if not p.event.startswith("topic:")][:3]
    return {
        "date": entry["created_at"][:10],
        "emotion": entry["emotion"], "confidence": round(entry["emotion_conf"] * 100),
        "secondary": entry.get("emotion_second"),
        "happened": list(dict.fromkeys(happened)), "upcoming": list(dict.fromkeys(upcoming)),
        "history": history,
        "helpers": [f"{h.label} (happy in {h.top_count} of {h.n} entries)" for h in helpers],
        "helper_labels": [h.label for h in helpers],
    }


def _context_lines(ctx: dict) -> str:
    lines = [f"Detected emotion: {ctx['emotion']} ({ctx['confidence']}%)"
             + (f", with some {ctx['secondary']}" if ctx["secondary"] else "")]
    if ctx["happened"]:
        lines.append("What happened (feeling in that sentence): " + ", ".join(ctx["happened"]))
    if ctx["upcoming"]:
        lines.append("Coming up: " + ", ".join(ctx["upcoming"]))
    if ctx["history"]:
        lines.append("What their journal history shows:\n" + "\n".join(f"- {h}" for h in ctx["history"]))
    if ctx["helpers"]:
        lines.append("What usually lifts their mood: " + "; ".join(ctx["helpers"]))
    return "\n".join(lines)


def key_sentences(body: str, events: list[dict], limit: int = 4, max_chars: int = 700) -> list[str]:
    """The sentences that carry the entry: those with events, then the rest, in order, capped."""
    from .preprocess import split_sentences
    sents = split_sentences(body)
    wanted = sorted({e["sentence_idx"] for e in events if e["sentence_idx"] < len(sents)})
    for i in range(len(sents)):
        if len(wanted) >= limit:
            break
        if i not in wanted:
            wanted.append(i)
    out, total = [], 0
    for i in sorted(wanted[:limit]):
        s = sents[i][:300]
        if total + len(s) > max_chars:
            break
        out.append(s)
        total += len(s)
    return out


# ---------------------------------------------------------------------------
# Template responder (no AI model needed; also the baseline in evaluation)
# ---------------------------------------------------------------------------
_AFFIRM = {
    "fear": ["Feeling nervous means this matters to you, and you showed up anyway.",
             "You've got through anxious days before; this one counts too."],
    "sadness": ["Being honest about a low day is a kind of strength.",
                "Days like this pass, even when it doesn't feel that way."],
    "anger": ["Your frustration makes sense; you can choose what to do with it.",
              "Noticing anger instead of acting on it is a skill, and you're using it."],
    "disgust": ["It's okay to be put off by things that cross your lines."],
    "joy": ["Good days are worth noticing; this one is yours.", "You earned this one."],
    "surprise": ["Unexpected days can teach you something about yourself."],
    "neutral": ["Ordinary days are part of the story too.", "Showing up to write is a good habit in itself."],
}
_NOUN = {"fear": "worry", "sadness": "sadness", "anger": "frustration", "disgust": "distaste", "joy": "happiness",
         "surprise": "surprise", "neutral": "calm"}
_MICRO = {
    "fear": "Write down the one worry that's loudest right now and one small step you could take on it.",
    "sadness": "Send a short message to one person you feel close to, even just to say hi.",
    "anger": "Before replying to anyone involved, give it ten minutes and a short walk.",
    "disgust": "Step away from whatever bothered you for a while and do something absorbing.",
    "joy": "Note what made today good so you can plan more of it.",
    "surprise": "Jot down what surprised you and what you'd like to do about it.",
    "neutral": "Pick one small thing tomorrow that you'd look forward to.",
}


def template_reply(ctx: dict, seed: int = 0) -> dict:
    rng = random.Random(seed)
    feel = FEELING_WORD.get(ctx["emotion"], ctx["emotion"])
    happened = [h.split(" (")[0].lower() for h in ctx["happened"]]
    if happened:
        about = happened[0] if len(happened) == 1 else ", ".join(happened[:-1]) + " and " + happened[-1]
        reflection = f"You wrote about {about}, and overall the day reads as {feel}"
    else:
        reflection = f"Overall this entry reads as {feel}"
    reflection += f", with some {_NOUN.get(ctx['secondary'], ctx['secondary'])} too." if ctx["secondary"] else "."
    if ctx["history"]:
        label, _, rest = ctx["history"][0].partition(": felt ")
        reflection += f" That fits your journal: with {label.lower()}, you've felt {rest.split(' (')[0]}."
    suggestions = []
    if ctx["helper_labels"] and VALENCE.get(ctx["emotion"]) != "positive":
        h = ctx["helpers"][0]
        suggestions.append(f"{h.split(' (')[0]} has lifted your mood before ({h.split('(')[1].rstrip(')')}). "
                           "Could you fit a little of it in tomorrow?")
    suggestions.append(_MICRO.get(ctx["emotion"], _MICRO["neutral"]))
    if ctx["upcoming"]:
        suggestions.append(f"For {ctx['upcoming'][0].lower()} coming up, decide tonight on one thing to prepare "
                           "and one thing to do afterwards to unwind.")
    return {"reflection": reflection, "affirmation": rng.choice(_AFFIRM.get(ctx["emotion"], _AFFIRM["neutral"])),
            "suggestions": suggestions[:3]}


_PLAN_STEPS = {
    "exam": "Tonight, pick the three topics most likely to come up and revise only those; then stop at a fixed time.",
    "interview": "Write short answers to the three questions you dread most and say them out loud once.",
    "presentation": "Rehearse your opening minute twice; a strong start carries the rest.",
    "meeting": "Write down the one point you want to make and one question you can ask.",
    "conflict": "Decide beforehand the one thing you need them to hear, and say it calmly.",
    "assignment": "Break it into the next three 30-minute tasks and start the first one today.",
    "results": "Decide now what you'll do in either case, so the result is information, not a verdict.",
    "boss": "Prepare a two-line update on what you've done and what you need.",
}


def template_plan(event: str, evidence: dict, helpers: list[str], day_name: str | None = None) -> dict:
    if day_name:
        intro = f"{day_name}s have often been hard for you, so it helps to go in with a plan."
        first = f"Tonight, lay out tomorrow: the one thing that must get done, and what can wait."
    elif event.startswith("topic:"):
        intro = "This has often been hard for you, so a small plan can help."
        first = f"Before it, write down what worries you most and one way you'd handle it."
    else:
        intro = f"{event_label(event)} have often been hard for you, so a little planning can help."
        first = _PLAN_STEPS.get(event, f"Before {event_phrase(event)}, write down what worries you most and one "
                                       "way you'd handle it.")
    steps = [first,
             "During it, slow your breathing for a few seconds whenever you notice tension: in for 4, out for 6.",
             f"Afterwards, plan some {helpers[0].lower()}: it has lifted your mood before." if helpers else
             "Afterwards, plan one small thing you enjoy, whatever happens."]
    return {"intro": intro, "steps": steps}


# ---------------------------------------------------------------------------
class Responder:
    def __init__(self, local, gemini, anonymizer: Anonymizer, helplines: list[dict]):
        self.local, self.gemini, self.anonymizer = local, gemini, anonymizer
        self.helplines = helplines

    # -------- which backends, in which order --------
    def chain(self, settings: dict, prefer: str | None = None) -> list[str]:
        gemini_ok = settings.get("gemini_consent") and self.gemini.available()
        if prefer == "gemini":
            if not settings.get("gemini_consent"):
                raise PermissionError("Turn on Gemini in Settings before using it.")
            return ["gemini", "local", "template"]
        if prefer == "local":
            return ["local", "template"]
        if prefer == "template":
            return ["template"]
        mode = settings.get("ai_mode", "local")
        if mode == "enhanced" and gemini_ok:
            return ["gemini", "local", "template"]
        if mode in ("local", "enhanced"):
            return ["local", "template"]
        return ["template"]

    def gemini_payload(self, entry: dict, events: list[dict], ctx: dict, hidden_names: list[str]):
        """Exactly what Gemini receives: anonymised key sentences + context. Returns (text, mapping)."""
        sents = key_sentences(entry["body"], events)
        # One pass over sentences + context so each placeholder means the same thing everywhere.
        sep = "\n\u241e\n"
        red = self.anonymizer.anonymize("\n".join(sents) + sep + _context_lines(ctx), hidden_names)
        sent_part, _, ctx_part = red.text.partition(sep)
        text = ("Key sentences from today's entry (names, places and numbers replaced; the full entry stays "
                f"on the user's device):\n\"\"\"\n{sent_part}\n\"\"\"\n\n{ctx_part}\n\nWrite the JSON now.")
        return text, red.mapping

    def respond(self, entry: dict, events: list[dict], insights, settings: dict, prefer: str | None = None) -> Reply:
        t0 = time.perf_counter()
        if crisis_check(entry["body"]).flagged:
            r = crisis_response(self.helplines)
            return Reply("crisis", r["reflection"], r["affirmation"], r["suggestions"], "safety",
                         "fixed safety message", "none", latency_ms=round((time.perf_counter() - t0) * 1000))

        ctx = build_context(entry, events, insights)
        errors = []
        for backend in self.chain(settings, prefer):
            start = time.perf_counter()
            try:
                if backend == "template":
                    out, model, privacy, sent = template_reply(ctx, seed=entry["id"]), "template (no AI model)", "on-device", None
                elif backend == "local":
                    if not self.local.available():
                        raise LLMError(self.local.status()["error"])
                    prompt = (f"Journal entry written on {ctx['date']}:\n\"\"\"\n{entry['body'][:4000]}\n\"\"\"\n\n"
                              f"{_context_lines(ctx)}\n\nWrite the JSON now.")
                    out = self.local.chat(SYSTEM_PROMPT, prompt, REFLECTION_SCHEMA)
                    model, privacy, sent = self.local.label, "on-device", None
                else:
                    sent, mapping = self.gemini_payload(entry, events, ctx, settings.get("hidden_names", []))
                    out = self.gemini.chat(SYSTEM_PROMPT, sent, REFLECTION_SCHEMA)
                    out = self._restore(out, mapping)
                    model, privacy = self.gemini.label, "anonymised-summary"
                clean = validate_reply(out)
                note = ("; ".join(errors) and f"Used {backend} because: " + "; ".join(errors)) or None
                return Reply("reflection", clean["reflection"], clean["affirmation"], clean["suggestions"],
                             backend, model, privacy, sent, round((time.perf_counter() - start) * 1000), note)
            except PermissionError:
                raise
            except Exception as exc:          # network, bad JSON, unsafe content... try the next one
                errors.append(f"{backend}: {exc}")
        raise LLMError("; ".join(errors))     # unreachable in practice: template never fails

    def plan(self, alert: dict, insights, settings: dict, prefer: str | None = None) -> dict:
        """3-step preparation plan for a heads-up. Only labels and counts are sent, never entry text."""
        helpers = [p.label for p in insights.lifts() if not p.event.startswith("topic:")][:2]
        ev = alert.get("event") or "day"
        evidence = alert.get("evidence") or {}
        if alert["kind"] == "day":
            situation = alert["message"].split(":")[0]
        else:
            situation = f"{event_phrase(ev)} ({alert['target_date'] or 'soon'})"
        prompt = (f"Upcoming: {situation}\nEvidence from their journal: {alert['message']}\n"
                  f"What usually lifts their mood: {', '.join(helpers) or 'not known yet'}\nWrite the JSON now.")
        errors = []
        for backend in self.chain(settings, prefer):
            start = time.perf_counter()
            try:
                if backend == "template":
                    day = alert["message"].split("s have been tough")[0] if alert["kind"] == "day" else None
                    out, model = template_plan(ev, evidence, helpers, day), "template (no AI model)"
                elif backend == "local":
                    if not self.local.available():
                        raise LLMError(self.local.status()["error"])
                    out, model = self.local.chat(PLAN_PROMPT, prompt, PLAN_SCHEMA), self.local.label
                else:
                    red = self.anonymizer.anonymize(prompt, settings.get("hidden_names", []))
                    out = self._restore(self.gemini.chat(PLAN_PROMPT, red.text, PLAN_SCHEMA), red.mapping)
                    model = self.gemini.label
                intro = _clip(out.get("intro"), 300)
                steps = [_clip(s, 300) for s in (out.get("steps") or []) if isinstance(s, str) and s.strip()][:3]
                if not intro or len(steps) < 2 or any(_UNSAFE.search(x) for x in [intro, *steps]):
                    raise LLMError("plan missing parts or failed the safety filter")
                return {"intro": intro, "steps": steps, "backend": backend, "model": model,
                        "latency_ms": round((time.perf_counter() - start) * 1000)}
            except PermissionError:
                raise
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
        raise LLMError("; ".join(errors))

    def _restore(self, out: dict, mapping: dict) -> dict:
        def fix(v):
            if isinstance(v, str):
                return self.anonymizer.restore(v, mapping)
            if isinstance(v, list):
                return [fix(x) for x in v]
            return v
        return {k: fix(v) for k, v in out.items()}


def _clip(s, n: int) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def validate_reply(out: dict) -> dict:
    reflection = _clip(out.get("reflection"), 700)
    affirmation = _clip(out.get("affirmation"), 220)
    sugg = out.get("suggestions") or []
    if isinstance(sugg, str):
        sugg = [s for s in re.split(r"\n+|(?<=\.)\s+(?=\d+[.)]|-)", sugg) if s.strip()]
    sugg = [_clip(re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", s), 280) for s in sugg if isinstance(s, str)]
    sugg = [s for s in sugg if s and not _UNSAFE.search(s)][:3]
    if len(reflection) < 20 or not sugg:
        raise LLMError("reply missing a reflection or suggestions")
    if _UNSAFE.search(reflection) or _UNSAFE.search(affirmation):
        raise LLMError("reply failed the safety filter (diagnosis or medication)")
    return {"reflection": reflection, "affirmation": affirmation, "suggestions": sugg}
