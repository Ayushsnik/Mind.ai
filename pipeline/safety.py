"""Crisis check that runs BEFORE any language model sees an entry.

If an entry suggests the writer may be thinking about suicide or self-harm, Mind.ai
does not generate an AI reflection (a model can say the wrong thing, and generic
tips are inappropriate). It shows a fixed, calm message with helplines instead.

The check is deliberately cautious: a false alarm costs the user one supportive
message; a miss could cost much more. Negated statements ("I don't want to die")
are excluded. English plus common romanised Hindi/Marathi phrasing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS = [
    r"\bkill(?:ing)? my ?self\b", r"\bsuicid(?:e|al)\b(?! (?:prevention|awareness|helpline|hotline|rates?|statistics))", r"\bend(?:ed|ing|s)? (?:my|this) life\b",
    r"\bend it all\b", r"\btake my (?:own )?life\b", r"\bwant(?:ed)? to die\b", r"\bwish i (?:was|were) dead\b",
    r"\bbetter off dead\b", r"\bno reason to (?:live|go on)\b", r"\bdon'?t want to (?:live|be alive|exist)\b",
    r"\bcan'?t go on\b", r"\bself[- ]?harm(?:ing)?\b", r"\bhurt(?:ing)? my ?self\b", r"\bcut(?:ting)? my ?self\b",
    r"\bnot worth living\b", r"\bdisappear forever\b", r"\bnobody would (?:care|miss me) if i\b",
    # romanised Hindi / Marathi
    r"\bmar(?: |-)?(?:jaana|jana|jaun|jau) (?:chahta|chahti|hai|aahe)\b", r"\bkhud ?kushi\b", r"\baatmahatya\b",
    r"\batmahatya\b", r"\bjeena nahi chahta\b", r"\bjeena nahi chahti\b", r"\bjagayche nahi\b",
]
_RX = re.compile("|".join(_PATTERNS), re.I)
_NEGATED = re.compile(r"\b(?:don'?t|do not|never|no longer|not)\s+(?:\w+\s+){0,2}$", re.I)


@dataclass
class CrisisCheck:
    flagged: bool
    reason: str | None = None     # kept internal; never shown or stored with the phrase


def check(text: str) -> CrisisCheck:
    for m in _RX.finditer(text or ""):
        before = text[max(0, m.start() - 30):m.start()]
        phrase = m.group(0).lower()
        # "I don't want to die" is the opposite of a risk statement; "don't want to live" is matched
        # by its own pattern, so only exclude the negation for phrases that are not themselves negative.
        if not phrase.startswith(("don", "can", "no ", "not ")) and _NEGATED.search(before):
            continue
        return CrisisCheck(True, "self-harm-language")
    return CrisisCheck(False)


def crisis_response(helplines: list[dict]) -> dict:
    lines = "; ".join(f"{h['name']} {h['number']}" + (f" ({h['note']})" if h.get("note") else "")
                      for h in helplines)
    return {
        "reflection": ("It sounds like you're carrying something really heavy right now, and you don't have "
                       "to carry it alone. What you're feeling matters, and talking to someone can help."),
        "affirmation": "Reaching out is a strong thing to do, not a weak one.",
        "suggestions": [
            f"If you are thinking about ending your life or hurting yourself, please talk to someone now: {lines}.",
            "Tell one person you trust how you're feeling today, even in a short message.",
            "If you can, stay with someone or go somewhere you feel safe for now.",
        ],
    }
