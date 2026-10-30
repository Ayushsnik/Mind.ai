"""Does the pattern engine find real patterns, and only real ones?

Real users can't give us months of journals with KNOWN causes of their moods, so
we generate them: 60 days of entries where some events are PLANTED to cause a
specific emotion most of the time (exam -> anxiety 85%), and NOISE events come
with random emotions. Everything goes through the real app pipeline (cleaning,
emotion model, event extraction, sentence attribution, mining, alerts).

Measured
  recall      planted patterns the engine recovers (right direction, right emotion)
  precision   of the patterns it reports for taxonomy events, how many are real
  noise FP    noise events wrongly reported as patterns
  alerts      warnings fire for upcoming planted-negative events (exam, Monday meeting)
              and stay quiet for positive ones (family dinner, hobby)

Run:
  python -m evaluation.synthetic_patterns                  # 5 seeds, prints + saves results
  python -m evaluation.synthetic_patterns --seeds 10 --noise 0.25
  python -m evaluation.synthetic_patterns --seed-demo      # fill YOUR app database with 60 demo days
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

FEELINGS = {
    "fear": ["really nervous", "anxious the whole time", "so worried", "scared and tense"],
    "joy": ["really happy", "great afterwards", "so proud", "grateful and happy"],
    "sadness": ["sad", "low and lonely", "really down and sad", "hurt"],
    "anger": ["angry", "so frustrated", "really annoyed", "furious"],
    "neutral": ["okay, nothing special", "fine, just a normal time", "neither good nor bad"],
}
NOISE_EMOTIONS = ["joy", "sadness", "anger", "fear", "neutral"]


@dataclass
class Planted:
    event: str              # taxonomy key the engine should report
    kind: str               # "drains" | "lifts"
    emotion: str
    clauses: list[str]
    p: float = 0.85         # how often the planted emotion appears (rest is random)


PLANTED = {
    "exam": Planted("exam", "drains", "fear", [
        "The DBMS exam was today", "Wrote the unit test this morning", "Gave my viva in the afternoon",
        "Spent all day on the end sem paper"]),
    "exercise": Planted("exercise", "lifts", "joy", [
        "Went to the gym after college", "Did a long workout in the evening", "Went for a run near the lake",
        "Played badminton with the hostel guys"]),
    "conflict": Planted("conflict", "drains", "anger", [
        "Had an argument with my roommate about cleaning", "Got into a fight with my roommate over the rent",
        "My roommate and I had a big quarrel"]),
    "family": Planted("family", "lifts", "joy", [
        "Called my mom in the evening", "Had a long video call with my parents", "Talked to my sister for an hour"]),
    "meeting": Planted("meeting", "drains", "fear", [
        "The Monday team meeting was this morning", "Had the weekly review meeting with the manager",
        "Presented my progress in the team meeting"]),
    "poor_sleep": Planted("poor_sleep", "drains", "sadness", [
        "Couldn't sleep last night", "Barely slept because of the noise", "Stayed up till 4 am"]),
}
# roommate is mentioned in every conflict clause, so it legitimately becomes an anger pattern too.
EXPECTED_SIDE_EFFECTS = {"roommate": ("drains", "anger")}

NOISE = {
    "entertainment": ["Watched a movie at night", "Finished two episodes of a series", "Played video games for a bit"],
    "commute": ["The local train was crowded", "Got stuck in traffic on the way back", "Took the bus to Dadar"],
    "outing": ["Went to a cafe near the station", "Walked around the mall", "Visited the beach for a while"],
}
FILLERS = ["Long day overall.", "Ate dinner late.", "Not much else happened.", "Wrote this before bed."]


def sentence(clause: str, emotion: str, rng) -> str:
    return f"{clause} and I felt {rng.choice(FEELINGS[emotion])}."


def generate(days=60, seed=0, noise_rate=0.15, start: date | None = None):
    """Returns [(created_at_iso, text)] - one entry per day, oldest first."""
    rng = random.Random(seed)
    start = start or date.today() - timedelta(days=days)
    exam_days = {d for block in (8, 28, 48) for d in range(block, block + 5)}
    out = []

    def planted(key):
        p = PLANTED[key]
        emo = p.emotion if rng.random() > noise_rate else rng.choice([e for e in NOISE_EMOTIONS if e != p.emotion])
        return sentence(rng.choice(p.clauses), emo, rng)

    for i in range(days):
        day = start + timedelta(days=i)
        wd = day.weekday()
        parts = []
        if i in exam_days:
            parts.append(planted("exam"))
            if rng.random() < 0.5:
                parts.append(planted("poor_sleep"))
        elif rng.random() < 0.12:
            parts.append(planted("poor_sleep"))
        if wd == 0:
            parts.append(planted("meeting"))
        if wd in (1, 3, 5) and rng.random() < 0.8:
            parts.append(planted("exercise"))
        if wd == 6:
            parts.append(planted("family"))
        if rng.random() < 0.13:
            parts.append(planted("conflict"))
        for key, clauses in NOISE.items():
            if rng.random() < 0.18:
                parts.append(sentence(rng.choice(clauses), rng.choice(NOISE_EMOTIONS), rng))
        # Upcoming mentions, so alerts can be tested
        if i + 1 in exam_days and i not in exam_days:
            parts.append("I have an exam tomorrow.")
        if not parts:
            parts.append(sentence("Spent the day at college", rng.choice(NOISE_EMOTIONS), rng))
        rng.shuffle(parts)
        parts.append(rng.choice(FILLERS))
        out.append((datetime.combine(day, datetime.min.time()).replace(hour=21, minute=30).isoformat(), " ".join(parts)))
    return out


def make_app(tmp: Path, backend: str | None):
    from app import create_app
    from config import Config

    class EvalConfig(Config):
        TESTING = True
        DATABASE_PATH = str(tmp / "eval.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp / "eval.key")
        MEDIA_DIR = str(tmp / "media")
        EMOTION_BACKEND = backend or Config.EMOTION_BACKEND
    return create_app(EvalConfig)


def run_once(seed: int, days: int, noise_rate: float, backend: str | None) -> dict:
    today = date.today()
    with tempfile.TemporaryDirectory() as d:
        app = make_app(Path(d), backend)
        ext = app.extensions["mindai"]
        for created, text in generate(days, seed, noise_rate, start=today - timedelta(days=days)):
            ext["create_entry"](text, None, created_at=created, refresh_alerts=False)
        ins = ext["insights"].insights()
        found = ins.by_event

        # ---- pattern recovery ----
        expected = {k: (p.kind, p.emotion) for k, p in PLANTED.items()} | EXPECTED_SIDE_EFFECTS
        recovered, detected = {}, {}
        for key, (kind, emo) in expected.items():
            p = found.get(key)
            direction_ok = bool(p and p.kind == kind and (kind == "lifts" or p.top_emotion == emo))
            detected[key] = direction_ok
            recovered[key] = direction_ok and p.established
        # What the engine asserts: established (statistically significant) patterns for taxonomy events.
        reported = [p for p in ins.events if p.established and not p.event.startswith("topic:")]
        # Correct = a planted pattern, or a real correlation the generator creates (e.g. "manager"
        # is only ever mentioned in the anxious Monday meeting, so Boss -> anxious is genuinely in the data).
        real_side = {"boss": ("drains", "fear"), "presentation": ("drains", "fear"), "chores": ("drains", "anger")}
        correct = [p for p in reported if recovered.get(p.event)
                   or (p.event in real_side and (p.kind, p.top_emotion) == real_side[p.event])]
        noise_fp = [p.event for p in reported if p.event in NOISE]

        # ---- alerts: one entry written today about tomorrow ----
        alert_cases = {"I have an exam tomorrow and need to revise.": True,
                       "There is a team meeting tomorrow.": True,
                       "Dinner with my family tomorrow.": False,
                       "Going to the gym tomorrow morning.": False}
        alert_hits = {}
        for text, should_fire in alert_cases.items():
            out = ext["create_entry"](text, None)
            fired = any(a["kind"] == "event" for a in out["alerts"])
            alert_hits[text] = fired == should_fire

        return {
            "seed": seed, "entries": ins.analysed_entries,
            "extractor": ext["insights"].extractor.name, "emotion_backend": ext["emotions"].status()["active"],
            "recall": sum(recovered.values()) / len(recovered),
            "detected": sum(detected.values()) / len(detected),
            "precision": (len(correct) / len(reported)) if reported else 0.0,
            "noise_false_positives": noise_fp,
            "emerging": [p.event for p in ins.emerging()],
            "missed": [k for k, ok in recovered.items() if not ok],
            "monday_hard": ins.days[0].kind == "hard",
            "alert_accuracy": sum(alert_hits.values()) / len(alert_hits),
            "alert_failures": [t for t, ok in alert_hits.items() if not ok],
            "combos": [f"{' + '.join(c.labels)} -> {c.outcome} ({c.confidence:.0%}, n={c.count})" for c in ins.combos],
            "patterns": {p.event: f"{p.kind}/{p.top_emotion} n={p.n} strength={p.strength}" for p in ins.events},
        }


def evaluate(seeds: int, days: int, noise_rate: float, backend: str | None):
    runs = []
    for s in range(seeds):
        r = run_once(s, days, noise_rate, backend)
        runs.append(r)
        print(f"seed {s}: recall {r['recall']:.2f}  precision {r['precision']:.2f}  alerts {r['alert_accuracy']:.2f}"
              f"  monday={r['monday_hard']}  missed={r['missed']}  noise_fp={r['noise_false_positives']}")
    mean = lambda k: statistics.mean(r[k] for r in runs)
    print(f"\n{days} days per run, {noise_rate:.0%} of planted events carry a random emotion, "
          f"extractor={runs[0]['extractor']}, emotion={runs[0]['emotion_backend']}")
    print(f"{'metric':<34}{'mean':>8}")
    print(f"{'Planted patterns established':<34}{mean('recall'):>8.2f}   (significant, used for alerts)")
    print(f"{'Planted patterns detected':<34}{mean('detected'):>8.2f}   (right direction, incl. emerging)")
    print(f"{'Pattern precision':<34}{mean('precision'):>8.2f}")
    print(f"{'Noise events reported as pattern':<34}{statistics.mean(len(r['noise_false_positives']) for r in runs):>8.2f}")
    print(f"{'Monday detected as a hard day':<34}{statistics.mean(r['monday_hard'] for r in runs):>8.2f}")
    print(f"{'Alert decisions correct':<34}{mean('alert_accuracy'):>8.2f}")
    print(f"{'Spurious combination rules':<34}{statistics.mean(len(r['combos']) for r in runs):>8.2f}"
          "   (no interactions are planted, so any rule is spurious)")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "synthetic_patterns.json").write_text(json.dumps(runs, indent=2))
    print(f"\nSaved {out / 'synthetic_patterns.json'}")
    return runs


def seed_demo(days: int, yes: bool):
    """Fill the real app database with demo history (for a viva demo)."""
    from app import create_app
    app = create_app()
    ext = app.extensions["mindai"]
    if ext["db"].count() and not yes:
        sys.exit(f"Your journal already has {ext['db'].count()} entries. Re-run with --yes to add demo entries anyway.")
    today = date.today()
    for created, text in generate(days, seed=42, start=today - timedelta(days=days)):
        ext["create_entry"](text, None, created_at=created, refresh_alerts=False)
    ext["create_entry"]("I have my DBMS viva tomorrow and the syllabus is huge.", "Viva tomorrow",
                        created_at=datetime.now().replace(microsecond=0).isoformat())
    print(f"Added {days + 1} demo entries. Open the app and check the Insights page and the alert on Write.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.15, help="share of planted events with a random emotion")
    ap.add_argument("--backend", choices=["auto", "roberta", "lexicon"], default=None)
    ap.add_argument("--seed-demo", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.seed_demo:
        seed_demo(a.days, a.yes)
    else:
        evaluate(a.seeds, a.days, a.noise, a.backend)
