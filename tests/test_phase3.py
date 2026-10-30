from datetime import date, datetime, timedelta

import pytest

from app import create_app
from config import TestConfig
from pipeline.alerts import build_alerts, when_phrase
from pipeline.events import KeywordExtractor, SpacyExtractor, resolve_date
from pipeline.patterns import analyze, combo_rules, wilson_lower

THU = date(2026, 10, 8)   # a Thursday

try:
    SPACY = SpacyExtractor()
except Exception:          # spaCy model not installed: those tests are skipped, fallback still tested
    SPACY = None
EXTRACTORS = [e for e in (SPACY, KeywordExtractor()) if e is not None]


def events(extractor, text, today=THU):
    return [(m.event, m.timing, m.target_date) for m in extractor.extract(text, today).mentions
            if not m.event.startswith("topic:")]


# ======================= dates =======================
@pytest.mark.parametrize("text,expected", [
    ("exam tomorrow", "2026-10-09"), ("day after tomorrow", "2026-10-10"), ("meeting on Monday", "2026-10-12"),
    ("next week", "2026-10-15"), ("in 3 days", "2026-10-11"), ("tonight", "2026-10-08"),
    ("this weekend", "2026-10-10"), ("next Thursday", "2026-10-15"),
])
def test_resolve_date(text, expected):
    future, when = resolve_date(text, THU)
    assert future and when.isoformat() == expected


def test_no_date_words():
    assert resolve_date("I went to college", THU) == (False, None)


# ======================= extraction (both extractors) =======================
@pytest.mark.parametrize("ex", EXTRACTORS, ids=lambda e: e.name)
def test_basic_events_and_sentence_index(ex):
    out = ex.extract("The exam was awful. After that I went to the gym and felt great.", THU)
    found = {(m.event, m.sentence_idx) for m in out.mentions}
    assert ("exam", 0) in found and ("exercise", 1) in found


@pytest.mark.parametrize("ex", EXTRACTORS, ids=lambda e: e.name)
def test_negation_cancels_event(ex):
    assert events(ex, "I didn't go to the gym today.") == []
    assert events(ex, "There was no exam this week.") == []


@pytest.mark.parametrize("ex", EXTRACTORS, ids=lambda e: e.name)
def test_negation_inside_trigger(ex):
    assert ("poor_sleep", "past", None) in events(ex, "Couldn't sleep last night.")


@pytest.mark.parametrize("ex", EXTRACTORS, ids=lambda e: e.name)
def test_upcoming_with_date(ex):
    assert ("exam", "upcoming", "2026-10-09") in events(ex, "I have a viva tomorrow and I am scared.")
    assert ("meeting", "upcoming", "2026-10-12") in events(ex, "There's a meeting on Monday.")


@pytest.mark.skipif(SPACY is None, reason="spaCy model not installed")
def test_spacy_specifics():
    # preference verbs don't cancel the event; attendance verbs do
    assert events(SPACY, "I didn't like the movie at all.") == [("entertainment", "past", None)]
    # clause-level timing: one past exam, one upcoming
    got = events(SPACY, "Yesterday's exam went badly and I have another exam on Friday.")
    assert ("exam", "past", None) in got and ("exam", "upcoming", "2026-10-09") in got
    # part-of-speech rules: "ran out" / "ran to class" are not exercise, "went for a run" is
    assert not any(e == "exercise" for e, *_ in events(SPACY, "I ran out of time and ran to class."))
    assert ("exercise", "past", None) in events(SPACY, "Went for a run in the evening.")
    # "meeting" lemmatised as the verb "meet" must still match
    assert ("meeting", "past", None) in events(SPACY, "Had the weekly review meeting with the manager.")
    # punctuation-based sentences keep an event with its feeling
    out = SPACY.extract("The Monday team meeting was this morning and I felt nervous.", THU)
    assert len(out.sentences) == 1
    # sentences without a verb don't hang (regression: infinite loop in _governor)
    assert SPACY.extract("My secret thought about the exam", THU).mentions


@pytest.mark.skipif(SPACY is None, reason="spaCy model not installed")
def test_topics_skip_generic_and_emotion_nouns():
    topics = [m.event for m in SPACY.extract("My dissertation is stressful. The anxiety about the dissertation "
                                             "ruined the day.", THU).mentions if m.event.startswith("topic:")]
    assert topics == ["topic:dissertation"]


# ======================= statistics =======================
def test_wilson_penalises_small_samples():
    assert wilson_lower(3, 3) < wilson_lower(18, 20) < 0.9
    assert wilson_lower(0, 0) == 0


def make_data(spec, start=date(2026, 8, 1)):
    """spec: list of (events, sentence emotion for each event, entry emotion) per day."""
    entries, mentions = [], []
    for i, (evs, entry_emotion) in enumerate(spec):
        eid = i + 1
        created = datetime.combine(start + timedelta(days=i), datetime.min.time()).isoformat()
        entries.append({"id": eid, "created_at": created, "emotion": entry_emotion})
        for ev, emo in evs:
            mentions.append({"entry_id": eid, "event": ev, "category": "x", "emotion": emo, "emotion_conf": 0.9,
                             "timing": "past", "target_date": None, "entry_created": created})
    return entries, mentions


def test_planted_patterns_found_and_noise_ignored():
    spec = []
    for i in range(40):
        evs = []
        if i % 2 == 0:
            evs.append(("exam", "fear" if i % 10 else "joy"))           # 90% fear
        if i % 3 == 0:
            evs.append(("exercise", "joy"))
        evs.append(("commute", ["joy", "sadness", "neutral", "anger", "fear"][i % 5]))   # pure noise
        spec.append((evs, "fear" if any(e == "exam" for e, _ in evs) else "neutral"))
    entries, mentions = make_data(spec)
    ins = analyze(entries, mentions)
    p = ins.by_event
    assert p["exam"].kind == "drains" and p["exam"].established and p["exam"].top_emotion == "fear"
    assert p["exercise"].kind == "lifts" and p["exercise"].established
    assert not p["commute"].established


def test_too_few_entries_shows_nothing():
    entries, mentions = make_data([([("exam", "fear")], "fear")] * 5)
    ins = analyze(entries, mentions, min_entries=8)
    assert not ins.enough_data and ins.events == []


def test_trend_detects_improvement():
    spec = [([("exam", "fear")], "fear")] * 6 + [([("exam", "fear" if i < 1 else "neutral")], "neutral") for i in range(6)]
    spec += [([("other", "neutral")], "neutral")] * 8
    entries, mentions = make_data(spec)
    ex = analyze(entries, mentions, min_rate=0.5, min_lift=1.0).by_event["exam"]
    assert ex.trend == "easier"


def test_combo_requires_real_interaction():
    # exam alone: 50% hard days. exam + poor sleep: always hard -> a genuine interaction.
    spec = []
    for i in range(60):
        if i % 3 == 0:
            spec.append(([("exam", "fear"), ("poor_sleep", "sadness")], "fear"))
        elif i % 3 == 1:
            spec.append(([("exam", "fear")], "fear" if i % 2 else "joy"))
        else:
            spec.append(([("poor_sleep", "sadness")], "sadness" if i % 2 else "joy"))
    entries, mentions = make_data(spec)
    rules = combo_rules(entries, mentions)
    assert any(set(r.events) == {"exam", "poor_sleep"} for r in rules)

    # no interaction: pairing a strong event with a random one must NOT produce a rule
    spec = [([("poor_sleep", "sadness"), ("commute", "neutral")] if i % 2 else [("poor_sleep", "sadness")],
             "sadness" if i % 7 else "joy") for i in range(40)]
    entries, mentions = make_data(spec)
    assert combo_rules(entries, mentions) == []


def test_day_pattern():
    spec = [([("x", "neutral")], "fear" if (date(2026, 8, 1) + timedelta(days=i)).weekday() == 0 else "joy")
            for i in range(35)]
    entries, mentions = make_data(spec)
    days = analyze(entries, mentions).days
    assert days[0].kind == "hard" and days[2].kind != "hard"


# ======================= alerts =======================
def test_when_phrase():
    assert when_phrase(THU, THU) == "today"
    assert when_phrase(THU + timedelta(days=1), THU) == "tomorrow"
    assert when_phrase(THU + timedelta(days=4), THU) == "on Monday"
    assert when_phrase(None, THU) == "coming up"


def test_alerts_only_for_established_negative_patterns():
    spec = [([("exam", "fear")], "fear")] * 10 + [([("exercise", "joy")], "joy")] * 10 + \
           [([("commute", "fear" if i % 2 else "joy")], "neutral") for i in range(10)]
    entries, mentions = make_data(spec)
    ins = analyze(entries, mentions)
    today = date(2026, 9, 1)
    upcoming = [{"entry_id": 99, "event": ev, "target_date": "2026-09-02", "entry_created": "2026-09-01T20:00:00"}
                for ev in ("exam", "exercise", "commute")]
    drafts = [d for d in build_alerts(ins, upcoming, today) if d.kind == "event"]
    assert [d.event for d in drafts] == ["exam"]
    assert "tomorrow" in drafts[0].message and "exercise" in drafts[0].suggestion
    # past target dates never alert
    old = [dict(upcoming[0], target_date="2026-08-20")]
    assert not [d for d in build_alerts(ins, old, today) if d.kind == "event"]


# ======================= app end to end =======================
@pytest.fixture
def app(tmp_path):
    class Cfg(TestConfig):
        DATABASE_PATH = str(tmp_path / "t.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp_path / "t.key")
        MEDIA_DIR = str(tmp_path / "media")
    return create_app(Cfg)


def test_end_to_end_alert_and_dismiss(app):
    c, ext = app.test_client(), app.extensions["mindai"]
    start = date.today() - timedelta(days=20)
    for i in range(20):
        created = datetime.combine(start + timedelta(days=i), datetime.min.time()).isoformat()
        text = ("The exam today made me so nervous and anxious." if i % 2 == 0
                else "Went to the gym and felt really happy.")
        ext["create_entry"](text, None, created_at=created, refresh_alerts=False)

    r = c.post("/api/entries", json={"body": "I have another exam tomorrow."})
    out = r.get_json()
    assert any(e["event"] == "exam" and e["timing"] == "upcoming" for e in out["events"])
    assert len(out["alerts"]) == 1 and "exam" in out["alerts"][0]["message"]

    html = c.get("/").data.decode()
    assert "You mentioned an exam tomorrow" in html and "What has helped before: exercise" in html
    insights = c.get("/insights").data.decode()
    assert "What weighs on you" in insights and "Exams" in insights and "Exercise" in insights
    assert "mention <strong>exams</strong>" in c.get("/entries?event=exam").data.decode()

    alert_id = out["alerts"][0]["id"]
    assert c.post(f"/alerts/{alert_id}/dismiss", data={"next": "//evil.com"}).headers["Location"] == "/"
    event_alerts = lambda: [a for a in c.get("/api/alerts").get_json() if a["kind"] == "event"]
    assert event_alerts() == []
    # the same situation doesn't alert again after being dismissed
    c.post("/api/entries", json={"body": "Still thinking about the exam tomorrow."})
    assert event_alerts() == []


def test_deleting_entry_removes_events_and_alerts(app):
    c = app.test_client()
    c.post("/api/entries", json={"body": "Had a fight with my roommate."})
    assert c.get("/api/entries/1/events").get_json()
    c.post("/entries/1/delete")
    assert app.extensions["mindai"]["db"].all_events() == []


def test_backfill_analyses_old_entries(app):
    ext = app.extensions["mindai"]
    db = ext["db"]
    entry_id = ext["create_entry"]("The exam was hard.", None)["id"]
    with db._conn() as conn:                     # simulate an entry saved before Phase 3
        conn.execute("DELETE FROM entry_events")
        conn.execute("UPDATE entries SET analysis_version = NULL")
    assert ext["insights"].status()["pending"] == 1
    r = app.test_client().post("/insights/reanalyse")
    assert r.status_code == 302 and db.events_for_entry(entry_id)


def test_insights_page_with_few_entries(app):
    c = app.test_client()
    c.post("/api/entries", json={"body": "Normal day at college."})
    html = c.get("/insights").data.decode()
    assert "Patterns appear after 8 entries. You have 1." in html
