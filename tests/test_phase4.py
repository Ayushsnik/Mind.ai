import json
from datetime import date, datetime, timedelta

import pytest

from app import create_app
from config import TestConfig
from pipeline.anonymize import Anonymizer
from pipeline.llm import LLMError, parse_json_reply
from pipeline.responder import template_plan, template_reply, validate_reply
from pipeline.safety import check
from tests.fake_llm import FakeLLM


# ======================= anonymiser =======================
@pytest.fixture(scope="module")
def anon():
    return Anonymizer()


@pytest.mark.parametrize("text,secrets", [
    ("I met Rohan and Priya at Phoenix Mall in Kurla.", ["Rohan", "Priya", "Phoenix Mall", "Kurla"]),
    ("Call me on 9876543210 or +91 98200 12345.", ["9876543210", "98200 12345"]),
    ("Sneha said it was fine.", ["Sneha"]),
    ("Email sneha.k@gmail.com, roll no CS-2041, Aadhaar 1234 5678 9012, PIN 400708.",
     ["sneha.k@gmail.com", "CS-2041", "1234 5678 9012", "400708"]),
    ("See https://drive.google.com/abc for notes.", ["https://drive.google.com/abc"]),
])
def test_anonymiser_removes_personal_details(anon, text, secrets):
    red = anon.anonymize(text)
    assert not [s for s in secrets if s in red.text]
    assert anon.restore(red.text, red.mapping) == text


def test_anonymiser_keeps_ordinary_words_and_is_consistent(anon):
    red = anon.anonymize("Exam was hard. Mom called. Traffic was bad. Rohan helped, and later Rohan called.")
    assert red.text.startswith("Exam was hard. Mom called. Traffic was bad.")
    assert red.text.count(list(red.mapping)[0]) == 2          # same person -> same placeholder


def test_hidden_names_always_removed(anon):
    red = anon.anonymize("chintu was annoying today", hidden_names=["Chintu"])
    assert "chintu" not in red.text.lower()


def test_restore_tolerates_missing_brackets(anon):
    assert anon.restore("Talk to PERSON_1 tomorrow", {"[PERSON_1]": "Rohan"}) == "Talk to Rohan tomorrow"


# ======================= crisis check =======================
@pytest.mark.parametrize("text", [
    "I just want to die.", "Sometimes I think about killing myself.", "I don't want to live anymore.",
    "Kabhi kabhi lagta hai mar jaana chahta hoon.", "I've been self-harming again.",
    "Maybe they'd be better off if I ended my life.", "I feel suicidal tonight.",
])
def test_crisis_detected(text):
    assert check(text).flagged


@pytest.mark.parametrize("text", [
    "I don't want to die, I want to live a long life.", "The exam killed me today lol.",
    "The movie was about suicide prevention awareness.", "My phone died.", "I will not hurt myself, I promise.",
])
def test_no_false_crisis(text):
    assert not check(text).flagged


# ======================= parsing & validation =======================
def test_parse_json_reply_tolerates_fences_and_prose():
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_reply('Sure! Here you go: {"a": 2} hope it helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_reply("no json here")


def test_validate_reply_cleans_and_blocks_unsafe():
    out = validate_reply({"reflection": "That sounds like a long, tiring day of revision.", "affirmation": "ok",
                          "suggestions": "1. Take a walk.\n2. Sleep early."})
    assert out["suggestions"] == ["Take a walk.", "Sleep early."]
    with pytest.raises(LLMError):
        validate_reply({"reflection": "You have depression and need treatment, clearly.", "affirmation": "",
                        "suggestions": ["Rest."]})
    out = validate_reply({"reflection": "That sounds like a long, tiring day of revision.", "affirmation": "",
                          "suggestions": ["Take 50 mg of melatonin.", "Go for a walk."]})
    assert out["suggestions"] == ["Go for a walk."]


def test_template_reply_uses_patterns():
    ctx = {"date": "2026-10-08", "emotion": "fear", "confidence": 80, "secondary": None,
           "happened": ["Exams (anxious)"], "upcoming": ["Exams"], "history": ["Exams: felt anxious in 15 of 19 entries"],
           "helpers": ["Exercise (happy in 16 of 17 entries)"], "helper_labels": ["Exercise"]}
    r = validate_reply(template_reply(ctx))
    assert "15 of 19" in r["reflection"] and r["suggestions"][0].startswith("Exercise")
    plan = template_plan("exam", {}, ["Exercise"])
    assert len(plan["steps"]) == 3 and "exercise" in plan["steps"][2]


# ======================= app with fake Ollama + Gemini =======================
@pytest.fixture
def fake():
    with FakeLLM() as f:
        yield f


@pytest.fixture
def app(tmp_path, fake):
    class Cfg(TestConfig):
        DATABASE_PATH = str(tmp_path / "t.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp_path / "t.key")
        MEDIA_DIR = str(tmp_path / "media")
        OLLAMA_HOST = fake.url
        GEMINI_API_KEY = "test-key"
        GEMINI_BASE_URL = fake.url + "/v1beta"
    return create_app(Cfg)


ENTRY = "Studied with Rohan at Dadar till late. My viva is tomorrow and I'm really nervous. Call Priya on 9876543210."


def test_local_reply_keeps_entry_on_device(app, fake):
    c = app.test_client()
    eid = c.post("/api/entries", json={"body": ENTRY}).get_json()["id"]
    r = c.post(f"/api/entries/{eid}/reflect", json={}).get_json()
    assert r["backend"] == "local" and r["privacy"] == "on-device" and r["sent"] is None
    assert len(r["suggestions"]) == 3 and not r["suggestions"][0].startswith("1.")
    assert all("generateContent" not in q["path"] for q in fake.requests)      # nothing went to Gemini


def test_gemini_needs_consent_and_only_sees_anonymised_summary(app, fake):
    c = app.test_client()
    eid = c.post("/api/entries", json={"body": ENTRY}).get_json()["id"]
    assert c.post(f"/api/entries/{eid}/reflect", json={"prefer": "gemini"}).status_code == 403

    # Enhanced mode without ticking consent is refused and falls back to local
    c.post("/settings", data={"ai_mode": "enhanced", "auto_reflect": "1"})
    assert c.post(f"/api/entries/{eid}/reflect", json={}).get_json()["backend"] == "local"

    c.post("/settings", data={"ai_mode": "enhanced", "gemini_consent": "1", "auto_reflect": "1"})
    r = c.post(f"/api/entries/{eid}/reflect", json={}).get_json()
    assert r["backend"] == "gemini" and r["privacy"] == "anonymised-summary"
    sent = json.dumps([q["body"] for q in fake.requests if "generateContent" in q["path"]])
    for secret in ("Rohan", "Dadar", "Priya", "9876543210"):
        assert secret not in sent
    assert "Rohan" in r["reflection"]                  # placeholder restored locally
    assert "temperature" not in sent                    # deprecated for Gemini 3
    html = c.get(f"/entries/{eid}").data.decode()
    assert "See exactly what was sent to Gemini" in html and "[PERSON_1]" in html


def test_fallback_chain(app, fake):
    c = app.test_client()
    eid = c.post("/api/entries", json={"body": "The exam was stressful."}).get_json()["id"]
    fake.mode = "garbage"
    r = c.post(f"/api/entries/{eid}/reflect", json={}).get_json()
    assert r["backend"] == "template" and "local:" in r["note"]
    fake.mode = "unsafe"
    assert c.post(f"/api/entries/{eid}/reflect", json={}).get_json()["backend"] == "template"


def test_crisis_entry_never_reaches_a_model(app, fake):
    c = app.test_client()
    c.post("/settings", data={"ai_mode": "enhanced", "gemini_consent": "1", "auto_reflect": "1"})
    eid = c.post("/api/entries", json={"body": "I can't do this anymore. I want to die."}).get_json()["id"]
    stored = c.get(f"/api/entries/{eid}/response").get_json()
    assert stored["kind"] == "crisis" and stored["backend"] == "safety"
    r = c.post(f"/api/entries/{eid}/reflect", json={"prefer": "gemini"}).get_json()
    assert r["kind"] == "crisis"
    assert not any("want to die" in json.dumps(q["body"]) for q in fake.requests)
    assert "14416" in c.get(f"/entries/{eid}").data.decode()


def test_ai_off_still_shows_crisis_and_offers_template(app, fake):
    c = app.test_client()
    c.post("/settings", data={"ai_mode": "off"})
    eid = c.post("/api/entries", json={"body": "Normal day at college."}).get_json()["id"]
    html = c.get(f"/entries/{eid}").data.decode()
    assert 'data-auto="0"' in html and "Get a note back" in html
    assert c.post(f"/api/entries/{eid}/reflect", json={}).get_json()["backend"] == "template"
    assert not fake.requests


def test_rating_and_stats(app):
    c = app.test_client()
    eid = c.post("/api/entries", json={"body": "Went to the gym, felt great."}).get_json()["id"]
    rid = c.post(f"/api/entries/{eid}/reflect", json={}).get_json()["id"]
    assert c.post(f"/api/responses/{rid}/rate", json={"rating": 1}).status_code == 200
    assert c.post(f"/api/responses/{rid}/rate", json={"rating": 5}).status_code == 400
    assert app.extensions["mindai"]["db"].response_stats()[0]["helpful"] == 1
    assert "How the notes have been rated" in c.get("/settings").data.decode()


def test_response_text_encrypted_at_rest(app):
    import sqlite3
    c = app.test_client()
    eid = c.post("/api/entries", json={"body": "Went to the gym, felt great."}).get_json()["id"]
    c.post(f"/api/entries/{eid}/reflect", json={})
    raw = sqlite3.connect(app.config["DATABASE_PATH"]).execute("SELECT body_encrypted FROM responses").fetchone()[0]
    assert b"reflection" not in raw


def test_alert_plan(app, fake):
    c, ext = app.test_client(), app.extensions["mindai"]
    start = date.today() - timedelta(days=20)
    for i in range(20):
        created = datetime.combine(start + timedelta(days=i), datetime.min.time()).isoformat()
        text = ("The exam today made me so nervous and anxious." if i % 2 == 0
                else "Went to the gym and felt really happy.")
        ext["create_entry"](text, None, created_at=created, refresh_alerts=False)
    alert = c.post("/api/entries", json={"body": "Another exam tomorrow."}).get_json()["alerts"][0]
    plan = c.post(f"/api/alerts/{alert['id']}/plan", json={}).get_json()
    assert plan["backend"] == "local" and len(plan["steps"]) == 3
    assert plan["intro"] in c.get("/").data.decode()          # stored plan replaces the button
    c.post("/settings", data={"ai_mode": "off"})
    plan = c.post(f"/api/alerts/{alert['id']}/plan", json={}).get_json()
    assert plan["backend"] == "template" and "exercise" in plan["steps"][2].lower()


def test_settings_store_hidden_names_encrypted(app):
    import sqlite3
    c = app.test_client()
    c.post("/settings", data={"ai_mode": "local", "hidden_names": "Chintu, Shreya\nVashi"})
    raw = sqlite3.connect(app.config["DATABASE_PATH"]).execute(
        "SELECT value FROM settings WHERE key='hidden_names'").fetchone()[0]
    assert b"Chintu" not in raw
    assert "Chintu, Shreya, Vashi" in c.get("/settings").data.decode()


def test_no_ollama_running_degrades_gracefully(tmp_path):
    class Cfg(TestConfig):
        DATABASE_PATH = str(tmp_path / "t.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp_path / "t.key")
        MEDIA_DIR = str(tmp_path / "media")
        OLLAMA_HOST = "http://127.0.0.1:9"        # nothing listens here
    c = create_app(Cfg).test_client()
    eid = c.post("/api/entries", json={"body": "A quiet day."}).get_json()["id"]
    r = c.post(f"/api/entries/{eid}/reflect", json={}).get_json()
    assert r["backend"] == "template" and "Ollama isn't running" in r["note"]
    assert "Ollama isn't running" in c.get("/settings").data.decode()
