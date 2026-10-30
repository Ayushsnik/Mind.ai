import sqlite3

import pytest

from app import create_app
from config import TestConfig
from pipeline.emotion import LexiconEmotionModel
from pipeline.preprocess import chunk_text, clean_text


@pytest.fixture
def app(tmp_path):
    class Cfg(TestConfig):
        DATABASE_PATH = str(tmp_path / "test.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp_path / "test.key")
    return create_app(Cfg)


@pytest.fixture
def client(app):
    return app.test_client()


# ---------- preprocessing ----------
def test_clean_keeps_negation_and_strips_voice_fillers():
    assert "not happy" in clean_text("I am   not happy", "text")
    assert clean_text("Um, I was uh tired", "voice") == "I was tired"
    assert clean_text("wow!!!!!") == "wow!!"


def test_chunking_respects_limit():
    text = ". ".join(["word " * 40] * 20)
    chunks = chunk_text(text, max_words=100)
    assert len(chunks) > 1 and all(len(c.split()) <= 100 for c in chunks)


# ---------- emotion fallback ----------
def test_lexicon_basic_and_negation():
    m = LexiconEmotionModel()
    assert m.predict("I am so worried and anxious about the exam").primary == "fear"
    assert m.predict("I was not happy at all").primary != "joy"
    assert m.predict("Went to college. Had lunch.").primary == "neutral"


# ---------- storage + encryption ----------
def test_entry_is_encrypted_at_rest(app, client):
    client.post("/api/entries", json={"body": "My secret thought about the exam"})
    raw = sqlite3.connect(app.config["DATABASE_PATH"]).execute(
        "SELECT body_encrypted FROM entries").fetchone()[0]
    assert b"secret" not in raw


# ---------- API ----------
def test_api_create_get_list(client):
    r = client.post("/api/entries", json={"body": "Feeling stressed and nervous today", "title": "Monday"})
    assert r.status_code == 201
    entry_id = r.get_json()["id"]
    got = client.get(f"/api/entries/{entry_id}").get_json()
    assert got["body"] == "Feeling stressed and nervous today"
    assert got["emotion"] == "fear" and got["title"] == "Monday"
    assert abs(sum(got["emotion_scores"].values()) - 1) < 1e-6
    assert len(client.get("/api/entries").get_json()) == 1


def test_api_rejects_empty_and_bad_type(client):
    assert client.post("/api/entries", json={"body": "   "}).status_code == 400
    assert client.post("/api/entries", json={"body": "hi", "input_type": "fax"}).status_code == 400


def test_analyze_does_not_save(client):
    client.post("/api/analyze", json={"text": "So happy and grateful today"})
    assert client.get("/api/health").get_json()["entries"] == 0


# ---------- pages ----------
def test_html_flow(client):
    assert client.get("/").status_code == 200
    r = client.post("/entries", data={"title": "Test", "body": "I miss home and feel lonely"},
                    follow_redirects=True)
    assert r.status_code == 200 and b"sadness" in r.data and b"I miss home" in r.data
    listing = client.get("/entries?emotion=sadness")
    assert b"Test" in listing.data
    assert client.post("/entries/1/delete", follow_redirects=True).status_code == 200
    assert client.get("/entries/1").status_code == 404


def test_empty_form_shows_error(client):
    r = client.post("/entries", data={"body": ""})
    assert r.status_code == 400 and b"Write something" in r.data


# ---------- RoBERTa aggregation logic (model mocked, no download needed) ----------
def test_roberta_chunk_weighting(monkeypatch):
    import sys, types
    from pipeline import emotion as em

    def fake_pipeline(*a, **k):
        def run(chunks):
            # first chunk: joy, all later chunks: sadness
            out = []
            for i, _ in enumerate(chunks):
                top = "joy" if i == 0 else "sadness"
                out.append([{"label": e, "score": 0.94 if e == top else 0.01} for e in em.EMOTIONS])
            return out
        return run

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))
    model = em.RobertaEmotionModel("fake", chunk_words=10)
    text = "Great start to the day honestly. " + "Then everything went badly wrong after lunch. " * 5
    r = model.predict(text)
    assert r.chunks > 1 and r.primary == "sadness"
    assert abs(sum(r.scores.values()) - 1) < 1e-6
