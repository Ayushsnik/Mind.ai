import io
import json
import sqlite3
import types

import numpy as np
import pytest
from PIL import Image

from app import create_app
from config import TestConfig
from db.crypto import Cipher
from db.media import MediaStore
from db.models import JournalDB
from pipeline import ocr as ocr_mod
from pipeline.metrics import cer, wer
from pipeline.ocr import LineResult, OCRError, OCRService, prepare, segment_lines
from pipeline.preprocess import clean_text
from pipeline.speech import SpeechError, SpeechService
from tests.make_test_page import TEXT, make_page

HAS_TESSERACT = OCRService().availability()["tesseract"]
needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract not installed")
TRUTH = clean_text("\n".join(TEXT), "image")


@pytest.fixture(scope="module")
def page_bytes(tmp_path_factory):
    path = tmp_path_factory.mktemp("img") / "page.png"
    make_page(str(path), angle=4.0)
    return path.read_bytes()


@pytest.fixture
def app(tmp_path):
    class Cfg(TestConfig):
        DATABASE_PATH = str(tmp_path / "t.sqlite3")
        ENCRYPTION_KEY_PATH = str(tmp_path / "t.key")
        MEDIA_DIR = str(tmp_path / "media")
    return create_app(Cfg)


@pytest.fixture
def client(app):
    return app.test_client()


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ======================= metrics & cleaning =======================
def test_metrics():
    assert cer("hello world", "hello world") == 0
    assert cer("abcd", "abed") == 0.25
    assert wer("I went to the gym", "I went to gym") == 0.2
    assert wer("Hello, World!", "hello world") == 0


def test_clean_ocr_text():
    out = clean_text("| was tired after the exam-\nination.\nIt rained.\n\nNext day", "image")
    assert out == "I was tired after the examination. It rained.\n\nNext day"
    assert "||" in clean_text("a || b", "image")      # only a lone pipe becomes I


# ======================= image preparation =======================
def test_prepare_deskews_and_segments(page_bytes):
    page = prepare(page_bytes)
    assert abs(page.skew - (-4.0)) <= 0.6             # page was rotated +4 degrees
    lines = segment_lines(page.ink)
    assert len(lines) == 5                            # ruled lines must not count as text
    assert [ln.paragraph_break for ln in lines] == [False, False, False, True, False]


def test_straight_page_has_no_skew(tmp_path):
    path = tmp_path / "straight.png"
    make_page(str(path), angle=0.0, shadow=False)
    assert abs(prepare(path.read_bytes()).skew) <= 0.4


def test_bad_inputs():
    with pytest.raises(OCRError):
        prepare(b"definitely not an image")
    with pytest.raises(OCRError):
        prepare(png_bytes(np.full((100, 100), 255, np.uint8)))          # too small
    svc = OCRService()
    svc._avail = {"tesseract": True, "trocr": False}
    with pytest.raises(OCRError, match="No writing"):
        svc.read(png_bytes(np.full((800, 600), 240, np.uint8)))         # blank page


# ======================= recognition =======================
@needs_tesseract
def test_tesseract_reads_tilted_shadowed_ruled_page(page_bytes):
    r = OCRService().read(page_bytes, "auto")
    assert r.engine == "tesseract" and r.mode == "printed"
    assert cer(TRUTH, clean_text(r.text, "image")) < 0.03
    assert r.preview.startswith("data:image/png;base64,")


def fake_trocr(svc, texts_confs):
    svc._avail = {"tesseract": svc.availability()["tesseract"], "trocr": True}
    svc.trocr.read_lines = lambda crops: [LineResult(*texts_confs[i % len(texts_confs)]) for i in range(len(crops))]


def test_handwritten_mode_uses_trocr_and_flags_unsure_lines(page_bytes):
    svc = OCRService()
    fake_trocr(svc, [("today was good", 0.9), ("i felt anxius", 0.3)])
    r = svc.read(page_bytes, "handwritten")
    assert r.engine == "trocr" and len(r.lines) == 5
    assert "\n\n" in r.text                                 # paragraph gap preserved
    assert any("Check lines 2, 4" in w for w in r.warnings)
    assert all("box" in ln for ln in r.lines)


def test_auto_mode_switches_to_trocr_when_tesseract_unsure(page_bytes):
    svc = OCRService()
    fake_trocr(svc, [("handwritten words", 0.8)])
    svc._avail["tesseract"] = True
    svc.tesseract.read_page = lambda img: ([(1, "garb1ed t3xt", 0.31)], 0.31)
    r = svc.read(page_bytes, "auto")
    assert r.engine == "trocr" and r.mode == "handwritten"


def test_auto_without_trocr_warns(page_bytes):
    svc = OCRService()
    svc._avail = {"tesseract": True, "trocr": False}
    svc.tesseract.read_page = lambda img: ([(1, "scribbles maybe words", 0.4)], 0.4)
    r = svc.read(page_bytes, "auto")
    assert r.engine == "tesseract" and any("TrOCR isn't installed" in w for w in r.warnings)


# ======================= speech (Whisper mocked) =======================
class FakeWhisper:
    def __init__(self, language="en", segments=None):
        self.language, self.calls = language, []
        self.segments = segments or [("I had a long day at college.", -0.3, 0.01),
                                     ("Thanks for watching!", -1.4, 0.9)]   # classic silence hallucination

    def transcribe(self, path, task, language, **kw):
        self.calls.append(task)
        segs = [types.SimpleNamespace(start=i * 3.0, end=i * 3.0 + 3, text=t, avg_logprob=lp, no_speech_prob=ns)
                for i, (t, lp, ns) in enumerate(self.segments)]
        info = types.SimpleNamespace(language=self.language, language_probability=0.97, duration=6.0)
        return iter(segs), info


def speech_with(fake):
    svc = SpeechService()
    svc._model = fake
    return svc


def test_transcribe_drops_hallucinated_silence():
    t = speech_with(FakeWhisper()).transcribe_bytes(b"x" * 5000)
    assert t.text == "I had a long day at college." and not t.translated


def test_non_english_is_translated():
    fake = FakeWhisper(language="hi", segments=[("I am very tired today.", -0.4, 0.02)])
    t = speech_with(fake).transcribe_bytes(b"x" * 5000)
    assert fake.calls == ["transcribe", "translate"] and t.translated and t.language == "hi"
    assert any("Hindi" in w for w in t.warnings)


def test_speech_errors():
    with pytest.raises(SpeechError):
        speech_with(FakeWhisper()).transcribe_bytes(b"tiny")
    with pytest.raises(SpeechError, match="No speech"):
        speech_with(FakeWhisper(segments=[("Thank you.", -1.5, 0.95)])).transcribe_bytes(b"x" * 5000)


# ======================= storage =======================
def test_media_store_roundtrip_and_traversal(tmp_path):
    store = MediaStore(str(tmp_path / "m"), Cipher(str(tmp_path / "k")))
    token = store.save_pending(b"secret audio", "audio/webm")
    assert b"secret" not in (tmp_path / "m" / "pending" / f"{token}.enc").read_bytes()
    rel, mime = store.claim(token)
    assert store.read(rel) == b"secret audio" and mime == "audio/webm"
    assert store.claim(token) is None                       # can't claim twice
    assert store.claim("../../etc/passwd") is None
    with pytest.raises(FileNotFoundError):
        store.read("../k")


def test_phase1_database_is_migrated(tmp_path):
    db_path, key = str(tmp_path / "old.sqlite3"), str(tmp_path / "k")
    cipher = Cipher(key)
    conn = sqlite3.connect(db_path)
    conn.executescript("""CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
        input_type TEXT NOT NULL DEFAULT 'text', title TEXT, body_encrypted BLOB NOT NULL, raw_encrypted BLOB,
        word_count INTEGER NOT NULL DEFAULT 0, emotion TEXT NOT NULL, emotion_conf REAL NOT NULL,
        emotion_second TEXT, emotion_scores TEXT NOT NULL, model_backend TEXT NOT NULL);""")
    conn.execute("INSERT INTO entries (created_at, body_encrypted, emotion, emotion_conf, emotion_scores, model_backend)"
                 " VALUES ('2026-10-01T10:00:00', ?, 'joy', 0.9, '{}', 'roberta')", (cipher.encrypt("old entry"),))
    conn.commit()
    conn.close()
    db = JournalDB(db_path, cipher)
    entry = db.get_entry(1)
    assert entry["body"] == "old entry" and entry["source_meta"] is None and entry["media_path"] is None


# ======================= API end to end =======================
@needs_tesseract
def test_photo_flow_end_to_end(client, app, page_bytes):
    r = client.post("/api/ocr", data={"image": (io.BytesIO(page_bytes), "page.png"), "mode": "auto"},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["media_token"] and out["text"].startswith("Today the DBMS exam")

    corrected = out["text"].replace("Rohan", "Rohit")       # user fixes a word
    resp = client.post("/entries", data={"body": corrected, "input_type": "image", "source_text": out["text"],
                                         "source_meta": json.dumps(out["meta"]), "media_token": out["media_token"]})
    assert resp.status_code == 302
    entry = client.get("/api/entries/1").get_json()
    assert entry["input_type"] == "image" and entry["source_meta"]["engine"] == "tesseract"
    assert 0 < entry["source_meta"]["user_correction_rate"] < 0.05
    assert client.get("/entries/1/media").data == page_bytes
    page = client.get("/entries/1").data
    assert b"Read from a photo by Tesseract" in page and b"You corrected" in page

    media_dir = app.config["MEDIA_DIR"]
    client.post("/entries/1/delete")
    import os
    assert not [f for f in os.listdir(media_dir) if f.endswith(".enc")]


def test_voice_flow_end_to_end(client, app):
    app.extensions["mindai"]["speech"]._model = FakeWhisper(
        segments=[("Um, I was so nervous before the viva, uh, but it went well.", -0.2, 0.01)])
    r = client.post("/api/transcribe", data={"audio": (io.BytesIO(b"x" * 4000), "rec.webm", "audio/webm")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["text"] == "I was so nervous before the viva, but it went well."
    resp = client.post("/api/entries", json={"body": out["text"], "input_type": "voice", "source_text": out["text"],
                                             "source_meta": out["meta"], "media_token": out["media_token"]})
    assert resp.status_code == 201
    entry = client.get("/api/entries/1").get_json()
    assert entry["source_meta"]["user_correction_rate"] == 0 and entry["media_mime"] == "audio/webm"
    assert b"Spoken for 0:06" in client.get("/entries/1").data


def test_upload_validation(client):
    r = client.post("/api/ocr", data={"image": (io.BytesIO(b"hi"), "notes.txt", "text/plain")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "Unsupported" in r.get_json()["error"]
    assert client.post("/api/transcribe", data={}, content_type="multipart/form-data").status_code == 400
    r = client.post("/api/ocr", data={"image": (io.BytesIO(b"x"), "a.png", "image/png"), "mode": "magic"},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_forged_meta_is_filtered(client):
    client.post("/api/entries", json={"body": "hello there friend", "input_type": "voice",
                                      "source_meta": {"engine": "whisper-small", "evil": "<script>", "nested": {"a": 1}}})
    assert client.get("/api/entries/1").get_json()["source_meta"] == {"engine": "whisper-small"}


def test_write_page_shows_modes(client):
    html = client.get("/").data
    assert b"Speak" in html and b"Photo of a page" in html
