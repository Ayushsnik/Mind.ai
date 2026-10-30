"""Central configuration. Override any value with an environment variable or a .env file."""
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-change-me")
    DATABASE_PATH = os.getenv("MINDAI_DB", str(INSTANCE_DIR / "mindai.sqlite3"))
    # Fernet key used to encrypt journal text at rest. Auto-generated on first run if missing.
    ENCRYPTION_KEY_PATH = os.getenv("MINDAI_KEY_PATH", str(INSTANCE_DIR / "secret.key"))

    # "auto"    -> try RoBERTa, fall back to the lexicon model if it can't load
    # "roberta" -> RoBERTa only (fail loudly if unavailable)
    # "lexicon" -> lightweight keyword model (tests / very low-RAM machines)
    EMOTION_BACKEND = os.getenv("MINDAI_EMOTION_BACKEND", "auto")
    EMOTION_MODEL = os.getenv(
        "MINDAI_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
    )
    # Long entries are split into chunks of roughly this many words (RoBERTa max is 512 tokens).
    EMOTION_CHUNK_WORDS = int(os.getenv("MINDAI_CHUNK_WORDS", "250"))

    MAX_ENTRY_CHARS = 20_000
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024        # largest upload (photo or recording)

    # ---- Phase 2: voice ----
    # tiny / base / small / medium. "small" is the best accuracy-to-speed trade-off on a laptop CPU;
    # use "base" if transcription feels slow.
    WHISPER_MODEL = os.getenv("MINDAI_WHISPER_MODEL", "small")
    WHISPER_DEVICE = os.getenv("MINDAI_WHISPER_DEVICE", "cpu")        # "cuda" with an NVIDIA GPU
    WHISPER_COMPUTE = os.getenv("MINDAI_WHISPER_COMPUTE", "int8")     # "float16" on GPU
    WHISPER_LANGUAGE = os.getenv("MINDAI_WHISPER_LANGUAGE") or None   # None = auto-detect
    WHISPER_TRANSLATE = os.getenv("MINDAI_WHISPER_TRANSLATE", "1") == "1"
    MAX_AUDIO_SECONDS = int(os.getenv("MINDAI_MAX_AUDIO_SECONDS", "600"))

    # ---- Phase 2: OCR ----
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")   # auto-detected on PATH or in Program Files
    TESSERACT_LANG = os.getenv("MINDAI_TESSERACT_LANG", "eng")
    # trocr-base-handwritten (~1.3 GB, best) or trocr-small-handwritten (~250 MB, faster, weaker)
    TROCR_MODEL = os.getenv("MINDAI_TROCR_MODEL", "microsoft/trocr-base-handwritten")
    TROCR_BEAMS = int(os.getenv("MINDAI_TROCR_BEAMS", "1"))           # 3-4 is a bit more accurate, slower
    # In auto mode, Tesseract confidence at or above this means "printed text"; below means handwriting.
    OCR_PRINTED_THRESHOLD = float(os.getenv("MINDAI_OCR_PRINTED_THRESHOLD", "0.70"))

    # Keep the original recording/photo (encrypted) with the entry.
    KEEP_MEDIA = os.getenv("MINDAI_KEEP_MEDIA", "1") == "1"
    MEDIA_DIR = os.getenv("MINDAI_MEDIA_DIR", str(INSTANCE_DIR / "media"))

    # ---- Phase 3: events, patterns, alerts ----
    SPACY_MODEL = os.getenv("MINDAI_SPACY_MODEL", "en_core_web_sm")
    EXTRACT_TOPICS = os.getenv("MINDAI_EXTRACT_TOPICS", "1") == "1"   # discover recurring personal nouns
    PATTERN_MIN_ENTRIES = int(os.getenv("MINDAI_PATTERN_MIN_ENTRIES", "8"))       # before any pattern is shown
    PATTERN_MIN_OCCURRENCES = int(os.getenv("MINDAI_PATTERN_MIN_OCCURRENCES", "3"))  # entries per event
    PATTERN_MIN_RATE = float(os.getenv("MINDAI_PATTERN_MIN_RATE", "0.6"))   # share of entries going one way
    PATTERN_MIN_LIFT = float(os.getenv("MINDAI_PATTERN_MIN_LIFT", "1.15"))  # vs. your usual mood
    COMBO_MIN_CONFIDENCE = float(os.getenv("MINDAI_COMBO_MIN_CONFIDENCE", "0.7"))
    ALERT_LOOKBACK_DAYS = int(os.getenv("MINDAI_ALERT_LOOKBACK_DAYS", "14"))

    # ---- Phase 4: AI responses ----
    # Local model through Ollama (https://ollama.com). Nothing leaves the device.
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("MINDAI_OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT = float(os.getenv("MINDAI_OLLAMA_TIMEOUT", "180"))     # CPU replies can take a while
    # Optional enhanced mode. Only used after the user opts in on the Settings page,
    # and only ever receives an anonymised summary.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or None
    GEMINI_MODEL = os.getenv("MINDAI_GEMINI_MODEL", "gemini-3.8-flash")  # or gemini-3.5-flash-lite
    GEMINI_BASE_URL = os.getenv("MINDAI_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    GEMINI_THINKING_LEVEL = os.getenv("MINDAI_GEMINI_THINKING_LEVEL", "low")  # empty = model default
    # Shown when an entry suggests a crisis. Override with JSON in MINDAI_HELPLINES for other countries.
    HELPLINES = json.loads(os.getenv("MINDAI_HELPLINES") or "null") or [
        {"name": "Tele-MANAS", "number": "14416", "note": "free, 24x7, many Indian languages"},
        {"name": "Emergency", "number": "112", "note": "if you are in immediate danger"},
    ]


class TestConfig(Config):
    TESTING = True
    EMOTION_BACKEND = "lexicon"
