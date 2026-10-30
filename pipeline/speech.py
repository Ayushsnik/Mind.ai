"""Speech-to-text with Whisper (faster-whisper implementation).

faster-whisper runs the same OpenAI Whisper weights through CTranslate2, which is
~4x faster on CPU and needs no separate ffmpeg install (audio is decoded with
PyAV), so browser recordings (webm/opus), mp3, m4a and wav all work on Windows.

Multilingual: Whisper detects the spoken language first. Because the emotion
model is English-only, non-English speech (Hindi, Marathi, ...) is translated to
English by Whisper itself (task="translate"); the detected language is stored
with the entry.
"""
from __future__ import annotations

import io
import logging
import os
import wave
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)


class SpeechError(ValueError):
    """Problem with the audio (shown to the user)."""


class SpeechUnavailable(RuntimeError):
    """faster-whisper isn't installed."""


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float
    translated: bool
    duration: float
    model: str
    segments: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Whisper's own per-segment quality signals (thresholds from the Whisper paper/code).
LOW_LOGPROB = -1.0
NO_SPEECH = 0.6


class SpeechService:
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8",
                 language: str | None = None, translate_to_english: bool = True, max_seconds: int = 600):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.translate = translate_to_english
        self.max_seconds = max_seconds
        self._model = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            self.error = "faster-whisper is not installed"
            return False

    def status(self) -> dict:
        ok = self.available()
        return {"available": ok, "model": self.model_size, "loaded": self._model is not None,
                "translate_to_english": self.translate, "error": None if ok else self.error}

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self.available():
                        raise SpeechUnavailable("Voice input needs faster-whisper. Run: pip install faster-whisper")
                    from faster_whisper import WhisperModel
                    try:
                        self._model = WhisperModel(self.model_size, device=self.device,
                                                   compute_type=self.compute_type)
                    except Exception as exc:     # first run: model download failed (network, blocked site...)
                        log.exception("Could not load the Whisper model")
                        raise SpeechUnavailable(
                            f"The Whisper '{self.model_size}' model isn't downloaded yet and the download failed "
                            f"({type(exc).__name__}). Check your internet connection and try again; see the "
                            "README section 'Model downloads fail' for other options.") from exc
        return self._model

    def transcribe_bytes(self, data: bytes, suffix: str = ".webm") -> Transcript:
        if len(data) < 1000:
            raise SpeechError("The recording is empty. Hold the button a little longer and speak clearly.")
        # The browser converts recordings to 16 kHz WAV before upload. Reading WAV needs only the
        # standard library, so audio works even where PyAV's compiled decoders are blocked.
        samples = wav_to_samples(data)
        if samples is not None:
            return self.transcribe_file(samples)
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return self.transcribe_file(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def transcribe_file(self, path) -> Transcript:
        """`path` is a file path, or a float32 numpy array of 16 kHz mono samples."""
        t0 = time.perf_counter()
        model = self.model
        common = dict(beam_size=5, vad_filter=True, vad_parameters={"min_silence_duration_ms": 700},
                      condition_on_previous_text=False)
        try:
            segments, info = model.transcribe(path, task="transcribe", language=self.language, **common)
        except Exception as exc:  # undecodable file, blocked decoder DLL, ...
            log.exception("Whisper could not process the audio")
            raise SpeechError("That audio couldn't be transcribed. Try recording again or upload a WAV/MP3. "
                              f"(Details: {type(exc).__name__}: {str(exc)[:160]})") from exc

        if info.duration > self.max_seconds:
            raise SpeechError(f"Recordings are limited to {self.max_seconds // 60} minutes.")

        translated = False
        # Language is detected before any decoding happens, so switching task here costs nothing extra.
        if self.translate and info.language != "en":
            segments, info = model.transcribe(path, task="translate", language=info.language, **common)
            translated = True

        segs, warnings = [], []
        for s in segments:
            segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                         "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3)})
        # Whisper hallucinates text on silence; drop segments it itself thinks are silence.
        spoken = [s for s in segs if not (s["no_speech_prob"] > NO_SPEECH and s["avg_logprob"] < LOW_LOGPROB)]
        text = " ".join(s["text"] for s in spoken).strip()
        if not text:
            raise SpeechError("No speech was detected in the recording. Check your microphone and try again.")

        unsure = [s for s in spoken if s["avg_logprob"] < LOW_LOGPROB]
        if unsure:
            times = ", ".join(f"{int(s['start'] // 60)}:{int(s['start'] % 60):02d}" for s in unsure[:5])
            warnings.append(f"Some parts were hard to hear (around {times}). Check those sentences.")
        if translated:
            warnings.append(f"Spoken in {LANGUAGE_NAMES.get(info.language, info.language)}; "
                            "translated to English for emotion detection.")

        return Transcript(text=text, language=info.language,
                          language_probability=round(info.language_probability, 3), translated=translated,
                          duration=round(info.duration, 1), model=f"whisper-{self.model_size}",
                          segments=spoken, warnings=warnings,
                          elapsed_ms=round((time.perf_counter() - t0) * 1000))


LANGUAGE_NAMES = {"hi": "Hindi", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
                  "bn": "Bengali", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
                  "en": "English"}


def wav_to_samples(data: bytes):
    """16-bit PCM WAV bytes -> float32 mono samples at 16 kHz (numpy), or None if not a WAV file."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    import numpy as np
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise SpeechError("That WAV file couldn't be read. Try recording again.") from exc
    if width != 2:
        return None                      # unusual WAV: let the full decoder handle it
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and len(audio):
        n = int(round(len(audio) * 16000 / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio