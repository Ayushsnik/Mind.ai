"""Mind.ai: Phase 1 (text + emotion + encrypted storage) and Phase 2 (voice + OCR)."""
import json
import mimetypes
from datetime import date, datetime
from io import BytesIO

from flask import (Flask, abort, flash, jsonify, redirect, render_template, request, send_file,
                   url_for)
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from db.crypto import Cipher
from db.media import MediaStore
from db.models import JournalDB
from pipeline.emotion import EMOTIONS, EmotionService
from pipeline.event_taxonomy import CATEGORY_LABELS
from pipeline.insights import InsightsService
from pipeline.anonymize import Anonymizer
from pipeline.llm import GeminiBackend, LLMError, OllamaBackend
from pipeline.responder import Responder
from pipeline.safety import check as crisis_check
from pipeline.metrics import cer
from pipeline.ocr import MODES as OCR_MODES
from pipeline.ocr import OCRError, OCRService, OCRUnavailable
from pipeline.preprocess import clean_text
from pipeline.speech import SpeechError, SpeechService, SpeechUnavailable

INPUT_TYPES = {"text", "voice", "image"}
AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/x-m4a", "audio/wav",
               "audio/x-wav", "audio/wave", "video/webm"}   # Chrome sometimes labels mic recordings video/webm
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    cipher = Cipher(app.config["ENCRYPTION_KEY_PATH"])
    db = JournalDB(app.config["DATABASE_PATH"], cipher)
    media = MediaStore(app.config["MEDIA_DIR"], cipher)
    media.cleanup_pending()
    emotions = EmotionService(backend=app.config["EMOTION_BACKEND"], model_name=app.config["EMOTION_MODEL"],
                              chunk_words=app.config["EMOTION_CHUNK_WORDS"])
    speech = SpeechService(model_size=app.config["WHISPER_MODEL"], device=app.config["WHISPER_DEVICE"],
                           compute_type=app.config["WHISPER_COMPUTE"], language=app.config["WHISPER_LANGUAGE"],
                           translate_to_english=app.config["WHISPER_TRANSLATE"],
                           max_seconds=app.config["MAX_AUDIO_SECONDS"])
    ocr = OCRService(tesseract_cmd=app.config["TESSERACT_CMD"], tesseract_lang=app.config["TESSERACT_LANG"],
                     trocr_model=app.config["TROCR_MODEL"], printed_threshold=app.config["OCR_PRINTED_THRESHOLD"],
                     trocr_beams=app.config["TROCR_BEAMS"])
    insights = InsightsService(db, emotions, spacy_model=app.config["SPACY_MODEL"], topics=app.config["EXTRACT_TOPICS"],
                               min_entries=app.config["PATTERN_MIN_ENTRIES"],
                               min_occurrences=app.config["PATTERN_MIN_OCCURRENCES"],
                               min_rate=app.config["PATTERN_MIN_RATE"], min_lift=app.config["PATTERN_MIN_LIFT"],
                               combo_conf=app.config["COMBO_MIN_CONFIDENCE"],
                               lookback_days=app.config["ALERT_LOOKBACK_DAYS"])
    local_llm = OllamaBackend(app.config["OLLAMA_HOST"], app.config["OLLAMA_MODEL"], app.config["OLLAMA_TIMEOUT"])
    gemini = GeminiBackend(app.config["GEMINI_API_KEY"], app.config["GEMINI_MODEL"], app.config["GEMINI_BASE_URL"],
                           app.config["GEMINI_THINKING_LEVEL"])
    anonymizer = Anonymizer(app.config["SPACY_MODEL"])
    responder = Responder(local_llm, gemini, anonymizer, app.config["HELPLINES"])
    app.extensions["mindai"] = {"db": db, "emotions": emotions, "speech": speech, "ocr": ocr, "media": media,
                                "insights": insights, "responder": responder}

    # ---------- Phase 4: settings ----------
    def settings() -> dict:
        names = db.get_setting("hidden_names", "", encrypted=True) or ""
        return {
            "ai_mode": db.get_setting("ai_mode", "local"),                  # off | local | enhanced
            "gemini_consent": db.get_setting("gemini_consent", "0") == "1",
            "auto_reflect": db.get_setting("auto_reflect", "1") == "1",
            "hidden_names": [n.strip() for n in names.split(",") if n.strip()],
        }

    def ai_status() -> dict:
        return {"local": local_llm.status(), "gemini": gemini.status()}

    def reflect(entry_id: int, prefer: str | None = None) -> dict:
        entry = db.get_entry(entry_id)
        reply = responder.respond(entry, db.events_for_entry(entry_id), insights.insights(), settings(), prefer)
        db.add_response(entry_id, reply)
        return db.latest_response(entry_id)

    # ---------- core logic (shared by the HTML form and the JSON API) ----------
    def create_entry(raw_text: str, title: str | None, input_type: str = "text", *,
                     source_text: str | None = None, source_meta: dict | None = None,
                     media_token: str | None = None, created_at: str | None = None,
                     refresh_alerts: bool = True) -> dict:
        if input_type not in INPUT_TYPES:
            raise ValueError(f"input_type must be one of {sorted(INPUT_TYPES)}")
        raw_text = (raw_text or "").strip()
        if not raw_text:
            raise ValueError("Write something before saving.")
        if len(raw_text) > app.config["MAX_ENTRY_CHARS"]:
            raise ValueError(f"Entries are limited to {app.config['MAX_ENTRY_CHARS']:,} characters.")
        # Voice transcripts get filler removal. OCR text was already cleaned before the
        # user reviewed it, so their edits are kept as-is.
        body = clean_text(raw_text, "voice" if input_type == "voice" else "text")

        meta = dict(source_meta or {})
        if source_text and input_type in ("voice", "image"):
            source_text = source_text[: app.config["MAX_ENTRY_CHARS"]]
            # How much the user had to fix the machine output: a real-world accuracy signal.
            meta["user_correction_rate"] = round(cer(source_text, body), 4)
        else:
            source_text = None

        stored = media.claim(media_token) if (media_token and app.config["KEEP_MEDIA"]) else None
        result = emotions.predict(body)
        entry_id = db.add_entry(body=body, raw=raw_text, title=(title or "").strip()[:120], input_type=input_type,
                                emotion=result, source_text=source_text, source_meta=meta or None,
                                media_path=stored[0] if stored else None, media_mime=stored[1] if stored else None,
                                created_at=created_at)
        # Phase 3: what happened, and does any of it call for an early warning?
        entry = db.get_entry(entry_id)
        events = insights.analyze_entry(entry_id, body, entry["created_at"])
        # Phase 4: a crisis message is stored immediately, whatever the AI settings.
        if crisis_check(body).flagged:
            reflect(entry_id)
        new_alerts = []
        if refresh_alerts:
            insights.refresh_alerts()
            new_alerts = db.alerts_for_entry(entry_id)
        return {"id": entry_id, "emotion": result.to_dict(), "events": insights.describe_events(events),
                "alerts": new_alerts}

    app.extensions["mindai"]["create_entry"] = create_entry   # used by evaluation/synthetic_patterns.py

    def parse_meta(raw) -> dict | None:
        if not raw:
            return None
        if isinstance(raw, dict):
            meta = raw
        else:
            try:
                meta = json.loads(raw)
            except (TypeError, ValueError):
                return None
        if not isinstance(meta, dict):
            return None
        allowed = {"engine", "mode", "confidence", "language", "language_probability", "translated",
                   "duration", "model", "skew", "lines"}
        return {k: v for k, v in meta.items() if k in allowed and isinstance(v, (str, int, float, bool, type(None)))}

    def uploaded(field: str, allowed: set[str]):
        file = request.files.get(field)
        if not file or not file.filename and not file.mimetype:
            return None, None, f"Choose a file to upload ({field})."
        data = file.read()
        mime = (file.mimetype or mimetypes.guess_type(file.filename or "")[0] or "").split(";")[0].lower()
        if mime not in allowed:
            guessed = (mimetypes.guess_type(file.filename or "")[0] or "").lower()
            mime = guessed if guessed in allowed else mime
        if mime not in allowed:
            return None, None, f"Unsupported file type ({mime or 'unknown'})."
        return data, mime, None

    # ---------- template helpers ----------
    @app.template_filter("nicedate")
    def nicedate(iso: str) -> str:
        return datetime.fromisoformat(iso).strftime("%a, %d %b %Y, %I:%M %p").replace(" 0", " ")

    @app.template_filter("shortdate")
    def shortdate(iso: str) -> str:
        return datetime.fromisoformat(iso).strftime("%d %b")

    @app.template_filter("pct")
    def pct(x) -> str:
        return "" if x is None else f"{round(float(x) * 100)}%"

    @app.template_filter("andjoin")
    def andjoin(items) -> str:
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    @app.template_filter("whendate")
    def whendate(iso: str | None) -> str:
        from pipeline.alerts import when_phrase
        return when_phrase(date.fromisoformat(iso) if iso else None, date.today())

    @app.context_processor
    def inject_globals():
        return {"EMOTIONS": EMOTIONS, "recent_strip": db.list_entries(limit=14), "CATEGORY_LABELS": CATEGORY_LABELS}

    def safe_next(default: str) -> str:
        nxt = request.values.get("next", "")
        return nxt if nxt.startswith("/") and not nxt.startswith("//") else default

    # ---------- pages ----------
    @app.get("/")
    def write():
        return render_template("write.html", caps=capabilities(), alerts=insights.refresh_alerts())

    @app.post("/entries")
    def save_entry():
        f = request.form
        try:
            out = create_entry(f.get("body", ""), f.get("title"), f.get("input_type", "text"),
                               source_text=f.get("source_text"), source_meta=parse_meta(f.get("source_meta")),
                               media_token=f.get("media_token"))
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("write.html", draft=f, caps=capabilities()), 400
        flash("Entry saved.", "ok")
        return redirect(url_for("view_entry", entry_id=out["id"]))

    @app.get("/entries")
    def list_entries():
        emotion = request.args.get("emotion")
        if emotion and emotion not in EMOTIONS:
            emotion = None
        event = request.args.get("event") or None
        from pipeline.patterns import event_label
        return render_template("entries.html", entries=db.list_entries(limit=200, emotion=emotion, event=event),
                               active=emotion, counts=db.emotion_counts(), event=event,
                               event_name=event_label(event) if event else None)

    @app.get("/entries/<int:entry_id>")
    def view_entry(entry_id):
        entry = db.get_entry(entry_id) or abort(404)
        ranked = sorted(entry["emotion_scores"].items(), key=lambda kv: kv[1], reverse=True)
        events = insights.describe_events(db.events_for_entry(entry_id))
        return render_template("entry.html", entry=entry, ranked=ranked, events=events,
                               alerts=db.alerts_for_entry(entry_id), response=db.latest_response(entry_id),
                               settings=settings(), ai=ai_status())

    # ---------- Phase 4: AI replies, plans, settings ----------
    @app.get("/entries/<int:entry_id>/reply")
    def reply_fragment(entry_id):
        if not db.get_entry(entry_id):
            abort(404)
        return render_template("_response.html", response=db.latest_response(entry_id), entry_id=entry_id,
                               settings=settings(), ai=ai_status())

    @app.post("/api/entries/<int:entry_id>/reflect")
    def api_reflect(entry_id):
        if not db.get_entry(entry_id):
            return jsonify(error="Entry not found."), 404
        prefer = (request.get_json(silent=True) or {}).get("prefer")
        if prefer not in (None, "local", "gemini", "template"):
            return jsonify(error="prefer must be local, gemini or template"), 400
        try:
            return jsonify(reflect(entry_id, prefer))
        except PermissionError as exc:
            return jsonify(error=str(exc)), 403
        except LLMError as exc:
            return jsonify(error=f"No reply could be written: {exc}"), 503

    @app.get("/api/entries/<int:entry_id>/response")
    def api_response(entry_id):
        r = db.latest_response(entry_id)
        return (jsonify(r), 200) if r else (jsonify(error="No reply yet."), 404)

    @app.post("/api/responses/<int:response_id>/rate")
    def api_rate(response_id):
        rating = (request.get_json(silent=True) or {}).get("rating")
        if rating not in (1, -1, 0):
            return jsonify(error="rating must be 1, -1 or 0"), 400
        ok = db.rate_response(response_id, rating or None)
        return (jsonify(ok=True), 200) if ok else (jsonify(error="Reply not found."), 404)

    @app.post("/api/alerts/<int:alert_id>/plan")
    def api_alert_plan(alert_id):
        alert = db.get_alert(alert_id)
        if not alert:
            return jsonify(error="Heads-up not found."), 404
        prefer = (request.get_json(silent=True) or {}).get("prefer")
        try:
            plan = responder.plan(alert, insights.insights(), settings(), prefer)
        except PermissionError as exc:
            return jsonify(error=str(exc)), 403
        except LLMError as exc:
            return jsonify(error=f"No plan could be written: {exc}"), 503
        db.set_alert_plan(alert_id, plan)
        return jsonify(plan)

    @app.get("/api/entries/<int:entry_id>/gemini-preview")
    def api_gemini_preview(entry_id):
        """Exactly what Gemini would receive for this entry (for transparency before opting in)."""
        entry = db.get_entry(entry_id)
        if not entry:
            return jsonify(error="Entry not found."), 404
        from pipeline.responder import build_context
        events = db.events_for_entry(entry_id)
        text, mapping = responder.gemini_payload(entry, events, build_context(entry, events, insights.insights()),
                                                 settings()["hidden_names"])
        return jsonify(sent=text, hidden=len(mapping))

    @app.route("/settings", methods=["GET", "POST"])
    def settings_page():
        if request.method == "POST":
            f = request.form
            mode = f.get("ai_mode", "local")
            if mode not in ("off", "local", "enhanced"):
                mode = "local"
            consent = f.get("gemini_consent") == "1"
            if mode == "enhanced" and not consent:
                flash("To use Gemini, tick the box confirming what will be shared.", "error")
                mode = "local"
            db.set_setting("ai_mode", mode)
            db.set_setting("gemini_consent", "1" if consent and mode == "enhanced" else "0")
            db.set_setting("auto_reflect", "1" if f.get("auto_reflect") == "1" else "0")
            names = ",".join(n.strip() for n in f.get("hidden_names", "").replace("\n", ",").split(",") if n.strip())
            db.set_setting("hidden_names", names[:2000], encrypted=True)
            flash("Settings saved.", "ok")
            return redirect(url_for("settings_page"))
        preview = None
        latest = db.list_entries(limit=1)
        if latest:
            from pipeline.responder import build_context
            e = latest[0]
            ev = db.events_for_entry(e["id"])
            preview = responder.gemini_payload(e, ev, build_context(e, ev, insights.insights()),
                                               settings()["hidden_names"])[0]
        return render_template("settings.html", s=settings(), ai=ai_status(), preview=preview,
                               stats=db.response_stats(), helplines=app.config["HELPLINES"])

    # ---------- Phase 3: insights and alerts ----------
    @app.get("/insights")
    def insights_page():
        data = insights.insights()
        return render_template("insights.html", data=data, alerts=insights.refresh_alerts(),
                               status=insights.status(), extractor=insights.extractor.name)

    @app.post("/insights/reanalyse")
    def reanalyse():
        n = insights.backfill(force=request.form.get("all") == "1")
        flash(f"Analysed {n} entr{'y' if n == 1 else 'ies'}.", "ok")
        return redirect(url_for("insights_page"))

    @app.post("/alerts/<int:alert_id>/dismiss")
    def dismiss_alert(alert_id):
        db.dismiss_alert(alert_id)
        return redirect(safe_next(url_for("write")))

    @app.get("/api/insights")
    def api_insights():
        return jsonify(insights.insights().to_dict() | {"status": insights.status()})

    @app.get("/api/alerts")
    def api_alerts():
        return jsonify(insights.refresh_alerts())

    @app.post("/api/alerts/<int:alert_id>/dismiss")
    def api_dismiss_alert(alert_id):
        return (jsonify(ok=True), 200) if db.dismiss_alert(alert_id) else (jsonify(error="Alert not found."), 404)

    @app.get("/api/entries/<int:entry_id>/events")
    def api_entry_events(entry_id):
        if not db.get_entry(entry_id):
            return jsonify(error="Entry not found."), 404
        return jsonify(insights.describe_events(db.events_for_entry(entry_id)))

    @app.post("/api/reanalyse")
    def api_reanalyse():
        return jsonify(analysed=insights.backfill(force=bool((request.get_json(silent=True) or {}).get("all"))))

    @app.get("/entries/<int:entry_id>/media")
    def entry_media(entry_id):
        entry = db.get_entry(entry_id)
        if not entry or not entry["media_path"]:
            abort(404)
        try:
            data = media.read(entry["media_path"])
        except Exception:
            abort(404)
        resp = send_file(BytesIO(data), mimetype=entry["media_mime"] or "application/octet-stream")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.post("/entries/<int:entry_id>/delete")
    def delete_entry(entry_id):
        result = db.delete_entry(entry_id)
        if result:
            if isinstance(result, str):
                media.delete(result)
            flash("Entry deleted.", "ok")
        return redirect(url_for("list_entries"))

    # ---------- Phase 2: voice and photo ----------
    def capabilities() -> dict:
        return {"ocr": ocr.status(), "speech": speech.status(), "keep_media": app.config["KEEP_MEDIA"],
                "ai": ai_status()}

    @app.get("/api/capabilities")
    def api_capabilities():
        return jsonify(capabilities())

    @app.post("/api/transcribe")
    def api_transcribe():
        data, mime, err = uploaded("audio", AUDIO_TYPES)
        if err:
            return jsonify(error=err), 400
        suffix = {"audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/ogg": ".ogg",
                  "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav"}.get(mime, ".webm")
        try:
            t = speech.transcribe_bytes(data, suffix)
        except SpeechUnavailable as exc:
            return jsonify(error=str(exc)), 503
        except SpeechError as exc:
            return jsonify(error=str(exc)), 422
        token = media.save_pending(data, "audio/webm" if mime == "video/webm" else mime) \
            if app.config["KEEP_MEDIA"] else None
        out = t.to_dict()
        out.update(text=clean_text(t.text, "voice"), media_token=token,
                   meta={"engine": t.model, "language": t.language, "language_probability": t.language_probability,
                         "translated": t.translated, "duration": t.duration})
        return jsonify(out)

    @app.post("/api/ocr")
    def api_ocr():
        data, mime, err = uploaded("image", IMAGE_TYPES)
        if err:
            return jsonify(error=err), 400
        mode = request.form.get("mode", "auto")
        if mode not in OCR_MODES:
            return jsonify(error=f"mode must be one of {', '.join(OCR_MODES)}"), 400
        try:
            r = ocr.read(data, mode)
        except OCRUnavailable as exc:
            return jsonify(error=str(exc)), 503
        except OCRError as exc:
            return jsonify(error=str(exc)), 422
        token = media.save_pending(data, mime) if app.config["KEEP_MEDIA"] else None
        out = r.to_dict()
        out.update(raw_text=r.text, text=clean_text(r.text, "image"), media_token=token,
                   meta={"engine": r.engine, "mode": r.mode, "confidence": r.confidence, "skew": r.skew,
                         "lines": len(r.lines)})
        return jsonify(out)

    # ---------- JSON API ----------
    @app.post("/api/entries")
    def api_create():
        d = request.get_json(silent=True) or {}
        try:
            out = create_entry(d.get("body", ""), d.get("title"), d.get("input_type", "text"),
                               source_text=d.get("source_text"), source_meta=parse_meta(d.get("source_meta")),
                               media_token=d.get("media_token"))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(out), 201

    @app.get("/api/entries")
    def api_list():
        limit = min(int(request.args.get("limit", 50)), 500)
        return jsonify(db.list_entries(limit=limit))

    @app.get("/api/entries/<int:entry_id>")
    def api_get(entry_id):
        entry = db.get_entry(entry_id)
        return (jsonify(entry), 200) if entry else (jsonify(error="Entry not found."), 404)

    @app.post("/api/analyze")
    def api_analyze():
        """Emotion preview without saving (live feedback while typing)."""
        text = clean_text((request.get_json(silent=True) or {}).get("text", ""))
        return jsonify(emotions.predict(text).to_dict())

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", entries=db.count(), emotion_model=emotions.status(), insights=insights.status(),
                       **capabilities())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify(error=f"That file is too large. The limit is {mb} MB."), 413

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return jsonify(error="Not found."), 404
        return render_template("404.html"), 404

    return app


if __name__ == "__main__":
    application = create_app()
    ext = application.extensions["mindai"]
    ext["emotions"].model   # warm the emotion model so the first save isn't slow
    print("Emotion model :", ext["emotions"].status()["active"])
    print("OCR engines   :", ext["ocr"].availability())
    print("Voice (Whisper):", ext["speech"].status()["available"])
    print("Event extractor:", ext["insights"].extractor.name)
    llm = ext["responder"]
    print("Local AI       :", "ready" if llm.local.available() else llm.local.status()["error"])
    print("Gemini         :", "key set (used only if you opt in)" if llm.gemini.available() else "not configured")
    pending = ext["insights"].status()["pending"]
    if pending:   # entries saved before Phase 3 (or before an extractor update)
        print(f"Analysing {pending} earlier entries for events…")
        ext["insights"].backfill()
    application.run(debug=True, use_reloader=False, threaded=True)
