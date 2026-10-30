"""SQLite storage. Plain sqlite3 keeps the dependency list short and the schema readable for the report."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,              -- ISO 8601, local time
    input_type      TEXT    NOT NULL DEFAULT 'text' CHECK (input_type IN ('text','voice','image')),
    title           TEXT,
    body_encrypted  BLOB    NOT NULL,              -- Fernet token of the cleaned text
    raw_encrypted   BLOB,                          -- original text/transcript before cleaning
    word_count      INTEGER NOT NULL DEFAULT 0,
    emotion         TEXT    NOT NULL,
    emotion_conf    REAL    NOT NULL,
    emotion_second  TEXT,
    emotion_scores  TEXT    NOT NULL,              -- JSON {emotion: prob}
    model_backend   TEXT    NOT NULL               -- which model labelled it (matters for evaluation)
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_emotion ON entries(emotion);

-- Phase 3: what happened in each entry. Labels only; the sentence text stays encrypted in entries.
CREATE TABLE IF NOT EXISTS entry_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id      INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    event         TEXT    NOT NULL,              -- taxonomy key or "topic:<noun>"
    category      TEXT    NOT NULL,
    sentence_idx  INTEGER NOT NULL,
    timing        TEXT    NOT NULL CHECK (timing IN ('past','upcoming')),
    target_date   TEXT,                          -- ISO date of an upcoming event, when known
    emotion       TEXT    NOT NULL,              -- emotion of the sentence the event is in
    emotion_conf  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_entry  ON entry_events(entry_id);
CREATE INDEX IF NOT EXISTS idx_events_event  ON entry_events(event);
CREATE INDEX IF NOT EXISTS idx_events_target ON entry_events(timing, target_date);

CREATE TABLE IF NOT EXISTS alerts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    key              TEXT    NOT NULL UNIQUE,    -- same situation + date never alerts twice
    kind             TEXT    NOT NULL,           -- event | day
    event            TEXT,
    target_date      TEXT,
    expires_on       TEXT    NOT NULL,
    message          TEXT    NOT NULL,
    suggestion       TEXT,
    evidence         TEXT,                       -- JSON
    source_entry_id  INTEGER REFERENCES entries(id) ON DELETE CASCADE,
    created_at       TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active','dismissed','expired'))
);

-- Phase 4: Mind.ai's written reply to an entry (text encrypted, like the entry itself).
CREATE TABLE IF NOT EXISTS responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id        INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    created_at      TEXT    NOT NULL,
    kind            TEXT    NOT NULL,            -- reflection | crisis
    backend         TEXT    NOT NULL,            -- local | gemini | template | safety
    model           TEXT    NOT NULL,
    privacy         TEXT    NOT NULL,            -- on-device | anonymised-summary | none
    body_encrypted  BLOB    NOT NULL,            -- JSON {reflection, affirmation, suggestions}
    sent_encrypted  BLOB,                        -- exactly what left the device (Gemini only)
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    note            TEXT,
    rating          INTEGER                      -- +1 helpful / -1 not helpful (user feedback)
);
CREATE INDEX IF NOT EXISTS idx_responses_entry ON responses(entry_id);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  BLOB
);
"""


# Columns added after a table was first created; migrate() adds them to older databases.
MIGRATION_COLUMNS = {
    "source_encrypted": "BLOB",   # machine output (OCR/ASR) before the user corrected it
    "source_meta": "TEXT",        # JSON: engine, confidence, language, correction rate...
    "media_path": "TEXT",         # encrypted original audio/photo in instance/media/
    "media_mime": "TEXT",
    "analysis_version": "INTEGER",  # Phase 3: event-extraction version that processed this entry
}
ALERT_MIGRATION_COLUMNS = {
    "plan": "TEXT",                 # Phase 4: JSON preparation plan ("Help me prepare")
}


class JournalDB:
    def __init__(self, path: str, cipher):
        self.path = path
        self.cipher = cipher
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn):
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(entries)")}
        for col, typ in MIGRATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(alerts)")}
        for col, typ in ALERT_MIGRATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} {typ}")

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def add_entry(self, *, body: str, raw: str, title: str | None, input_type: str, emotion,
                  source_text: str | None = None, source_meta: dict | None = None,
                  media_path: str | None = None, media_mime: str | None = None,
                  created_at: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO entries (created_at, input_type, title, body_encrypted, raw_encrypted,
                       word_count, emotion, emotion_conf, emotion_second, emotion_scores, model_backend,
                       source_encrypted, source_meta, media_path, media_mime)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    created_at or datetime.now().isoformat(timespec="seconds"), input_type, title or None,
                    self.cipher.encrypt(body), self.cipher.encrypt(raw) if raw != body else None,
                    len(body.split()), emotion.primary, emotion.confidence, emotion.secondary,
                    json.dumps(emotion.scores), emotion.backend,
                    self.cipher.encrypt(source_text) if source_text else None,
                    json.dumps(source_meta) if source_meta else None, media_path, media_mime,
                ),
            )
            return cur.lastrowid

    def _row_to_dict(self, row, include_body=True) -> dict:
        d = {
            "id": row["id"], "created_at": row["created_at"], "input_type": row["input_type"],
            "title": row["title"], "word_count": row["word_count"], "emotion": row["emotion"],
            "emotion_conf": row["emotion_conf"], "emotion_second": row["emotion_second"],
            "emotion_scores": json.loads(row["emotion_scores"]), "model_backend": row["model_backend"],
            "source_meta": json.loads(row["source_meta"]) if row["source_meta"] else None,
            "media_path": row["media_path"], "media_mime": row["media_mime"],
        }
        if include_body:
            d["body"] = self.cipher.decrypt(row["body_encrypted"])
            d["source_text"] = self.cipher.decrypt(row["source_encrypted"]) if row["source_encrypted"] else None
        return d

    def get_entry(self, entry_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_entries(self, limit: int = 50, offset: int = 0, emotion: str | None = None,
                     event: str | None = None) -> list[dict]:
        sql, where, args = "SELECT * FROM entries", [], []
        if emotion:
            where.append("emotion = ?")
            args.append(emotion)
        if event:
            where.append("id IN (SELECT entry_id FROM entry_events WHERE event = ?)")
            args.append(event)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_entry(self, entry_id: int) -> str | bool:
        """Delete an entry. Returns its media path (so the caller can delete the file), True, or False."""
        with self._conn() as conn:
            row = conn.execute("SELECT media_path FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return row["media_path"] or True

    def emotion_counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT emotion, COUNT(*) n FROM entries GROUP BY emotion").fetchall()
        return {r["emotion"]: r["n"] for r in rows}

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    # ======================= Phase 3: events =======================
    def replace_events(self, entry_id: int, rows: list[dict], version: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM entry_events WHERE entry_id = ?", (entry_id,))
            conn.executemany(
                """INSERT INTO entry_events (entry_id, event, category, sentence_idx, timing, target_date,
                       emotion, emotion_conf) VALUES (?,?,?,?,?,?,?,?)""",
                [(entry_id, r["event"], r["category"], r["sentence_idx"], r["timing"], r["target_date"],
                  r["emotion"], r["emotion_conf"]) for r in rows])
            conn.execute("UPDATE entries SET analysis_version = ? WHERE id = ?", (version, entry_id))

    def entries_needing_analysis(self, version: int) -> list[int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM entries WHERE analysis_version IS NULL OR analysis_version < ? "
                                "ORDER BY id", (version,)).fetchall()
        return [r["id"] for r in rows]

    def analysis_meta(self) -> list[dict]:
        """Lightweight rows (no decryption) for analysed entries: id, created_at, emotion."""
        with self._conn() as conn:
            rows = conn.execute("SELECT id, created_at, emotion FROM entries WHERE analysis_version IS NOT NULL "
                                "ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def all_events(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ev.*, e.created_at AS entry_created FROM entry_events ev "
                                "JOIN entries e ON e.id = ev.entry_id ORDER BY e.created_at").fetchall()
        return [dict(r) for r in rows]

    def events_for_entry(self, entry_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM entry_events WHERE entry_id = ? ORDER BY sentence_idx, id",
                                (entry_id,)).fetchall()
        return [dict(r) for r in rows]

    def upcoming_events(self, since_written: str) -> list[dict]:
        """Upcoming-event mentions from entries written on/after `since_written` (ISO date)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT ev.*, e.created_at AS entry_created FROM entry_events ev "
                                "JOIN entries e ON e.id = ev.entry_id "
                                "WHERE ev.timing = 'upcoming' AND e.created_at >= ?", (since_written,)).fetchall()
        return [dict(r) for r in rows]

    def data_version(self) -> tuple:
        """Changes whenever entries or events change; used to cache the pattern analysis."""
        with self._conn() as conn:
            a = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(SUM(analysis_version), 0) "
                             "FROM entries").fetchone()
            b = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM entry_events").fetchone()
        return tuple(a) + tuple(b)

    # ======================= Phase 3: alerts =======================
    def upsert_alert(self, a) -> None:
        """Insert a new alert, or refresh the wording of an active one. Dismissed alerts stay dismissed."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO alerts (key, kind, event, target_date, expires_on, message, suggestion, evidence,
                       source_entry_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET message = excluded.message, suggestion = excluded.suggestion,
                       evidence = excluded.evidence, expires_on = excluded.expires_on
                   WHERE alerts.status = 'active'""",
                (a.key, a.kind, a.event, a.target_date, a.expires_on, a.message, a.suggestion,
                 json.dumps(a.evidence), a.source_entry_id, datetime.now().isoformat(timespec="seconds")))

    def expire_alerts(self, today: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE alerts SET status = 'expired' WHERE status = 'active' AND expires_on < ?", (today,))

    @staticmethod
    def _alert(r) -> dict:
        return dict(r) | {"evidence": json.loads(r["evidence"] or "{}"),
                          "plan": json.loads(r["plan"]) if r["plan"] else None}

    def active_alerts(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM alerts WHERE status = 'active' "
                                "ORDER BY COALESCE(target_date, expires_on), id").fetchall()
        return [self._alert(r) for r in rows]

    def alerts_for_entry(self, entry_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM alerts WHERE source_entry_id = ? AND status = 'active'",
                                (entry_id,)).fetchall()
        return [self._alert(r) for r in rows]

    def get_alert(self, alert_id: int) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(r) if r else None

    def set_alert_plan(self, alert_id: int, plan: dict) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE alerts SET plan = ? WHERE id = ?", (json.dumps(plan), alert_id))

    def dismiss_alert(self, alert_id: int) -> bool:
        with self._conn() as conn:
            return conn.execute("UPDATE alerts SET status = 'dismissed' WHERE id = ? AND status = 'active'",
                                (alert_id,)).rowcount > 0

    # ======================= Phase 4: responses =======================
    def add_response(self, entry_id: int, reply) -> int:
        body = {"reflection": reply.reflection, "affirmation": reply.affirmation, "suggestions": reply.suggestions}
        with self._conn() as conn:
            return conn.execute(
                """INSERT INTO responses (entry_id, created_at, kind, backend, model, privacy, body_encrypted,
                       sent_encrypted, latency_ms, note) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (entry_id, datetime.now().isoformat(timespec="seconds"), reply.kind, reply.backend, reply.model,
                 reply.privacy, self.cipher.encrypt(json.dumps(body)),
                 self.cipher.encrypt(reply.sent) if reply.sent else None, reply.latency_ms, reply.note)).lastrowid

    def _response(self, r) -> dict:
        body = json.loads(self.cipher.decrypt(r["body_encrypted"]) or "{}")
        return {"id": r["id"], "entry_id": r["entry_id"], "created_at": r["created_at"], "kind": r["kind"],
                "backend": r["backend"], "model": r["model"], "privacy": r["privacy"], "latency_ms": r["latency_ms"],
                "note": r["note"], "rating": r["rating"],
                "sent": self.cipher.decrypt(r["sent_encrypted"]) if r["sent_encrypted"] else None, **body}

    def latest_response(self, entry_id: int) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM responses WHERE entry_id = ? ORDER BY id DESC LIMIT 1",
                             (entry_id,)).fetchone()
        return self._response(r) if r else None

    def all_responses(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM responses ORDER BY id").fetchall()
        return [self._response(r) for r in rows]

    def rate_response(self, response_id: int, rating: int | None) -> bool:
        with self._conn() as conn:
            return conn.execute("UPDATE responses SET rating = ? WHERE id = ?", (rating, response_id)).rowcount > 0

    def response_stats(self) -> list[dict]:
        """Per backend: how many replies, average speed, and how users rated them (for the report)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT backend, COUNT(*) n, AVG(latency_ms) avg_ms,
                          SUM(rating = 1) helpful, SUM(rating = -1) not_helpful
                   FROM responses WHERE kind = 'reflection' GROUP BY backend""").fetchall()
        return [dict(r) for r in rows]

    # ======================= Phase 4: settings =======================
    def get_setting(self, key: str, default=None, encrypted: bool = False):
        with self._conn() as conn:
            r = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not r or r["value"] is None:
            return default
        return self.cipher.decrypt(r["value"]) if encrypted else r["value"]

    def set_setting(self, key: str, value, encrypted: bool = False) -> None:
        stored = self.cipher.encrypt(str(value)) if encrypted else value
        with self._conn() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, stored))
