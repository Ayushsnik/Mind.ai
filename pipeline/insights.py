"""Glue for Phase 3: analyse entries, mine patterns, keep alerts up to date."""
from __future__ import annotations

import logging
import threading
from datetime import date, timedelta

from .alerts import build_alerts
from .events import load_extractor
from .patterns import Insights, analyze, event_label

log = logging.getLogger(__name__)

# Bump when extraction logic changes so old entries are re-analysed automatically.
ANALYSIS_VERSION = 2   # 2: names can no longer become topics


class InsightsService:
    def __init__(self, db, emotions, *, spacy_model="en_core_web_sm", topics=True, min_entries=8,
                 min_occurrences=3, min_rate=0.6, min_lift=1.15, combo_conf=0.7, lookback_days=14):
        self.db, self.emotions = db, emotions
        self.spacy_model, self.topics = spacy_model, topics
        self.params = dict(min_entries=min_entries, min_occurrences=min_occurrences, min_rate=min_rate,
                           min_lift=min_lift, combo_conf=combo_conf)
        self.lookback_days = lookback_days
        self._extractor = None
        self._lock = threading.Lock()
        self._cache: tuple[tuple, Insights] | None = None

    @property
    def extractor(self):
        if self._extractor is None:
            with self._lock:
                if self._extractor is None:
                    self._extractor = load_extractor(self.spacy_model, self.topics)
        return self._extractor

    def status(self) -> dict:
        return {"extractor": self._extractor.name if self._extractor else "not loaded yet",
                "analysis_version": ANALYSIS_VERSION,
                "pending": len(self.db.entries_needing_analysis(ANALYSIS_VERSION))}

    # ---------------- per entry ----------------
    def analyze_entry(self, entry_id: int, body: str, created_at: str) -> list[dict]:
        ext = self.extractor.extract(body, date.fromisoformat(created_at[:10]))
        idxs = sorted({m.sentence_idx for m in ext.mentions})
        # Emotion of each sentence that contains an event (sentence-level attribution).
        results = dict(zip(idxs, self.emotions.predict_many([ext.sentences[i] for i in idxs])))
        rows = []
        for m in ext.mentions:
            r = results[m.sentence_idx]
            rows.append(m.to_dict() | {"emotion": r.primary, "emotion_conf": round(r.confidence, 4)})
        self.db.replace_events(entry_id, rows, ANALYSIS_VERSION)
        return rows

    def backfill(self, force: bool = False, progress=None) -> int:
        """Analyse entries saved before Phase 3 (or all entries when force=True)."""
        ids = ([e["id"] for e in self.db.list_entries(limit=10**9)] if force
               else self.db.entries_needing_analysis(ANALYSIS_VERSION))
        for n, entry_id in enumerate(ids, 1):
            entry = self.db.get_entry(entry_id)
            if entry:
                self.analyze_entry(entry_id, entry["body"], entry["created_at"])
            if progress:
                progress(n, len(ids))
        return len(ids)

    # ---------------- across entries ----------------
    def insights(self) -> Insights:
        version = self.db.data_version()
        if self._cache and self._cache[0] == version:
            return self._cache[1]
        result = analyze(self.db.analysis_meta(), self.db.all_events(), total_entries=self.db.count(),
                         **self.params)
        self._cache = (version, result)
        return result

    def refresh_alerts(self, today: date | None = None) -> list[dict]:
        today = today or date.today()
        self.db.expire_alerts(today.isoformat())
        since = (today - timedelta(days=self.lookback_days)).isoformat()
        for draft in build_alerts(self.insights(), self.db.upcoming_events(since), today):
            self.db.upsert_alert(draft)
        return self.db.active_alerts()

    @staticmethod
    def describe_events(rows: list[dict]) -> list[dict]:
        """Events of one entry for display: one chip per event, upcoming ones with their date."""
        seen, out = set(), []
        for r in rows:
            key = (r["event"], r["timing"], r["target_date"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r | {"label": event_label(r["event"])})
        return out
