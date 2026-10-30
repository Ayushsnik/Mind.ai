# Mind.ai

AI journal: write, speak, or photograph a journal page. Mind.ai detects the emotion, works out *what happened*, learns which situations affect you, and warns you before the hard ones come round again. Everything is encrypted on your own machine.

| Phase | Status | What it adds |
|---|---|---|
| 1 | Done | Typed entries, RoBERTa emotion detection, encrypted SQLite storage |
| 2 | Done | Voice (Whisper) and photo of a page (OpenCV + TrOCR / Tesseract) with a review step |
| 3 | Done | Event extraction, event-emotion patterns (statistics + Apriori), predictive alerts, Insights page |
| 4 | Done | Notes back on entries: local Llama (Ollama) by default, opt-in Gemini with anonymisation, crisis safety, "Help me prepare" plans |
| 5 | Next | Dashboard: mood timeline, event-emotion heatmap, PDF export |

## Setup (Windows)

```powershell
cd mind-ai                         # the folder that contains app.py
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The terminal prints which engines are active.

### Install Tesseract (for printed pages)

pip only installs the Python wrapper; the Tesseract program is separate.

1. Download the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Install with the default folder (`C:\Program Files\Tesseract-OCR`). The app finds it there automatically.
3. Installed somewhere else? Set `TESSERACT_CMD` in `.env` (copy `.env.example` to `.env`).

### First-run downloads

Each model downloads once from Hugging Face, then works offline:

| Model | Size | Downloaded when |
|---|---|---|
| Emotion (DistilRoBERTa) | ~330 MB | app start |
| spaCy English (small) | ~13 MB | installed by `pip install -r requirements.txt` |
| Whisper small | ~480 MB | first recording |
| TrOCR base handwritten | ~1.3 GB | first handwritten page |

The first recording or photo is slow because of this. After that, expect roughly 5-15 s to transcribe a minute of speech and 10-30 s for a handwritten page on a laptop CPU. To trade accuracy for speed, set `MINDAI_WHISPER_MODEL=base` or `MINDAI_TROCR_MODEL=microsoft/trocr-small-handwritten` in `.env`.

Check everything at http://localhost:5000/api/health.

### Set up the AI notes (Phase 4)

Without any setup, notes are written from your own patterns with fixed wording. For real AI notes:

**On this device (recommended).** Your entries never leave your computer.

1. Install Ollama from https://ollama.com.
2. In a terminal, run `ollama pull llama3.2:3b`. This is about a 2 GB download and needs about 4 GB of free RAM while running.
3. Keep the Ollama app running. Mind.ai finds it automatically. Replies take roughly 10-40 s on a laptop CPU.

**Enhanced, with Gemini (optional).**

1. Get a key at https://aistudio.google.com/apikey.
2. Add `GEMINI_API_KEY=...` to `.env`.
3. Restart, then choose *Enhanced* on the **Settings** page and tick the consent box.

Gemini never receives your entry. It gets a few key sentences with names, places, phone numbers, IDs, emails and links replaced by placeholders, plus emotion labels. Settings shows a preview of exactly that, and each note has a "See exactly what was sent" link.

## Using it

**Type:** write, and the mood preview updates as you go.

**Speak:** press *Start recording*, talk, press stop. You can also upload a voice memo (m4a, mp3, wav, webm). The transcript lands in the editor so you can fix misheard words before saving. Hindi, Marathi and other languages are detected and translated to English, because the emotion model is English-only.

**Photo of a page:** choose or drop a photo. The page is cleaned up and read, then shown next to the text with every detected line boxed and numbered. Red boxes are lines the reader was unsure about. Fix the text, then save.

> The microphone only works on `localhost` or HTTPS (a browser rule). If you open the app from a phone over Wi-Fi (`http://192.168.x.x:5000`), recording is blocked, but uploading a voice memo or a photo works.

**Insights:** after 8 entries, this page shows which situations weigh on you, which lift you, combinations that are harder together, and how your week usually goes. Each entry page lists *what happened* (click an event to see every entry about it).

**Notes back:** after you save, Mind.ai writes a short reflection, an encouraging line, and 2-3 small suggestions. Where possible, the suggestions are drawn from what your journal shows helps you. Rate each note helpful or not; the ratings feed the comparison on the Settings page.

**Heads-ups:** mention something coming up ("viva tomorrow", "meeting on Monday") that has usually been hard for you, and a heads-up appears on the entry and on the Write page, with what has helped you before. *Help me prepare* turns it into a 3-step plan (before, during, after).

> Try it without weeks of journaling: `python -m evaluation.synthetic_patterns --seed-demo` fills your journal with 60 generated days (use a fresh install, or it asks first). Delete `instance/` to start clean again.

## How events and patterns work (Phase 3)

**1. What happened (`pipeline/events.py`, `pipeline/event_taxonomy.py`)**

- About 30 life situations (exams, meetings, family, conflict, exercise, poor sleep, commute...) are matched on word *lemmas* with spaCy, so "exams", "viva" and "end sem paper" all count as *exams*.
- Part-of-speech rules stop false hits: "went for a run" is exercise, "ran out of time" isn't.
- **Negation:** "didn't go to the gym" is not a gym event; "didn't like the movie" still is (the movie happened).
- **Timing:** each event is past or upcoming, read at the level of its own clause. "Yesterday's exam went badly and I have another exam on Friday" gives one past exam and one upcoming exam dated Friday. Date words understood include tomorrow, day after tomorrow, weekday names, next week, this weekend, and in N days.
- **Personal topics:** recurring nouns outside the list (your thesis, a club, a place) are picked up automatically. This is where stopword removal is used.
- **Design rule:** events are situations, never emotion words. "Stressed" would trivially correlate with fear and teach nothing.

**2. Sentence-level emotion.** Each event takes the emotion of *the sentence it is in*, not the whole entry. "The exam was awful, but the gym after was great" links exam to fear and gym to joy. Using the entry's overall emotion would wrongly tie the gym to the exam's anxiety.

**3. Patterns (`pipeline/patterns.py`)**

| Measure | Meaning |
|---|---|
| confidence | share of entries about the event that went negative (or happy) |
| lift | that share ÷ how often *any* event goes that way for you. 1.5× = worse than your usual |
| strength | Wilson lower bound of the confidence. Small samples are penalised: 3 of 3 gives 0.53, 9 of 10 gives 0.65 |
| established | strength > your baseline, which is a one-sided significance test at the 5% level |

Only *established* patterns are claimed or used for heads-ups. Weaker ones appear under "Still forming". A trend (getting easier or harder) is reported once an event has 6+ entries.

**Combinations:** Apriori (mlxtend) runs over each day's set of events. A pair is reported only when its lower confidence bound beats *its stronger part alone*. Without that test, pairing any strongly negative event with a random one produces "rules" by chance.

**Weekdays:** a day is flagged when its negative share is both high and clearly above your average.

**4. Heads-ups (`pipeline/alerts.py`).** An upcoming mention of an established negative event creates a heads-up until the date passes, with the evidence and the situations that most reliably lift your mood. A hard weekday tomorrow does the same. Dismissed heads-ups never come back for the same situation and date.

### Evaluation: does it find real patterns, and only real ones?

`python -m evaluation.synthetic_patterns` generates journals with **planted** causes: exams cause anxiety 85% of the time, the gym causes happiness, roommate arguments cause anger, Monday meetings cause anxiety, and so on. It mixes in **noise** events with random emotions, then runs everything through the real pipeline.

Results (5 random journals each, keyword emotion model, spaCy extractor):

| Metric | 60 days | 90 days |
|---|---|---|
| Planted patterns detected (right direction and emotion) | 1.00 | 1.00 |
| Planted patterns established (significant, used for heads-ups) | 0.94 | 1.00 |
| Precision of established patterns | 1.00 | 1.00 |
| Noise events reported as patterns | 0 | 0 |
| Hard weekday (Monday) found | 1.00 | 1.00 |
| Heads-up decisions correct (2 should fire, 2 shouldn't) | 0.85 | 1.00 |
| Spurious combination rules (none planted) | 0.00 | 0.33 |

At 60 days, the misses are patterns with only 5-9 mentions that point the right way but can't yet be told apart from chance. The engine shows them as "still forming" instead of guessing. Run it again with RoBERTa installed (`--backend roberta`) for the numbers to report.

## How the AI notes work (Phase 4)

```
entry ──► crisis check ──► flagged? ──► fixed safety message + helplines (no model, ever)
              │
              ▼
        context: emotion, what happened, your patterns, what lifts you   (Phase 3)
              │
     ┌────────┼──────────────────────────┐
     ▼        ▼                          ▼
  Gemini   local model (Ollama)       template
  (opt-in) full entry, on device      no AI; also the evaluation baseline
  anonymised summary only
     │        │                          │
     └────────┴──► validate: JSON shape, length, no diagnosis or medication ──► store encrypted
                   anything fails → next backend in the chain
```

- **Backend order** depends on Settings. *Enhanced* tries Gemini, then local, then template. *On this device* tries local, then template. *No AI* uses the template only. Gemini is never used without consent, even if requested directly through the API.
- **Anonymisation (`pipeline/anonymize.py`)** has three layers, because spaCy's small model alone misses many Indian names and reads phone numbers as dates:
  1. Patterns: emails, links, Indian and international phone numbers, Aadhaar-style and roll numbers, PIN codes.
  2. spaCy entities.
  3. Capitalised proper nouns and sentence-initial names, plus your own "names to always hide".

  Each value gets a stable placeholder. The mapping stays on the device, and the reply is restored before you see it.
- **Safety (`pipeline/safety.py`):** suicide and self-harm language, in English and romanised Hindi/Marathi, with negation handled ("I don't want to die" is not flagged), skips every model. It shows Tele-MANAS 14416 (free, 24×7) and 112 instead. Model replies that diagnose or mention medication are rejected.
- **Prompts:** in `pipeline/responder.py`. Models must reply in JSON. Ollama gets a JSON schema; Gemini gets `responseMimeType: application/json`. Temperature is not sent to Gemini 3 models, as Google now recommends.

### Evaluation

`python -m evaluation.eval_responses` uses 20 realistic sample entries with every personal detail annotated:

| Check | Result (needs no model) |
|---|---|
| Personal details removed before Gemini | **26 of 26 (100%)**: names, places, phones, Aadhaar, email, roll no. |
| Crisis entries routed to the safety message, reaching no model | **2 of 2** |

With Ollama running and/or a Gemini key set, it also measures per backend: median reply time, replies without fallback, suggestions grounded in your patterns, and specificity. It writes `rating_sheet.csv`, a **blind** A/B/C sheet for 3-5 classmates to score empathy, specificity and helpfulness (1-5) without knowing which model wrote what. Then `python -m evaluation.eval_responses --score rating_sheet.csv` gives the comparison table for the Results slide: template vs local vs Gemini.

## How the OCR pipeline works (`pipeline/ocr.py`)

1. **Load:** decode, fix phone EXIF rotation, resize to ~2000 px.
2. **Clean:** grayscale, illumination normalisation (removes shadows and paper tint), denoise, adaptive threshold.
3. **Deskew:** test angles from -15° to +15° and keep the one that gives the sharpest horizontal ink profile.
4. **Remove ruled lines:** morphological opening with a long thin kernel keeps only notebook rules and the margin line, which are then subtracted.
5. **Segment lines:** horizontal projection profile, gap merging, splitting of touching lines, paragraph detection from larger gaps. Required because TrOCR reads one line at a time.
6. **Recognise:**
   - *Printed* → Tesseract on the whole cleaned page.
   - *Handwritten* → TrOCR on each line crop, with a per-line confidence from token probabilities.
   - *Not sure* (auto) → Tesseract first; if its confidence is below 70% the page is treated as handwriting and TrOCR reads it.
7. **Text cleanup:** rejoin wrapped lines into paragraphs, merge hyphenated line breaks, fix `|` read as `I`.

Whatever the machine read is stored alongside your corrected text. The entry page shows how much you corrected (character error rate between the two), which is real-world accuracy data you collect just by using the app.

## What is stored

| Data | Where | Encrypted |
|---|---|---|
| Entry text, original transcript/OCR output | `instance/mindai.sqlite3` | Yes |
| Original recording / photo | `instance/media/*.enc` | Yes |
| Emotion labels, engine, confidence, language | SQLite | No (needed for analytics) |
| Detected events (e.g. "exam", "conflict") and heads-up messages | SQLite | No (needed for pattern mining). The sentences themselves stay encrypted. Capitalised words (likely names) never become topics. |
| Mind.ai's notes, and what was sent to Gemini | `responses` table | Yes |
| Settings; "names to always hide" | `settings` table | Names: yes |

Set `MINDAI_KEEP_MEDIA=0` to not keep recordings and photos at all. Deleting an entry deletes its media. `instance/secret.key` decrypts everything; losing it means losing the entries, so back it up and never commit it.

Older databases are upgraded automatically on first start, and earlier entries are analysed for events when the app starts.

## Evaluation (numbers for the report)

**OCR: Tesseract vs the TrOCR pipeline**

1. Put page photos and exact typed transcriptions in `evaluation/data/ocr/` as pairs: `page01.jpg` + `page01.txt`.
2. Aim for 15-20 handwritten pages from different people, plus a few printed ones.
3. Run `python -m evaluation.eval_ocr`.

This prints the mean CER/WER per engine and saves `evaluation/results/ocr_results.csv`.

**Speech: Whisper WER**

1. Put recordings and exact transcripts in `evaluation/data/audio/`: `clip01.m4a` + `clip01.txt`.
2. Aim for about 25 clips of 20-60 s, with different speakers, a quiet room and a noisy one.
3. Run `python -m evaluation.eval_wer` (or `python -m evaluation.eval_wer base small` to compare sizes).

## Files

| Path | Purpose |
|---|---|
| `app.py` | Flask routes: pages, `/api/transcribe`, `/api/ocr`, entries API |
| `config.py` | All settings, overridable from `.env` |
| `pipeline/preprocess.py` | Text cleanup shared by all inputs (stopwords deliberately kept) |
| `pipeline/emotion.py` | RoBERTa emotion service with chunking, plus keyword fallback |
| `pipeline/speech.py` | Whisper transcription, language detection, translation, hallucination filter |
| `pipeline/ocr.py` | Image cleanup, deskew, rule removal, line segmentation, TrOCR + Tesseract |
| `pipeline/metrics.py` | CER / WER (Levenshtein) |
| `pipeline/event_taxonomy.py` | The situations Mind.ai looks for |
| `pipeline/events.py` | Event extraction: matching, negation, timing, topics |
| `pipeline/patterns.py` | Event statistics, Apriori combinations, weekday patterns |
| `pipeline/alerts.py` | Heads-up generation |
| `pipeline/insights.py` | Ties it together: per-entry analysis, caching, backfill |
| `pipeline/llm.py` | Ollama and Gemini clients (standard library only) |
| `pipeline/responder.py` | Prompts, context, backend chain, validation, plans, template replies |
| `pipeline/anonymize.py` | Removes personal details before Gemini |
| `pipeline/safety.py` | Crisis check and safety message |
| `tests/fake_llm.py` | Fake Ollama + Gemini server used by tests (`python -m tests.fake_llm` for a UI demo) |
| `db/models.py` | SQLite schema and migration |
| `db/crypto.py`, `db/media.py` | Encryption; encrypted media storage |
| `evaluation/` | OCR, speech and pattern evaluation scripts, demo data |
| `tests/` | `python -m pytest` (93 tests) |

## API

| Method | Route | Body |
|---|---|---|
| POST | `/api/transcribe` | multipart `audio` file → transcript, language, warnings, `media_token` |
| POST | `/api/ocr` | multipart `image` file, `mode` = auto / handwritten / printed → text, line boxes, preview, `media_token` |
| POST | `/api/entries` | `{"body", "title", "input_type", "source_text", "source_meta", "media_token"}` |
| GET | `/api/entries`, `/api/entries/<id>` | |
| GET | `/entries/<id>/media` | original recording or photo (decrypted) |
| POST | `/api/analyze` | `{"text"}`: emotion preview, nothing saved |
| GET | `/api/insights` | all patterns, combinations, weekday stats |
| GET | `/api/alerts` | active heads-ups |
| POST | `/api/alerts/<id>/dismiss` | |
| GET | `/api/entries/<id>/events` | what happened in one entry |
| POST | `/api/reanalyse` | `{"all": true}` re-runs event extraction on every entry |
| POST | `/api/entries/<id>/reflect` | `{"prefer": "local" / "gemini" / "template"}` (optional) → note |
| GET | `/api/entries/<id>/response` | latest note |
| POST | `/api/responses/<id>/rate` | `{"rating": 1 / -1 / 0}` |
| POST | `/api/alerts/<id>/plan` | 3-step preparation plan |
| GET | `/api/entries/<id>/gemini-preview` | exactly what Gemini would receive |
| GET | `/api/health`, `/api/capabilities` | |
