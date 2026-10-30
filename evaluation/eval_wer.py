"""Speech-to-text evaluation: Word Error Rate of Whisper on your own recordings.

Data layout (evaluation/data/audio/):
    clip01.wav   clip01.txt     <- recording + exactly what was said
    clip02.m4a   clip02.txt
Record ~25 clips of 20-60 s each: different speakers, Indian-English accents, a quiet
room and a noisy one. Phone voice memos (m4a) are fine.

Run:   python -m evaluation.eval_wer                 (uses WHISPER_MODEL from config)
       python -m evaluation.eval_wer base small      (compare model sizes)
Writes evaluation/results/wer_results.csv.
"""
import csv
import sys
import time
from pathlib import Path

from config import Config
from pipeline.metrics import wer
from pipeline.speech import SpeechError, SpeechService

DATA = Path(__file__).parent / "data" / "audio"
OUT = Path(__file__).parent / "results"
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac"}


def main():
    sizes = sys.argv[1:] or [Config.WHISPER_MODEL]
    clips = [(a, a.with_suffix(".txt")) for a in sorted(DATA.iterdir())
             if a.suffix.lower() in AUDIO_EXT and a.with_suffix(".txt").exists()]
    if not clips:
        sys.exit(f"No audio + .txt pairs found in {DATA}. See the docstring for the layout.")
    OUT.mkdir(exist_ok=True)
    rows = []
    for size in sizes:
        # Evaluate transcription only (no translation), against English references.
        svc = SpeechService(model_size=size, device=Config.WHISPER_DEVICE, compute_type=Config.WHISPER_COMPUTE,
                            language="en", translate_to_english=False)
        print(f"\nwhisper-{size}: {len(clips)} clips")
        for audio, ref in clips:
            t0 = time.perf_counter()
            try:
                t = svc.transcribe_file(str(audio))
                hyp, dur = t.text, t.duration
            except SpeechError as exc:
                hyp, dur = "", 0.0
                print(f"  {audio.name}: {exc}")
            elapsed = time.perf_counter() - t0
            score = wer(ref.read_text(encoding="utf-8"), hyp)
            rows.append({"model": f"whisper-{size}", "clip": audio.name, "wer": round(score, 4),
                         "audio_seconds": dur, "processing_seconds": round(elapsed, 1),
                         "real_time_factor": round(elapsed / dur, 2) if dur else ""})
            print(f"  {audio.name:<20} WER {score:.3f}   {elapsed:.1f}s for {dur:.0f}s of audio")

    with open(OUT / "wer_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'model':<16}{'mean WER':>10}{'speed (x real time)':>22}")
    for size in sizes:
        sel = [r for r in rows if r["model"] == f"whisper-{size}"]
        rtf = [r["real_time_factor"] for r in sel if r["real_time_factor"] != ""]
        print(f"whisper-{size:<8}{sum(r['wer'] for r in sel) / len(sel):>10.3f}"
              f"{(sum(rtf) / len(rtf) if rtf else 0):>22.2f}")
    print(f"\nSaved {OUT / 'wer_results.csv'}")


if __name__ == "__main__":
    main()
