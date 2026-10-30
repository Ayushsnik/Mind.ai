"""Download every model Mind.ai uses, once, so the app can run fully offline afterwards.

    python -m scripts.download_models            # all models
    python -m scripts.download_models whisper    # just one: emotion | whisper | trocr

If huggingface.co is blocked on your network, run this first in PowerShell:
    $env:HF_ENDPOINT = "https://hf-mirror.com"
"""
import os
import sys
import time

from config import Config


def emotion():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    AutoTokenizer.from_pretrained(Config.EMOTION_MODEL)
    AutoModelForSequenceClassification.from_pretrained(Config.EMOTION_MODEL)


def whisper():
    from faster_whisper import WhisperModel
    WhisperModel(Config.WHISPER_MODEL, device="cpu", compute_type="int8")


def trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TrOCRProcessor.from_pretrained(Config.TROCR_MODEL)
    VisionEncoderDecoderModel.from_pretrained(Config.TROCR_MODEL)


STEPS = {"emotion": ("Emotion model (RoBERTa, ~330 MB)", emotion),
         "whisper": (f"Whisper '{Config.WHISPER_MODEL}' (~480 MB for small)", whisper),
         "trocr": ("Handwriting model (TrOCR, ~1.3 GB)", trocr)}


def main():
    wanted = sys.argv[1:] or list(STEPS)
    if os.getenv("HF_HUB_OFFLINE") == "1":
        sys.exit("HF_HUB_OFFLINE=1 is set, so nothing can be downloaded. Remove it from .env for this run.")
    print("Downloading from", os.getenv("HF_ENDPOINT", "https://huggingface.co"))
    failed = []
    for key in wanted:
        label, fn = STEPS[key]
        for attempt in range(1, 4):            # flaky connections often succeed on a retry
            print(f"\n{label}: attempt {attempt} ...", flush=True)
            try:
                fn()
                print(f"{label}: ready")
                break
            except Exception as exc:
                print(f"  failed: {type(exc).__name__}: {str(exc)[:200]}")
                time.sleep(3)
        else:
            failed.append(key)
    if failed:
        sys.exit(f"\nCouldn't download: {', '.join(failed)}. Try another network or "
                 '$env:HF_ENDPOINT = "https://hf-mirror.com", then run this again.')
    print("\nAll models downloaded. You can add HF_HUB_OFFLINE=1 to .env to run fully offline.")


if __name__ == "__main__":
    main()