"""OCR evaluation: Character Error Rate for Tesseract alone vs the full pipeline with TrOCR.

Data layout (evaluation/data/ocr/):
    page01.jpg   page01.txt     <- photo + exact transcription you typed by hand
    page02.jpg   page02.txt
    ...
Aim for 15-20 handwritten pages from different team members plus a few printed ones.

Run:   python -m evaluation.eval_ocr
Writes evaluation/results/ocr_results.csv and prints a summary table for the report.
"""
import csv
import sys
import time
from pathlib import Path

from config import Config
from pipeline.metrics import cer, wer
from pipeline.ocr import OCRService
from pipeline.preprocess import clean_text

DATA = Path(__file__).parent / "data" / "ocr"
OUT = Path(__file__).parent / "results"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    pairs = [(img, img.with_suffix(".txt")) for img in sorted(DATA.iterdir())
             if img.suffix.lower() in IMAGE_EXT and img.with_suffix(".txt").exists()]
    if not pairs:
        sys.exit(f"No image + .txt pairs found in {DATA}. See the docstring for the layout.")
    svc = OCRService(tesseract_cmd=Config.TESSERACT_CMD, trocr_model=Config.TROCR_MODEL,
                     trocr_beams=Config.TROCR_BEAMS)
    engines = [e for e, ok in svc.availability().items() if ok]
    print(f"{len(pairs)} pages, engines: {', '.join(engines)}\n")

    OUT.mkdir(exist_ok=True)
    rows = []
    for img, txt in pairs:
        truth = clean_text(txt.read_text(encoding="utf-8"), "image")
        t0 = time.perf_counter()
        outputs = svc.compare(img.read_bytes())
        elapsed = time.perf_counter() - t0
        row = {"page": img.name, "seconds_all_engines": round(elapsed, 1)}
        for engine, text in outputs.items():
            hyp = clean_text(text, "image")
            row[f"{engine}_cer"] = round(cer(truth, hyp), 4)
            row[f"{engine}_wer"] = round(wer(truth, hyp), 4)
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

    with open(OUT / "ocr_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "page", k)))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean over all pages (lower is better)")
    print(f"{'engine':<12}{'CER':>8}{'WER':>8}")
    for e in engines:
        cers = [r[f"{e}_cer"] for r in rows if f"{e}_cer" in r]
        wers = [r[f"{e}_wer"] for r in rows if f"{e}_wer" in r]
        print(f"{e:<12}{sum(cers) / len(cers):>8.3f}{sum(wers) / len(wers):>8.3f}")
    print(f"\nSaved {OUT / 'ocr_results.csv'}")


if __name__ == "__main__":
    main()
