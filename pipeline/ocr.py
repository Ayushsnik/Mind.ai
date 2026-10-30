"""OCR for photographed journal pages.

Pipeline
--------
1. Load      : decode bytes, apply EXIF rotation (phone photos), resize to ~2000 px.
2. Clean     : grayscale -> illumination normalisation (removes shadows and paper
               tint) -> denoise -> adaptive threshold.
3. Rules     : remove the printed horizontal/vertical lines of ruled notebooks,
               which otherwise wreck both segmentation and Tesseract.
4. Deskew    : find the angle that makes the horizontal ink projection sharpest.
5. Segment   : split the page into text lines using the horizontal projection
               profile. TrOCR reads ONE line at a time, so this step is required.
6. Recognise : Tesseract (fast, good on printed text) and/or TrOCR
               (transformer, good on handwriting), chosen by `mode`.

Every step is a separate function so each can be shown/measured in the report.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

MODES = ("auto", "handwritten", "printed")


class OCRError(ValueError):
    """Problem with the input image (shown to the user)."""


class OCRUnavailable(RuntimeError):
    """No OCR engine is installed."""


# ----------------------------------------------------------------------------
# 1-4. Image preparation
# ----------------------------------------------------------------------------
def load_image(data: bytes, long_side: int = 2000) -> np.ndarray:
    """Bytes -> BGR array, EXIF-rotated, long side scaled towards `long_side`."""
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as exc:
        raise OCRError("That file isn't an image we can read. Use a JPG or PNG photo.") from exc
    if min(pil.size) < 200:
        raise OCRError("The image is too small. Use a photo at least 200 px on each side.")
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    scale = min(scale, 2.0)  # upscale small scans a bit, never wildly
    if abs(scale - 1) > 0.05:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)
    return img


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low values mean a blurry photo."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Estimate the paper background and divide it out (kills shadows, yellow paper)."""
    k = max(15, (min(gray.shape) // 40) | 1)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k)
    norm = cv2.divide(gray, background, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold. Returns an INK MASK: text = 255, paper = 0."""
    block = max(25, (min(gray.shape) // 50) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)


def remove_ruled_lines(ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove long straight horizontal/vertical strokes (notebook rules, margins).

    Text strokes are short, so an opening with a long thin kernel keeps only rules.
    Returns the cleaned mask and how many rule pixels were removed.
    """
    h, w = ink.shape
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 8))))
    rules = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    cleaned = cv2.bitwise_and(ink, cv2.bitwise_not(rules))
    # Re-close small breaks the removal cut into letters crossing a rule.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    return cleaned, int(np.count_nonzero(rules))


def _projection_sharpness(ink_small: np.ndarray, angle: float) -> float:
    h, w = ink_small.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(ink_small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rotated.sum(axis=1, dtype=np.float64)
    return float(np.var(np.diff(profile)))


def estimate_skew(ink: np.ndarray, max_angle: float = 15.0) -> float:
    """Angle (degrees) to rotate by so text lines become horizontal.

    Projection-profile method: when lines are level, the row-sum profile has the
    sharpest peaks/valleys. Coarse search, then a fine search around the best.
    """
    if np.count_nonzero(ink) < 500:
        return 0.0
    scale = 800 / max(ink.shape)
    small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else ink
    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    best = max(coarse, key=lambda a: _projection_sharpness(small, a))
    fine = np.arange(best - 1, best + 1.01, 0.2)
    best = max(fine, key=lambda a: _projection_sharpness(small, a))
    return round(float(best), 2)


def rotate(img: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    if abs(angle) < 0.1:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = fill if img.ndim == 2 else (fill, fill, fill)
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


@dataclass
class PreparedPage:
    gray: np.ndarray          # deskewed, illumination-normalised grayscale (for TrOCR)
    ink: np.ndarray           # deskewed ink mask, rules removed (for segmentation)
    binary: np.ndarray        # deskewed black-on-white image (for Tesseract)
    skew: float
    blur: float
    rules_removed: int


def prepare(data: bytes) -> PreparedPage:
    img = load_image(data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = blur_score(gray)
    gray = denoise(normalize_illumination(gray))
    # Deskew BEFORE removing rules: a tilted rule doesn't match a horizontal kernel.
    skew = estimate_skew(binarize(gray))
    gray = rotate(gray, skew, fill=255)
    ink, rules = remove_ruled_lines(binarize(gray))
    # Remove isolated specks (sensor noise, paper texture).
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return PreparedPage(gray=gray, ink=ink, binary=cv2.bitwise_not(ink), skew=skew,
                        blur=blur, rules_removed=rules)


# ----------------------------------------------------------------------------
# 5. Line segmentation
# ----------------------------------------------------------------------------
@dataclass
class Line:
    y0: int
    y1: int
    x0: int
    x1: int
    paragraph_break: bool = False   # True if a large gap precedes this line

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _runs(mask: np.ndarray) -> list[list[int]]:
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    return runs


def _split_tall(run, profile, median_h):
    """Two touching lines produce one tall run: split it at the weakest row."""
    a, b = run
    if b - a < 1.8 * median_h:
        return [run]
    lo, hi = a + int(0.3 * median_h), b - int(0.3 * median_h)
    if hi <= lo:
        return [run]
    cut = lo + int(np.argmin(profile[lo:hi]))
    return _split_tall([a, cut], profile, median_h) + _split_tall([cut, b], profile, median_h)


def segment_lines(ink: np.ndarray, pad: int = 8) -> list[Line]:
    h, w = ink.shape
    profile = (ink > 0).sum(axis=1).astype(np.float64)
    if profile.max() == 0:
        return []
    win = max(3, h // 300)
    profile = np.convolve(profile, np.ones(win) / win, mode="same")
    thr = max(2.0, 0.08 * np.percentile(profile[profile > 0], 95))
    runs = _runs(profile > thr)
    if not runs:
        return []

    heights = sorted(b - a for a, b in runs)
    median_h = max(heights[len(heights) // 2], 4)
    # Merge runs separated by tiny gaps (i-dots, accents, broken strokes).
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < 0.35 * median_h:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    heights = sorted(b - a for a, b in merged)
    median_h = max(heights[len(heights) // 2], 4)

    split = []
    for run in merged:
        split.extend(_split_tall(run, profile, median_h))
    # Drop slivers that are much shorter than a real line and carry little ink.
    kept = [r for r in split if (r[1] - r[0]) >= 0.35 * median_h
            or (ink[r[0]:r[1]] > 0).sum() > 0.5 * median_h * median_h]

    lines = []
    for a, b in kept:
        cols = np.where((ink[a:b] > 0).any(axis=0))[0]
        if cols.size == 0:
            continue
        lines.append(Line(max(0, a - pad), min(h, b + pad),
                          max(0, int(cols[0]) - pad), min(w, int(cols[-1]) + 1 + pad)))
    if len(lines) > 2:
        gaps = [lines[i + 1].y0 - lines[i].y1 for i in range(len(lines) - 1)]
        median_gap = max(sorted(gaps)[len(gaps) // 2], 1)
        for i, g in enumerate(gaps):
            if g > max(1.8 * median_gap, 0.9 * median_h):
                lines[i + 1].paragraph_break = True
    return lines


def lines_to_text(texts: list[str], lines: list[Line]) -> str:
    """Join recognised lines; blank line where a paragraph gap was detected."""
    out = []
    for text, line in zip(texts, lines):
        text = text.strip()
        if not text:
            continue
        if out and line.paragraph_break:
            out.append("")
        out.append(text)
    return "\n".join(out)


def preview_image(page: PreparedPage, lines: list[Line], flagged: set[int] = frozenset(),
                  width: int = 900) -> str:
    """PNG data URI of the cleaned page with numbered line boxes (for the review screen)."""
    vis = cv2.cvtColor(page.gray, cv2.COLOR_GRAY2BGR)
    # Draw at a size that stays readable after the image is shrunk to `width`.
    k = max(1.0, vis.shape[1] / width)
    thick, font = max(2, round(2 * k)), 0.75 * k
    for i, ln in enumerate(lines):
        colour = (47, 64, 200) if i in flagged else (197, 71, 35)   # BGR: red for unsure, blue otherwise
        cv2.rectangle(vis, (ln.x0, ln.y0), (ln.x1, ln.y1), colour, thick)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        gap = round(8 * k)
        lx = ln.x0 - tw - gap if ln.x0 - tw - gap >= 0 else ln.x1 + gap
        cv2.putText(vis, label, (lx, (ln.y0 + ln.y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font, colour, thick, cv2.LINE_AA)
    scale = width / vis.shape[1]
    if scale < 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", vis)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


# ----------------------------------------------------------------------------
# 6. Recognition engines
# ----------------------------------------------------------------------------
@dataclass
class LineResult:
    text: str
    confidence: float | None   # 0..1, None when the engine can't tell


class TesseractEngine:
    name = "tesseract"
    WINDOWS_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                     r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]

    def __init__(self, cmd: str | None = None, lang: str = "eng"):
        self.lang = lang
        self.error: str | None = None
        try:
            import pytesseract
        except ImportError:
            self.pt, self.error = None, "pytesseract is not installed"
            return
        self.pt = pytesseract
        cmd = cmd or shutil.which("tesseract") or next((p for p in self.WINDOWS_PATHS if os.path.exists(p)), None)
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def available(self) -> bool:
        if self.pt is None:
            return False
        try:
            self.pt.get_tesseract_version()
            return True
        except Exception as exc:
            self.error = "Tesseract program not found. Install it (see README) or set TESSERACT_CMD."
            log.debug("tesseract unavailable: %s", exc)
            return False

    def read_page(self, binary: np.ndarray) -> tuple[list[tuple[int, str, float]], float]:
        """Whole-page OCR. Returns [(paragraph_id, line_text, conf)], mean word confidence 0..1."""
        data = self.pt.image_to_data(binary, lang=self.lang, config="--psm 4",
                                     output_type=self.pt.Output.DICT)
        groups: dict[tuple, list] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not word.strip() or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append((word, conf))
            confs.append(conf)
        out = []
        for key in sorted(groups):
            words = groups[key]
            out.append((key[0] * 1000 + key[1], " ".join(w for w, _ in words),
                        sum(c for _, c in words) / len(words) / 100))
        mean = (sum(confs) / len(confs) / 100) if confs else 0.0
        return out, mean


class TrOCREngine:
    name = "trocr"

    def __init__(self, model_name: str, batch_size: int = 6, num_beams: int = 1):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_beams = num_beams
        self._model = self._processor = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            self.error = "torch/transformers not installed"
            return False

    def _load(self):
        with self._lock:
            if self._model is None:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                self._processor = TrOCRProcessor.from_pretrained(self.model_name)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name).eval()

    def read_lines(self, crops: list[np.ndarray]) -> list[LineResult]:
        import torch
        self._load()
        results: list[LineResult] = []
        for start in range(0, len(crops), self.batch_size):
            batch = [Image.fromarray(c).convert("RGB") for c in crops[start:start + self.batch_size]]
            pixels = self._processor(images=batch, return_tensors="pt").pixel_values
            with torch.no_grad():
                out = self._model.generate(pixels, max_new_tokens=96, num_beams=self.num_beams,
                                           output_scores=True, return_dict_in_generate=True)
            texts = self._processor.batch_decode(out.sequences, skip_special_tokens=True)
            confs = self._confidences(out)
            results.extend(LineResult(t, c) for t, c in zip(texts, confs))
        return results

    def _confidences(self, out) -> list[float | None]:
        """Geometric-mean token probability per line, a usable 'how sure' signal."""
        try:
            scores = self._model.compute_transition_scores(
                out.sequences, out.scores, getattr(out, "beam_indices", None), normalize_logits=True)
            probs = []
            for row in scores:
                valid = row[torch_isfinite(row)]
                probs.append(float(valid.mean().exp()) if valid.numel() else None)
            return probs
        except Exception:
            return [None] * len(out.sequences)


def torch_isfinite(t):
    import torch
    return torch.isfinite(t) & (t != 0)


# ----------------------------------------------------------------------------
# Service used by the app
# ----------------------------------------------------------------------------
@dataclass
class OCRResult:
    text: str
    engine: str
    mode: str                     # what the page was treated as: printed / handwritten
    confidence: float | None
    lines: list[dict]
    skew: float
    warnings: list[str] = field(default_factory=list)
    preview: str = ""
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self, include_preview: bool = True) -> dict:
        d = asdict(self)
        if not include_preview:
            d.pop("preview")
        return d


class OCRService:
    def __init__(self, tesseract_cmd: str | None = None, tesseract_lang: str = "eng",
                 trocr_model: str = "microsoft/trocr-base-handwritten",
                 printed_threshold: float = 0.70, low_line_conf: float = 0.55, blur_threshold: float = 60.0,
                 trocr_beams: int = 1):
        self.tesseract = TesseractEngine(tesseract_cmd, tesseract_lang)
        self.trocr = TrOCREngine(trocr_model, num_beams=trocr_beams)
        self.printed_threshold = printed_threshold
        self.low_line_conf = low_line_conf
        self.blur_threshold = blur_threshold
        self._avail: dict | None = None

    def availability(self) -> dict:
        if self._avail is None:
            self._avail = {"tesseract": self.tesseract.available(), "trocr": self.trocr.available()}
        return self._avail

    def status(self) -> dict:
        a = self.availability()
        return {**a, "trocr_model": self.trocr.model_name,
                "tesseract_error": None if a["tesseract"] else self.tesseract.error,
                "trocr_error": None if a["trocr"] else self.trocr.error}

    # -- engine runners --
    def _run_tesseract(self, page: PreparedPage, lines: list[Line]) -> tuple[str, float, list[dict]]:
        rows, mean = self.tesseract.read_page(page.binary)
        out, prev_par = [], None
        for par, text, conf in rows:
            if prev_par is not None and par != prev_par:
                out.append("")
            out.append(text)
            prev_par = par
        return "\n".join(out).strip(), mean, [{"text": t, "confidence": round(c, 3)} for _, t, c in rows]

    def _run_trocr(self, page: PreparedPage, lines: list[Line]) -> tuple[str, float | None, list[dict]]:
        crops = [page.gray[ln.y0:ln.y1, ln.x0:ln.x1] for ln in lines]
        results = self.trocr.read_lines(crops)
        text = lines_to_text([r.text for r in results], lines)
        confs = [r.confidence for r in results if r.confidence is not None]
        mean = sum(confs) / len(confs) if confs else None
        line_dicts = [{"text": r.text.strip(), "confidence": None if r.confidence is None else round(r.confidence, 3),
                       "box": [ln.x0, ln.y0, ln.x1, ln.y1]} for r, ln in zip(results, lines)]
        return text, mean, line_dicts

    def read(self, data: bytes, mode: str = "auto", include_preview: bool = True) -> OCRResult:
        if mode not in MODES:
            raise OCRError(f"mode must be one of {', '.join(MODES)}")
        avail = self.availability()
        if not any(avail.values()):
            raise OCRUnavailable("No OCR engine is installed. Install Tesseract or torch + transformers (see README).")

        t0 = time.perf_counter()
        page = prepare(data)
        lines = segment_lines(page.ink)
        t_prep = time.perf_counter()
        warnings = []
        if page.blur < self.blur_threshold:
            warnings.append("The photo looks blurry. A sharper photo will read much better.")
        if not lines:
            raise OCRError("No writing was found in this image. Make sure the page fills the photo and is well lit.")

        engine = text = None
        conf, line_dicts, treated_as = None, [], mode
        # Printed pages: Tesseract. Handwriting: TrOCR. Auto: let Tesseract's confidence decide.
        if mode in ("auto", "printed") and avail["tesseract"]:
            t_text, t_conf, t_lines = self._run_tesseract(page, lines)
            looks_printed = t_conf >= self.printed_threshold and len(t_text.split()) >= 3
            if mode == "printed" or looks_printed or not avail["trocr"]:
                engine, text, conf, line_dicts = "tesseract", t_text, t_conf, t_lines
                treated_as = "printed" if (mode == "printed" or looks_printed) else "handwritten"
                if treated_as == "handwritten":
                    warnings.append("This looks handwritten, but TrOCR isn't installed, so Tesseract read it. "
                                    "Expect mistakes; install torch + transformers for handwriting.")
        if engine is None:
            if avail["trocr"]:
                text, conf, line_dicts = self._run_trocr(page, lines)
                engine, treated_as = "trocr", "handwritten"
                if mode == "printed":
                    warnings.append("Tesseract isn't installed, so TrOCR read this printed page.")
            else:
                text, conf, line_dicts = self._run_tesseract(page, lines)
                engine, treated_as = "tesseract", mode if mode != "auto" else "handwritten"
                warnings.append("TrOCR isn't installed, so Tesseract read this handwritten page. Expect mistakes.")
        t_rec = time.perf_counter()

        flagged = {i for i, ln in enumerate(line_dicts)
                   if engine == "trocr" and ln["confidence"] is not None and ln["confidence"] < self.low_line_conf}
        if flagged:
            warnings.append("Check line" + ("s " if len(flagged) > 1 else " ")
                            + ", ".join(str(i + 1) for i in sorted(flagged)) + ": the reader wasn't sure about them.")
        if not text.strip():
            raise OCRError("The page was found but no words could be read. Try a closer, brighter photo.")

        return OCRResult(
            text=text, engine=engine, mode=treated_as,
            confidence=None if conf is None else round(conf, 3), lines=line_dicts, skew=page.skew,
            warnings=warnings,
            preview=preview_image(page, lines, flagged) if include_preview else "",
            timings_ms={"prepare": round((t_prep - t0) * 1000), "recognise": round((t_rec - t_prep) * 1000)},
        )

    def compare(self, data: bytes) -> dict:
        """Run every available engine on the same page (used by evaluation/eval_ocr.py)."""
        page = prepare(data)
        lines = segment_lines(page.ink)
        out = {}
        if self.availability()["tesseract"]:
            out["tesseract"] = self._run_tesseract(page, lines)[0]
        if self.availability()["trocr"]:
            out["trocr"] = self._run_trocr(page, lines)[0]
        return out
