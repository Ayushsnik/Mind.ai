"""Error rates used for evaluation and for measuring how much users correct OCR/ASR output.

CER = character edits / reference length   (OCR metric)
WER = word edits / reference word count    (speech-to-text metric)
Both use Levenshtein distance, the standard definition in OCR/ASR papers.
"""
import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        reference, hypothesis = _normalize(reference), _normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    if normalize:
        strip = lambda t: re.sub(r"[^\w\s']", " ", _normalize(t))
        reference, hypothesis = strip(reference), strip(hypothesis)
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)
