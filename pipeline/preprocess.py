"""Text cleaning shared by every input type (text, voice, OCR).

Important: we do NOT remove stopwords here. Transformer models need the full
sentence — dropping "not" turns "not happy" into "happy". Stopword removal is
only used later for event/keyword extraction (Phase 3).
"""
import re
import unicodedata

# Spoken fillers that Whisper transcribes but carry no emotional meaning.
_FILLERS = re.compile(r"\b(?:um+|uh+|erm+|hmm+|uh-huh|you know|i mean)\b[,.]?\s*", re.IGNORECASE)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_REPEATED_PUNCT = re.compile(r"([!?.])\1{2,}")
# OCR artefacts
_PIPE_AS_I = re.compile(r"(?<![|\w])[|](?![|\w])")          # " | was" -> " I was"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")                    # "exam-\nination" -> "examination"
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")                  # single newline inside a paragraph


def clean_text(text: str, input_type: str = "text") -> str:
    """Normalize unicode and whitespace; strip fillers from voice transcripts."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))
    if input_type == "voice":
        text = _FILLERS.sub("", text)
    if input_type == "image":
        text = _PIPE_AS_I.sub("I", text)
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
        text = _SOFT_BREAK.sub(" ", text)    # OCR gives one line per written line; rejoin into prose
    text = _REPEATED_PUNCT.sub(r"\1\1", text)   # "!!!!!" -> "!!" (keeps intensity, drops noise)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    """Group sentences into chunks under max_words so each fits RoBERTa's 512-token limit."""
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if words > max_words:  # one giant run-on sentence: hard split it
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            tokens = sentence.split()
            for i in range(0, len(tokens), max_words):
                chunks.append(" ".join(tokens[i:i + max_words]))
            continue
        if count + words > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks
