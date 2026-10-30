"""Encrypted storage for the original audio recordings and page photos.

Flow: /api/ocr or /api/transcribe stores the upload as *pending* and returns a
token. When the user saves the entry, the token is *claimed* and the file moves
into instance/media/. Pending files the user never saved are deleted after a day.
"""
import os
import re
import secrets
import time
from pathlib import Path

_TOKEN = re.compile(r"^[a-f0-9]{32}$")

MIME_EXT = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/heic": ".heic",
}


class MediaStore:
    def __init__(self, root: str, cipher):
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.pending.mkdir(parents=True, exist_ok=True)
        self.cipher = cipher

    def save_pending(self, data: bytes, mime: str) -> str:
        token = secrets.token_hex(16)
        (self.pending / f"{token}.enc").write_bytes(self.cipher.encrypt_bytes(data))
        (self.pending / f"{token}.mime").write_text(mime or "application/octet-stream")
        return token

    def claim(self, token: str) -> tuple[str, str] | None:
        """Move a pending upload into permanent storage. Returns (relative_path, mime)."""
        if not token or not _TOKEN.match(token):
            return None
        enc, mime_file = self.pending / f"{token}.enc", self.pending / f"{token}.mime"
        if not enc.exists():
            return None
        mime = mime_file.read_text() if mime_file.exists() else "application/octet-stream"
        dest = self.root / f"{token}.enc"
        enc.replace(dest)
        mime_file.unlink(missing_ok=True)
        return dest.name, mime

    def read(self, rel_path: str) -> bytes:
        path = (self.root / rel_path).resolve()
        if path.parent != self.root.resolve():
            raise FileNotFoundError(rel_path)
        return self.cipher.decrypt_bytes(path.read_bytes())

    def delete(self, rel_path: str | None) -> None:
        if rel_path and _TOKEN.match(Path(rel_path).stem):
            (self.root / rel_path).unlink(missing_ok=True)

    def cleanup_pending(self, max_age_hours: float = 24) -> int:
        cutoff, removed = time.time() - max_age_hours * 3600, 0
        for f in self.pending.iterdir():
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed += f.suffix == ".enc"
        return removed
