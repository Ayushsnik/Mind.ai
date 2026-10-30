"""Encryption at rest for journal text (Fernet = AES-128-CBC + HMAC-SHA256).

The key lives in instance/secret.key (created on first run, chmod 600).
Lose the key and the entries can't be decrypted — back it up separately
from the database. Emotion labels/scores stay unencrypted so analytics
can run without decrypting every entry.
"""
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class Cipher:
    def __init__(self, key_path: str):
        path = Path(key_path)
        env_key = os.getenv("MINDAI_ENCRYPTION_KEY")
        if env_key:
            key = env_key.encode()
        elif path.exists():
            key = path.read_bytes().strip()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            key = Fernet.generate_key()
            path.write_bytes(key)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass  # Windows: chmod is mostly a no-op
        self._fernet = Fernet(key)

    def encrypt(self, text: str) -> bytes:
        return self._fernet.encrypt(text.encode("utf-8"))

    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt_bytes(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)

    def decrypt(self, token: bytes) -> str:
        try:
            return self._fernet.decrypt(token).decode("utf-8")
        except InvalidToken:
            return "[This entry can't be decrypted with the current key.]"
