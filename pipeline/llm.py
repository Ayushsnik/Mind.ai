"""Language-model backends. Each takes a system prompt + user prompt and returns parsed JSON.

    OllamaBackend   local model (default llama3.2:3b) through the Ollama app on this machine.
                    Nothing leaves the device.
    GeminiBackend   Google Gemini over the internet. Only ever receives anonymised summaries
                    (see responder.py) and only when the user has opted in.

Plain urllib keeps the dependency list unchanged. Both backends ask for JSON output and
parse defensively, because small local models sometimes wrap JSON in prose.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMError(f"Couldn't reach the model: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("The model server returned something that isn't JSON.") from exc


def parse_json_reply(text: str) -> dict:
    """Parse a model reply that should be a JSON object (tolerates ```json fences and extra prose)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise LLMError("The model didn't return JSON.")
        try:
            out = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned malformed JSON.") from exc
    if not isinstance(out, dict):
        raise LLMError("The model returned JSON that isn't an object.")
    return out


class OllamaBackend:
    name = "local"

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:3b", timeout: float = 180):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._status: tuple[float, dict] | None = None

    @property
    def label(self) -> str:
        return f"{self.model} (on this device)"

    def status(self, max_age: float = 30) -> dict:
        """Is Ollama running and is the model pulled? Cached briefly so pages stay fast."""
        if self._status and time.time() - self._status[0] < max_age:
            return self._status[1]
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=1.5) as r:
                names = {m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])}
            pulled = self.model in names or f"{self.model}:latest" in names
            st = {"running": True, "model_pulled": pulled, "model": self.model,
                  "error": None if pulled else f"Run: ollama pull {self.model}"}
        except Exception:
            st = {"running": False, "model_pulled": False, "model": self.model,
                  "error": "Ollama isn't running. Install it from ollama.com and start it."}
        self._status = (time.time(), st)
        return st

    def available(self) -> bool:
        s = self.status()
        return s["running"] and s["model_pulled"]

    def chat(self, system: str, user: str, schema: dict | None = None) -> dict:
        payload = {"model": self.model, "stream": False, "format": schema or "json",
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": 0.6, "num_ctx": 4096}}
        data = _post_json(f"{self.host}/api/chat", payload, timeout=self.timeout)
        return parse_json_reply((data.get("message") or {}).get("content", ""))


class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str = "gemini-3.8-flash",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 thinking_level: str | None = "low", timeout: float = 60):
        self.api_key = api_key or None
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.thinking_level = thinking_level or None
        self.timeout = timeout

    @property
    def label(self) -> str:
        return f"Gemini ({self.model})"

    def status(self) -> dict:
        return {"configured": bool(self.api_key), "model": self.model,
                "error": None if self.api_key else "Add GEMINI_API_KEY to your .env file."}

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, schema: dict | None = None) -> dict:
        if not self.api_key:
            raise LLMError("No Gemini API key configured.")
        # Gemini 3 models: no temperature/top_p (deprecated); JSON via responseMimeType.
        config = {"responseMimeType": "application/json"}
        if self.thinking_level:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": config}
        data = _post_json(f"{self.base_url}/models/{self.model}:generateContent", payload,
                          headers={"x-goog-api-key": self.api_key}, timeout=self.timeout)
        try:
            cand = data["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError, TypeError) as exc:
            reason = (data.get("promptFeedback") or {}).get("blockReason") or "no answer returned"
            raise LLMError(f"Gemini didn't answer ({reason}).") from exc
        return parse_json_reply(text)
