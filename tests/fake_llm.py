"""A tiny HTTP server that speaks the Ollama and Gemini APIs, for tests and offline demos.

It records every request body, so tests can check exactly what would have left the device.
Run standalone for a UI demo:  python -m tests.fake_llm   (serves on port 11434 like Ollama)
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self, port: int = 0, model: str = "llama3.2:3b"):
        self.model = model
        self.requests: list[dict] = []
        self.mode = "ok"            # ok | garbage | unsafe | error
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    return self._send(200, {"models": [{"name": outer.model}]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = json.loads(raw or b"{}")
                outer.requests.append({"path": self.path, "body": req, "headers": dict(self.headers)})
                if outer.mode == "error":
                    return self._send(500, {"error": "boom"})
                if self.path == "/api/chat":
                    prompt = req["messages"][-1]["content"]
                    return self._send(200, {"message": {"role": "assistant", "content": outer.reply(prompt, req)}})
                m = re.match(r"/v1beta/models/([^:]+):generateContent", self.path)
                if m:
                    if self.headers.get("x-goog-api-key") != "test-key":
                        return self._send(403, {"error": {"message": "bad key"}})
                    prompt = req["contents"][0]["parts"][0]["text"]
                    return self._send(200, {"candidates": [{"content": {"parts": [{"text": outer.reply(prompt, req)}]}}]})
                self._send(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, prompt: str, req: dict) -> str:
        if self.mode == "garbage":
            return "Sure! Here is some text that is not JSON at all."
        if self.mode == "unsafe":
            return json.dumps({"reflection": "You have depression and should take medication for it, honestly.",
                               "affirmation": "ok", "suggestions": ["Take 50 mg of something."]})
        is_plan = '"steps"' in json.dumps(req)
        placeholder = re.search(r"\[(?:PERSON|NAME)_\d+\]", prompt)
        who = placeholder.group(0) if placeholder else "the people around you"
        if is_plan:
            return json.dumps({"intro": "This has been hard before, so a plan makes sense.",
                               "steps": ["Revise the three most likely topics tonight.", "Breathe slowly when tense.",
                                         "Go to the gym afterwards; it has helped before."]})
        return "```json\n" + json.dumps({
            "reflection": f"It sounds like a draining day, and the time with {who} stayed on your mind. "
                          "Feeling tense before something that matters is understandable.",
            "affirmation": "You noticed what you needed, and that counts.",
            "suggestions": ["1. Take a 20-minute walk before you start revising.",
                            "- Write down the one topic that worries you most.",
                            "Message a friend tonight."]}) + "\n```"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    print("Fake Ollama/Gemini on http://127.0.0.1:11434 (Ctrl+C to stop)")
    with FakeLLM(11434) as f:
        f.thread.join()
