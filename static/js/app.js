// Mind.ai write page: typing with live mood preview, voice recording, photo OCR + review.
(() => {
  const form = document.getElementById("entry-form");
  if (!form) return;
  const $ = (id) => document.getElementById(id);
  const body = $("body"), dot = $("live-dot"), label = $("live-label"), count = $("count");
  const statusEl = $("status"), notes = $("notes"), discardBtn = $("discard");
  const fields = { type: $("input_type"), source: $("source_text"), meta: $("source_meta"), token: $("media_token") };

  $("today").textContent = new Date().toLocaleDateString(undefined, {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
  });

  // ---------------- live mood preview (typing) ----------------
  let timer, lastSent = "";
  const words = (t) => (t.trim() ? t.trim().split(/\s+/).length : 0);

  async function analyze() {
    const text = body.value.trim();
    if (words(text) < 5 || text === lastSent) return;
    lastSent = text;
    label.textContent = "Reading…";
    try {
      const r = await (await fetch("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      })).json();
      dot.className = "dot e-" + r.primary;
      label.textContent = `Reads as ${r.primary}` + (r.secondary ? ` with some ${r.secondary}` : "") +
        ` (${Math.round(r.confidence * 100)}%)`;
    } catch {
      label.textContent = "Mood preview unavailable. Your entry will still save.";
    }
  }
  body.addEventListener("input", () => {
    const n = words(body.value);
    count.textContent = `${n} word${n === 1 ? "" : "s"}`;
    clearTimeout(timer);
    timer = setTimeout(analyze, 900);
  });
  body.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") form.requestSubmit();
  });

  // ---------------- modes ----------------
  const panels = { voice: $("panel-voice"), image: $("panel-image") };
  document.querySelectorAll(".mode").forEach((btn) => btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => {
      const on = b === btn;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on);
    });
    for (const [mode, panel] of Object.entries(panels)) panel.hidden = btn.dataset.mode !== mode;
  }));

  // ---------------- shared helpers ----------------
  let busy = false;
  function setStatus(msg, kind = "") {
    statusEl.hidden = !msg;
    statusEl.textContent = msg || "";
    statusEl.className = "status" + (kind ? " status-" + kind : "");
  }
  function setNotes(list) {
    notes.innerHTML = "";
    (list || []).forEach((t) => { const li = document.createElement("li"); li.textContent = t; notes.appendChild(li); });
    notes.hidden = !list || !list.length;
  }
  function setBusy(on) {
    busy = on;
    form.querySelectorAll("button, input[type=file]").forEach((el) => { if (el.id !== "rec-btn") el.disabled = on; });
  }
  function confirmReplace() {
    return !body.value.trim() || confirm("Replace what's in the entry now with the new text?");
  }
  function applyResult(type, text, meta, token, noteList) {
    body.value = text;
    fields.type.value = type;
    fields.source.value = text;            // machine output, kept to measure your corrections
    fields.meta.value = JSON.stringify(meta || {});
    fields.token.value = token || "";
    setNotes(noteList);
    discardBtn.hidden = false;
    body.dispatchEvent(new Event("input"));
    body.focus();
    body.setSelectionRange(0, 0);
  }
  function reset() {
    Object.values(fields).forEach((f) => (f.value = ""));
    fields.type.value = "text";
    body.value = "";
    setNotes([]);
    setStatus("");
    $("scan").hidden = true;
    form.classList.remove("wide");
    const drop = $("drop");
    if (drop) { drop.classList.remove("compact"); drop.querySelector(".drop-main").textContent = "Choose a photo of your journal page"; }
    const audio = $("audio-preview"); if (audio) { audio.hidden = true; audio.removeAttribute("src"); }
    discardBtn.hidden = true;
    body.dispatchEvent(new Event("input"));
  }
  discardBtn.addEventListener("click", () => { if (confirm("Clear this entry and start over?")) reset(); });

  async function postFile(url, fieldName, blob, filename, extra = {}) {
    const fd = new FormData();
    fd.append(fieldName, blob, filename);
    for (const [k, v] of Object.entries(extra)) fd.append(k, v);
    const res = await fetch(url, { method: "POST", body: fd });
    let data = {};
    try { data = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) throw new Error(data.error || `Something went wrong (${res.status}). Try again.`);
    return data;
  }
  // Convert any audio the browser can play (webm, m4a, mp3...) to 16 kHz mono 16-bit WAV.
  // The server can read WAV with Python's standard library, so no compiled audio decoder is needed.
  async function toWav16k(blob) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    let decoded;
    try { decoded = await ctx.decodeAudioData(await blob.arrayBuffer()); } finally { ctx.close(); }
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * 16000)), 16000);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    const pcm = (await offline.startRendering()).getChannelData(0);
    const buf = new ArrayBuffer(44 + pcm.length * 2);
    const v = new DataView(buf);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i++) {
      const x = Math.max(-1, Math.min(1, pcm[i]));
      v.setInt16(44 + i * 2, x < 0 ? x * 0x8000 : x * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  // ---------------- voice ----------------
  const recBtn = $("rec-btn");
  if (recBtn) {
    let recorder = null, chunks = [], startedAt = 0, tick = null;
    const recLabel = $("rec-label"), recTime = $("rec-time"), audioPreview = $("audio-preview");

    async function transcribe(blob, filename) {
      if (!confirmReplace()) return;
      audioPreview.src = URL.createObjectURL(blob);
      audioPreview.hidden = false;
      setBusy(true);
      setStatus("Transcribing… The first time, Whisper downloads its model, which can take a few minutes.", "busy");
      try {
        let upload = blob, name = filename;
        try {
          upload = await toWav16k(blob);
          name = filename.replace(/\.[^.]+$/, "") + ".wav";
        } catch { /* browser can't decode it: send the original and let the server try */ }
        const r = await postFile("/api/transcribe", "audio", upload, name);
        applyResult("voice", r.text, r.meta, r.media_token, r.warnings);
        setStatus(`Transcribed ${fmtTime(r.duration)} of audio. Check the text below, fix anything misheard, then save.`, "ok");
      } catch (err) {
        setStatus(err.message, "error");
      } finally {
        setBusy(false);
      }
    }

    recBtn.addEventListener("click", async () => {
      if (busy) return;
      if (recorder && recorder.state === "recording") { recorder.stop(); return; }
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("This browser can't record here. Open the app at http://localhost:5000, or upload a recording instead.", "error");
        return;
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        setStatus("Microphone access was blocked. Allow it in the browser's address bar, then try again.", "error");
        return;
      }
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
        .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunks = [];
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(tick);
        recBtn.classList.remove("on");
        recBtn.setAttribute("aria-pressed", "false");
        recLabel.textContent = "Record again";
        const type = (recorder.mimeType || "audio/webm").split(";")[0];
        const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
        transcribe(new Blob(chunks, { type }), `recording.${ext}`);
      };
      recorder.start(1000);
      startedAt = Date.now();
      recTime.textContent = "0:00";
      tick = setInterval(() => {
        const s = (Date.now() - startedAt) / 1000;
        recTime.textContent = fmtTime(s);
        if (s >= 600) recorder.stop();          // matches the 10-minute server limit
      }, 250);
      recBtn.classList.add("on");
      recBtn.setAttribute("aria-pressed", "true");
      recLabel.textContent = "Stop and transcribe";
      setStatus("Recording. Speak naturally; press stop when you're done.", "busy");
    });

    $("audio-file").addEventListener("change", (e) => {
      const file = e.target.files[0];
      e.target.value = "";
      if (file) transcribe(file, file.name);
    });
  }

  // ---------------- photo / OCR ----------------
  const imageInput = $("image-file");
  if (imageInput) {
    const drop = $("drop");

    async function readPage(file) {
      if (!file.type.startsWith("image/")) { setStatus("That isn't an image. Choose a JPG or PNG photo.", "error"); return; }
      if (!confirmReplace()) return;
      const mode = form.querySelector("input[name=ocr_mode]:checked")?.value || "auto";
      setBusy(true);
      setStatus("Reading your page… Handwriting takes longer, and the first run downloads the model.", "busy");
      try {
        const r = await postFile("/api/ocr", "image", file, file.name, { mode });
        $("scan-img").src = r.preview;
        $("scan-caption").textContent = `${r.lines.length} line${r.lines.length === 1 ? "" : "s"} found` +
          (Math.abs(r.skew) >= 0.5 ? `, straightened by ${Math.abs(r.skew)}°` : "") + ".";
        $("scan").hidden = false;
        form.classList.add("wide");
        drop.classList.add("compact");
        drop.querySelector(".drop-main").textContent = "Use a different photo";
        applyResult("image", r.text, r.meta, r.media_token, r.warnings);
        const who = r.engine === "trocr" ? "the handwriting reader (TrOCR)" : "Tesseract";
        const conf = r.confidence != null ? `, ${Math.round(r.confidence * 100)}% confident` : "";
        setStatus(`Read by ${who}${conf}. Compare it with your page and fix any mistakes before saving.`, "ok");
      } catch (err) {
        setStatus(err.message, "error");
      } finally {
        setBusy(false);
      }
    }

    imageInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      e.target.value = "";
      if (file) readPage(file);
    });
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) readPage(f); });
  }

  // Restore state after a failed save (server re-renders the form with the draft).
  if (fields.type.value !== "text") discardBtn.hidden = false;
  body.dispatchEvent(new Event("input"));
})();