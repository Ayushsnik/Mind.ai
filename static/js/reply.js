// Mind.ai notes back on entries, ratings, and "Help me prepare" plans for heads-ups.
(() => {
  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
    });
    let data = {};
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) throw new Error(data.error || `Something went wrong (${res.status}).`);
    return data;
  }

  // ---------------- note back on an entry ----------------
  async function reflect(section, prefer) {
    const id = section.dataset.entry;
    const waiting = section.querySelector("[data-waiting]");
    const errorEl = section.querySelector("[data-error]");
    section.querySelectorAll("button").forEach((b) => (b.disabled = true));
    if (waiting) waiting.hidden = false;
    else {
      const p = document.createElement("p");
      p.className = "status status-busy";
      p.textContent = prefer === "gemini" ? "Asking Gemini…" : "Writing a note back…";
      section.appendChild(p);
    }
    errorEl.hidden = true;
    try {
      await postJSON(`/api/entries/${id}/reflect`, prefer ? { prefer } : {});
      const html = await (await fetch(`/entries/${id}/reply`)).text();
      section.outerHTML = html;
      bind(document.getElementById("reply"));
    } catch (err) {
      section.querySelectorAll(".status-busy").forEach((el) => (el.hidden = true));
      errorEl.textContent = err.message;
      errorEl.hidden = false;
      section.querySelectorAll("button").forEach((b) => (b.disabled = false));
    }
  }

  function bind(section) {
    if (!section) return;
    section.querySelectorAll("[data-reflect]").forEach((btn) =>
      btn.addEventListener("click", () => reflect(section, btn.dataset.reflect || null)));
    section.querySelectorAll("[data-rate]").forEach((btn) =>
      btn.addEventListener("click", async () => {
        const pressed = btn.getAttribute("aria-pressed") === "true";
        const rating = pressed ? 0 : Number(btn.dataset.rate);
        try {
          await postJSON(`/api/responses/${btn.dataset.response}/rate`, { rating });
          section.querySelectorAll("[data-rate]").forEach((b) =>
            b.setAttribute("aria-pressed", String(rating !== 0 && b === btn)));
        } catch { /* rating is optional; ignore failures */ }
      }));
    if (section.dataset.auto === "1") reflect(section, null);
  }
  bind(document.getElementById("reply"));

  // ---------------- "Help me prepare" on heads-ups ----------------
  document.querySelectorAll(".prepare").forEach((btn) =>
    btn.addEventListener("click", async () => {
      const box = document.getElementById(`plan-${btn.dataset.alert}`);
      btn.disabled = true;
      btn.textContent = "Making a plan…";
      try {
        const plan = await postJSON(`/api/alerts/${btn.dataset.alert}/plan`);
        box.innerHTML = "";
        const intro = document.createElement("p");
        intro.textContent = plan.intro;
        const ol = document.createElement("ol");
        plan.steps.forEach((s) => { const li = document.createElement("li"); li.textContent = s; ol.appendChild(li); });
        box.append(intro, ol);
        box.hidden = false;
        btn.remove();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = "Try again";
        box.hidden = false;
        box.textContent = err.message;
      }
    }));
})();
