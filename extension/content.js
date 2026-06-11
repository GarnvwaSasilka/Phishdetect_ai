/**
 * PhishDetect AI — Gmail content script
 * Injects an Analyse button into each expanded email.
 */

const DEFAULT_API = "http://localhost:8000";
const INJECTED    = "data-pd-injected";
const VERDICT_CLS = "pd-verdict-badge";

// ── Observe DOM for new expanded messages ─────────────────────────────────────
const observer = new MutationObserver(debounce(scanMessages, 400));
observer.observe(document.body, { childList: true, subtree: true });

// Also catch Gmail hash-based navigation
window.addEventListener("hashchange", () => setTimeout(scanMessages, 600));
setTimeout(scanMessages, 1500); // initial load

// ── Graceful shutdown when extension is reloaded mid-session ─────────────────
// "Extension context invalidated" fires when the extension updates/reloads
// while this content script is still alive in an open tab.
function _isContextValid() {
  try { return !!chrome.runtime?.id; } catch { return false; }
}

// ── Find all expanded email bodies and inject buttons ─────────────────────────
function scanMessages() {
  // Gmail renders each message in a [data-message-id] container.
  // Only inject into *expanded* messages that have a visible body.
  document.querySelectorAll('[data-message-id]').forEach(tryInject);
}

function tryInject(msgEl) {
  if (msgEl.hasAttribute(INJECTED)) return;

  // The readable body lives in .a3s (innermost text container)
  const bodyEl = msgEl.querySelector(".a3s.aiL, .a3s");
  if (!bodyEl || !bodyEl.offsetParent) return; // not visible / collapsed

  msgEl.setAttribute(INJECTED, "1");

  const btn = document.createElement("button");
  btn.className = "pd-btn";
  btn.innerHTML = shieldSVG() + " Analyse";
  btn.addEventListener("click", () => runAnalysis(msgEl, bodyEl, btn));

  // Insert the button above the email body
  bodyEl.before(btn);
}

// ── Core analysis call ────────────────────────────────────────────────────────
async function runAnalysis(msgEl, bodyEl, btn) {
  btn.disabled = true;
  btn.innerHTML = spinnerSVG() + " Analysing…";

  // Remove stale verdict
  msgEl.querySelector(`.${VERDICT_CLS}`)?.remove();

  const text    = bodyEl.innerText || "";
  const sender  = msgEl.querySelector(".gD[email]")?.getAttribute("email") || "";
  const subject = document.querySelector(".hP")?.innerText || "";
  const apiUrl  = await getApiUrl();

  try {
    const res = await fetch(`${apiUrl}/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, sender, subject }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    renderVerdict(msgEl, bodyEl, data);

  } catch (err) {
    if (err.message?.includes("Extension context invalidated")) {
      observer.disconnect();
      return;
    }
    renderError(msgEl, bodyEl, err.message);

  } finally {
    if (_isContextValid()) {
      btn.disabled = false;
      btn.innerHTML = shieldSVG() + " Re-analyse";
    }
  }
}

// ── Render verdict badge ───────────────────────────────────────────────────────
function renderVerdict(msgEl, bodyEl, data) {
  const { prediction, confidence, risk, signals = [] } = data;

  const isAI    = prediction.includes("ai");
  const isPhish = prediction.includes("phishing");
  const cls     = isAI ? "pd-ai" : isPhish ? "pd-phish" : "pd-legit";

  // Friendly, direct verdict copy
  const label   = isAI    ? "AI-Crafted Phishing"
                : isPhish ? "Phishing Detected"
                :           "Looks Legitimate";
  const sublabel = isAI    ? "Sophisticated AI-generated attack"
                 : isPhish ? "Classic credential-harvesting attempt"
                 :           "No phishing signals found";

  const sigHtml = signals.length
    ? `<div class="pd-sigs">${signals.slice(0, 3).map(s =>
        `<span class="pd-sig ${s.weight > 0 ? "pd-sig-warn" : "pd-sig-ok"}">${s.label}</span>`
      ).join("")}</div>`
    : "";

  const badge = document.createElement("div");
  badge.className = `${VERDICT_CLS} ${cls}`;
  badge.innerHTML = `
    <div class="pd-header-row">
      <div class="pd-row">
        <span class="pd-label">${label}</span>
        <span class="pd-conf">${confidence}%</span>
        <span class="pd-risk">${risk}</span>
      </div>
      <span class="pd-brand-tag">PhishDetect AI</span>
    </div>
    <div style="font-size:11px;color:#9E8068;margin-top:1px">${sublabel}</div>
    ${sigHtml}
  `;

  bodyEl.before(badge);
}

function renderError(msgEl, bodyEl, msg) {
  const badge = document.createElement("div");
  badge.className = `${VERDICT_CLS} pd-error`;
  badge.textContent = `PhishDetect: cannot reach API — ${msg}`;
  bodyEl.before(badge);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function getApiUrl() {
  if (!_isContextValid()) return DEFAULT_API;
  return new Promise((resolve, reject) => {
    try {
      chrome.storage.sync.get("apiUrl", ({ apiUrl }) => {
        if (chrome.runtime.lastError) {
          resolve(DEFAULT_API);
          return;
        }
        resolve(apiUrl || DEFAULT_API);
      });
    } catch {
      // Context invalidated — shut down gracefully
      observer.disconnect();
      resolve(DEFAULT_API);
    }
  });
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function shieldSVG() {
  // Magnifying glass — matching the detective cat's prop
  return `<svg width="12" height="12" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="11" cy="11" r="7"/><path d="m21 21-4.35-4.35"/></svg>`;
}

function spinnerSVG() {
  return `<span class="pd-spinner"></span>`;
}
