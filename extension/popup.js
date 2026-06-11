const DEFAULT_API    = "http://localhost:8000";
const DEFAULT_DASH   = "http://localhost:8501";

const apiInput    = document.getElementById("apiUrl");
const saveBtn     = document.getElementById("saveBtn");
const savedMsg    = document.getElementById("savedMsg");
const statusDot   = document.getElementById("statusDot");
const statusText  = document.getElementById("statusText");
const dashBtn     = document.getElementById("dashboardBtn");
const feedBtn     = document.getElementById("feedBtn");

// ── Load saved values ─────────────────────────────────────────────────────────
chrome.storage.sync.get(["apiUrl", "dashUrl"], ({ apiUrl, dashUrl }) => {
  apiInput.value = apiUrl || DEFAULT_API;
  dashBtn.href   = dashUrl || DEFAULT_DASH;
  feedBtn.href   = chrome.runtime.getURL("feed.html");
  pingApi(apiInput.value);
});

// ── Save settings ─────────────────────────────────────────────────────────────
saveBtn.addEventListener("click", () => {
  const url = apiInput.value.trim().replace(/\/$/, "");
  chrome.storage.sync.set({ apiUrl: url }, () => {
    savedMsg.textContent = "Saved.";
    setTimeout(() => (savedMsg.textContent = ""), 1800);
    pingApi(url);
  });
});

// ── Ping API health ───────────────────────────────────────────────────────────
async function pingApi(base) {
  statusDot.className  = "dot";
  statusText.textContent = "Checking API…";
  try {
    const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusDot.className    = "dot ok";
      statusText.textContent = "API connected";
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch {
    statusDot.className    = "dot fail";
    statusText.textContent = "API unreachable — is it running?";
  }
}
