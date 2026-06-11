"""
PhishDetect AI — FastAPI backend
Run: uvicorn api:app --reload --port 8000
"""
import asyncio, json, os, sys
from collections import deque
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from phishdetect_util import predict_email

_HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="PhishDetect AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Scan log & SSE broadcast ──────────────────────────────────────────────────
scan_log: deque = deque(maxlen=500)
_listeners: set  = set()
_loop            = None


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_running_loop()


def _broadcast(entry: dict):
    """Push a new scan entry to all connected SSE clients (called from sync thread)."""
    if _loop:
        for q in list(_listeners):
            _loop.call_soon_threadsafe(q.put_nowait, entry)


# ── Models ────────────────────────────────────────────────────────────────────
class EmailPayload(BaseModel):
    text:    str
    sender:  str = ""
    subject: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/feed", include_in_schema=False)
def serve_feed():
    return FileResponse(os.path.join(_HERE, "extension", "feed.html"), media_type="text/html")


@app.get("/logo.png", include_in_schema=False)
def serve_logo():
    return FileResponse(os.path.join(_HERE, "logo.png"), media_type="image/png")


@app.post("/analyze")
def analyze(payload: EmailPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Email text is empty")

    full = ""
    if payload.sender:
        full += f"From: {payload.sender}\n"
    if payload.subject:
        full += f"Subject: {payload.subject}\n"
    full += f"\n{payload.text}"

    result = predict_email(full)

    entry = {
        "id":         len(scan_log) + 1,
        "ts":         datetime.now(timezone.utc).isoformat(),
        "sender":     payload.sender  or "—",
        "subject":    payload.subject or "(no subject)",
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 1),
        "risk":       result["risk_level"],
        "signals":    result.get("signals", []),
    }
    scan_log.append(entry)
    _broadcast(entry)

    return {
        "prediction":    result["prediction"],
        "confidence":    round(result["confidence"], 1),
        "risk":          result["risk_level"],
        "probabilities": result["probabilities"],
        "signals":       result.get("signals", []),
        "used_ensemble": result.get("used_ensemble", False),
    }


@app.get("/scans")
def get_scans(limit: int = 100):
    items = list(scan_log)
    return {
        "scans": list(reversed(items))[:limit],
        "total": len(items),
    }


@app.get("/scans/stream")
async def stream_scans():
    """Server-Sent Events — streams new scan results in real time."""
    q: asyncio.Queue = asyncio.Queue()
    _listeners.add(q)

    async def generator():
        # Real data event (not just a comment) so Chrome flushes headers immediately
        yield 'data: {"type":"connected"}\n\n'

        # Replay last 50 entries so the page isn't blank on load
        for entry in list(scan_log)[-50:]:
            yield f"data: {json.dumps(entry)}\n\n"

        try:
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield 'data: {"type":"ping"}\n\n'   # keep connection alive
        finally:
            _listeners.discard(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )
