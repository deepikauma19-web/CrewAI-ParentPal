"""
ParentPal API server (FastAPI) + React front end.

    python3 server.py        ->  open http://localhost:8000

Endpoints
    GET  /                 React app (index.html next to server.py)
    GET  /api/config       ages, countries, models, samples, WHO sources, key status
    POST /api/ask          runs the crew and streams live progress (Server-Sent Events)
    GET  /api/journal      saved journal entries (newest first)
    GET  /api/visit-sheet  printable doctor-visit question list (.txt)
    POST /api/kb/refresh   re-fetch the WHO pages
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import crew as pp

import os

HERE = Path(__file__).resolve().parent
# The React page lives NEXT TO server.py, e.g.
#   /Users/deepika/Documents/CrewAI - Parent Pal/index.html
# Override with the PARENTPAL_INDEX environment variable if you move it elsewhere.
INDEX = Path(os.getenv("PARENTPAL_INDEX", HERE / "index.html"))
if not INDEX.exists() and (HERE / "static" / "index.html").exists():
    INDEX = HERE / "static" / "index.html"  # older folder layout still works
app = FastAPI(title="ParentPal API")


class AskRequest(BaseModel):
    query: str
    age: str = pp.CHILD_AGES[2]
    country: str = "India"
    model: str = pp.MODELS[0]
    temperature: float = 0.3
    n_results: int = 3
    live_all: bool = False
    use_cache: bool = True


@app.get("/")
def index():
    if not INDEX.exists():
        return PlainTextResponse(f"index.html not found at {INDEX}. Put it next to server.py.",
                                 status_code=500)
    return FileResponse(INDEX)


@app.get("/api/config")
def config():
    kb = pp.load_knowledge(pp.KB_SOURCES, live_all=False)
    return {
        "ages": pp.CHILD_AGES,
        "countries": list(pp.EMERGENCY_NUMBERS),
        "models": pp.MODELS,
        "samples": pp.SAMPLE_QUERIES,
        "agents": [{"icon": i, "name": n, "desc": d} for i, n, d in pp.AGENTS_UI],
        "keys": {k: pp.key_status(k) for k in ("OPENAI_API_KEY", "SERPER_API_KEY")},
        "sources": [{"title": s["title"], "url": s["url"], "mode": s["mode"]} for s in kb],
        "snapshot_date": pp.KB_SNAPSHOT_DATE,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    """Stream events while the crew works. Each line is `data: {json}`."""
    events: queue.Queue = queue.Queue()

    def worker():
        try:
            result = pp.run_question(req.query, req.age, req.country, req.model, req.temperature,
                                     req.n_results, req.live_all, req.use_cache, events.put)
            events.put(result)
        except Exception as exc:  # report the error to the UI instead of a silent 500
            events.put({"type": "error", "message": str(exc)[:500]})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # stops proxies/browsers timing out on long runs
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/journal")
def journal():
    return pp.read_journal()


@app.get("/api/visit-sheet")
def visit_sheet(name: str = ""):
    text = pp.visit_sheet(pp.read_journal(), name)
    filename = f"doctor_visit_{datetime.now():%Y%m%d}.txt"
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/api/kb/refresh")
def refresh_kb():
    pp.fetch_page.clear()
    return config()["sources"]


if __name__ == "__main__":
    print("\n  ParentPal is running →  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")