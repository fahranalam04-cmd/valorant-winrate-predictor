"""The live dashboard: python -m valwr.dash

A local page that shows the current match -- both win probabilities, both teams
ranked by potential, and the factors driving the prediction -- updating over a
websocket without a refresh.

It computes nothing. Every number comes from `live.state.poll_once`, the same
dictionary the terminal view renders, so the two cannot drift apart. A second
copy of the prediction logic is the failure `live/predict.py` was written to
avoid, and it would fail silently here too: every field would still be present
and still look plausible.

**Bound to 127.0.0.1 and nothing else.** docs/ETHICS-AND-TOS.md forbids
exposing an endpoint that looks up arbitrary players; this serves the match you
are in and nothing more, and there is no route that takes a puuid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import webbrowser
from pathlib import Path

# Imported at MODULE level, deliberately. `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves those against the
# module namespace. With `WebSocket` imported inside build_app instead, the
# name was not there to resolve, so FastAPI fell back to treating the `socket`
# parameter as a *query parameter* -- and rejected every handshake with
# 403 Forbidden and "loc: ['query', 'socket'], Field required".
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from valwr.live import state as st

HOST = "127.0.0.1"          # never 0.0.0.0 -- see the module docstring
PORT = 8787
POLL_SECONDS = 5.0

STATIC = Path(__file__).resolve().parent / "static"


def build_app(no_fetch: bool = False, deadline: float = st.DEFAULT_DEADLINE):
    app = FastAPI(title="valwr live", docs_url=None, redoc_url=None,
                  openapi_url=None)   # minimal surface: two routes, no schema
    app.state.ctx = None
    app.state.error = None

    def context():
        """Open the live context lazily, so the page can explain a failure.

        Opening at import time would mean starting the server with VALORANT
        closed produced a traceback in the terminal and no page at all.
        """
        if app.state.ctx is None and app.state.error is None:
            try:
                app.state.ctx = st.open_context(no_fetch=no_fetch,
                                                deadline=deadline)
            except st.NotReady as e:
                app.state.error = str(e)
        return app.state.ctx

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        last: str | None = None
        try:
            while True:
                ctx = context()
                if ctx is None:
                    await socket.send_text(json.dumps(
                        {"status": "error", "message": app.state.error}))
                    await asyncio.sleep(POLL_SECONDS)
                    # The game may start later; clear the error and retry.
                    app.state.error = None
                    continue

                # poll_once blocks on HTTP and SQLite, so keep it off the event
                # loop or the socket stops responding while it resolves players.
                state = await asyncio.to_thread(st.poll_once, ctx)
                if state is None:
                    await socket.send_text(json.dumps({"status": "lobby"}))
                    last = None
                else:
                    top1 = ctx.index.top1_rate if ctx.index else None
                    payload = {"status": "match", "state": state,
                               "top1_rate": top1,
                               "fresh": state["match_id"] != last}
                    last = state["match_id"]
                    await socket.send_text(json.dumps(payload))
                await asyncio.sleep(POLL_SECONDS)
        except WebSocketDisconnect:
            return

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.dash")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-fetch", action="store_true",
                    help="cache only; never spend API quota")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--deadline", type=float, default=st.DEFAULT_DEADLINE)
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://{HOST}:{args.port}/"
    print(f"  dashboard on {url}")
    print("  bound to localhost only -- not reachable from your network.")
    print("  Ctrl+C to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)

    uvicorn.run(build_app(no_fetch=args.no_fetch, deadline=args.deadline),
                host=HOST, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
