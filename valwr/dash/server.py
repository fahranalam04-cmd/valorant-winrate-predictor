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
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    @asynccontextmanager
    async def lifespan(_app):
        yield
        _app.state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="valwr live", docs_url=None, redoc_url=None,
                  openapi_url=None,   # minimal surface: two routes, no schema
                  lifespan=lifespan)
    app.state.ctx = None
    app.state.error = None

    # ONE worker, and always the same one. A SQLite connection belongs to the
    # thread that opened it, and `asyncio.to_thread` draws from a pool with no
    # such guarantee -- so the context was opened on the event-loop thread and
    # every poll ran on some pool worker. SQLite refuses that outright:
    #
    #   ProgrammingError: SQLite objects created in a thread can only be used
    #   in that same thread.
    #
    # Every poll raised it, the page showed the error and never recovered, and
    # the suite stayed green because the websocket test accepted
    # `status == "error"` as a pass -- it was written to tolerate VALORANT
    # being closed, and tolerated a completely dead dashboard too.
    #
    # A single dedicated worker gives the connection one owner for the life of
    # the process, and serialises polls for free.
    app.state.pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="valwr-poll")

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
        # no-store, deliberately. Cached, the page outlives the server that
        # served it: with the dashboard stopped the browser happily renders a
        # stale copy whose websocket can never connect, so it reads as "the app
        # is broken" rather than "nothing is running". That cost a real
        # debugging session.
        return FileResponse(STATIC / "index.html",
                            headers={"Cache-Control": "no-store, max-age=0"})

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        last: str | None = None
        loop = asyncio.get_running_loop()
        # Both of these run on app.state.pool's single thread: `context` opens
        # the SQLite connection, `poll_once` uses it, and they must agree on
        # which thread that is.
        pool = app.state.pool
        try:
            while True:
                ctx = await loop.run_in_executor(pool, context)
                if ctx is None:
                    await socket.send_text(json.dumps(
                        {"status": "error", "message": app.state.error}))
                    await asyncio.sleep(POLL_SECONDS)
                    # The game may start later; clear the error and retry.
                    app.state.error = None
                    continue

                # poll_once blocks on HTTP and SQLite, so keep it off the event
                # loop or the socket stops responding while it resolves players.
                # Any failure is reported to the page rather than closing the
                # socket: a dropped connection renders as "disconnected" with
                # no cause, which is the least useful thing it could say.
                try:
                    state = await loop.run_in_executor(pool, st.poll_once, ctx)
                except Exception as e:                  # noqa: BLE001
                    await socket.send_text(json.dumps(
                        {"status": "error",
                         "message": f"{type(e).__name__}: {e}"}))
                    await asyncio.sleep(POLL_SECONDS)
                    continue
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
    print("  Keep this window open. Ctrl+C to stop.\n")

    server = uvicorn.Server(uvicorn.Config(
        build_app(no_fetch=args.no_fetch, deadline=args.deadline),
        host=HOST, port=args.port, log_level="warning"))

    if not args.no_browser:
        # Opened only once the port is accepting. Firing it before
        # `uvicorn.run` raced the bind: the browser hit a closed port, and if
        # it happened to hold a cached copy of the page it rendered that
        # instead -- a live-looking dashboard with a websocket to nowhere.
        def open_when_up():
            for _ in range(100):
                if getattr(server, "started", False):
                    webbrowser.open(url)
                    return
                time.sleep(0.1)
        threading.Thread(target=open_when_up, daemon=True).start()

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
