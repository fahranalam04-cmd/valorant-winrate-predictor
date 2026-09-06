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
from fastapi.staticfiles import StaticFiles

from valwr.live import state as st

HOST = "127.0.0.1"          # never 0.0.0.0 -- see the module docstring
PORT = 8787
POLL_SECONDS = 5.0

STATIC = Path(__file__).resolve().parent / "static"
AGENTS = STATIC / "agents"
MAPS = STATIC / "maps"


def _demo_payload() -> dict:
    """The demo match, with agent UUIDs filled in where a database exists."""
    from valwr.dash.demo import demo_state
    conn = None
    try:
        from valwr import config
        from valwr.store import schema
        s = config.load(require_key=False)
        if s.database_path.exists():
            conn = schema.connect(s.database_path)
    except Exception:                                # noqa: BLE001
        conn = None                                  # lettered tiles, still fine
    try:
        return {"status": "match", "state": demo_state(conn),
                "top1_rate": 0.296, "fresh": True}
    finally:
        if conn is not None:
            conn.close()


def _replay_payload(match_id: str) -> dict:
    """One finished match, rebuilt as the dashboard would have shown it."""
    import joblib

    from valwr import config
    from valwr.dash.replay import replay_state
    from valwr.rating import potential as pot
    from valwr.store import schema

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    bundle = joblib.load(s.database_path.parent.parent / "models" / "model.joblib")
    try:
        index = pot.PerfIndex.load()
    except FileNotFoundError:
        index = None
    row = conn.execute("SELECT puuid FROM players WHERE lower(tag) = lower(?) "
                       "ORDER BY last_seen_at DESC LIMIT 1",
                       (getattr(s, "riot_tag", "") or "",)).fetchone()
    me = row["puuid"] if row else ""
    try:
        return {"status": "match",
                "state": replay_state(conn, match_id, bundle, index, me),
                "top1_rate": index.top1_rate if index else None, "fresh": True}
    finally:
        conn.close()


def build_app(no_fetch: bool = False, deadline: float = st.DEFAULT_DEADLINE,
              demo: bool = False, match: str | None = None):
    @asynccontextmanager
    async def lifespan(_app):
        yield
        _app.state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="valwr live", docs_url=None, redoc_url=None,
                  openapi_url=None,   # minimal surface: two routes, no schema
                  lifespan=lifespan)
    app.state.ctx = None
    app.state.error = None
    app.state.demo_state = None
    app.state.replay_state = None

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

    # Agent artwork, and nothing else. This is the only route besides the page
    # and the socket, and it is deliberately a directory of static PNGs: it
    # takes no query, names no player, and reveals nothing about anyone. The
    # constraint in docs/ETHICS-AND-TOS.md is that no endpoint may look a
    # player up, and a file server for Riot's own art does not.
    #
    # Absent until tools/fetch_agent_art.py has run, so the mount is
    # conditional -- StaticFiles raises at construction on a missing directory,
    # which would turn "no art yet" into "no dashboard at all".
    if AGENTS.is_dir():
        app.mount("/agents", StaticFiles(directory=AGENTS), name="agents")
    if MAPS.is_dir():
        app.mount("/maps", StaticFiles(directory=MAPS), name="maps")

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
                if match:
                    # A finished match, rebuilt from history. Nothing is
                    # fetched and nothing about the game client is consulted.
                    if app.state.replay_state is None:
                        try:
                            app.state.replay_state = _replay_payload(match)
                        except Exception as e:          # noqa: BLE001
                            app.state.replay_state = {
                                "status": "error",
                                "message": f"{type(e).__name__}: {e}"}
                    await socket.send_text(json.dumps(app.state.replay_state))
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                if demo:
                    # The synthetic state goes through the same renderer as a
                    # real match. Built once -- it never changes -- and the only
                    # database read is `ref_agents`, a static table of agent
                    # names and UUIDs, so the bundled artwork resolves. No
                    # player row is touched and no client call is made.
                    if app.state.demo_state is None:
                        app.state.demo_state = _demo_payload()
                    await socket.send_text(json.dumps(app.state.demo_state))
                    await asyncio.sleep(POLL_SECONDS)
                    continue
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
    ap.add_argument("--demo", action="store_true",
                    help="serve an invented match, to see the page without "
                         "playing one; touches nothing real")
    ap.add_argument("--match", metavar="ID",
                    help="replay a finished match from history, scored only "
                         "on what was knowable before it started")
    ap.add_argument("--deadline", type=float, default=st.DEFAULT_DEADLINE)
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://{HOST}:{args.port}/"
    print(f"  dashboard on {url}")
    print("  bound to localhost only -- not reachable from your network.")
    print("  Keep this window open. Ctrl+C to stop.\n")

    server = uvicorn.Server(uvicorn.Config(
        build_app(no_fetch=args.no_fetch, deadline=args.deadline,
                  demo=args.demo, match=args.match),
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
