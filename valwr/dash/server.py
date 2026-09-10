"""The live dashboard: python -m valwr.dash

A local page that shows the current match -- both win probabilities, both teams
ranked by potential, and the factors driving the prediction -- updating over a
websocket without a refresh.

It computes nothing. Every number comes from `live.state.poll_once`, the same
dictionary the terminal view renders, so the two cannot drift apart. A second
copy of the prediction logic is the failure `live/predict.py` was written to
avoid, and it would fail silently here too: every field would still be present
and still look plausible.

**Bound to 127.0.0.1 by default.** docs/ETHICS-AND-TOS.md forbids exposing an
endpoint that looks up arbitrary players; this serves the match you are in and
nothing more, and there is no route that takes a puuid.

`--host` widens that binding, for the one case that needs it: reading the lobby
on your phone while sat at the same desk. It is opt-in, never the default, and
it says what it is exposing and to whom. What goes over the wire is the current
match -- other players' gamertags and their stored statistics -- so it belongs
on a network you control and nowhere else. The rule it still satisfies is the
one that matters: nothing here can be asked about a player who is not in the
match being served.

**A bind address is not an access control.** Any web page open in the same
browser can reach 127.0.0.1, and websockets are exempt from the same-origin
policy, so `LocalOnly` refuses a request addressed to a domain name (DNS
rebinding) and a websocket opened from any other origin.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

# Imported at MODULE level, deliberately. `from __future__ import annotations`
# turns every annotation into a string, and FastAPI resolves those against the
# module namespace. With `WebSocket` imported inside build_app instead, the
# name was not there to resolve, so FastAPI fell back to treating the `socket`
# parameter as a *query parameter* -- and rejected every handshake with
# 403 Forbidden and "loc: ['query', 'socket'], Field required".
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketClose

from valwr.live import state as st

HOST = "127.0.0.1"          # never 0.0.0.0 -- see the module docstring
PORT = 8787
POLL_SECONDS = 5.0

STATIC = Path(__file__).resolve().parent / "static"
AGENTS = STATIC / "agents"
MAPS = STATIC / "maps"


def host_allowed(host: str | None) -> bool:
    """Loopback by name, or any IP literal. Never a domain name.

    DNS rebinding points an attacker's domain at 127.0.0.1 after their page
    has loaded, and the browser then treats this server as that domain's own
    origin -- free to load the page and open the socket. The request still
    names the host it was sent to, so an unrecognised name is refused. An IP
    literal cannot be rebound, which is what lets phone.bat's
    http://192.168.x.x:8787/ through.
    """
    if not host:
        return False
    try:
        name = urlsplit(f"//{host}").hostname
    except ValueError:
        return False
    if not name:
        return False
    if name == "localhost":
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def refusal(kind: str, host: str | None, origin: str | None) -> str | None:
    """Why a request is refused, or None to let it through."""
    if not host_allowed(host):
        return f"refused: {host!r} is not a local address"
    # Browsers attach Origin to every websocket handshake and a page cannot
    # remove it, so this is what stops another site open in the same browser
    # from reading the match. A client sending no Origin is not a web page,
    # and anything that is not a web page could reach the port regardless.
    if kind == "websocket" and origin is not None:
        try:
            netloc = urlsplit(origin).netloc
        except ValueError:
            netloc = ""
        if netloc.lower() != host.lower():
            return f"refused: websocket from origin {origin!r}"
    return None


class LocalOnly:
    """ASGI middleware applying `refusal` to every request and handshake."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                       for k, v in scope.get("headers", [])}
            reason = refusal(scope["type"], headers.get("host"),
                             headers.get("origin"))
            if reason:
                if scope["type"] == "http":
                    response = PlainTextResponse(reason, status_code=403)
                    await response(scope, receive, send)
                else:
                    await WebSocketClose(code=1008)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def content_policy(host: str) -> str:
    """The page's Content-Security-Policy.

    The page is one file with inline script and style -- no build step, by
    design -- so this cannot forbid inline script; escaping every value is
    what stops an injection. What it does is make one worthless: no external
    script, image, font or connection is allowed, so nothing on the page can
    be sent anywhere, and no other site can frame it.
    """
    return ("default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; img-src 'self'; "
            f"connect-src 'self' ws://{host}; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'")


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
    app.add_middleware(LocalOnly)
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
    def index(request: Request):
        # no-store, deliberately. Cached, the page outlives the server that
        # served it: with the dashboard stopped the browser happily renders a
        # stale copy whose websocket can never connect, so it reads as "the app
        # is broken" rather than "nothing is running". That cost a real
        # debugging session.
        #
        # The host is safe to echo into the policy: LocalOnly has already
        # refused anything that is not localhost or an IP literal.
        host = request.headers.get("host", HOST)
        return FileResponse(STATIC / "index.html", headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Security-Policy": content_policy(host),
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
        })

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
    ap.add_argument("--host", default=HOST,
                    help="interface to bind. Defaults to 127.0.0.1; pass "
                         "0.0.0.0 to read it on a phone on the same network")
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
    if args.host == HOST:
        print("  bound to localhost only -- not reachable from your network.")
    else:
        import socket
        try:
            lan = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan = args.host
        print(f"  BOUND TO {args.host} -- reachable from your network.")
        print(f"  On a phone on the same wifi:  http://{lan}:{args.port}/")
        print("  This serves the current match, including other players'")
        print("  gamertags and statistics. Only do this on a network you")
        print("  control, and stop it when you are done.")
    print("  Keep this window open. Ctrl+C to stop.\n")

    server = uvicorn.Server(uvicorn.Config(
        build_app(no_fetch=args.no_fetch, deadline=args.deadline,
                  demo=args.demo, match=args.match),
        host=args.host, port=args.port, log_level="warning"))

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
