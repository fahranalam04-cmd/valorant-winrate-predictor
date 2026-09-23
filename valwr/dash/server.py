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

from valwr.live import outcomes, review
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


def port_file() -> Path | None:
    """Where the last working port is remembered, beside the database."""
    try:
        from valwr import config
        return config.load(require_key=False).database_path.parent / "dashboard-port"
    except Exception:                                # noqa: BLE001
        return None


def remembered_port() -> int | None:
    """The port this dashboard last ran on, if it is still a sane one."""
    path = port_file()
    if path is None or not path.exists():
        return None
    try:
        port = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return port if 1024 <= port <= 65535 else None


def remember_port(port: int) -> None:
    """Record the port so the next run can land on it again.

    Every tab this dashboard opens is an absolute address with a port in it --
    one per match, kept so they can be read after the game. If the next run
    binds somewhere else, every one of those tabs is dead: the browser shows
    ERR_CONNECTION_REFUSED and there is nothing the page can do about it,
    because nothing is listening to serve it. Coming back to the same port is
    what makes yesterday's tabs work today.
    """
    path = port_file()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(port), encoding="utf-8")
    except OSError:
        pass                                         # not worth failing over


def preferred_port(explicit: int | None, remembered: int | None) -> int:
    """Which port to try first: what was asked for, then last time's, then the
    default."""
    if explicit is not None:
        return explicit
    return remembered or PORT


def port_free(host: str, port: int) -> bool:
    """Whether this address can be bound, asked before uvicorn tries.

    Uvicorn's own failure is a raw WinError 10048 several lines into its log,
    which reads as a crash rather than as "one of these is already running".
    Note the missing SO_REUSEADDR: on Windows it lets a second process bind a
    port already in use, which is the opposite of the question being asked.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def dashboard_at(port: int, host: str = HOST) -> bool:
    """Whether the thing holding the port is another copy of this dashboard."""
    import httpx
    probe = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        r = httpx.get(f"http://{probe}:{port}/", timeout=1.5)
    except httpx.HTTPError:
        return False
    return r.status_code == 200 and "valwr live" in r.text


def free_port(host: str, start: int, tries: int = 20) -> int | None:
    """The first bindable port at or after `start`."""
    for port in range(start, start + tries):
        if port_free(host, port):
            return port
    return None


def _open_db():
    """A read connection for the request thread. Short-lived on purpose.

    The poll's connection belongs to the pool thread and cannot be shared;
    these routes are read-only and open their own.
    """
    try:
        from valwr import config
        from valwr.store import schema
        s = config.load(require_key=False)
        if not s.database_path.exists():
            return None
        return schema.connect(s.database_path)
    except Exception:                                # noqa: BLE001
        return None


def recorded(match_id: str) -> bool:
    """Whether this dashboard predicted that match."""
    conn = _open_db()
    if conn is None:
        return False
    try:
        return conn.execute(
            "SELECT 1 FROM live_predictions WHERE match_id = ?",
            (match_id,)).fetchone() is not None
    finally:
        conn.close()


def _review_payload(match_id: str) -> dict:
    """A recorded match: what was predicted, and the result once it lands."""
    conn = _open_db()
    if conn is None:
        return {"status": "error", "message": "no database"}
    try:
        got = review.compare(conn, match_id)
    finally:
        conn.close()
    if got is None:
        return {"status": "error", "message": "that match is not recorded here"}
    return {"status": "match", "state": got["state"], "review": got,
            "top1_rate": None, "fresh": False}


def recent_rows() -> list[dict]:
    """Recently recorded matches, owning the connection it reads them with.

    Deliberately not taken from the live context: the moment the game closes
    there is no context, and that is exactly when a player wants the match they
    just finished. Tying this to the context meant the page offered nothing but
    "VALORANT is not running".
    """
    try:
        from valwr import config
        from valwr.store import schema
        s = config.load(require_key=False)
        if not s.database_path.exists():
            return []
        conn = schema.connect(s.database_path)
        try:
            return review.recent(conn)
        finally:
            conn.close()
    except Exception:                                # noqa: BLE001
        return []                                    # never break the poll


def open_match_tab(base_url: str, match_id: str) -> bool:
    """Open this match's own page, and say so in the console.

    Printed because a tab that fails to appear is otherwise invisible: there
    was no way to tell a browser that refused from a dashboard that never
    tried. Failure is caught rather than raised -- this runs inside the poll
    loop, and losing the websocket because a browser would not launch would
    take the live view down with it.
    """
    url = f"{base_url}m/{match_id}"
    try:
        ok = webbrowser.open(url)
    except Exception as e:                           # noqa: BLE001
        print(f"  could not open a tab for this match ({e}).")
        print(f"  open it yourself: {url}")
        return False
    if ok:
        print(f"  this match has its own tab: {url}")
    else:
        print("  no browser would open a tab for this match.")
        print(f"  open it yourself: {url}")
    return bool(ok)


def _demo_top1() -> float | None:
    """How often the shipped score picks the best player, from the index.

    None when no index is fitted, and the page then says nothing rather than
    quoting a figure it cannot stand behind.
    """
    from valwr.rating import roleindex
    try:
        return roleindex.RoleIndex.load().top1_rate
    except (FileNotFoundError, ValueError, KeyError):
        return None


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
        # Read from the index rather than written here: a literal in this
        # file is how the live view once advertised an accuracy two retrains
        # out of date, with nothing to catch it.
        return {"status": "match", "state": demo_state(conn),
                "top1_rate": _demo_top1(), "fresh": True}
    finally:
        if conn is not None:
            conn.close()


def _replay_payload(match_id: str) -> dict:
    """One finished match, rebuilt as the dashboard would have shown it."""
    import joblib

    from valwr import config
    from valwr.dash.replay import replay_state
    from valwr.rating import potential as pot
    from valwr.rating import roleindex
    from valwr.store import schema

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    bundle = joblib.load(s.database_path.parent.parent / "models" / "model.joblib")
    try:
        index = pot.PerfIndex.load()
    except FileNotFoundError:
        index = None
    try:
        role_index = roleindex.RoleIndex.load()
    except FileNotFoundError:
        role_index = None
    row = conn.execute("SELECT puuid FROM players WHERE lower(tag) = lower(?) "
                       "ORDER BY last_seen_at DESC LIMIT 1",
                       (getattr(s, "riot_tag", "") or "",)).fetchone()
    me = row["puuid"] if row else ""
    try:
        return {"status": "match",
                "state": replay_state(conn, match_id, bundle, index, me,
                                      role_index=role_index),
                # The rate has to come from whichever index scored the players
                # on this page, or a pinned tab quotes one score's accuracy
                # beside another score's numbers.
                "top1_rate": (role_index.top1_rate if role_index
                              else index.top1_rate if index else None),
                "fresh": True}
    finally:
        conn.close()


def poll_and_record(ctx) -> dict | None:
    """One poll, with the prediction logged. Runs on the pool's thread.

    Recording here rather than in the websocket keeps every database call on
    the thread that owns the connection, which is the rule the poll pool
    exists to enforce.
    """
    state = st.poll_once(ctx)
    if state is not None:
        outcomes.record(ctx.conn, state)
    return state


SETTLE_EVERY_SECONDS = 60


def settle_tick(no_fetch: bool = False) -> dict[str, int]:
    """One pass at collecting results, owning everything it needs.

    Deliberately independent of the live context. Settling used to run inside
    the websocket loop, which meant it required both a browser tab to be open
    AND the game to be running -- and when the game was closed the loop gave up
    before it ever reached this step. Finishing a match and quitting VALORANT
    is the most ordinary thing a player does, and it was the one case where the
    result was never fetched.
    """
    from valwr import config
    from valwr.collect.client import HenrikClient
    from valwr.collect.limiter import TokenBucket
    from valwr.store import schema
    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    client = None
    try:
        if not no_fetch:
            try:
                full = config.load()
                client = HenrikClient(full.henrik_api_key, conn=conn,
                                      limiter=TokenBucket(full.requests_per_minute))
            except Exception:                        # noqa: BLE001
                client = None        # no key: the crawler's own matches still settle
        return outcomes.settle_pending(conn, client, s.region)
    finally:
        if client is not None:
            client.close()
        conn.close()


def build_app(no_fetch: bool = False, deadline: float = st.DEFAULT_DEADLINE,
              demo: bool = False, match: str | None = None,
              settle: bool = False):
    """The app. `settle` starts the background collector, and only `main` asks
    for it.

    Off by default because building an app must not have side effects on the
    real database. It did: every test that constructed one started a task that
    opened the live database and wrote to it, which held the write lock and
    killed two long backfills mid-run with "database is locked".
    """
    async def settling(_app):
        """Collect results forever, whatever else the dashboard is doing."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                got = await loop.run_in_executor(None, settle_tick, no_fetch)
                if got.get("settled"):
                    print(f"  scored {got['settled']} finished match(es); "
                          f"their comparisons are ready.")
            except asyncio.CancelledError:
                raise
            except Exception as e:                   # noqa: BLE001
                print(f"  could not collect results this time ({e}).")
            await asyncio.sleep(SETTLE_EVERY_SECONDS)

    @asynccontextmanager
    async def lifespan(_app):
        task = None
        if settle and not (demo or match):
            task = asyncio.create_task(settling(_app))
        yield
        if task is not None:
            task.cancel()
        _app.state.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="valwr live", docs_url=None, redoc_url=None,
                  openapi_url=None,   # minimal surface: two routes, no schema
                  lifespan=lifespan)
    app.add_middleware(LocalOnly)
    app.state.ctx = None
    app.state.error = None
    app.state.demo_state = None
    app.state.replay_state = None
    # Matches this process has already opened a tab for, so reconnecting a
    # page does not reopen one.
    app.state.opened = set()
    app.state.base_url = None       # set by main(), once the port is known
    app.state.settled_at = 0.0

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

    @app.get("/m/{match_id}")
    def pinned(request: Request, match_id: str):
        """One match, on its own address, so a tab can stay on it.

        Serves the same page. It reads the id out of its own path and asks the
        socket for that match rather than the current one. Only matches this
        dashboard recorded resolve: it is not a lookup surface for anything
        else, which is the constraint in docs/ETHICS-AND-TOS.md.
        """
        if not recorded(match_id):
            return PlainTextResponse("no such match on this dashboard",
                                     status_code=404)
        return index(request)

    @app.get("/results")
    def results(request: Request):
        host = request.headers.get("host", HOST)
        return FileResponse(STATIC / "results.html", headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Security-Policy": content_policy(host),
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
        })

    @app.get("/api/scorecard")
    def scorecard():
        """Every recorded prediction, scored. No parameters, nothing to look up."""
        conn = _open_db()
        if conn is None:
            return {"recorded": 0, "settled": 0, "pending": 0,
                    "insights": ["no database yet"], "matches": []}
        try:
            return review.scorecard(conn)
        finally:
            conn.close()

    @app.websocket("/ws")
    async def ws(socket: WebSocket, pinned: str | None = None):
        await socket.accept()
        if pinned:
            # A tab pinned to one match: what was predicted at that loading
            # screen, frozen because that is the claim being scored, with the
            # result filled in once it lands.
            try:
                while True:
                    await socket.send_text(json.dumps(_review_payload(pinned)))
                    await asyncio.sleep(POLL_SECONDS * 4)
            except WebSocketDisconnect:
                return
        if not demo and not match:
            # The first poll resolves ten players and can spend its whole
            # deadline doing it. Without this the page sits on "connecting"
            # and reads as broken rather than busy.
            await socket.send_text(json.dumps(
                {"status": "working",
                 "message": "reading the match and looking up players"}))
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
                    # No game client, which is the normal state right after a
                    # match. The recorded ones still have to be reachable, so
                    # they ride along with the error rather than leaving the
                    # page with nothing but "VALORANT is not running".
                    recent = await loop.run_in_executor(None, recent_rows)
                    await socket.send_text(json.dumps(
                        {"status": "error", "message": app.state.error,
                         "recent": recent}))
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
                    state = await loop.run_in_executor(pool, poll_and_record, ctx)
                except st.NotReady as e:
                    # The client went away -- closed, restarted, or its session
                    # could not be renewed. Drop the context so the next tick
                    # opens a fresh one; the page recovers on its own when the
                    # game comes back, which is the whole point of polling.
                    app.state.ctx = None
                    app.state.error = None
                    # The game being closed is the normal state right after a
                    # match, and the recorded ones have to stay reachable then
                    # above all.
                    recent = await loop.run_in_executor(None, recent_rows)
                    await socket.send_text(json.dumps(
                        {"status": "error", "message": f"{e} Retrying.",
                         "recent": recent}))
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                except Exception as e:                  # noqa: BLE001
                    recent = await loop.run_in_executor(None, recent_rows)
                    await socket.send_text(json.dumps(
                        {"status": "error",
                         "message": f"{type(e).__name__}: {e}",
                         "recent": recent}))
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                if state is None:
                    # The client just left a match: that is the moment it
                    # ended, and the only signal this side has for it. Asking
                    # now saves the player watching "waiting for the result"
                    # while a schedule anchored on the loading screen catches
                    # up.
                    if last is not None:
                        await loop.run_in_executor(
                            pool, outcomes.match_ended, ctx.conn, ctx.client,
                            ctx.settings.region, last)
                    # Between matches, hand over the ones already recorded.
                    # Without this the page says "waiting for a match" and the
                    # game just played is unreachable unless the tab the server
                    # popped open was caught at the time.
                    recent = await loop.run_in_executor(None, recent_rows)
                    await socket.send_text(json.dumps(
                        {"status": "lobby", "recent": recent}))
                    last = None
                else:
                    # Every match gets a tab of its own, opened once. This tab
                    # keeps following the current match; the new one keeps this
                    # match to compare against later.
                    if (app.state.base_url
                            and state["match_id"] not in app.state.opened):
                        app.state.opened.add(state["match_id"])
                        open_match_tab(app.state.base_url, state["match_id"])
                    # The figure has to belong to the score being shown.
                    # Quoting the old index's rate beside a per-role score
                    # would advertise an accuracy this page does not have.
                    top1 = (ctx.role_index.top1_rate if ctx.role_index
                            else ctx.index.top1_rate if ctx.index else None)
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
    ap.add_argument("--port", type=int, default=None,
                    help=f"default {PORT}, or whatever the last run used -- "
                         f"tabs opened then point at that one")
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
    ap.add_argument("--no-tabs", action="store_true",
                    help="do not open a tab of its own for each match")
    args = ap.parse_args(argv)

    # Come back to the port the last run used, so the per-match tabs it opened
    # still resolve. Only when nothing was asked for explicitly.
    last = remembered_port()
    args.port = preferred_port(args.port, last)
    if last is not None and last != PORT and args.port == last:
        print(f"  reusing port {last} from the last run, so tabs opened then "
              f"still work.")

    # A second launch is the common case: the first window is still open, and
    # its browser tab is still connected. Bind-time failure told the user
    # nothing useful, so the two cases are separated here.
    if not port_free(args.host, args.port):
        if dashboard_at(args.port, args.host):
            running = f"http://{HOST}:{args.port}/"
            print(f"  a dashboard is already running on {running}")
            print("  opening that one rather than starting a second copy.")
            print("  close its window first if you want a fresh start.\n")
            if not args.no_browser:
                webbrowser.open(running)
            return 0
        moved = free_port(args.host, args.port + 1)
        if moved is None:
            print(f"  port {args.port} is busy, and so are the next 20.")
            print("  close whatever is using them, or pass --port.")
            return 1
        print(f"  port {args.port} is in use by something else; "
              f"using {moved} instead.")
        print("  tabs opened by an earlier run pointed at the old port and "
              "will not load;")
        print("  close whatever is holding it and restart to get them back.")
        args.port = moved

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
    print(f"  scorecard: {url}results")
    print("  Keep this window open. Ctrl+C to stop.\n")

    # Remembered once the port is settled, so the next run lands here and the
    # tabs this run is about to open keep working.
    remember_port(args.port)

    # Predictions left stranded by the old settle schedule, which spent every
    # retry while the match was still being played. They are still settleable.
    try:
        from valwr import config as _cfg
        from valwr.store import schema as _schema
        _conn = _schema.connect(_cfg.load(require_key=False).database_path)
        freed = outcomes.unstick(_conn)
        _conn.commit()
        # "Played best" changed meaning in 13.06. Re-judge what is already
        # stored so the scorecard is not averaging two definitions.
        moved = outcomes.rescore(_conn)
        if moved:
            print(f"  re-judged {moved} match(es) against the current "
                  f"definition of the best game.")
        _conn.close()
        if freed:
            print(f"  {freed} earlier match(es) can be scored again; their "
                  f"results are fetched shortly.")
    except Exception:                                # noqa: BLE001
        pass                                         # never block a launch

    app = build_app(no_fetch=args.no_fetch, deadline=args.deadline,
                    demo=args.demo, match=args.match, settle=True)
    if not (args.no_tabs or args.no_browser or args.demo or args.match):
        app.state.base_url = url
    server = uvicorn.Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning"))

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

    try:
        server.run()
    except OSError as e:
        # Something grabbed the port between the check above and the bind.
        print(f"\n  could not start on {args.host}:{args.port} -- {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
