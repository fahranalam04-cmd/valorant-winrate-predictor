"""Build the interactive dashboard demo that GitHub Pages serves.

    python tools/build_pages_demo.py [--out site]

The real page, fed the invented demo match instead of the local server.
Nothing is re-implemented: `valwr/dash/static/index.html` is copied whole, and
one script is placed ahead of its own, standing in for the websocket the page
would open to `python -m valwr.dash`. Because the page's script is untouched,
what the demo shows is what the dashboard shows.

The players are the invented ones from `valwr/dash/demo.py`. No real gamertag,
PUUID or statistic is involved; the only database table read is the static
agent list, and only so the artwork resolves.

Controls change what changes in a real match: the map, your side, agent select
or in game, how much of the lobby is known, and waiting in the menus. `?clean`
hides them, which is how the README images are captured. The same switches are
available to scripts as `window.__demo.set(key, value)`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from valwr.dash.demo import demo_state

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "valwr" / "dash" / "static" / "index.html"
AGENTS = ROOT / "valwr" / "dash" / "static" / "agents"
MAPS = ROOT / "valwr" / "dash" / "static" / "maps"
INDEX = ROOT / "models" / "perf_index.json"
REPO_URL = "https://github.com/fahranalam04-cmd/valorant-winrate-predictor"

# Written into every build, and required before an existing directory is
# replaced -- so a mistyped --out cannot wipe something that is not a demo.
MARKER = ".valwr-demo"

# The competitive pool, offered where the splash art exists locally.
POOL = ["Ascent", "Abyss", "Bind", "Breeze", "Corrode", "Fracture", "Haven",
        "Icebox", "Lotus", "Pearl", "Split", "Sunset"]

# The demo page makes no network request at all, so it can say so.
POLICY = ("default-src 'none'; script-src 'unsafe-inline'; "
          "style-src 'unsafe-inline'; img-src 'self'; base-uri 'none'; "
          "form-action 'none'")

SHIM = r"""<style>
.demo-bar{position:fixed;left:12px;bottom:12px;z-index:50;display:flex;
  flex-wrap:wrap;align-items:center;gap:8px 16px;max-width:calc(100vw - 24px);
  padding:10px 16px 10px 14px;background:rgba(7,10,14,.94);
  border:1px solid rgba(255,255,255,.14);border-left:3px solid var(--atk,#ff4655);
  color:var(--bone,#ece8e1);font:12px/1.35 Bahnschrift,"DIN Alternate","Segoe UI",system-ui,sans-serif}
.demo-bar b{color:var(--atk,#ff4655);font-weight:600;text-transform:uppercase;
  letter-spacing:.14em;font-size:11px}
.demo-bar label{display:flex;align-items:center;gap:6px;color:var(--slate,#8b97a3);
  text-transform:uppercase;letter-spacing:.1em;font-size:10.5px}
.demo-bar select{background:#0f151d;color:var(--bone,#ece8e1);
  border:1px solid rgba(255,255,255,.22);padding:3px 6px;font:inherit;
  font-size:12px;text-transform:none;letter-spacing:0}
.demo-bar select:focus-visible{outline:2px solid var(--atk,#ff4655);outline-offset:1px}
.demo-bar .fine{flex-basis:100%;color:var(--slate,#8b97a3);font-size:11px}
.demo-bar a{color:var(--bone,#ece8e1)}
.demo-room{height:150px}
body.demo-clean .demo-bar,body.demo-clean .demo-room{display:none}
@media (max-width:720px){.demo-bar{left:6px;right:6px;bottom:6px;gap:6px 10px}
  .demo-room{height:230px}}
</style>
<script>
(function(){
"use strict";
const BASE = __STATE__;
const MAPS = __MAPS__;
const TOP1 = __TOP1__;
const qs = new URLSearchParams(location.search);
const view = {map: MAPS.includes(qs.get("map")) ? qs.get("map") : BASE.map,
              side: qs.get("side") === "Red" ? "Red" : "Blue",
              phase: qs.get("phase") === "pregame" ? "pregame" : "coregame",
              known: qs.get("known") === "thin" ? "thin" : "full",
              status: qs.get("status") === "lobby" ? "lobby" : "match"};
let socket = null;

/* The four switches, applied to a copy of the invented match. */
function build(){
  const s = JSON.parse(JSON.stringify(BASE));
  s.map = view.map;
  s.phase = view.phase;
  for (const p of s.players) if (p.detail) p.detail.map.name = view.map;

  if (view.known === "thin"){
    // A lobby of strangers: only four of the ten have history. Parties are
    // inferred from shared history, so nobody unknown can be in one.
    const keep = new Set(["demo-00", "demo-03", "demo-05", "demo-08"]);
    for (const p of s.players) if (!keep.has(p.puuid))
      Object.assign(p, {score: null, career: null, recent: null, detail: null,
                        flag: null, reason: "no history"});
    s.parties = s.parties.filter(g => g.members.every(m => keep.has(m)));
    s.coverage = keep.size;
    s.confidence = "low";
  }

  const you = s.players.find(p => p.team === view.side);
  for (const p of s.players) p.is_you = p === you;
  s.own_team = view.side;
  s.enemy_team = view.side === "Blue" ? "Red" : "Blue";
  const blue = BASE.prediction.win_probability;
  s.prediction.win_probability = blue;
  s.prediction.own_probability = view.side === "Blue" ? blue : 1 - blue;
  return {status: "match", state: s, top1_rate: TOP1, fresh: false};
}

function send(){
  if (!socket || !socket.onmessage) return;
  const msg = view.status === "lobby" ? {status: "lobby"} : build();
  socket.onmessage({data: JSON.stringify(msg)});
}

/* Stands in for the connection to `python -m valwr.dash`. The page assigns
   its handlers after constructing the socket, so delivery waits a tick. */
window.WebSocket = class {
  constructor(){ socket = this; setTimeout(() => { if (this.onopen) this.onopen(); send(); }, 50); }
  close(){}
};

window.__demo = {
  set(key, value){ if (key in view){ view[key] = value; sync(); send(); } },
  view: () => ({...view}),
};

const bar = document.createElement("div");
bar.className = "demo-bar";
bar.setAttribute("role", "region");
bar.setAttribute("aria-label", "Demo controls");
const option = (v, text) => `<option value="${v}">${text}</option>`;
bar.innerHTML = `<b>Interactive demo</b>
  <label>Map <select data-k="map">${MAPS.map(m => option(m, m)).join("")}</select></label>
  <label>You are <select data-k="side">${option("Blue", "Blue · defending")}${option("Red", "Red · attacking")}</select></label>
  <label>Phase <select data-k="phase">${option("coregame", "In game")}${option("pregame", "Agent select")}</select></label>
  <label>Players known <select data-k="known">${option("full", "8 of 10")}${option("thin", "4 of 10")}</select></label>
  <label>Status <select data-k="status">${option("match", "In a match")}${option("lobby", "In the menus")}</select></label>
  <span class="fine">Invented players. Click any row for the full card; Esc closes it.
    The real dashboard runs on your own PC while you play &mdash;
    <a href="__REPO__">source and setup</a>.
    Agent and map art &copy; Riot Games. Not affiliated with or endorsed by Riot Games.</span>`;

function sync(){
  for (const el of bar.querySelectorAll("select")) el.value = view[el.dataset.k];
}
for (const el of bar.querySelectorAll("select"))
  el.addEventListener("change", () => window.__demo.set(el.dataset.k, el.value));
sync();

const room = document.createElement("div");
room.className = "demo-room";
document.body.appendChild(room);
document.body.appendChild(bar);
if (qs.has("clean")) document.body.classList.add("demo-clean");
})();
</script>
"""


def _slug(name: str) -> str:
    """The page's own rule: lowercase, anything but a-z0-9 removed."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _json_for_script(value) -> str:
    # "</" inside a script would end it early; JSON allows the escaped form.
    return json.dumps(value).replace("</", "<\\/")


def build(out: Path, conn=None) -> dict:
    """Write the demo into `out`. Returns what was written."""
    if out.exists():
        if not (out / MARKER).exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not a demo build; refusing "
                             f"to replace it")
        shutil.rmtree(out)
    (out / "agents").mkdir(parents=True)
    (out / "maps").mkdir()

    state = demo_state(conn)
    maps = [m for m in POOL if (MAPS / f"{_slug(m)}-splash.jpg").exists()]
    if state["map"] not in maps:
        maps.insert(0, state["map"])
    top1 = None
    if INDEX.exists():
        top1 = json.loads(INDEX.read_text(encoding="utf-8")).get("top1_rate")

    written = []
    for p in state["players"]:
        for kind in ("row", "portrait"):
            src = AGENTS / f"{p['agent_id']}-{kind}.png"
            if p.get("agent_id") and src.exists():
                shutil.copy2(src, out / "agents" / src.name)
                written.append(src.name)
    for m in maps:
        src = MAPS / f"{_slug(m)}-splash.jpg"
        if src.exists():
            shutil.copy2(src, out / "maps" / src.name)
            written.append(src.name)

    page = PAGE.read_text(encoding="utf-8")
    shim = (SHIM.replace("__STATE__", _json_for_script(state))
                .replace("__MAPS__", _json_for_script(maps))
                .replace("__TOP1__", _json_for_script(top1))
                .replace("__REPO__", REPO_URL))
    at = page.index("<script>")
    html = page[:at] + shim + page[at:]
    html = html.replace(
        "<head>", f'<head>\n<meta http-equiv="Content-Security-Policy" '
                  f'content="{POLICY}">', 1)
    html = re.sub(r"<title>(.*?)</title>",
                  lambda m: f"<title>{m.group(1)} — interactive demo</title>",
                  html, count=1)
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / MARKER).write_text("built by tools/build_pages_demo.py\n",
                              encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"out": out, "assets": written, "maps": maps, "bytes": size,
            "art": bool(written)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_pages_demo")
    ap.add_argument("--out", default=str(ROOT / "site"))
    args = ap.parse_args(argv)

    conn = None
    try:
        from valwr import config
        from valwr.store import schema
        s = config.load(require_key=False)
        if s.database_path.exists():
            conn = schema.connect(s.database_path)
    except Exception:                                # noqa: BLE001
        conn = None
    info = build(Path(args.out), conn)
    print(f"wrote {info['out']}  ({info['bytes'] / 1e6:.1f} MB, "
          f"{len(info['assets'])} art files, maps: {', '.join(info['maps'])})")
    if not info["art"]:
        print("  no artwork found -- run tools/fetch_agent_art.py first, or the "
              "demo shows lettered tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
