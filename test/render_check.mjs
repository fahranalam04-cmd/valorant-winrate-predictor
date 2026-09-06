// Execute the dashboard page's own JavaScript against a real state dictionary
// and assert on the HTML it builds.
//
//     node test/render_check.mjs <index.html> <state.json>
//
// The page is the one part of this project a Python test cannot reach, and it
// is where a rename in `poll_once` would surface as a blank column rather than
// an exception. The state comes from `valwr.dash.demo.demo_state()`, which
// test_state.py separately pins to the exact key set `poll_once` returns, so
// this cannot drift from the real shape.
//
// Every branch is exercised, not just the happy path: lobby, error, a match
// with no prediction yet, a non-bomb mode, players with no history, and both
// the deep and shallow history cases -- the shallow one being the common case,
// since the median player in this database has 8 stored matches.
//
// Prints one line per check and exits non-zero if any fail.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const [, , pagePath, statePath] = process.argv;
const html = readFileSync(pagePath, "utf8");
const code = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const state = JSON.parse(readFileSync(statePath, "utf8"));

const els = {};
for (const id of ["pulse", "map", "sub", "stage", "foot", "themes"])
  els[id] = { id, innerHTML: "", textContent: "", className: "", scrollTop: 0,
              addEventListener(t, fn){ (this._h ||= {})[t] = fn; } };
const handlers = {};
const body = { dataset: {}, style: { _v: {},
  setProperty(k, v){ this._v[k] = v; } } };
globalThis.document = {
  body,
  getElementById: id => els[id] || null,
  addEventListener: (t, fn) => { (handlers[t] ||= []).push(fn); },
  // The picker rewrites its own aria-pressed after every switch; the harness
  // only needs that call not to throw.
  querySelectorAll: () => [],
};
// localStorage does not exist under node, so reading it throws -- which is
// exactly what a private window does, and the page must fall back not break.
globalThis.WebSocket = class { constructor(){ this.onopen = null; } };
globalThis.location = { host: "127.0.0.1:8788" };
globalThis.setTimeout = () => {};

vm.runInThisContext(code);
render({ status: "match", state, top1_rate: 0.296 });
const out = els.stage.innerHTML;

let bad = 0;
const ck = (n, ok, extra) => { if (!ok) bad++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}${!ok && extra ? "  -> " + extra : ""}`); };

const known = state.players.filter(p => p.score !== null);
const unknown = state.players.filter(p => p.score === null);
const first = known[0];

ck("map in the header", els.map.textContent === state.map);
ck("one button per player",
   (out.match(/<button class="row/g) || []).length === state.players.length);
ck("attack column present", /class="team atkc"/.test(out));
ck("defence column present", /class="team defc"/.test(out));
ck("side labels", /attacking · first half/.test(out) && /defending · first half/.test(out));
ck("known rows marked", (out.match(/class="row known/g) || []).length === known.length);
ck("unknown rows marked", (out.match(/class="row unknown/g) || []).length === unknown.length);
ck("YOU marker", /class="tag">YOU/.test(out));
ck("acs on row", out.includes(first.career.acs.toFixed(1)));
ck("k/d on row", out.includes(first.career.kd.toFixed(2)));
ck("last-20 on row", out.includes(first.recent.kd.toFixed(2)));
ck("hs% on row", out.includes((first.career.headshot_rate * 100).toFixed(1) + "%"));
ck("no K/D/A triple on row",
   !out.includes(`${first.career.kills}/${first.career.deaths}/${first.career.assists}</b>`));
// Rows take the small -row art and only the panel hero takes the tall
// portrait. Ten full portraits on screen was 7 MB of decode on first paint.
ck("rows use the row-sized art", out.includes(`-row.png`));
ck("rows do NOT pull the tall portrait", !out.includes(`-portrait.png`));
ck("odds bar", /class="oddsbar"/.test(out));
ck("factors panel", /What moves the prediction/.test(out));
ck("empty panel prompts", /Select any player/.test(out));
ck("every distinct reason rendered",
   [...new Set(state.players.map(p => p.reason))].every(r => out.includes(r)));

// --- interaction -------------------------------------------------------
const fire = puuid => { for (const fn of handlers.click || [])
  fn({ target: { closest: s => s.includes("button.row")
        ? { dataset: { puuid } } : null } }); };
// Clicking is a toggle, so selecting a player who may already be selected has
// to clear first or the test silently asserts against a closed panel.
const escape = () => { for (const fn of handlers.keydown || [])
  fn({ key: "Escape", target: {} }); };
const select = puuid => { escape(); fire(puuid); };

fire(first.puuid);
let p = els.stage.innerHTML;
ck("panel opens on click", p.includes(first.name) && /class="panel"/.test(p));
ck("panel portrait", p.includes(`/agents/${first.agent_id}-portrait.png`));
ck("panel score", new RegExp(`<b style="color:[^"]+">${first.score}</b>`).test(p));
ck("record & form section", /record &amp; form/.test(p));
ck("on-map section", p.includes(`on ${state.map}`));
ck("last matches section", /last matches/.test(p));
ck("row shows selected", /class="row known sel"/.test(p));
ck("aria-pressed set", /aria-pressed="true"/.test(p));

// a player whose stored history is shorter than the window
const shallow = known.find(x => x.detail && x.detail.career.games <= 20);
if (shallow){
  select(shallow.puuid);
  const q = els.stage.innerHTML;
  ck("short history collapses the two columns",
     /fall inside the last/.test(q) && !/>last \d+<\/th>/.test(q));
}

// a deep-history player keeps both columns
const deep = known.find(x => x.detail && x.detail.career.games > 20);
if (deep){
  select(deep.puuid);
  const q = els.stage.innerHTML;
  ck("deep history shows career vs form", /<th>last \d+<\/th>/.test(q));
}

// unknown player
const u = unknown[0];
if (u){
  select(u.puuid);
  ck("unknown player explained, not crashed",
     /No match history/.test(els.stage.innerHTML));
}

// deselect
select(first.puuid); fire(first.puuid);
ck("clicking twice closes the panel", /Select any player/.test(els.stage.innerHTML));

// esc
select(first.puuid);
for (const fn of handlers.keydown || []) fn({ key: "Escape", target: {} });
ck("escape closes the panel", /Select any player/.test(els.stage.innerHTML));

// lobby / error branches must not throw
render({ status: "lobby" });
ck("lobby branch", els.map.textContent === "STANDBY");
render({ status: "error", message: "VALORANT is not running" });
ck("error branch", /not running/.test(els.stage.innerHTML));

// non-standard mode drops the side labels
const dm = JSON.parse(JSON.stringify(state));
dm.standard_mode = false;
render({ status: "match", state: dm, top1_rate: 0.296 });
ck("no side labels outside bomb defusal",
   !/attacking · first half/.test(els.stage.innerHTML));

// no prediction yet
const np = JSON.parse(JSON.stringify(state));
np.prediction = null;
render({ status: "match", state: np, top1_rate: 0.296 });
ck("missing prediction says so",
   /Not enough of the roster/.test(els.stage.innerHTML));

// --- themes ------------------------------------------------------------
render({ status: "match", state, top1_rate: 0.296 });
ck("picker built", /data-theme="midnight"/.test(els.themes.innerHTML));
ck("five themes offered",
   (els.themes.innerHTML.match(/<button data-theme=/g) || []).length === 5);
ck("falls back to midnight when storage is unreadable",
   body.dataset.theme === "midnight");
ck("default layout is stacked", body.dataset.layout === "stack");

applyTheme("splash");
ck("switching sets the theme", body.dataset.theme === "splash");
ck("splash switches the layout too", body.dataset.layout === "split");
ck("splash paints the map",
   (body.style._v["--mapart"] || "")
     .includes(`/maps/${state.map.toLowerCase()}-splash.jpg`));

applyTheme("tactical");
ck("tactical is also split", body.dataset.layout === "split");
applyTheme("bone");
ck("daylight stacks again", body.dataset.layout === "stack");
applyTheme("not-a-theme");
ck("an unknown theme falls back rather than blanking the page",
   body.dataset.theme === "midnight");
applyTheme("midnight");

console.log(bad ? `\n${bad} FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
