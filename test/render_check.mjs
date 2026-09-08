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
// Rank and party, the two things a lobby is read for before the numbers.
ck("every player shows a rank",
   (out.match(/class="rank /g) || []).length === state.players.length);
ck("the rank is the short form", out.includes(`>${first.rank.short}<`));
{
  const grouped = (state.parties || []).reduce((n, g) => n + g.members.length, 0);
  ck("a marker for every partied player",
     (out.match(/class="party"/g) || []).length === grouped);
  const inferred = (state.parties || []).some(g => g.source === "inferred");
  ck(inferred ? "inferred parties are qualified in the header"
              : "exact parties need no qualifier",
     /class="partynote"/.test(out) === inferred);
}
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
ck("panel body is two grouped columns",
   (p.match(/class="pcol"/g) || []).length === 2);
// The per-map block passes no form window, so the "same games" note must not
// render under it. It printed "fall inside the last 0" beneath every map.
ck("the map block never claims a form window", !/the last 0/.test(p));

// a player whose stored history is shorter than the window
const shallow = known.find(x => x.detail && x.detail.career.games <= 20);
if (shallow){
  select(shallow.puuid);
  const q = els.stage.innerHTML;
  ck("short history collapses the two columns",
     /fall inside the last/.test(q) && !/>last \d+<\/th>/.test(q));
  ck("and names the real window, not zero",
     /fall inside the last 20/.test(q));
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

// --- which side a factor favours ---------------------------------------
// `predict.top_factors` signs toward TEAM_A (Blue), not toward the viewer.
// Reading the raw sign as "toward you" was backwards in every match played on
// Red -- half of them -- and nothing caught it until a replay showed five
// positive factors labelled "toward you" beside a 37.9% chance of winning.
{
  const onBlue = JSON.parse(JSON.stringify(state));
  onBlue.own_team = "Blue"; onBlue.enemy_team = "Red";
  onBlue.prediction.factors = [{name: "d_acs", value: 0.2}];
  render({ status: "match", state: onBlue, top1_rate: 0.296 });
  ck("a positive factor helps you when you are Blue",
     /toward you/.test(els.stage.innerHTML));

  const onRed = JSON.parse(JSON.stringify(onBlue));
  onRed.own_team = "Red"; onRed.enemy_team = "Blue";
  render({ status: "match", state: onRed, top1_rate: 0.296 });
  ck("the same factor helps THEM when you are Red",
     /toward them/.test(els.stage.innerHTML)
     && !/toward you/.test(els.stage.innerHTML));

  const negRed = JSON.parse(JSON.stringify(onRed));
  negRed.prediction.factors = [{name: "d_acs", value: -0.2}];
  render({ status: "match", state: negRed, top1_rate: 0.296 });
  ck("and a negative factor helps you when you are Red",
     /toward you/.test(els.stage.innerHTML));
}
render({ status: "match", state, top1_rate: 0.296 });

// --- the map background ------------------------------------------------
render({ status: "match", state, top1_rate: 0.296 });
const slug = state.map.toLowerCase().replace(/[^a-z0-9]/g, "");
ck("the background is the map being played",
   (body.style._v["--mapart"] || "").includes(`/maps/${slug}-splash.jpg`));

// Every map has to swap the art, which is the whole point of the theme.
for (const name of ["Pearl", "Bind", "Fracture", "Icebox"]){
  const other = JSON.parse(JSON.stringify(state));
  other.map = name;
  render({ status: "match", state: other, top1_rate: 0.296 });
  ck(`${name} paints its own art`,
     (body.style._v["--mapart"] || "")
       .includes(`/maps/${name.toLowerCase()}-splash.jpg`));
}

const nameless = JSON.parse(JSON.stringify(state));
nameless.map = null;
render({ status: "match", state: nameless, top1_rate: 0.296 });
ck("an unknown map paints nothing rather than a broken url",
   body.style._v["--mapart"] === "none");

render({ status: "match", state, top1_rate: 0.296 });

console.log(bad ? `\n${bad} FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
