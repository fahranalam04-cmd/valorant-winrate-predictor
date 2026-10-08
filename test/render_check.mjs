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
globalThis.location = { host: "127.0.0.1:8788", pathname: "/" };
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
{
  const runs = state.players.filter(p => p.streak);
  ck("a run is badged on each row that has one, and no other",
     runs.length > 0 && (out.match(/class="run (won|lost)"/g) || []).length === runs.length
     && runs.every(p => out.includes(`${p.streak.result} ${p.streak.count} in a row`)));
  // The two-column board has no room beside the longest Riot IDs.
  const names = out.match(/<div class="nm">[\s\S]*?<\/div>/g) || [];
  ck("on the line under the name, never beside it",
     names.length === state.players.length
     && names.every(n => !n.includes('class="run'))
     && (out.match(/<span class="ag">[^<]*<\/span><span class="run/g) || []).length === runs.length);
}
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
// Damage per round, not combat score: patch 13.06 removed ACS from the game's
// own scoreboard, so it left this one too.
ck("damage per round on row", out.includes(first.career.adr.toFixed(1)));
ck("and the row no longer prints combat score",
   !/\bACS\b/.test(out) && !out.includes(first.career.acs.toFixed(1)));
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
ck("panel portrait", p.includes(`src="/agents/${first.agent_id}-portrait.png"`));
// Artwork is addressed against the directory the page is served from, which is
// "/" here. A hardcoded "/agents/" would break the demo under GitHub Pages'
// /valorant-winrate-predictor/; a plain relative "agents/" broke a match's own
// page at /m/<id>. Both cases are checked where pinnedMatch() is tested.
ck("art is addressed from the page's own directory",
   p.includes('src="/agents/') && artURL({ agent_id: "x" }, "row")
     === "/agents/x-row.png");
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

// Between matches the page must still offer a way back into the game just
// played. Without this it says "waiting for a match" and the only route back
// is a pop-up tab the browser may have refused or the player may have closed.
render({ status: "lobby", recent: [
  { match_id: "aaa-111", map: "Split", made_at: Math.floor(Date.now() / 1000) - 600,
    settled: true, own_won: 0, correct: 1, score: "6-13" },
  { match_id: "bbb-222", map: "Lotus", made_at: Math.floor(Date.now() / 1000) - 60,
    settled: false, own_won: null, correct: null, score: null },
]});
{
  const q = els.stage.innerHTML;
  ck("recorded matches are listed between games",
     /href="\/m\/aaa-111"/.test(q) && /href="\/m\/bbb-222"/.test(q));
  ck("each says how it went", /lost/.test(q) && /6-13/.test(q)
     && /called it/.test(q));
  ck("one still waiting says so", /waiting for the result/.test(q));
  ck("and the scorecard is reachable", /href="\/results"/.test(q));
}
render({ status: "lobby" });
ck("no recorded matches yet is not an empty box",
   !/recentrow/.test(els.stage.innerHTML));
render({ status: "error", message: "VALORANT is not running" });
ck("error branch", /not running/.test(els.stage.innerHTML));

// Closing the game is what a player does straight after a match, so this is
// exactly when the recorded ones have to stay reachable.
render({ status: "error", message: "VALORANT is not running", recent: [
  { match_id: "ccc-333", map: "Ascent", made_at: Math.floor(Date.now() / 1000) - 900,
    settled: true, own_won: 1, correct: 1, score: "13-8" },
]});
ck("with the game closed, the recorded matches are still listed",
   /href="\/m\/ccc-333"/.test(els.stage.innerHTML)
   && /not running/.test(els.stage.innerHTML));

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
   (body.style._v["--mapart"] || "") === `url("/maps/${slug}-splash.jpg")`);

// Every map has to swap the art, which is the whole point of the theme.
for (const name of ["Pearl", "Bind", "Fracture", "Icebox"]){
  const other = JSON.parse(JSON.stringify(state));
  other.map = name;
  render({ status: "match", state: other, top1_rate: 0.296 });
  ck(`${name} paints its own art`,
     (body.style._v["--mapart"] || "")
       .includes(`maps/${name.toLowerCase()}-splash.jpg`));
}

const nameless = JSON.parse(JSON.stringify(state));
nameless.map = null;
render({ status: "match", state: nameless, top1_rate: 0.296 });
ck("an unknown map paints nothing rather than a broken url",
   body.style._v["--mapart"] === "none");

render({ status: "match", state, top1_rate: 0.296 });

// --- hostile input ------------------------------------------------------
// Other players choose their own names, and every number here was stored from
// an API response that SQLite did not type-check. Plant markup in every field
// the page prints -- text and numbers alike -- and require that none of it
// arrives as markup, while proving it did flow through (escaped).
{
  const EVIL = '<img src=x onerror=alert(1)>';
  const h = JSON.parse(JSON.stringify(state));
  Object.assign(h, { mode: EVIL, confidence: EVIL, coverage: EVIL,
                     model: EVIL, warnings: [EVIL] });
  for (const g of h.parties || []) g.label = EVIL;
  for (const q of h.players){
    Object.assign(q, { name: EVIL, agent: EVIL, reason: EVIL, role: EVIL });
    if (q.score !== null) q.score = EVIL;
    if (q.rank) Object.assign(q.rank, { short: EVIL, name: EVIL });
    if (q.career) Object.assign(q.career, { games: EVIL, kills: EVIL, deaths: EVIL, assists: EVIL });
    if (q.flag) q.flag.note = EVIL;
    const d = q.detail;
    if (!d) continue;
    for (const c of d.components || []) Object.assign(c, { label: EVIL, note: EVIL });
    // Forced into the branch that prints the gate: it only renders for a
    // player whose map record is too thin to count, and the demo's is not --
    // so planting markup there alone tested nothing.
    Object.assign(d.map, { name: EVIL, gate: EVIL, games: d.map.games || 3,
                           counts_toward_score: false });
    for (const a of d.map.agents || []) Object.assign(a, { agent: EVIL, games: EVIL });
    for (const f of d.form || []) Object.assign(f, { map: EVIL, agent: EVIL, ago: EVIL, kills: EVIL });
    if (d.freshness) Object.assign(d.freshness, { label: EVIL, games_known: EVIL });
  }
  escape();
  render({ status: "match", state: h, top1_rate: 0.296 });
  const rows = els.stage.innerHTML + els.sub.innerHTML;
  fire(h.players.find(q => q.detail).puuid);
  const panel = els.stage.innerHTML;
  const all = rows + panel;
  ck("hostile values never render as markup", !all.includes("<img src=x"),
     all.slice(Math.max(0, all.indexOf("<img src=x") - 80), all.indexOf("<img src=x") + 40));
  ck("hostile values still reach the page, escaped",
     (all.match(/&lt;img src=x/g) || []).length > 20);
  escape();
  render({ status: "error", message: EVIL });
  ck("an error message is escaped too", !els.stage.innerHTML.includes("<img src=x"));
  render({ status: "match", state, top1_rate: 0.296 });
}

// A tab pinned to one match reads the id out of its own address.
location.pathname = "/m/abc%20123";
ck("a pinned tab reads its match from the address", pinnedMatch() === "abc 123");

// Artwork is addressed relative to the page, and a match's own page is served
// one level deeper at /m/<id>. Plain relative paths sent every portrait and the
// map background to /m/agents/... and /m/maps/..., so the saved tab -- the one
// kept specifically to read after the game -- was the only page with no art.
{
  const art = artURL({ agent_id: "jett-uuid" }, "row");
  ck("a pinned tab resolves artwork above /m/", art === "/agents/jett-uuid-row.png");
  ck("and never into /m/", !art.includes("/m/"));
}
location.pathname = "/valorant-winrate-predictor/m/xyz";
ck("the same holds under a subdirectory, as on Pages",
   artURL({ agent_id: "sage" }, "row")
   === "/valorant-winrate-predictor/agents/sage-row.png");
location.pathname = "/valorant-winrate-predictor/";
ck("the demo keeps its subdirectory prefix",
   artURL({ agent_id: "sage" }, "row")
   === "/valorant-winrate-predictor/agents/sage-row.png");
location.pathname = "/";
ck("the live tab is not pinned to anything", pinnedMatch() === "");
ck("and addresses artwork beside itself",
   artURL({ agent_id: "sage" }, "row") === "/agents/sage-row.png");

// The client reports a mode as an identifier, not a label.
ck("the mode is readable", modeLabel("BombGameMode.BombGameMode_C") === "Standard");
ck("an unknown mode still prints something",
   modeLabel("/Game/GameModes/Weird/WeirdGameMode.WeirdGameMode_C") === "Weird");
ck("a missing mode does not print undefined", modeLabel(null) === "—");

// The first thing the socket sends, before a poll that may take 25 seconds.
render({ status: "working", message: "reading the match and looking up players" });
ck("the opening message says what it is doing",
   els.map.textContent === "STARTING"
   && /reading the match/.test(els.stage.innerHTML));
// --- a match that has been played ---------------------------------------
// The pinned tab keeps the prediction and fills in the result, which is the
// whole point of a tab per match.
{
  const rv = {
    settled: true, own_team: state.own_team,
    predicted: { own_probability: 0.573 }, correct: 1,
    actual: { winner: state.own_team, own_won: 1, score: "13-9" },
    top_pick: { hit: 0, picked: "Meridian#na1", actually_best: "Yarrow#na2" },
    summary: { winner_called: 1, top_pick_hit: 0, rated: 8, within_one: 5,
               order: 0.42 },
    players: state.players.map((q, i) => Object.assign({}, q, {
      played: true, predicted_score: q.score,
      predicted_rank: q.score === null ? null : i + 1,
      acs: 210 + i, kd: 1.15, kills: 17, deaths: 14, assists: 5,
      headshot_rate: 0.241, actual_rank: 10 - i,
      // What the page knew beforehand, which is what the block compares
      // the match against. Player 3 has none of it on purpose.
      career_acs: i === 3 ? null : 198 + i, career_kd: i === 3 ? null : 1.04,
      career_hs: i === 3 ? null : 0.212, career_games: i === 3 ? 0 : 40 + i,
      recent_kd: 1.11,
      // One player below their own average, because a bare "8" in red
      // once read as a gain of eight.
      acs_delta: i === 3 ? null : i === 4 ? -8.0 : 12.0,
      kd_delta: i === 3 ? null : 0.11,
      hs_delta: i === 3 ? null : 0.029,
      versus_usual: i === 3 ? null : "above their usual",
      place_delta: q.score === null ? null : (i + 1) - (10 - i) })),
  };
  render({ status: "match", state, review: rv, top1_rate: 0.296 });
  const q = els.stage.innerHTML;
  ck("a played match shows the result", /class="result won"/.test(q) && q.includes("13-9"));
  ck("a live match links to its own page",
     /class="ownpage"/.test(q) && /href="\/m\/demo"/.test(q));
  ck("and whether the call was right", /called it/.test(q));
  ck("and who actually played best", q.includes("Yarrow#na2"));
  ck("the summary says what it got right",
     /Winner <span class="beat">right/.test(q) && /5 of 8 players/.test(q)
     && /order \+0.42/.test(q));
  ck("every player who played gets their own block",
     (q.match(/class="pblock/g) || []).length === state.players.length);
  ck("both teams are grouped",
     />Your team</.test(q) && />Enemy team</.test(q));
  ck("each block compares before against this match",
     /K \/ D \/ A/.test(q) && /Headshots/.test(q)
     && />Before/.test(q) && />This match</.test(q) && />Change</.test(q));
  ck("the career line is the one the page held beforehand",
     q.includes("198.0") && q.includes("1.04") && q.includes("21.2%"));
  ck("changes are signed and coloured",
     /class="d up">\+12/.test(q) && /\+2.9pt/.test(q));
  ck("a loss keeps its minus sign",
     /class="d dn">−8/.test(q));
  ck("placing is spelled out rather than bracketed",
     /exactly where the score put them|places? (better|worse) than/.test(q));
  // Both rankings run across the lobby. "#6 of five" was the bug.
  ck("a rank says what it is out of",
     /Ranked <b>#1<\/b> of the 8 players/.test(q)
     && /of 10 on match\s+impact/.test(q));
  // The order has been by match impact since patch 13.06 took combat score
  // off the game's scoreboard; the sentences kept naming the old measure.
  ck("placing is named for what it is measured by",
     !/on combat\s+score/.test(q) && /Placing\s+is by match impact/.test(q));
  ck("and how they played against their own average is stated",
     /above their usual/.test(q));
  ck("a player with no history still renders, saying nothing",
     (q.match(/class="d mute">/g) || []).length >= 3);
  ck("the reason the score gave is kept beside the outcome",
     /class="pwhy"/.test(q));

  escape();
  ck("the result survives closing the card",
     /class="result won"/.test(els.stage.innerHTML)
     && /class="compare"/.test(els.stage.innerHTML));
  fire(first.puuid);
  ck("and survives opening another player",
     /class="compare"/.test(els.stage.innerHTML));
  const nextMatch = JSON.parse(JSON.stringify(state));
  nextMatch.match_id = "a-different-match";
  render({ status: "match", state: nextMatch, top1_rate: 0.296 });
  ck("a different match does not inherit the last result",
     !/class="compare"/.test(els.stage.innerHTML));
  render({ status: "match", state, review: rv, top1_rate: 0.296 });

  // With the game's own Performance Score, the lobby is ranked by it and
  // says so, and each block shows the number the end-of-game screen did.
  rv.best_by = "performance score";
  rv.players.forEach((q, i) => { q.performance_score = 300 - i * 10; });
  render({ status: "match", state, review: rv, top1_rate: 0.296 });
  const ps = els.stage.innerHTML;
  ck("ranked on Performance Score when the game gave one",
     /of 10 on Performance\s+Score/.test(ps) && !/on match\s+impact/.test(ps));
  ck("each block shows its Performance Score", /<td>Performance Score<\/td><td>—<\/td><td>300<\/td>/.test(ps));
  ck("and the note names the measure", /the game's own Performance Score/.test(ps));
  delete rv.best_by;
  rv.players.forEach(q => { delete q.performance_score; });

  rv.settled = false;
  render({ status: "match", state, review: rv, top1_rate: 0.296 });
  ck("before the result lands it says it is waiting",
     /class="result waiting"/.test(els.stage.innerHTML));
}

render({ status: "match", state, top1_rate: 0.296 });

console.log(bad ? `\n${bad} FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
