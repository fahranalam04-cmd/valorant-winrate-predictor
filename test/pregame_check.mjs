// Agent select, rendered by the dashboard page's own JavaScript.
//
//     node test/pregame_check.mjs <index.html> <state.json> <match.json>
//
// The state is `valwr.dash.demo.demo_state(phase="pregame")`: your team only,
// two players still picking, one still being looked up. The second is the same
// match once it has loaded, for the moment this tab moves on to it. Agent select leaves
// about a minute to lock in, so the checks are about what can be read at a
// glance -- every number on the card, and the difference between a player
// still being looked up and one with nothing to find.
//
// Prints one line per check and exits non-zero if any fail.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const [, , pagePath, statePath, matchPath] = process.argv;
const html = readFileSync(pagePath, "utf8");
const code = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const state = JSON.parse(readFileSync(statePath, "utf8"));

const els = {};
for (const id of ["pulse", "map", "sub", "stage", "foot", "themes"])
  els[id] = { id, innerHTML: "", textContent: "", className: "", scrollTop: 0,
              addEventListener(t, fn){ (this._h ||= {})[t] = fn; } };
const handlers = {};
globalThis.document = {
  body: { dataset: {}, style: { setProperty(){} } },
  getElementById: id => els[id] || null,
  addEventListener: (t, fn) => { (handlers[t] ||= []).push(fn); },
  querySelectorAll: () => [],
};
globalThis.WebSocket = class { constructor(){ this.onopen = null; } };
globalThis.location = { host: "127.0.0.1:8788", pathname: "/" };
globalThis.setTimeout = () => {};
vm.runInThisContext(code);

let bad = 0;
const ck = (n, ok, extra) => { if (!ok) bad++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}${!ok && extra ? "  -> " + extra : ""}`); };
const count = (s, re) => (s.match(re) || []).length;

render({ status: "match", state, top1_rate: 0.296 });
let out = els.stage.innerHTML;

const ours = state.players.filter(p => p.team === state.own_team);
const known = ours.filter(p => p.score !== null);
const pending = ours.filter(p => state.lookup.pending.includes(p.puuid));
const first = known.find(p => p.detail && p.detail.form.length);

ck("labelled as agent select", /Agent select/.test(els.sub.innerHTML));
ck("counted out of your team, not ten",
   els.sub.innerHTML.includes(`${state.coverage}/${ours.length} known`));
ck("one card per teammate", count(out, /<button class="pcard/g) === ours.length);
ck("no scoreboard rows", count(out, /<button class="row/g) === 0);
ck("no odds bar without an enemy to predict against", !/class="odds"/.test(out));
ck("no 'not enough known' blaming the lookup", !/Not enough of the roster/.test(out));
ck("says how many are still being looked up",
   out.includes(`looking up ${pending.length}`));
ck("no link to a match page that does not exist yet", !/class="ownpage"/.test(out));
{
  const runs = ours.filter(p => p.streak);
  ck("a teammate on a run is badged on their card",
     runs.length > 0 && (out.match(/class="run (won|lost)"/g) || []).length === runs.length
     && runs.every(p => out.includes(`${p.streak.result} ${p.streak.count} in a row`)));
}

// --- a known player: every number, the one they would be read for --------
{
  const r = first.recent, m = first.detail.map, f = first.detail.form[0];
  const per = n => (n / r.games).toFixed(1);
  const pct = v => Math.round(v * 100) + "%";
  ck("the 0-100 rating", out.includes(`<div class="score">${first.score}</div>`));
  ck("labelled with how many games it covers", out.includes(`last ${r.games}`));
  ck("K/D/A per game over those games",
     out.includes(`${per(r.kills)}/${per(r.deaths)}/${per(r.assists)}`));
  ck("K/D", out.includes(r.kd.toFixed(2)));
  ck("headshot rate", out.includes(`>${pct(r.headshot_rate)}<`));
  ck("damage per round", out.includes(`>${Math.round(r.adr)}<`));
  ck("win rate", out.includes(`>${pct(r.win_rate)}<`));
  ck("this map, named", out.includes(`On ${state.map}`));
  ck("their most played agent here, with games",
     out.includes(`${m.agents[0].agent} ×${m.agents[0].games}`));
  ck("their record here", out.includes(`${m.wins}–${m.losses}`));
  ck("last game result and score",
     out.includes(`${f.won ? "W" : "L"} ${f.rounds_won}–${f.rounds_lost}`));
  ck("last game K/D/A", out.includes(`${f.kills}/${f.deaths}/${f.assists}`));
  ck("last game headshots", out.includes(`HS ${pct(f.headshot_rate)}`));
  ck("last game, how long ago", out.includes(f.ago));
  ck("labelled as competitive", /Last comp/.test(out));
}

// --- players not locked in yet --------------------------------------------
{
  const picking = ours.filter(p => !p.agent);
  const guessed = picking.filter(p => p.likely && !p.is_you);
  ck("players still picking say so",
     count(out, />selecting</g) === picking.length - guessed.length);
  ck("or say what they are likely to play, from their last 20",
     guessed.every(p => out.includes(
       `likely ${p.likely.role} ${p.likely.games}/${p.likely.of}`)));
  const hovering = ours.filter(p => p.agent && p.selection === "selected");
  ck("a hovered agent is not shown as a locked one",
     count(out, /class="hov">hovering</g) === hovering.length && hovering.length > 0);
}

// --- still being looked up, versus nothing to find ------------------------
ck("a pending player is shown as looking up",
   /Looking up their last 20/.test(out) && !/No competitive history/.test(out));
render({ status: "match", top1_rate: 0.296,
         state: { ...state, lookup: { pending: [], remaining: 0 } } });
out = els.stage.innerHTML;
ck("once the lookup is done, the same player has no history",
   /No competitive history/.test(out) && !/Looking up their last 20/.test(out));
ck("and the team bar says everyone is looked up", /all looked up/.test(out));

// --- the team's roles so far ---------------------------------------------
{
  render({ status: "match", state, top1_rate: 0.296 });
  const o = els.stage.innerHTML, yp = state.your_picks;
  ck("a slot for each of the four roles", count(o, /class="slot( open)?"/g) === 4);
  ck("a role nobody has is called open",
     /class="slot open"><b>Initiator<\/b><span>open<\/span>/.test(o));
  ck("a hovered agent is marked as hovering in the strip", /\(hovering\)/.test(o));
  ck("an undecided teammate counts toward their likely role",
     /likely, 12\/20/.test(o) && /likely, 9\/20/.test(o));
  ck("your picks that would fill the open role are marked",
     count(o, /class="fill">open role/g)
     === [...yp.agents, ...yp.few].filter(a => a.role === "Initiator").length);
}

// --- your picks, in the panel until a teammate is clicked -----------------
{
  render({ status: "match", state, top1_rate: 0.296 });
  const o = els.stage.innerHTML, yp = state.your_picks;
  ck("your picks fill the panel", /class="panel picks"/.test(o) && /Your picks/.test(o));
  ck("every pick named, best first",
     yp.agents.every(a => o.includes(`<b>${a.agent}</b>`))
     && o.indexOf(`<b>${yp.agents[0].agent}</b>`) < o.indexOf(`<b>${yp.agents[1].agent}</b>`));
  ck("how well you play it, in words", /▲▲ well above/.test(o) && /≈ your usual/.test(o));
  ck("this map's record beside it", o.includes(`on ${yp.map}`));
  ck("agents played once or twice sit under their own divider",
     /Played once or twice/.test(o)
     && (o.match(/<tr class="few">/g) || []).length === yp.few.length
     && o.indexOf("Played once or twice") > o.indexOf(`<b>${yp.agents.at(-1).agent}</b>`));
  ck("and say how many games they rest on",
     yp.few.every(a => o.includes(`in ${a.games} game${a.games === 1 ? "" : "s"}`)));
}

// --- a newer account, with only agents played once or twice --------------
{
  const yp = state.your_picks;
  render({ status: "match", top1_rate: 0.296,
           state: { ...state, your_picks: { ...yp, agents: [] } } });
  const o = els.stage.innerHTML;
  ck("still gets its picks, not the generic hint",
     /class="panel picks"/.test(o) && /Played once or twice/.test(o));
  render({ status: "match", state, top1_rate: 0.296 });
}

// --- the full breakdown is still one click away ---------------------------
for (const fn of handlers.click || [])
  fn({ target: { closest: s => s.includes("button.pcard")
        ? { dataset: { puuid: first.puuid } } : null } });
out = els.stage.innerHTML;
ck("clicking a card opens the panel",
   /class="panel"/.test(out) && /record &amp; form/.test(out));
ck("and marks the card selected", /class="pcard known sel"/.test(out));
ck("a teammate's card replaces your picks", !/class="panel picks"/.test(out));
for (const fn of handlers.keydown || []) fn({ key: "Escape", target: {} });
ck("and ESC brings them back", /class="panel picks"/.test(els.stage.innerHTML));

// --- the match loads: this same tab moves on to it -------------------------
{
  const loaded = JSON.parse(readFileSync(matchPath, "utf8"));
  render({ status: "match", state, top1_rate: 0.296 });
  render({ status: "loading" });
  const held = els.stage.innerHTML;
  ck("while the match loads the agent-select cards stay up",
     count(held, /<button class="pcard/g) === ours.length);
  ck("and the line under the map says why", /match is loading/.test(els.sub.textContent));
  render({ status: "match", state: loaded, top1_rate: 0.296, fresh: false });
  const o = els.stage.innerHTML;
  ck("the match replaces agent select in the same tab",
     /Live/.test(els.sub.innerHTML) && !/Agent select/.test(els.sub.innerHTML));
  ck("no agent-select cards are left behind",
     count(o, /<button class="pcard/g) === 0 && !/class="panel picks"/.test(o)
     && !/class="slot( open)?"/.test(o));
  ck("both teams on the scoreboard", count(o, /<button class="row/g) === 10);
  ck("with the win probability", /class="odds"/.test(o));
  ck("and the link to the match's own page, now that it has one",
     o.includes(`href="/m/${loaded.match_id}"`));
}

console.log(bad ? `\n${bad} FAILED` : "\nall passed");
process.exit(bad ? 1 : 0);
