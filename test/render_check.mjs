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
// Prints one line per check and exits non-zero if any fail.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const [, , pagePath, statePath] = process.argv;
const html = readFileSync(pagePath, "utf8");
const code = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));
const state = JSON.parse(readFileSync(statePath, "utf8"));

// --- the smallest DOM this page touches -------------------------------
const els = {};
for (const id of ["live", "mapname", "meta", "main", "foot"])
  els[id] = { id, innerHTML: "", textContent: "", className: "", scrollTop: 0 };

const handlers = {};
globalThis.document = {
  getElementById: id => els[id] || null,
  addEventListener: (type, fn) => { (handlers[type] ||= []).push(fn); },
};
globalThis.WebSocket = class { constructor(){ this.onopen = null; } };
globalThis.location = { host: "127.0.0.1:8787" };
globalThis.setTimeout = () => {};

vm.runInThisContext(code);
render({ status: "match", state, top1_rate: 0.296 });

const out = els.main.innerHTML;
const known = state.players.filter(p => p.score !== null);
const unknown = state.players.filter(p => p.score === null);
const first = known[0];

let bad = 0;
const check = (name, ok) => {
  if (!ok) bad++;
  console.log(`  ${ok ? "ok  " : "FAIL"}  ${name}`);
};

check("map is named in the header", els.mapname.textContent === state.map);
check("a row per player",
      (out.match(/<tr tabindex/g) || []).length === state.players.length);
check("both teams rendered",
      /class="team red"/.test(out) && /class="team blue"/.test(out));
check("attack/defence sides labelled",
      /attacking/.test(out) && /defending/.test(out));
check("known players marked known",
      (out.match(/class="known/g) || []).length === known.length);
check("unknown players marked unknown",
      (out.match(/class="unknown/g) || []).length === unknown.length);
check("your own row is tagged", /YOU<\/span>/.test(out));
check("career ACS on the row", out.includes(String(first.career.acs)));
check("K/D on the row", out.includes(first.career.kd.toFixed(2)));
check("last-20 K/D on the row", out.includes(first.recent.kd.toFixed(2)));
check("K/D/A is NOT on the row",
      !out.includes(`${first.career.kills}/${first.career.deaths}/${first.career.assists}`));
check("headshot % on the row",
      out.includes((first.career.headshot_rate * 100).toFixed(1) + "%"));
check(first.agent_id ? "agent art addressed by uuid"
                    : "no uuid, so a lettered tile is used instead",
      first.agent_id ? out.includes(`/agents/${first.agent_id}-icon.png`)
                     : !out.includes("/agents/") && /class="ico">[A-Z]/.test(out));
check("unknown players show no numbers",
      (out.match(/>--</g) || []).length >= unknown.length * 4);

// Selecting a player must go through the real delegated click handler.
const row = { dataset: { puuid: first.puuid } };
for (const fn of handlers.click || [])
  fn({ target: { closest: s => s.includes("data-puuid") ? row : null } });
const p = els.main.innerHTML;

check("panel names the selected player", p.includes(first.name));
check(first.agent_id ? "panel shows the full portrait"
                    : "panel omits art when there is no uuid",
      first.agent_id ? p.includes(`/agents/${first.agent_id}-portrait.png`)
                     : !p.includes("-portrait.png"));
check("panel compares career against form", /career &amp; recent form/.test(p));
check("panel has an on-map section", p.includes(`on ${state.map}`));
check("panel shows this map's record",
      p.includes(`${first.detail.map.wins}W`));
check("selected row is highlighted", / sel"/.test(p));

// Clicking the same row again clears it, which is how you close the panel.
for (const fn of handlers.click || [])
  fn({ target: { closest: s => s.includes("data-puuid") ? row : null } });
check("clicking again deselects", /Select a player/.test(els.main.innerHTML));

console.log(bad ? `\n${bad} check(s) FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
