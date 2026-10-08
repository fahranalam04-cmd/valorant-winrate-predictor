// Execute the built GitHub Pages demo -- its stand-in socket and the real page
// script together -- and assert on what renders when each control is used.
//
//     node test/pages_check.mjs <site/index.html>
//
// The Python tests can confirm the page script was copied verbatim; only
// running it shows the stand-in actually drives it.
import { readFileSync } from "node:fs";
import vm from "node:vm";

const html = readFileSync(process.argv[2], "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);

const els = {};
for (const id of ["pulse", "map", "sub", "stage", "foot"])
  els[id] = { id, innerHTML: "", textContent: "", className: "", scrollTop: 0 };
const handlers = {};
const timers = [];
const stub = () => ({ className: "", innerHTML: "", dataset: {},
  setAttribute(){}, querySelectorAll: () => [], addEventListener(){} });
globalThis.window = globalThis;
globalThis.document = {
  body: { style: { _v: {}, setProperty(k, v){ this._v[k] = v; } },
          classList: { add(){} }, appendChild(){} },
  getElementById: id => els[id] || null,
  addEventListener: (t, fn) => { (handlers[t] ||= []).push(fn); },
  createElement: stub,
};
// Pages serves the demo under /valorant-winrate-predictor/, which the page
// reads when deciding whether it is pinned to one match. It is not.
globalThis.location = { host: "example.github.io", search: "",
                        pathname: "/valorant-winrate-predictor/" };
globalThis.setTimeout = fn => { timers.push(fn); };
const flush = () => { while (timers.length) timers.shift()(); };

for (const code of scripts) vm.runInThisContext(code);
flush();

let bad = 0;
const ck = (n, ok) => { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); };
const rows = () => (els.stage.innerHTML.match(/<button class="row/g) || []).length;
const cards = () => (els.stage.innerHTML.match(/<button class="pcard/g) || []).length;

ck("two scripts: the stand-in, then the page", scripts.length === 2);
ck("the stand-in delivers the match on open", els.map.textContent === "Ascent");
ck("all ten players render", rows() === 10);
ck("the connection shows as live", els.pulse.className === "pulse");

window.__demo.set("map", "Lotus");
ck("changing the map re-renders it", els.map.textContent === "Lotus");
// Addressed from the directory the page is served from, so the demo works
// under the Pages sub-path and a match's own page at /m/<id> works too.
// This harness serves the page from /valorant-winrate-predictor/, as Pages
// does, so the art has to carry that prefix. A rooted "/maps/" would point at
// the wrong site; a bare "maps/" breaks a match's own page at /m/<id>.
ck("and swaps the background art, under the sub-path it is served from",
   document.body.style._v["--mapart"]
   === 'url("/valorant-winrate-predictor/maps/lotus-splash.jpg")');

const runs = () => (els.stage.innerHTML.match(/class="run (won|lost)"/g) || []).length;
ck("your team's runs are badged", runs() === 2
   && /won 4 in a row/.test(els.stage.innerHTML));

window.__demo.set("side", "Red");
ck("switching side moves 'you' to Red", /you are on Red/.test(els.stage.innerHTML));
ck("and the runs badged are the new side's", runs() === 2
   && !/won 4 in a row/.test(els.stage.innerHTML)
   && /won 3 in a row/.test(els.stage.innerHTML));

window.__demo.set("known", "thin");
ck("a thin lobby shows four known", /4\/10 known/.test(els.sub.innerHTML));
ck("and nobody without history is on a run", runs() === 0);
ck("and low confidence", /low<\/span>|low confidence/.test(els.sub.innerHTML));

window.__demo.set("phase", "pregame");
ck("agent select is labelled", /Agent select/.test(els.sub.innerHTML));
ck("agent select shows your team as cards", cards() === 5);
ck("and nothing of the hidden enemy team", rows() === 0);
ck("without claiming a prediction it cannot make",
   !/class="odds"/.test(els.stage.innerHTML));
ck("with players still picking", />selecting</.test(els.stage.innerHTML));
ck("and strangers still being looked up, not written off",
   /Looking up their last 20/.test(els.stage.innerHTML)
   && !/No competitive history/.test(els.stage.innerHTML));
window.__demo.set("phase", "coregame");
ck("leaving it restores the full lobby", rows() === 10 && cards() === 0);

window.__demo.set("status", "finished");
// An earlier check switched sides, so this lobby is lost rather than won.
ck("after the game it shows the result", /class="result (won|lost)"/.test(els.stage.innerHTML));
ck("and the scoreboard beside the prediction", /class="compare"/.test(els.stage.innerHTML));
{
  const q = els.stage.innerHTML;
  ck("with a block for every player who played",
     (q.match(/class="pblock/g) || []).length === 10);
  ck("each one comparing their career line against the match",
     />Before/.test(q) && />This match</.test(q) && /Headshots/.test(q));
  ck("and saying how they played against their own average",
     /their usual/.test(q));
}

{
  const q = els.stage.innerHTML;
  // Patch 13.06 took combat score off the game's scoreboard. The demo's
  // invented result still read it, so the published page and the README
  // image showed a comparison the real one stopped making.
  ck("the comparison is in damage per round, not combat score",
     /Damage \/ round/.test(q) && !/Combat score/.test(q));
  ck("and placing is named for match impact",
     /on match\s+impact/.test(q) && !/on combat\s+score/.test(q));
}
// Agent select has no prediction to score. A finished match was played, so
// its result must not depend on where the phase switch was left.
window.__demo.set("phase", "pregame");
ck("a result renders whichever phase the switch was left on",
   /class="result (won|lost)"/.test(els.stage.innerHTML));
window.__demo.set("phase", "coregame");

window.__demo.set("status", "lobby");
ck("the menus show standby", els.map.textContent === "STANDBY");

window.__demo.set("status", "match");
ck("and a match comes back", rows() === 10);

console.log(bad ? `\n${bad} FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
