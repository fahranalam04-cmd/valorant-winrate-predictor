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
globalThis.location = { host: "example.github.io", search: "" };
globalThis.setTimeout = fn => { timers.push(fn); };
const flush = () => { while (timers.length) timers.shift()(); };

for (const code of scripts) vm.runInThisContext(code);
flush();

let bad = 0;
const ck = (n, ok) => { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); };
const rows = () => (els.stage.innerHTML.match(/<button class="row/g) || []).length;

ck("two scripts: the stand-in, then the page", scripts.length === 2);
ck("the stand-in delivers the match on open", els.map.textContent === "Ascent");
ck("all ten players render", rows() === 10);
ck("the connection shows as live", els.pulse.className === "pulse");

window.__demo.set("map", "Lotus");
ck("changing the map re-renders it", els.map.textContent === "Lotus");
ck("and swaps the background art",
   document.body.style._v["--mapart"] === 'url("maps/lotus-splash.jpg")');

window.__demo.set("side", "Red");
ck("switching side moves 'you' to Red", /you are on Red/.test(els.stage.innerHTML));

window.__demo.set("known", "thin");
ck("a thin lobby shows four known", /4\/10 known/.test(els.sub.innerHTML));
ck("and low confidence", /low<\/span>|low confidence/.test(els.sub.innerHTML));

window.__demo.set("phase", "pregame");
ck("agent select is labelled", /Agent select/.test(els.sub.innerHTML));

window.__demo.set("status", "lobby");
ck("the menus show standby", els.map.textContent === "STANDBY");

window.__demo.set("status", "match");
ck("and a match comes back", rows() === 10);

console.log(bad ? `\n${bad} FAILED` : "\nall checks passed");
process.exit(bad ? 1 : 0);
