"""Re-derive what the project claims, and report anything that disagrees.

    python tools/audit.py

This project has shipped the same class of defect four times, and every one was
invisible until something unrelated exposed it:

  * a comment saying "the crawler normalises inline" that was false of the
    caller, so every live fetch silently stored nothing
  * a hardcoded "30.5%" in the live output that went wrong at the next retrain
  * an inlined scaler that stopped matching the model it explained
  * documented figures left behind by three retrains

The common thread is a claim nothing re-checks. This checks the ones that can
be checked mechanically.

**What it cannot check**, and this is the important limitation: whether a
comment tells the truth about the code beneath it. Three of the four defects
above were exactly that, and catching them needs reading. This tool narrows the
surface; it does not replace the review.

It is read-only. It never writes, retrains, or fetches.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

sys.path.insert(0, ".")

ROOT = pathlib.Path(".")
DOCS = [ROOT / "README.md", ROOT / "docs" / "MODEL-CHOICE.md",
        ROOT / "docs" / "SANDBOX.md", ROOT / "docs" / "MODELING.md"]

# Prose that is *meant* to name an old number -- "an earlier bundle had it at
# 0.5138" is a deliberate historical reference. An audit that flagged those
# would be switched off within a week, so anything on a line matching these is
# left alone.
HISTORICAL = re.compile(
    r"\b(was|were|previously|earlier|used to|before the fix|old|"
    r"an earlier|prior to|at the time)\b", re.I)


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def problem(self, where: str, msg: str) -> None:
        self.problems.append(f"{where}: {msg}")

    def note(self, where: str, msg: str) -> None:
        self.notes.append(f"{where}: {msg}")


# --- 1. frozen artefacts -----------------------------------------------

def check_artefacts(rep: Report) -> None:
    """Anything refitted leaves copies behind. This has bitten twice."""
    model = ROOT / "models" / "model.joblib"
    if not model.exists():
        rep.note("models/model.joblib", "absent; skipping artefact checks")
        return
    import joblib
    bundle = joblib.load(model)
    norms_as_of = bundle.get("norms_as_of")

    idx = ROOT / "models" / "perf_index.json"
    if idx.exists():
        d = json.loads(idx.read_text(encoding="utf-8"))
        if norms_as_of and d.get("as_of", 0) < norms_as_of:
            rep.problem("models/perf_index.json",
                        f"fitted at {d.get('as_of')}, older than the bundle's "
                        f"norms at {norms_as_of} -- rebuild with "
                        f"tools/build_perf_index.py")
        if d.get("top1_rate") is None:
            rep.note("models/perf_index.json",
                     "top1_rate unmeasured; the live view will omit the figure "
                     "(tools/validate_potential.py --write-index)")

    bench = ROOT / "reports" / "sandbox" / "static_benchmark.json"
    if bench.exists():
        d = json.loads(bench.read_text(encoding="utf-8"))
        if norms_as_of and d.get("norms_as_of", 0) < norms_as_of:
            rep.problem("reports/sandbox/static_benchmark.json",
                        f"frozen against norms {d.get('norms_as_of')}, bundle "
                        f"is {norms_as_of} -- `compare` will report drift that "
                        f"is really staleness; re-freeze with `sandbox benchmark`")

    dash = ROOT / "reports" / "sandbox" / "dashboard_data.json"
    results = ROOT / "reports" / "results.json"
    if dash.exists() and results.exists():
        d = json.loads(dash.read_text(encoding="utf-8"))
        r = json.loads(results.read_text(encoding="utf-8"))
        if r.get("shipped") and d.get("model") != r["shipped"]:
            rep.problem("reports/sandbox/dashboard_data.json",
                        f"built from '{d.get('model')}' but the shipped model "
                        f"is '{r['shipped']}' -- re-export")
        # The win probabilities in this file survive an index refit; the
        # per-player 0-100 scores do not. Raising MIN_MAP_GAMES from 4 to 6
        # moved 1,489 of its 1,620 scores while the model name still matched,
        # so checking the model alone declared a stale file healthy.
        if idx.exists():
            live = json.loads(idx.read_text(encoding="utf-8")).get("as_of")
            got = d.get("index_as_of")
            if got is None:
                rep.problem("reports/sandbox/dashboard_data.json",
                            "records no index_as_of, so nothing can tell "
                            "whether its player scores are current -- "
                            "re-export with tools/export_dashboard.py")
            elif live and got != live:
                rep.problem("reports/sandbox/dashboard_data.json",
                            f"player scores came from the index at {got}, but "
                            f"the index is now {live} -- re-export with "
                            f"tools/export_dashboard.py")


# --- 2. figures quoted in prose ----------------------------------------

def check_figures(rep: Report) -> None:
    """Claims about the held-out test set, against what the last run produced.

    Deliberately narrow. A first version flagged every `0.6xxx` in the docs and
    produced 27 false positives out of 29 findings: the docs legitimately quote
    *validation* log losses, an equal-rank subset, and live-path replays, none
    of which appear in results.json. A check with that signal-to-noise gets
    switched off, which is worse than not having it.

    So this checks only claims that are unambiguous -- the size of the test set,
    and the figures attached to the shipped model. Everything else is prose the
    author owns.
    """
    results = ROOT / "reports" / "results.json"
    if not results.exists():
        rep.note("reports/results.json", "absent; skipping figure checks")
        return
    r = json.loads(results.read_text(encoding="utf-8"))
    n_test, shipped = r.get("n_test"), r.get("shipped")
    by_name = {m["name"]: m for m in r["results"]}

    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        # Matched across the whole document, not line by line: prose wraps, and
        # the stale claim this was written to catch had "on 6,132" ending one
        # line and "matches:" starting the next.
        for m in re.finditer(r"(?:on|of)\s+\*{0,2}([\d,]{4,})\*{0,2}\s+"
                             r"(?:held-out\s+)?matches", text):
            line_no = text[:m.start()].count(chr(10)) + 1
            context = chr(10).join(
                text.splitlines()[max(0, line_no - 2):line_no + 1])
            if HISTORICAL.search(context):
                continue
            got = int(m.group(1).replace(",", ""))
            if n_test and got not in (n_test, r.get("n_train"), r.get("n_val")):
                rep.problem(f"{doc}:{line_no}",
                            f"describes {m.group(1)} matches; the last run used "
                            f"{n_test:,} test / {r.get('n_train', 0):,} train")
    if shipped and shipped in by_name:
        s = by_name[shipped]
        rep.note("reports/results.json",
                 f"shipped '{shipped}': log loss {s['log_loss']:.4f}, "
                 f"auc {s['auc']:.3f}, acc {s['accuracy'] * 100:.1f}% "
                 f"on {n_test:,} test matches")


# --- 3. population claims ----------------------------------------------

def check_population(rep: Report) -> None:
    """Percentages about the dataset drift every time the crawler runs."""
    from valwr import config
    from valwr.store import schema
    try:
        s = config.load(require_key=False)
        if not s.database_path.exists():
            rep.note("database", "absent; skipping population checks")
            return
        conn = schema.connect(s.database_path)
    except Exception as e:                       # noqa: BLE001
        rep.note("database", f"unreadable ({type(e).__name__}); skipped")
        return

    import collections
    total = conn.execute(
        "SELECT COUNT(DISTINCT puuid) FROM match_players").fetchone()[0]
    counts = collections.Counter(
        n for _, n in conn.execute(
            "SELECT puuid, COUNT(*) FROM match_players GROUP BY puuid"))
    matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    if not total:
        return
    one = counts[1] / total * 100
    five = sum(v for k, v in counts.items() if k >= 5) / total * 100
    rep.note("database", f"{matches:,} matches, {total:,} players, "
                         f"{one:.1f}% seen once, {five:.1f}% seen 5+ times")

    # Any doc quoting a player total that is not the live one is stale.
    for doc in DOCS:
        if not doc.exists():
            continue
        for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if HISTORICAL.search(line):
                continue
            for found in re.findall(r"\b(\d{2,3},\d{3})\s+players\b", line):
                if int(found.replace(",", "")) != total:
                    rep.problem(f"{doc}:{i}",
                                f"says '{found} players'; the database holds "
                                f"{total:,}")


# --- 4. constants named in two places ----------------------------------

def check_constants(rep: Report) -> None:
    """A number written down twice drifts the moment one copy is tuned."""
    from valwr.rating import potential as P
    from valwr.live import resolve as R

    watched = {
        "MIN_MAP_GAMES": P.MIN_MAP_GAMES,
        "MIN_DOMINANCE_GAMES": P.MIN_DOMINANCE_GAMES,
        "STALE_AFTER_SECONDS": R.STALE_AFTER_SECONDS,
        "THIN_HISTORY": R.THIN_HISTORY,
    }
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for name, value in watched.items():
            for m in re.finditer(rf"`?{name}`?\s*(?:=|is|of)\s*(\d+)", text):
                if int(m.group(1)) != value:
                    line = text[:m.start()].count("\n") + 1
                    rep.problem(f"{doc}:{line}",
                                f"{name} documented as {m.group(1)}, code says "
                                f"{value}")


# --- 5. dead imports ----------------------------------------------------

def check_imports(rep: Report) -> None:
    for p in sorted(ROOT.glob("valwr/**/*.py")) + sorted(ROOT.glob("tools/*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            rep.problem(str(p), f"does not parse: {e}")
            continue
        imported: dict[str, int] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imported[(a.asname or a.name).split(".")[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name != "*":
                        imported[a.asname or a.name] = n.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                v = n
                while isinstance(v, ast.Attribute):
                    v = v.value
                if isinstance(v, ast.Name):
                    used.add(v.id)
        for name, line in sorted(imported.items(), key=lambda kv: kv[1]):
            if name not in used and name != "annotations":
                rep.problem(f"{p}:{line}", f"unused import '{name}'")


CHECKS = (
    ("frozen artefacts", check_artefacts),
    ("figures in prose", check_figures),
    ("population claims", check_population),
    ("duplicated constants", check_constants),
    ("dead imports", check_imports),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="audit")
    ap.add_argument("--quiet", action="store_true",
                    help="problems only; suppress the informational notes")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    rep = Report()
    for name, fn in CHECKS:
        try:
            fn(rep)
        except Exception as e:                   # noqa: BLE001
            rep.problem(name, f"the check itself failed: {type(e).__name__}: {e}")

    if rep.notes and not args.quiet:
        print("\n  Context")
        print("  " + "-" * 60)
        for n in rep.notes:
            print(f"    {n}")

    print("\n  Disagreements")
    print("  " + "-" * 60)
    if not rep.problems:
        print("    none -- every mechanically checkable claim matches")
    for p in rep.problems:
        print(f"    {p}")

    print(f"\n  {len(rep.problems)} problem(s) across {len(CHECKS)} checks.")
    print("  This cannot tell you whether a comment describes the code "
          "beneath it.\n  That still needs reading, and it is where the worst "
          "bugs in this project\n  have lived.")
    return 1 if rep.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
