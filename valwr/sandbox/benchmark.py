"""Static benchmark persistence and comparison.

Static scenarios are long-lived benchmarks, so a run can be frozen and a later
run compared against it. That shows how the model evolved -- which scenarios
moved, which features moved, and whether any favourite flipped side.

**These are not pytest golden values.** Retraining legitimately changes every
probability, and a benchmark that made retraining fail would simply be deleted
the first time it was inconvenient. The comparison is a lens on model
evolution, not a lock on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from valwr.sandbox.schema import ScenarioResult

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = REPO / "reports" / "sandbox" / "static_benchmark.json"

# Below this, a probability change is rounding rather than news.
NOISE_FLOOR = 0.0005


def save(results: list[ScenarioResult], bundle: dict,
         path: Path | None = None) -> Path:
    path = path or DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": bundle.get("best"),
        "norms_as_of": bundle.get("norms_as_of"),
        "n_features": len(bundle.get("columns", [])),
        "scenarios": {
            r.scenario: {
                "category": r.category,
                "model": r.model,
                "probability": round(r.probability, 6),
                "mirror_probability": (None if r.mirror_probability is None
                                       else round(r.mirror_probability, 6)),
                "expect": r.expect,
                "factors": [[k, round(v, 6)] for k, v in r.factors],
                "features": {k: round(v, 6) for k, v in sorted(r.features.items())},
            }
            for r in results
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict:
    path = path or DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no benchmark at {path}; run `python -m valwr.sandbox benchmark`")
    return json.loads(path.read_text(encoding="utf-8"))


def compare(old: dict, results: list[ScenarioResult], top: int = 15) -> str:
    """Human-readable diff between a saved benchmark and a fresh run."""
    previous = old.get("scenarios", {})
    lines = [
        "Benchmark comparison",
        "====================",
        f"  saved model : {old.get('model')}  "
        f"(norms as of {old.get('norms_as_of')})",
        f"  scenarios   : {len(previous)} saved, {len(results)} current",
        "",
    ]

    added = [r.scenario for r in results if r.scenario not in previous]
    removed = [n for n in previous if n not in {r.scenario for r in results}]
    if added:
        lines.append(f"  new scenarios     : {len(added)} "
                     f"({', '.join(added[:5])}{'...' if len(added) > 5 else ''})")
    if removed:
        lines.append(f"  removed scenarios : {len(removed)} "
                     f"({', '.join(removed[:5])}{'...' if len(removed) > 5 else ''})")
    if added or removed:
        lines.append("")

    deltas, flips = [], []
    for r in results:
        before = previous.get(r.scenario)
        if not before:
            continue
        was, now = before["probability"], r.probability
        delta = now - was
        if abs(delta) > NOISE_FLOOR:
            deltas.append((abs(delta), r.scenario, was, now))
        if (was > 0.5) != (now > 0.5) and min(abs(was - 0.5), abs(now - 0.5)) > 0.002:
            flips.append((r.scenario, was, now))

    lines += [f"  scenarios moved by more than {NOISE_FLOOR}: {len(deltas)}", ""]
    if deltas:
        deltas.sort(reverse=True)
        lines += [f"  {'scenario':<34}{'was':>8}{'now':>8}{'delta':>9}",
                  "  " + "-" * 57]
        for _, name, was, now in deltas[:top]:
            lines.append(f"  {name[:33]:<34}{was * 100:>7.1f}%{now * 100:>7.1f}%"
                         f"{(now - was) * 100:>+8.1f}")
        if len(deltas) > top:
            lines.append(f"  ... and {len(deltas) - top} more")
        lines.append("")

    lines.append(f"  favourite flipped side: {len(flips)}")
    for name, was, now in flips[:top]:
        lines.append(f"    {name:<34}{was * 100:>6.1f}% -> {now * 100:.1f}%")

    biggest = _largest_feature_moves(previous, results)
    if biggest:
        lines += ["", "  largest feature-vector changes:"]
        for name, feature, was, now in biggest[:top]:
            lines.append(f"    {name[:26]:<28}{feature.replace('d_', ''):<24}"
                         f"{was:>10.4f} -> {now:.4f}")
    return "\n".join(lines)


def _largest_feature_moves(previous: dict, results: list[ScenarioResult]):
    moves = []
    for r in results:
        before = previous.get(r.scenario)
        if not before:
            continue
        for feature, now in r.features.items():
            was = before.get("features", {}).get(feature)
            if was is None:
                continue
            if abs(now - was) > 1e-6:
                moves.append((abs(now - was), r.scenario, feature, was, now))
    moves.sort(reverse=True)
    return [(name, feature, was, now) for _, name, feature, was, now in moves]
