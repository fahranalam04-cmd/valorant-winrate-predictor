"""What the model's numbers have meant, so the page can say them honestly.

The win model is calibrated but modest. Across its 10,304 test matches the
lowest tenth of its calls averaged 40% and the highest tenth 60%, and they meant
what they said -- when it rated a side at 60%, that side won 61% -- but it
picked the winner 54% of the time. A bare "58%" beside the word "favoured"
invited a reading the record does not support. So the page puts beside each
prediction what happened the times the model made the same call in testing,
and how often it picks the winner with as many players known as this lobby
has.

The figures are the reliability table and coverage strata train.py writes to
reports/results.json. They are passed on only when that file describes the
model actually loaded -- the same model, tested on the same number of matches.
Otherwise the page shows the prediction with no record rather than another
model's.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "reports" / "results.json"


def _load(path: Path | None) -> dict | None:
    try:
        return json.loads((path or RESULTS).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def from_results(results: dict | None) -> dict | None:
    """The record in the shape the page reads: `bins` of [predicted, won, n]
    for Blue, and how often it picks the winner by players `known`."""
    if not results or not results.get("reliability"):
        return None
    return {
        "n_test": results.get("n_test"),
        "bins": [[round(float(m), 4), round(float(o), 4), int(n)]
                 for m, o, n in results["reliability"]],
        "strata": [{"known": s["name"], "n": int(s["n"]),
                    "accuracy": round(float(s["accuracy"]), 4)}
                   for s in results.get("coverage_strata") or []],
    }


def track_record(bundle: dict, path: Path | None = None) -> dict | None:
    """The record, if reports/results.json is this bundle's."""
    results = _load(path)
    if results is None:
        return None
    best = bundle.get("best")
    tested = ((bundle.get("metrics") or {}).get(best) or {}).get("n")
    if results.get("shipped") != best or results.get("n_test") != tested:
        return None
    return from_results(results)


def published_record(path: Path | None = None) -> dict | None:
    """For the demo, which has no model loaded: the shipped model's record."""
    return from_results(_load(path))
