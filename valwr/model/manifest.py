"""Which models these are, written down where git can see it.

    python -m valwr.model.manifest      # describe the models on disk

`models/` is not committed -- it is built from a database nobody else has -- so
nothing in the repository said which model the documented numbers came from,
or noticed when the files on disk stopped being that model. A retrain that
refitted one index and not the other left the dashboard scoring players
against a reference older than its model, and every check passed.

reports/model_manifest.json holds a fingerprint of each file the dashboard
loads, the facts each one carries (when it was fitted, on how much, how well
it measured), and -- when tools/rebuild.py wrote it -- the commit, database
and library versions it was built from. tools/preflight.py warns when the
files on disk are not the ones it describes; tools/audit.py checks the
reports and the docs against it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PATH = ROOT / "reports" / "model_manifest.json"
# Everything the dashboard loads from models/, in the order it matters.
ARTIFACTS = ("model.joblib", "role_index.json", "perf_index.json")
TEST_METRICS = ("n", "log_loss", "auc", "accuracy", "brier", "ece")
NOT_RECORDED = {"by": None,
                "note": "described after the fact: how these files were "
                        "built was not recorded"}


def _number(v):
    if hasattr(v, "item"):                  # a numpy scalar, int or float
        v = v.item()
    return round(v, 6) if isinstance(v, float) else v


def fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def facts(path: Path) -> dict:
    """What a model file says about itself, in the terms the docs use."""
    if path.suffix == ".joblib":
        import joblib
        b = joblib.load(path)
        test = (b.get("metrics") or {}).get(b.get("best")) or {}
        return {"model": b.get("best"), "features": len(b.get("columns") or ()),
                "norms_as_of": b.get("norms_as_of"),
                "test": {k: _number(test[k]) for k in TEST_METRICS if k in test}}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: _number(d.get(k)) for k in ("as_of", "n", "top1_rate")}


def describe(models: Path, built: dict | None = None) -> dict:
    artifacts = {}
    for name in ARTIFACTS:
        path = models / name
        if path.exists():
            artifacts[name] = {**fingerprint(path), **facts(path)}
    return {"artifacts": artifacts, "built": built or NOT_RECORDED}


def write(models: Path, built: dict | None = None,
          path: Path | None = None) -> dict:
    manifest = describe(models, built)
    path = path or PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return manifest


def load(path: Path | None = None) -> dict | None:
    try:
        return json.loads((path or PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def mismatches(models: Path, manifest: dict) -> list[str]:
    """The described files that are missing from disk or are not those files."""
    out = []
    for name, want in (manifest.get("artifacts") or {}).items():
        path = models / name
        if not path.exists() or fingerprint(path)["sha256"] != want.get("sha256"):
            out.append(name)
    return out


def main(argv=None) -> int:
    from valwr import config
    models = config.load(require_key=False).models_path
    # The files a rebuild described are still those files: keep what it
    # recorded about how they were built, which this cannot know.
    held = load()
    kept = (held or {}).get("built") if held and not mismatches(models, held) \
        else None
    got = write(models, kept)
    for name, a in got["artifacts"].items():
        print(f"  {name:<16} {a['sha256'][:12]}  {a['bytes']:>9,} bytes")
    shown = PATH.relative_to(ROOT) if PATH.is_relative_to(ROOT) else PATH
    print(f"  wrote {shown}")
    if not got["artifacts"]:
        print("  no models found -- nothing described")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
