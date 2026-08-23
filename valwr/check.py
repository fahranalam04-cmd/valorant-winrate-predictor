"""Phase 0 smoke test: python -m valwr.check

Proves the environment, the database, the reference data, and the HenrikDev key
all work. Degrades cleanly -- everything that does not need the API key still
runs and reports without one, so this is useful before the key is configured.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from valwr import config
from valwr.collect.client import HenrikClient, HenrikError
from valwr.store import reference, schema


def _rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")
    print("-" * max(len(title), 40))


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_environment() -> None:
    _rule("Environment")
    v = sys.version_info
    _ok(f"python {v.major}.{v.minor}.{v.micro}")
    for name in ("httpx", "pandas", "numpy", "sklearn", "lightgbm", "xgboost", "shap"):
        try:
            mod = __import__(name)
            _ok(f"{name} {getattr(mod, '__version__', '?')}")
        except ImportError:
            _fail(f"{name} not installed")


def check_config() -> config.Settings | None:
    _rule("Config")
    try:
        s = config.load(require_key=False)
    except config.ConfigError as e:
        _fail(str(e))
        return None

    _ok(f"region={s.region} platform={s.platform}")
    _ok(f"tier={s.henrik_tier} -> {s.requests_per_minute} req/min")

    if s.henrik_api_key:
        _ok(f"HENRIK_API_KEY set ({len(s.henrik_api_key)} chars)")
    else:
        _warn("HENRIK_API_KEY not set -- API checks will be skipped")

    if s.riot_name and s.riot_tag:
        _ok(f"riot id = {s.riot_id}")
    else:
        _warn("RIOT_NAME / RIOT_TAG not set -- account check will be skipped")

    return s


def check_database(s: config.Settings):
    _rule("Database")
    conn = schema.connect(s.database_path)
    schema.create_all(conn)
    # Idempotency is a real requirement, not an assumption -- prove it here.
    schema.create_all(conn)
    _ok(f"{s.database_path} (schema created, re-run is a no-op)")
    for table, n in schema.table_counts(conn).items():
        print(f"         {table:<16} {n:>8}")
    return conn


def check_reference(conn) -> None:
    _rule("Reference data (valorant-api.com, no key needed)")
    try:
        counts = reference.load_all(conn)
    except Exception as e:
        _fail(f"could not load reference data: {e}")
        return

    for k, n in counts.items():
        _ok(f"{k}: {n}")

    if counts["agents"] < 20:
        _warn(f"expected 20+ agents, got {counts['agents']}")
    if counts["maps"] < 10:
        _warn(f"expected 10+ maps, got {counts['maps']}")

    roles = reference.agent_roles(conn)
    by_role: dict[str, list[str]] = {}
    for agent, role in sorted(roles.items()):
        by_role.setdefault(role or "unknown", []).append(agent)
    print()
    for role, agents in sorted(by_role.items()):
        print(f"         {role:<12} {len(agents):>2}  {', '.join(agents[:6])}"
              f"{' ...' if len(agents) > 6 else ''}")


def _fmt_match(m: dict) -> str:
    """Tolerant extractor for the smoke test.

    Phase 2 does the real parsing. This only needs to prove the call worked, so
    it accepts either a nested object or a bare string for map/agent rather than
    asserting a schema.
    """
    def name_of(v):
        if isinstance(v, dict):
            return v.get("name") or v.get("displayName") or "?"
        return v or "?"

    meta = m.get("metadata", {})
    map_name = name_of(meta.get("map"))
    started = meta.get("started_at") or meta.get("game_start_iso") or ""
    if isinstance(started, (int, float)):
        started = datetime.fromtimestamp(started, timezone.utc).strftime("%Y-%m-%d")
    else:
        started = str(started)[:10]
    return f"{started:<12} {map_name}"


def check_api(s: config.Settings, conn) -> None:
    _rule("HenrikDev API")

    if not s.henrik_api_key:
        _warn("skipped: no HENRIK_API_KEY in .env")
        print("         Get one at https://api.henrikdev.xyz/dashboard/")
        return
    if not (s.riot_name and s.riot_tag):
        _warn("skipped: no RIOT_NAME / RIOT_TAG in .env")
        return

    try:
        with HenrikClient(s.henrik_api_key, conn=conn) as client:
            acct = client.account(s.riot_name, s.riot_tag).get("data", {})
            puuid = acct.get("puuid")
            if not puuid:
                _fail(f"no puuid returned for {s.riot_id} -- check the Riot ID")
                return
            _ok(f"{acct.get('name')}#{acct.get('tag')}  level {acct.get('account_level')}")
            _ok(f"puuid {puuid}")

            resp = client.matches(s.region, s.platform, puuid, size=5, mode="competitive")
            matches = resp.get("data", []) or []
            _ok(f"fetched {len(matches)} recent competitive matches")
            for m in matches:
                print(f"         {_fmt_match(m)}")

            if matches:
                # Useful for Phase 2: record the actual response shape rather
                # than guessing at it later.
                print(f"\n         match keys: {sorted(matches[0].keys())}")

    except HenrikError as e:
        _fail(str(e))
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")


def main() -> int:
    print("valwr Phase 0 check")
    check_environment()
    s = check_config()
    if s is None:
        return 1
    conn = check_database(s)
    check_reference(conn)
    check_api(s, conn)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
