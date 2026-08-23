"""Configuration loading.

Every setting comes from `.env`. Nothing is hardcoded and no key ever appears
in source. Failures here are the first thing anyone cloning this repo will hit,
so they explain themselves rather than raising a bare KeyError.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Stated ceilings from HenrikDev's docs. The collector's limiter applies its own
# headroom below these -- bursting at exactly the stated limit trips it.
TIER_LIMITS = {"basic": 30, "enhanced": 90}

VALID_REGIONS = {"na", "eu", "ap", "kr", "latam", "br"}
VALID_PLATFORMS = {"pc", "console"}


class ConfigError(RuntimeError):
    """Raised with an explanation of how to fix the problem, not just what it is."""


@dataclass(frozen=True)
class Settings:
    henrik_api_key: str
    henrik_tier: str
    region: str
    platform: str
    riot_name: str
    riot_tag: str
    database_path: Path
    dashboard_host: str
    dashboard_port: int
    anthropic_api_key: str | None

    @property
    def requests_per_minute(self) -> int:
        return TIER_LIMITS[self.henrik_tier]

    @property
    def riot_id(self) -> str:
        return f"{self.riot_name}#{self.riot_tag}"


def _require(name: str, hint: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set in .env\n\n  {hint}\n\n"
            f"If .env doesn't exist yet: cp .env.example .env"
        )
    return value


def load(require_key: bool = True) -> Settings:
    """Read settings from .env.

    `require_key=False` skips the HenrikDev key check, for the parts of the
    project that only touch valorant-api.com (no auth) or the local database.
    """
    load_dotenv(REPO_ROOT / ".env")

    if require_key:
        api_key = _require(
            "HENRIK_API_KEY",
            "Generate one at https://api.henrikdev.xyz/dashboard/ "
            "(requires joining their Discord).",
        )
    else:
        api_key = os.getenv("HENRIK_API_KEY", "").strip()

    tier = os.getenv("HENRIK_TIER", "basic").strip().lower()
    if tier not in TIER_LIMITS:
        raise ConfigError(
            f"HENRIK_TIER={tier!r} is not recognised. "
            f"Use one of: {', '.join(sorted(TIER_LIMITS))}"
        )

    region = os.getenv("REGION", "na").strip().lower()
    if region not in VALID_REGIONS:
        raise ConfigError(
            f"REGION={region!r} is not a valid region. "
            f"Use one of: {', '.join(sorted(VALID_REGIONS))}"
        )

    platform = os.getenv("PLATFORM", "pc").strip().lower()
    if platform not in VALID_PLATFORMS:
        raise ConfigError(
            f"PLATFORM={platform!r} is not valid. "
            f"Use one of: {', '.join(sorted(VALID_PLATFORMS))}"
        )

    db_path = Path(os.getenv("DATABASE_PATH", "data/valwr.db"))
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    return Settings(
        henrik_api_key=api_key,
        henrik_tier=tier,
        region=region,
        platform=platform,
        riot_name=os.getenv("RIOT_NAME", "").strip(),
        riot_tag=os.getenv("RIOT_TAG", "").strip().lstrip("#"),
        database_path=db_path,
        dashboard_host=os.getenv("DASHBOARD_HOST", "127.0.0.1").strip(),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8000")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
    )
