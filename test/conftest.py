"""Every test runs as a fresh clone would, whatever this machine holds.

CI runs the suite with no .env, no trained model and no database, and the
suite is built to pass that way. Nothing enforced it locally: `config.load`
read the developer's .env, so tests saw the real database path, the real
models directory and the real API key. Seven settler tests passed only because
trained models sat in models/ -- they failed the day the database moved to
another drive, and would have failed in CI on the next push.

So the .env is never read here, and every setting it could supply is pinned to
something harmless. A test that needs a setting sets it itself.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="valwr-tests-"))

ISOLATED = {
    "HENRIK_API_KEY": "",
    "HENRIK_TIER": "basic",
    "REGION": "na",
    "PLATFORM": "pc",
    "RIOT_NAME": "",
    "RIOT_TAG": "",
    # Neither exists: anything that opens "the configured database" or looks
    # for "the trained model" finds nothing, exactly as on a fresh clone.
    "DATABASE_PATH": str(_SANDBOX / "valwr.db"),
    "MODELS_PATH": str(_SANDBOX / "models"),
}


def pytest_configure(config):
    # Before collection, so nothing -- not even a module imported while
    # collecting -- gets to read the developer's settings first.
    os.environ.update(ISOLATED)
    from valwr import config as settings
    settings.load_dotenv = lambda *a, **k: False


def pytest_unconfigure(config):
    shutil.rmtree(_SANDBOX, ignore_errors=True)
