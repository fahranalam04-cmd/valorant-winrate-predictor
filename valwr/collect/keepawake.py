"""Keep the machine awake for the duration of a crawl.

The first overnight run died because the laptop slept ten minutes after the
keyboard went quiet. The obvious fix -- `powercfg /change standby-timeout-ac 0`
-- changes how the machine behaves for everything, permanently, and has to be
remembered and undone.

Windows exposes a better mechanism: a process can declare that the system is
busy on its behalf. The request lives only as long as this process, reverts
automatically if it crashes, and leaves the user's power settings untouched.

Display sleep is deliberately NOT blocked -- there is no reason to keep a
screen lit all night to download JSON.
"""

from __future__ import annotations

import ctypes
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000040


class KeepAwake:
    """Context manager. A no-op on platforms without the API."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.platform == "win32"
        self.active = False

    def __enter__(self) -> "KeepAwake":
        if not self.enabled:
            return self
        try:
            ok = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            self.active = bool(ok)
        except (AttributeError, OSError):
            self.active = False
        return self

    def __exit__(self, *exc) -> None:
        if self.active:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except (AttributeError, OSError):
                pass
            self.active = False

    @property
    def status(self) -> str:
        if not self.enabled:
            return "not supported on this platform"
        return "sleep blocked while crawling" if self.active else "FAILED -- machine may sleep"
