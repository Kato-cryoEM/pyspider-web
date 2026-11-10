"""User-scope configuration persisted in ``~/.pyspider_config.json``.

Currently just the default directory for auto-saved node outputs — kept
generic so more knobs (e.g. GPU device index) can slot in later without
touching the API surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path.home() / ".pyspider_config.json"
_DEFAULTS: dict[str, Any] = {
    "default_save_dir": str(Path.home() / "pyspider_saved"),
}
_ALLOWED_KEYS = set(_DEFAULTS.keys())


def _load() -> dict:
    if _CONFIG_PATH.is_file():
        try:
            with _CONFIG_PATH.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {**_DEFAULTS, **{k: v for k, v in data.items()
                                        if k in _ALLOWED_KEYS}}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULTS)


def get_settings() -> dict:
    return _load()


def update_settings(patch: dict) -> dict:
    current = _load()
    for k, v in patch.items():
        if k in _ALLOWED_KEYS and v is not None:
            current[k] = v
    try:
        with _CONFIG_PATH.open("w") as f:
            json.dump(current, f, indent=2)
    except OSError as e:
        raise RuntimeError(f"failed to write {_CONFIG_PATH}: {e}")
    return current
