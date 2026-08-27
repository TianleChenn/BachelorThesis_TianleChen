from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_env_file_paths() -> tuple[Path, ...]:
    """Return existing project configuration files in load-priority order."""
    candidates = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example")
    return tuple(path for path in candidates if path.is_file())


def get_env_file_path() -> Path | None:
    """Return the highest-priority existing project configuration file."""
    paths = get_env_file_paths()
    return paths[0] if paths else None


def _load_env_file(path: Path) -> None:
    """Fill missing process variables from one UTF-8/BOM-compatible file."""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_local_env() -> Path | None:
    """Load .env, then use .env.example only to fill still-missing values.

    Values already present in the operating-system/process environment always
    win. Values loaded from .env are present before .env.example is read, so
    the public template can never replace local credentials or configuration.
    """
    paths = get_env_file_paths()
    for path in paths:
        _load_env_file(path)
    return paths[0] if paths else None
