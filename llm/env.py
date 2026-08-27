from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ENV_FILE = Path(".env")
FALLBACK_ENV_FILE = Path(".env.example")


def get_env_file_paths() -> tuple[Path, ...]:
    """Return existing project configuration files in load-priority order."""
    candidates = (PROJECT_ROOT / PRIMARY_ENV_FILE, PROJECT_ROOT / FALLBACK_ENV_FILE)
    return tuple(path for path in candidates if path.is_file())


def get_env_file_path() -> Path | None:
    """Return the highest-priority existing project configuration file."""
    paths = get_env_file_paths()
    return paths[0] if paths else None


def _load_env_file(
    path: Path,
    *,
    override: bool,
    ignore_blank: bool,
) -> None:
    """Load one UTF-8/BOM-compatible file with explicit merge behavior."""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'").strip()
        if not key or (ignore_blank and not value):
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def load_local_env() -> Path | None:
    """Load the authoritative local file, or the public template as fallback.

    A readable .env is the only file loaded and its values replace same-name
    process values. If .env is missing or unreadable, .env.example supplies
    nonblank defaults without replacing process configuration.
    """
    env_path = PROJECT_ROOT / PRIMARY_ENV_FILE
    try:
        _load_env_file(env_path, override=True, ignore_blank=False)
    except (OSError, UnicodeError):
        pass
    else:
        return env_path

    example_path = PROJECT_ROOT / FALLBACK_ENV_FILE
    try:
        _load_env_file(example_path, override=False, ignore_blank=True)
    except (OSError, UnicodeError):
        return None
    return example_path
