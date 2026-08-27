from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_env_file_path() -> Path | None:
    """Return the centrally selected project environment file."""
    explicit = os.getenv("THESIS_ENV_FILE", "").strip()
    if explicit:
        path = (PROJECT_ROOT / explicit).resolve()
        try:
            path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Configured environment file must be inside the project root: {explicit}"
            ) from exc
        if not path.is_file():
            raise RuntimeError(
                f"Configured environment file does not exist: {explicit}"
            )
        return path

    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        return env_path

    example_path = PROJECT_ROOT / ".env.example"
    if example_path.is_file():
        return example_path

    return None


def load_local_env() -> Path | None:
    """Load the selected UTF-8 environment file without overriding process values."""
    env_path = get_env_file_path()
    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path
