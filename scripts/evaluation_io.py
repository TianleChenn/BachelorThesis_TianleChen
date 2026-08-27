"""Neutral JSON and JSONL helpers used by current evaluation workflows."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("cloud", "collaboration", "local_edge", "blocked")


def project_path(value) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_json(value):
    return json.loads(project_path(value).read_text(encoding="utf-8"))


def write_json(value, payload):
    path = project_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def sha256_file(value) -> str:
    return hashlib.sha256(project_path(value).read_bytes()).hexdigest()


def normalized(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def append_jsonl(value, row) -> None:
    path = project_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(value) -> list[dict]:
    path = project_path(value)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
