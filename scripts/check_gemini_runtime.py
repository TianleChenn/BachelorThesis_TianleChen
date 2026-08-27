"""Diagnose the configured Gemini runtime without exposing credentials."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env
from llm.model_clients import call_gemini_cloud_model


def _loaded(variable: str) -> str:
    return "LOADED" if os.getenv(variable, "").strip() else "NOT SET"


def _sanitize_error(error: object) -> str:
    text = str(error or "Unknown provider error")
    secret = os.getenv("LLM_GEMINI_API_KEY", "").strip()
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    return text[:800]


def main() -> int:
    env_path = load_local_env()
    model = os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash").strip()
    result = call_gemini_cloud_model(
        [{"role": "user", "content": "Reply only with OK"}],
        max_tokens=4096,
    )

    print(f"Environment file: {env_path.name if env_path is not None else 'Not found'}")
    print("Gemini:")
    print(f"api_key: {_loaded('LLM_GEMINI_API_KEY')}")
    print(f"model: {model}")
    print(f"requested_model: {result.requested_model}")
    print(f"actual_model: {result.actual_model}")
    print(f"endpoint: {result.endpoint}")
    print(f"success: {result.success}")
    print(f"finish_reason: {result.finish_reason}")
    print(f"input_tokens: {result.input_tokens}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"total_tokens: {result.total_tokens}")
    print(f"content_state: {result.content_state}")
    print(f"error: {_sanitize_error(result.error) if result.error else 'None'}")
    print(f"Gemini call: {'SUCCESS' if result.success else 'FAILURE'}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
