"""Diagnose the configured Gemini runtime without exposing credentials."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env
from llm.model_clients import call_gemini_cloud_model


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def main() -> int:
    load_local_env()
    provider = os.getenv("LLM_GEMINI_PROVIDER", "openai_compatible").strip()
    model = os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash").strip()
    base_url = os.getenv("LLM_GEMINI_BASE_URL", DEFAULT_BASE_URL).strip()
    api_key_loaded = bool(os.getenv("LLM_GEMINI_API_KEY"))

    print(f"provider={provider}")
    print(f"model={model}")
    print(f"base_url={base_url}")
    print(f"api_key_loaded={api_key_loaded}")

    result = call_gemini_cloud_model([
        {"role": "user", "content": "Reply only with OK"},
    ])
    print(f"success={result.success}")
    print(f"unavailable={result.unavailable}")
    print(f"requested_model={result.requested_model}")
    print(f"actual_model={result.actual_model}")
    print(f"endpoint={result.endpoint}")
    print(f"http_or_sanitized_error={result.error or 'None'}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
