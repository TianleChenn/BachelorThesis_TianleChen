"""Run minimal credential-safe checks of the configured cloud runtimes."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env
from llm.model_clients import call_gemini_cloud_model, call_privacy_risk_model


SECRET_VARIABLES = (
    "OPENAI_API_KEY",
    "LLM_GEMINI_API_KEY",
)


def _loaded(variable: str) -> str:
    return "LOADED" if os.getenv(variable, "").strip() else "NOT SET"


def _sanitize_error(error: object) -> str:
    text = str(error or "Unknown provider error")
    for variable in SECRET_VARIABLES:
        secret = os.getenv(variable, "").strip()
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    return text[:800]


def _report_result(label: str, result: object) -> bool:
    success = bool(getattr(result, "success", False))
    print(f"{label}: {'SUCCESS' if success else 'FAILURE'}")
    if not success:
        print(f"{label} error: {_sanitize_error(getattr(result, 'error', None))}")
    return success


def _print_gemini_result(result: object) -> bool:
    success = bool(getattr(result, "success", False))
    print("Gemini:")
    print(f"api_key: {_loaded('LLM_GEMINI_API_KEY')}")
    print(f"model: {os.getenv('LLM_GEMINI_MODEL', '').strip() or 'gemini-3.5-flash'}")
    print(f"requested_model: {getattr(result, 'requested_model', None)}")
    print(f"actual_model: {getattr(result, 'actual_model', None)}")
    print(f"endpoint: {getattr(result, 'endpoint', None)}")
    print(f"success: {success}")
    print(f"finish_reason: {getattr(result, 'finish_reason', None)}")
    print(f"input_tokens: {getattr(result, 'input_tokens', None)}")
    print(f"output_tokens: {getattr(result, 'output_tokens', None)}")
    print(f"total_tokens: {getattr(result, 'total_tokens', None)}")
    print(f"content_state: {getattr(result, 'content_state', None)}")
    error = getattr(result, "error", None)
    print(f"error: {_sanitize_error(error) if error else 'None'}")
    print(f"Gemini call: {'SUCCESS' if success else 'FAILURE'}")
    return success


def main() -> int:
    env_path = load_local_env()
    print(f"Environment file: {env_path.name if env_path is not None else 'Not found'}")
    print(f"OPENAI_API_KEY: {_loaded('OPENAI_API_KEY')}")
    print(f"LLM_GEMINI_API_KEY: {_loaded('LLM_GEMINI_API_KEY')}")
    print(f"GPT-4.1 model: {os.getenv('LLM_STRONG_MODEL', '').strip() or 'gpt-4.1'}")
    print(
        "Gemini model: "
        f"{os.getenv('LLM_GEMINI_MODEL', '').strip() or 'gemini-3.5-flash'}"
    )
    print()
    print("GPT-4.1 Privacy Assessor is executed before privacy routing.")
    print(
        "If it fails, prism_route intentionally selects safe_local_fallback / "
        "local_edge, so the cost-aware Cloud/Local router and Gemini may never run."
    )

    messages = [{"role": "user", "content": "Reply only with OK"}]
    privacy_result = call_privacy_risk_model(messages, max_tokens=8)
    gemini_result = call_gemini_cloud_model(messages, max_tokens=4096)
    privacy_ok = _report_result("GPT-4.1 privacy-assessor call", privacy_result)
    gemini_ok = _print_gemini_result(gemini_result)
    return 0 if privacy_ok and gemini_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
