"""Print a credential-safe summary of the centrally selected API configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.env import load_local_env


def _loaded(variable: str) -> str:
    return "LOADED" if bool(os.getenv(variable)) else "NOT SET"


def main() -> int:
    env_path = load_local_env()

    print("=" * 60)
    print("API CONFIGURATION CHECK")
    print("=" * 60)
    print("\nEnvironment file:")
    print(f"  {env_path if env_path is not None else 'Not found'}")

    print("\nOpenAI / GPT-4.1")
    print(f"  Model: {os.getenv('LLM_STRONG_MODEL', 'gpt-4.1')}")
    print(f"  OPENAI_API_KEY: {_loaded('OPENAI_API_KEY')}")

    print("\nGemini")
    print(f"  Model: {os.getenv('LLM_GEMINI_MODEL', 'gemini-3.5-flash')}")
    print(f"  LLM_GEMINI_API_KEY: {_loaded('LLM_GEMINI_API_KEY')}")

    print("\nClaude")
    print(f"  Model: {os.getenv('LLM_CLAUDE_MODEL', 'claude-sonnet-5')}")
    print(f"  LLM_CLAUDE_API_KEY: {_loaded('LLM_CLAUDE_API_KEY')}")

    print("\nLocal Model")
    print(f"  Model: {os.getenv('LLM_LOCAL_MODEL', 'Ministral-3-8B-Local')}")
    print(f"  LLM_LOCAL_API_KEY: {_loaded('LLM_LOCAL_API_KEY')}")

    print("\nNo network calls performed: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
