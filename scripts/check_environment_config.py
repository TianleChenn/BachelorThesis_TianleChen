from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm.env as env_module


ACTIVE_ENVIRONMENT_VARIABLES = (
    "OPENAI_API_KEY",
    "LLM_STRONG_MODEL",
    "PRIVACY_RISK_CACHE_ENABLED",
    "LLM_GEMINI_PROVIDER",
    "LLM_GEMINI_MODEL",
    "LLM_GEMINI_BASE_URL",
    "LLM_GEMINI_API_KEY",
    "LLM_CLAUDE_PROVIDER",
    "LLM_CLAUDE_MODEL",
    "LLM_CLAUDE_BASE_URL",
    "LLM_CLAUDE_API_KEY",
    "LLM_LOCAL_PROVIDER",
    "LLM_LOCAL_MODEL",
    "LLM_LOCAL_BASE_URL",
    "LLM_LOCAL_API_KEY",
    "LLM_LOCAL_DEVICE",
    "LLM_LOCAL_MAX_NEW_TOKENS",
    "LLM_LOCAL_MODEL_ID",
    "LLM_PRIVACY_GATER_MODEL_PATH",
    "LLAMA_SERVER_PATH",
    "EVAL_OPENAI_MODEL",
    "OPENAI_STRONG_MODEL",
    "EVAL_GEMINI_MODEL",
    "EVAL_CLAUDE_MODEL",
)

LEGACY_OPTIONAL_ENVIRONMENT_VARIABLES = (
    "LLM_PRIVACY_DATASET_MODEL",
    "LLM_WEAK_MODEL",
    "LLM_WEAK_PROVIDER",
    "LLM_WEAK_PROVIDER_MODEL_ID",
    "LLM_WEAK_BASE_URL",
    "LLM_WEAK_API_KEY",
    "ROUTELLM_CPT_TARGET",
    "LLM_JUDGE_MODEL",
)

SECRET_ENVIRONMENT_VARIABLES = {
    "OPENAI_API_KEY",
    "LLM_GEMINI_API_KEY",
    "LLM_CLAUDE_API_KEY",
    "LLM_LOCAL_API_KEY",
    "LLM_WEAK_API_KEY",
}


def read_variable_names(path: Path) -> tuple[str, ...]:
    """Read variable names only; values are intentionally discarded."""
    if not path.is_file():
        return ()
    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def unsafe_template_secret_names(path: Path) -> tuple[str, ...]:
    """Return only the names of secret fields containing unsafe public values."""
    unsafe: list[str] = []
    if not path.is_file():
        return ()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        value = value.strip('"').strip("'")
        safe_local_sentinel = name == "LLM_LOCAL_API_KEY" and value.lower() == "none"
        if name in SECRET_ENVIRONMENT_VARIABLES and value and not safe_local_sentinel:
            unsafe.append(name)
    return tuple(unsafe)


def is_configured(name: str) -> bool:
    """Return False for missing or blank values without exposing the value."""
    return bool(os.getenv(name, "").strip())


def _status(name: str) -> str:
    return "configured" if is_configured(name) else "missing"


def _safe_setting(name: str) -> str:
    """Show an explicitly non-secret setting, with a clear missing marker."""
    value = os.getenv(name, "").strip()
    return value if value else "missing"


def _print_name_list(title: str, names: tuple[str, ...] | list[str]) -> None:
    print(f"{title}: {len(names)}")
    for name in names:
        print(f"  - {name}")


def main() -> int:
    root = env_module.PROJECT_ROOT
    env_path = root / ".env"
    example_path = root / ".env.example"
    env_names = read_variable_names(env_path)
    example_names = read_variable_names(example_path)

    env_module.load_local_env()

    print("Environment Configuration")
    print("=========================")
    print()
    print("Configuration source:")
    print(f".env: {'present' if env_path.is_file() else 'missing'}")
    print(f".env.example: {'present' if example_path.is_file() else 'missing'}")

    print()
    print("OpenAI Privacy Assessor")
    print(f"OPENAI_API_KEY: {_status('OPENAI_API_KEY')}")
    print(f"LLM_STRONG_MODEL: {_safe_setting('LLM_STRONG_MODEL')}")

    print()
    print("Gemini Cloud Model")
    print(f"LLM_GEMINI_MODEL: {_safe_setting('LLM_GEMINI_MODEL')}")
    print(f"LLM_GEMINI_BASE_URL: {_status('LLM_GEMINI_BASE_URL')}")
    print(f"LLM_GEMINI_API_KEY: {_status('LLM_GEMINI_API_KEY')}")

    print()
    print("Claude Evaluation Model")
    print(f"LLM_CLAUDE_MODEL: {_safe_setting('LLM_CLAUDE_MODEL')}")
    print(f"LLM_CLAUDE_BASE_URL: {_status('LLM_CLAUDE_BASE_URL')}")
    print(f"LLM_CLAUDE_API_KEY: {_status('LLM_CLAUDE_API_KEY')}")

    print()
    print("Local Model")
    print(f"LLM_LOCAL_MODEL: {_safe_setting('LLM_LOCAL_MODEL')}")
    print(f"LLM_LOCAL_BASE_URL: {_safe_setting('LLM_LOCAL_BASE_URL')}")

    env_set = set(env_names)
    example_set = set(example_names)
    only_env = tuple(name for name in env_names if name not in example_set)
    only_example = tuple(name for name in example_names if name not in env_set)
    missing_active = tuple(
        name for name in ACTIVE_ENVIRONMENT_VARIABLES if name not in example_set
    )
    unsafe_template_secrets = unsafe_template_secret_names(example_path)
    shared_env_order = tuple(name for name in env_names if name in example_set)
    shared_example_order = tuple(name for name in example_names if name in env_set)
    shared_order_pass = shared_env_order == shared_example_order

    print()
    print("Variable-name comparison (values are never shown)")
    _print_name_list("Keys only in .env", only_env)
    _print_name_list("Keys only in .env.example", only_example)
    _print_name_list("Active runtime keys missing from .env.example", missing_active)
    _print_name_list("Unsafe non-empty secret fields in .env.example", unsafe_template_secrets)
    print(f"Shared variable order: {'PASS' if shared_order_pass else 'FAIL'}")

    structure_pass = (
        example_path.is_file()
        and not missing_active
        and not unsafe_template_secrets
        and shared_order_pass
    )
    cloud_credentials_pass = all(
        is_configured(name)
        for name in ("OPENAI_API_KEY", "LLM_GEMINI_API_KEY", "LLM_CLAUDE_API_KEY")
    )

    print()
    print("Result:")
    print(f"Environment structure: {'PASS' if structure_pass else 'FAIL'}")
    print(f"Cloud credentials: {'PASS' if cloud_credentials_pass else 'NOT CONFIGURED'}")
    if not cloud_credentials_pass:
        print("Copy .env.example to .env and provide your own API credentials.")

    return 0 if structure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
