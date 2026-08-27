"""Prompt-only variants for restricted code-generation ablation."""

from __future__ import annotations

import hashlib
from llm.code_generation_pools import ANALYSIS_METHOD_POOL
from llm.code_generation_prompt import build_code_generation_messages

PROMPT_VERSIONS = ("basic", "interface_guided", "full")
PROMPT_DISPLAY_NAMES = {"basic": "Basic", "interface_guided": "Interface Guided", "full": "Full"}

_BASIC_RULES = """You translate a user's sports-analysis request into one restricted Python API call.

Return exactly one Python assignment:

result = analysis.<method>(...)

Supported analysis methods:
{method_names}

Rules:
- Return code only and no explanatory prose.
- Return exactly one assignment and exactly one supported analysis.<method>(...) call.
- No imports or arbitrary Python statements.
- No file or network access.
- No exec or eval.
"""


def _basic_system_prompt() -> str:
    return _BASIC_RULES.format(method_names=", ".join(ANALYSIS_METHOD_POOL))


def _interface_guided_system_prompt() -> str:
    skeletons = []
    for method, details in ANALYSIS_METHOD_POOL.items():
        arguments = ",\n    ".join(f"{name}=..." for name in details["arguments"])
        skeletons.append(f"result = analysis.{method}(\n    {arguments}\n)")
    return (_basic_system_prompt() + "\nRESTRICTED ANALYSIS CALL INTERFACES:\n\n"
            + "\n\n".join(skeletons))


def build_codegen_ablation_messages(prompt_version: str, user_request: str) -> list[dict[str, str]]:
    request = str(user_request or "").strip()
    if prompt_version == "full":
        return build_code_generation_messages(request)
    if prompt_version == "basic":
        system = _basic_system_prompt()
    elif prompt_version == "interface_guided":
        system = _interface_guided_system_prompt()
    else:
        raise ValueError(f"Unknown code-generation prompt version: {prompt_version}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User request:\n{request}"},
    ]


def prompt_sha256(prompt_version: str) -> str:
    system = build_codegen_ablation_messages(prompt_version, "")[0]["content"]
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


def prompt_metadata() -> dict:
    return {
        version: {
            "sha256": prompt_sha256(version),
            "system_prompt": build_codegen_ablation_messages(version, "")[0]["content"],
        }
        for version in PROMPT_VERSIONS
    }
