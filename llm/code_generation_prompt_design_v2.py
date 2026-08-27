"""Independent prompt definitions for Restricted Code Generation Design V2."""

from __future__ import annotations

import hashlib
import json

from llm.code_generation_pools import ANALYSIS_METHOD_POOL, ALLOWED_VALUE_POOL
from llm.code_generation_prompt import build_code_generation_messages

PROMPT_VERSIONS = ("basic_interface", "defined", "full")
PROMPT_DISPLAY_NAMES = {
    "basic_interface": "Basic Interface",
    "defined": "Defined",
    "full": "Full",
}

_BASE_RULES = """Translate the user request into exactly one Restricted Analysis Call.

Required output format:

result = analysis.<method>(...)

Syntax and safety requirements:
- Return exactly one assignment containing exactly one supported analysis.<method>(...) call.
- Use keyword arguments and literal values only.
- Infer the appropriate method and parameter contents from the user request.
- Return code only, with no explanatory prose.
- No imports or arbitrary Python statements.
- No dataframe, file, or network access.
- No exec or eval.
- No loops or function/class definitions.
"""


def _interface_skeletons() -> str:
    rows = []
    for method, details in ANALYSIS_METHOD_POOL.items():
        arguments = ",\n    ".join(f"{name}=..." for name in details["arguments"])
        rows.append(f"result = analysis.{method}(\n    {arguments}\n)")
    return "\n\n".join(rows)


def _basic_interface_system_prompt() -> str:
    return (_BASE_RULES + "\nSupported Restricted Analysis Call interfaces:\n\n"
            + _interface_skeletons())


_PARAMETER_DEFINITIONS = {
    "predictors": "Athlete-domain predictors included in a regression.",
    "variables": "Athlete domains included in an analysis.",
    "filters": "Cohort conditions restricting which athletes are included.",
    "target": "The outcome analyzed by a method.",
    "controls": "Adjustment-variable specifications used by a regression.",
    "group": "The population grouping option used by a predefined analysis.",
    "group_field": "The field defining comparison groups.",
    "groups": "The groups compared by an analysis.",
    "method": "The supported calculation option for an analysis.",
    "visualization": "Whether the analysis should return its supported visualization.",
    "correlation_threshold": "The threshold used by the network-style analysis.",
    "variance_iterations": "The iteration count used by Figure 1 variance calculations.",
    "iterations": "The iteration count used by variance analysis.",
    "max_athletes": "The maximum number of anonymous profiles displayed.",
    "reference_group": "The comparison population used as a reference.",
    "subject_token": "The protected token representing the current anonymous subject.",
    "output_mode": "The supported protected output representation.",
}


def _defined_system_prompt() -> str:
    method_definitions = "\n".join(
        f"- {method}: {details['description']}" for method, details in ANALYSIS_METHOD_POOL.items()
    )
    used_parameters = dict.fromkeys(
        name for details in ANALYSIS_METHOD_POOL.values() for name in details["arguments"]
    )
    parameter_definitions = "\n".join(
        f"- {name}: {_PARAMETER_DEFINITIONS[name]}" for name in used_parameters
    )
    return (
        _basic_interface_system_prompt()
        + "\n\nConcise method definitions:\n"
        + method_definitions
        + "\n\nConcise parameter definitions:\n"
        + parameter_definitions
        + "\n\nAllowed values:\n"
        + json.dumps(ALLOWED_VALUE_POOL, ensure_ascii=False, indent=2)
        + "\n\nBoolean parameters use the Python literals true or false as True or False."
    )


def build_codegen_prompt_design_v2_messages(
    prompt_version: str, user_request: str
) -> list[dict[str, str]]:
    request = str(user_request or "").strip()
    if prompt_version == "full":
        return build_code_generation_messages(request)
    if prompt_version == "basic_interface":
        system = _basic_interface_system_prompt()
    elif prompt_version == "defined":
        system = _defined_system_prompt()
    else:
        raise ValueError(f"Unknown V2 prompt version: {prompt_version}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User request:\n{request}"},
    ]


def prompt_sha256(prompt_version: str) -> str:
    system = build_codegen_prompt_design_v2_messages(prompt_version, "")[0]["content"]
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


def prompt_metadata() -> dict:
    return {
        version: {
            "display_name": PROMPT_DISPLAY_NAMES[version],
            "sha256": prompt_sha256(version),
            "system_prompt": build_codegen_prompt_design_v2_messages(version, "")[0]["content"],
        }
        for version in PROMPT_VERSIONS
    }
