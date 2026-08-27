"""The single provider-independent restricted code-generation prompt."""

from __future__ import annotations

import json

from llm.code_generation_pools import ANALYSIS_METHOD_POOL, ALLOWED_VALUE_POOL


CODE_GENERATION_PROMPT_VERSION = "pool_typed_schema_v3"
RESTRICTED_CODE_MAX_TOKENS = 4096


_GENERIC_RULES = """You translate a user's sports-analysis request into one restricted Python API call.

Return exactly one Python assignment:

result = analysis.<method>(...)

Choose the appropriate analysis method and arguments from the supplied method and value pools based only on the user request.

Rules:
- Return code only.
- Return exactly one assignment.
- Use exactly one analysis.<method>(...) call.
- Use keyword arguments only.
- Use literal values only.
- No imports or arbitrary Python.
- No dataframe or raw athlete measurement access.
- No file or network access.
- No exec or eval.
- No loops or function/class definitions.
- Do not invent method names, variable names, filter fields, or values outside the pools.
- Never use a real athlete identifier.
- For an individual request, only CURRENT_SUBJECT may be used as the subject token.

Infer the analysis method and arguments yourself from the user request.

Before returning the code, make sure that every explicit requirement in the user request is represented in the selected method and arguments. Do not omit any stated variables, filters, controls, groups, analysis types, or numeric settings.

When the user explicitly specifies a value or condition, use the corresponding allowed value from the supplied pools exactly rather than replacing it with a default.

Argument construction rules:

- Represent every explicit requirement in the user request in the generated arguments.
- Do not replace explicitly requested information with None or omit it.
- For list arguments, preserve every explicitly requested item or specification.
- When a request asks for multiple model specifications, represent each specification separately in the corresponding list.
- When a request asks for all items from a named pool, include every item from that pool exactly once.
- Use an empty filter dictionary only when the request has no cohort restriction.
- When the request explicitly gives a value, prefer that value over an API default.
- Before returning the code, silently check that the selected method and all explicit user constraints are represented. Return code only."""

_PREDEFINED_ANALYSIS_RULES = """

Predefined regression contracts:
- Table 1 must include exactly four control specifications, each exactly once: no controls, sex only, age only, and sex plus age. Encode them as controls=[[], ["sex"], ["age"], ["sex", "age"]].
- The order of the four Table 1 control specifications is not semantically important, and the order of "sex" and "age" inside the combined specification is not semantically important.
- Table 2 must use all eight public-domain predictors, group="all", and the requested cohort filters. Its predefined backend models do not take a controls argument.
"""


def build_code_generation_messages(user_request: str) -> list[dict[str, str]]:
    system = (
        _GENERIC_RULES
        + _PREDEFINED_ANALYSIS_RULES
        + "\n\nANALYSIS METHOD POOL:\n"
        + json.dumps(ANALYSIS_METHOD_POOL, ensure_ascii=False, indent=2)
        + "\n\nALLOWED VALUE POOL:\n"
        + json.dumps(ALLOWED_VALUE_POOL, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User request:\n{str(user_request or '').strip()}"},
    ]


def get_code_generation_prompt_preview(user_request: str) -> dict[str, str]:
    messages = build_code_generation_messages(user_request)
    return {
        "system_message": messages[0]["content"],
        "user_message": messages[1]["content"],
    }
