"""Print the exact provider-independent code-generation prompt without calling an LLM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.code_generation_prompt import CODE_GENERATION_PROMPT_VERSION,get_code_generation_prompt_preview


EXAMPLES = [
    (
        "Generate the four logistic regression models corresponding to Table 1 "
        "for all athletes using all eight public domains, with elite_status as the "
        "binary target. Use four model specifications: no controls, sex, age, and "
        "both sex and age."
    ),
    (
        "Calculate pairwise Pearson correlations among all eight public domains "
        "for junior national team athletes."
    ),
]

FORBIDDEN = [
    "Required contract",
    "requested_analysis",
    "required_filters",
    "Expected method",
    "Expected arguments",
    "request_mismatches",
    "result = analysis.table1(",
    "result = analysis.correlation(",
]


def main() -> None:
    print("Prompt version:")
    print(CODE_GENERATION_PROMPT_VERSION)
    combined = []
    for index, request in enumerate(EXAMPLES, start=1):
        preview = get_code_generation_prompt_preview(request)
        print("=" * 80)
        print(f"EXAMPLE {index}")
        print("=" * 80)
        print("SYSTEM MESSAGE")
        print(preview["system_message"])
        print("\nUSER MESSAGE")
        print(preview["user_message"])
        combined.extend(preview.values())
    prompt_text = "\n".join(combined)
    leaks = [value for value in FORBIDDEN if value in prompt_text]
    if leaks:
        raise AssertionError(f"Gold-answer leakage detected: {leaks}")
    print("\nCompleteness instruction present: YES" if "every explicit requirement" in prompt_text else "\nCompleteness instruction present: NO")
    print("Exact-value instruction present: YES" if "use the corresponding allowed value" in prompt_text else "Exact-value instruction present: NO")
    print("Gold contract exposed: NO")
    print("requested_analysis exposed: NO")
    print("requested_filters exposed: NO")


if __name__ == "__main__":
    main()
