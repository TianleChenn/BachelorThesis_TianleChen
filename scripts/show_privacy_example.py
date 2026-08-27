"""Print one live Method C privacy assessment for a paper screenshot."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.llm_privacy_assessor import assess_privacy_with_llm


PROMPT = (
    "Generate the four logistic regression models corresponding to Table 1 "
    "for athletes aged 20 and above using all eight public domains, "
    "with models using no controls, sex, age, and both sex and age."
)


def main() -> int:
    print("=" * 70)
    print("METHOD C - LLM PRIVACY ASSESSMENT")
    print("=" * 70)
    print()
    print("User Request:")
    print(PROMPT)
    print()

    # Bypass the cache so the output comes from a fresh call to Method C's
    # production privacy-assessment entry point.
    result = assess_privacy_with_llm(PROMPT, use_cache=False)

    if not result.success:
        print("Privacy assessment failed:")
        print(result.error)
        return 1

    print("Privacy Assessment Result")
    print("-" * 70)
    print(f"Privacy Risk Score : {result.privacy_risk_score:.4f}")
    print(f"Subject Scope      : {result.subject_scope:.4f}")
    print(f"Data Sensitivity   : {result.data_sensitivity:.4f}")
    print(f"Disclosure Level   : {result.disclosure_level:.4f}")
    print(f"Blocked Request    : {result.blocked_request}")
    print()
    print(f"Requested Model    : {result.requested_model}")
    print(f"Actual Model       : {result.actual_model}")
    print(f"Provider           : {result.provider}")
    print(f"Cache Used         : {result.cache_used}")
    print()
    print("Explanation:")
    print(result.explanation)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
