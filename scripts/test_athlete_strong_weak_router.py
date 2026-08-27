"""Smoke-test the saved project-specific preference router without model calls."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_strong_weak_router import AthleteStrongWeakRouter


REQUEST = (
    "Generate the four logistic regression models corresponding to Table 1 for "
    "athletes aged 20 and above using all eight public domains, with models using "
    "no controls, sex, age, and both sex and age."
)


def main() -> None:
    result = AthleteStrongWeakRouter().predict(REQUEST)
    decision = str(result["decision"])
    print("User request:")
    print(REQUEST)
    print("\nStrong Model Probability:")
    print(f"{float(result['strong_model_probability']):.4f}")
    print("\nThreshold:")
    print(f"{float(result['threshold']):.4f}")
    print("\nDecision:")
    print(decision.title())
    print("\nSelected Model:")
    print("GPT-4.1" if decision == "strong" else "Ministral-3-8B")


if __name__ == "__main__":
    main()
