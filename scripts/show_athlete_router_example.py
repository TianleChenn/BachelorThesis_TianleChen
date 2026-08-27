"""Show one saved Strong/Weak evaluation and the offline athlete-router decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.athlete_strong_weak_router import AthleteStrongWeakRouter


CALIBRATION_RESULTS = ROOT / "artifacts" / "athlete_router_calibration_results.json"
PROMPT_ID = "routellm_cal_033"


def load_saved_sample() -> dict:
    if not CALIBRATION_RESULTS.exists():
        raise FileNotFoundError(
            f"Saved calibration results were not found: {CALIBRATION_RESULTS}"
        )
    payload = json.loads(CALIBRATION_RESULTS.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    sample = next(
        (row for row in rows if str(row.get("prompt_id")) == PROMPT_ID),
        None,
    )
    if sample is None:
        raise LookupError(f"Prompt ID {PROMPT_ID!r} was not found in the saved results")
    return sample


def main() -> int:
    sample = load_saved_sample()

    print("=" * 80)
    print("REAL STRONG / WEAK ROUTER EVALUATION EXAMPLE")
    print("=" * 80)
    print("\nPrompt ID:")
    print(sample["prompt_id"])
    print("\nUser Request:")
    print(sample["prompt"])
    print("\nRequested Analysis:")
    print(sample["requested_analysis"])
    print("\nPrivacy Route:")
    print(sample["privacy_route"])

    print("\n" + "=" * 80)
    print("STEP 1: STRONG / WEAK REFERENCE EVALUATION")
    print("=" * 80)

    strong_score = float(sample["strong_score"])
    weak_score = float(sample["weak_score"])
    reference_preference = str(sample["preferred_model"]).lower()

    print(f"GPT-4.1 Judge Score:        {strong_score}")
    print(f"Ministral-3-8B Judge Score: {weak_score}")
    print("\nJudge Reason:")
    print(sample.get("judge_reason"))
    print("\nReference Preference:")
    print(reference_preference.upper())

    strong_code = (sample.get("strong") or {}).get("generated_code")
    weak_code = (sample.get("weak") or {}).get("generated_code")
    print("\nStrong Model Generated Code:")
    print(strong_code)
    print("\nWeak Model Generated Code:")
    print(weak_code)

    print("\n" + "=" * 80)
    print("STEP 2: NEW ATHLETE ROUTER")
    print("=" * 80)

    prediction = AthleteStrongWeakRouter().predict(
        prompt=sample["prompt"],
        requested_analysis=sample["requested_analysis"],
        difficulty=sample.get("difficulty"),
        privacy_route=sample["privacy_route"],
        filters=sample.get("filters") or {},
        requires_code=True,
        router_prompt_source="saved_calibration_example",
    )

    p_strong = float(prediction["p_strong"])
    threshold = float(prediction["threshold"])
    selected_tier = str(prediction["selected_tier"]).lower()
    execution_model = prediction["execution_model"]

    print(f"Strong Model Probability:   {p_strong:.4f}")
    print(f"Threshold:                  {threshold:.4f}")
    print("\nRouter Decision:")
    print(selected_tier.upper())
    print("\nSelected Execution Model:")
    print(execution_model)

    print("\n" + "=" * 80)
    print("STEP 3: FINAL EVALUATION")
    print("=" * 80)

    correct = selected_tier == reference_preference
    print(f"Reference Preference: {reference_preference.upper()}")
    print(f"Router Selection:     {selected_tier.upper()}")
    print(f"Correct Decision:     {correct}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"""
Strong Judge Score     = {strong_score}
Weak Judge Score       = {weak_score}

Reference Preference   = {reference_preference.upper()}

Strong Probability     = {p_strong:.4f}
Threshold              = {threshold:.4f}

Router Selection       = {selected_tier.upper()}
Execution Model        = {execution_model}

Correct Routing        = {correct}
""".strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
