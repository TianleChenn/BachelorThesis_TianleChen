"""Run each supported analysis through the real Local Edge generator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.generated_code_verifier import inspect_generated_code, verify_and_execute_generated_code
from llm.code_generator import generate_code
from llm.code_generation_prompt import build_code_generation_messages
from llm.code_generator import _call_for_channel


REQUESTS = {
    "table1": "Run Table 1 for the selected cohort using all standardized domains.",
    "table2": "Run Table 2 for the selected cohort using all standardized domains.",
    "figure1": "Generate Figure 1 for the selected cohort using all standardized domains.",
    "figure2": "Generate Figure 2 showing 50 anonymous athletes for the selected cohort.",
    "correlation": "Run correlation for the selected cohort using all standardized domains.",
    "variance_analysis": "Run variance analysis for the selected cohort using all standardized domains.",
    "individual_profile": "Generate the protected standardized profile for CURRENT_SUBJECT.",
}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--inspect":
        name=sys.argv[2]
        messages=build_code_generation_messages(REQUESTS[name])
        call=_call_for_channel("local",messages)
        print(call.content or call.error)
        print(f"MODEL CALL: {'PASS' if call.success else 'FAIL'}")
        if not call.success or not call.content:
            print("STRUCTURE VALIDATION: FAIL")
            print("REQUEST MATCH: FAIL")
            print(f"VALIDATION ERROR: {call.error or 'No model output.'}")
            return 1
        inspected=inspect_generated_code(
            call.content,user_request=REQUESTS[name],requested_analysis=name,requested_filters={}
        )
        print(f"STRUCTURE VALIDATION: {'PASS' if inspected.structure_validation_passed else 'FAIL'}")
        print(f"REQUEST MATCH: {'PASS' if inspected.request_match_passed else 'FAIL'}")
        print(f"VALIDATION ERROR: {inspected.validation_error or 'None'}")
        return 0 if inspected.request_match_passed else 1
    failures = []
    selected = {name: REQUESTS[name] for name in sys.argv[1:] if name in REQUESTS} or REQUESTS
    for requested_analysis, prompt in selected.items():
        generated = generate_code(
            prompt, {"selected_model": None}, {"route": "local_edge", "blocked": False},
            requested_analysis=requested_analysis, requested_filters={},
        )
        if not generated.code:
            print(f"{requested_analysis}: GENERATION FAIL ({generated.failure_stage})")
            failures.append(requested_analysis)
            continue
        execution = verify_and_execute_generated_code(
            generated.code, user_request=prompt,requested_analysis=requested_analysis,
            requested_filters={},
            subject_reference="Athlete_003" if requested_analysis == "individual_profile" else None,
        )
        if not execution.executed:
            print(f"{requested_analysis}: EXECUTION FAIL ({execution.failure_stage})")
            failures.append(requested_analysis)
            continue
        print(f"{requested_analysis}: PASS (repair={'yes' if generated.generation_retry_used else 'no'})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
