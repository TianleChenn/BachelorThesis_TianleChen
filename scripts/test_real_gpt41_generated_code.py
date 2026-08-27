from pprint import pprint

from sports.service import handle_user_request


def main():
    print("=" * 80, flush=True)
    print("REAL PROJECT FLOW TEST", flush=True)
    print("=" * 80, flush=True)

    user_request = (
        "Generate the four logistic regression models corresponding to Table 1 "
        "for all athletes using all eight public domains, with models using "
        "no controls, sex, age, and both sex and age."
    )

    print("\n[1] User request:", flush=True)
    print(user_request, flush=True)

    print("\n[2] Calling normal project pipeline...", flush=True)
    print(
        "User Request -> Privacy Router -> Cost-aware Router -> "
        "LLM Code Generation -> Verifier -> Local Execution",
        flush=True,
    )

    # ============================================================
    # THIS IS THE NORMAL PROJECT FLOW
    # ============================================================

    response = handle_user_request(
        user_prompt=user_request,
        use_openai=True,
        requested_analysis="table1",
        analysis_filters={},
    )

    print("\n[3] Project pipeline returned successfully.", flush=True)

    # ============================================================
    # PRIVACY ROUTING
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("PRIVACY ROUTING", flush=True)
    print("=" * 80, flush=True)

    privacy = response.get("privacy_decision") or {}

    print("Privacy route:", privacy.get("route"), flush=True)

    # ============================================================
    # COST-AWARE ROUTING
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("COST-AWARE ROUTING", flush=True)
    print("=" * 80, flush=True)

    model_decision = response.get("model_decision") or {}
    audit = response.get("pipeline_audit") or {}

    print(
        "Strong probability:",
        model_decision.get("athlete_router_probability")
        or model_decision.get("strong_model_probability"),
        flush=True,
    )

    print(
        "Threshold:",
        model_decision.get("athlete_router_threshold")
        or model_decision.get("threshold"),
        flush=True,
    )

    print(
        "Selected tier:",
        model_decision.get("selected_tier"),
        flush=True,
    )

    print(
        "Selected model:",
        model_decision.get("selected_model"),
        flush=True,
    )

    # ============================================================
    # REAL LLM CODE GENERATION
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("REAL LLM CODE GENERATION", flush=True)
    print("=" * 80, flush=True)

    code_generation = response.get("code_generation") or {}

    print(
        "Requested model:",
        code_generation.get("requested_model"),
        flush=True,
    )

    print(
        "Actual model:",
        code_generation.get("actual_model"),
        flush=True,
    )

    print(
        "Provider:",
        code_generation.get("provider"),
        flush=True,
    )

    print("\nGenerated Python code:", flush=True)
    print("-" * 80, flush=True)

    generated_code = response.get("generated_code")

    print(generated_code, flush=True)

    print("-" * 80, flush=True)

    # ============================================================
    # NEW GENERATED CODE VERIFIER
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("GENERATED CODE VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    verification = (
        response.get("code_verification")
        or response.get("code_execution")
        or {}
    )

    def read_value(name):
        if name in verification:
            return verification.get(name)
        return audit.get(name)

    print(
        "Structure validation:",
        read_value("structure_validation_passed"),
        flush=True,
    )

    print(
        "Request match:",
        read_value("request_match_passed"),
        flush=True,
    )

    print(
        "Local execution:",
        read_value("local_execution_passed"),
        flush=True,
    )

    print(
        "Result validation:",
        read_value("result_validation_passed"),
        flush=True,
    )

    print(
        "Fully correct:",
        read_value("fully_correct"),
        flush=True,
    )

    print(
        "Failure stage:",
        verification.get("failure_stage")
        or response.get("failure_stage"),
        flush=True,
    )

    print("\nRequest mismatches:", flush=True)
    pprint(verification.get("request_mismatches") or [])

    # ============================================================
    # RESULT
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("LOCAL ANALYSIS RESULT", flush=True)
    print("=" * 80, flush=True)

    result = response.get("result")

    if isinstance(result, dict):
        print("Analysis:", result.get("analysis"), flush=True)
        print("Filters:", result.get("filters"), flush=True)
        print("Result keys:", list(result.keys()), flush=True)
    else:
        print(result, flush=True)

    # ============================================================
    # FINAL STATUS
    # ============================================================

    print("\n" + "=" * 80, flush=True)
    print("FINAL STATUS", flush=True)
    print("=" * 80, flush=True)

    print("Final allowed:", response.get("allowed"), flush=True)
    print("Fully correct:", read_value("fully_correct"), flush=True)

    actual_model = (
        code_generation.get("actual_model")
        or model_decision.get("selected_model")
    )

    print("Actual generation model:", actual_model, flush=True)

    if not generated_code:
        print("\n❌ FAILED: No Python code was generated.", flush=True)
        return

    if read_value("fully_correct") is True:
        print(
            "\n✅ PASSED: LLM generated code correctly answered "
            "the user request and executed successfully.",
            flush=True,
        )
    else:
        print(
            "\n❌ FAILED: Generated code did not pass the complete verifier.",
            flush=True,
        )


if __name__ == "__main__":
    main()