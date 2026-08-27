from pprint import pprint

from llm import code_generator
from llm.generated_code_verifier import verify_and_execute_generated_code
from sports.config import PREDICTORS
from sports.service import handle_user_request


# ============================================================
# TEST REQUEST
#
# This request was previously routed to:
# Privacy Route = Cloud
# Cost Router = Strong
# GPT-4.1
# ============================================================

USER_REQUEST = (
    "Generate the four logistic regression models corresponding to Table 1 "
    "for all athletes using all eight public domains, with elite_status as "
    "the binary target. Use four model specifications: no controls, sex, "
    "age, and both sex and age."
)

# ============================================================
# IMPORTANT:
#
# These two values are LOCAL ground truth.
# They are allowed to go to the local verifier.
# They MUST NOT be exposed as hidden answers to GPT-4.1.
# ============================================================

REQUESTED_ANALYSIS = "table1"
REQUESTED_FILTERS = {}


def as_dict(obj):
    """Support either dataclass result or dictionary result."""

    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)

    raise TypeError(f"Cannot convert result to dict: {type(obj)}")


def main():

    print("\n" + "=" * 100)
    print("NEW PROMPT + GENERATED CODE VERIFIER END-TO-END TEST")
    print("=" * 100)

    # ============================================================
    # PART 1
    # SPY ON THE REAL CODE-GENERATION GPT-4.1 CALL
    # ============================================================

    print("\n[PART 1] Preparing prompt spy...")

    if not hasattr(code_generator, "call_strong_model"):
        raise RuntimeError(
            "llm.code_generator no longer exposes call_strong_model. "
            "The provider integration has changed and this test script "
            "needs to patch the new provider entry point."
        )

    real_call_strong_model = code_generator.call_strong_model

    captured = {
        "messages": None,
        "call_count": 0,
    }

    def spy_call_strong_model(messages, *args, **kwargs):

        captured["messages"] = messages
        captured["call_count"] += 1

        print("\n" + "=" * 100)
        print("ACTUAL CODE-GENERATION MESSAGES SENT TO GPT-4.1")
        print("=" * 100)

        for index, message in enumerate(messages, start=1):

            role = str(message.get("role") or "").upper()
            content = str(message.get("content") or "")

            print(f"\n--- MESSAGE {index}: {role} ---")
            print(content)

        print("\n" + "=" * 100)
        print("END OF ACTUAL GPT-4.1 INPUT")
        print("=" * 100)

        # Continue with the REAL GPT-4.1 API call.
        return real_call_strong_model(
            messages,
            *args,
            **kwargs,
        )

    # Replace the actual function used by production code.
    code_generator.call_strong_model = spy_call_strong_model

    try:

        print("\n[PART 2] Running NORMAL project pipeline...")
        print()
        print("User Request")
        print("    ↓")
        print("Privacy Router")
        print("    ↓")
        print("Cost-aware Router")
        print("    ↓")
        print("GPT-4.1 Code Generation")
        print("    ↓")
        print("Generated Code Verifier")
        print("    ↓")
        print("Local RestrictedAnalysisAPI")
        print()

        response = handle_user_request(
            user_prompt=USER_REQUEST,
            use_openai=True,
            requested_analysis=REQUESTED_ANALYSIS,
            analysis_filters=REQUESTED_FILTERS,
        )

    finally:

        # ALWAYS restore the production function.
        code_generator.call_strong_model = real_call_strong_model

    # ============================================================
    # PART 3
    # CONFIRM THE NEW PROMPT WAS ACTUALLY USED
    # ============================================================

    print("\n" + "=" * 100)
    print("PART 3 — VERIFY ACTIVE PROMPT")
    print("=" * 100)

    messages = captured.get("messages")

    if not messages:
        raise AssertionError(
            "FAILED: No code-generation GPT-4.1 messages were captured."
        )

    print(f"\nCaptured GPT-4.1 code-generation calls: {captured['call_count']}")

    joined_prompt = "\n".join(
        str(message.get("content") or "")
        for message in messages
    )

    # ------------------------------------------------------------
    # 3A. All seven methods must be available
    # ------------------------------------------------------------

    expected_methods = [
        "table1",
        "table2",
        "figure1",
        "figure2",
        "correlation",
        "variance_analysis",
        "individual_profile",
    ]

    print("\n[3A] Analysis Method Pool")

    for method in expected_methods:

        present = method in joined_prompt

        print(
            f"{method:25s}"
            f"{'FOUND ✅' if present else 'MISSING ❌'}"
        )

        assert present, (
            f"New Method Pool is not active. "
            f"Missing method: {method}"
        )

    # ------------------------------------------------------------
    # 3B. All eight public domains must be available
    # ------------------------------------------------------------

    print("\n[3B] Allowed Value Pool — Public Domains")

    assert len(PREDICTORS) == 8, (
        f"Expected 8 public domains, found {len(PREDICTORS)}"
    )

    for domain in PREDICTORS:

        present = domain in joined_prompt

        print(
            f"{domain:35s}"
            f"{'FOUND ✅' if present else 'MISSING ❌'}"
        )

        assert present, (
            f"Allowed Value Pool is incomplete. "
            f"Missing domain: {domain}"
        )

    # ------------------------------------------------------------
    # 3C. Natural-language request must be present
    # ------------------------------------------------------------

    print("\n[3C] Natural-language request")

    assert USER_REQUEST in joined_prompt

    print("User request found ✅")

    # ------------------------------------------------------------
    # 3D. Old gold-answer leakage must NOT be present
    # ------------------------------------------------------------

    print("\n[3D] Gold-answer leakage check")

    forbidden_phrases = [
        "Required contract",
        "required_contract",
        "requested_analysis",
        "required_filters",
        "Expected method",
        "Expected arguments",
        "request_mismatches",
        "Generate code only for requested analysis",
        "Do not use another analysis",
        "MANDATORY UI ARGUMENT",
    ]

    for phrase in forbidden_phrases:

        leaked = phrase.lower() in joined_prompt.lower()

        print(
            f"{phrase:48s}"
            f"{'LEAKED ❌' if leaked else 'NOT PRESENT ✅'}"
        )

        assert not leaked, (
            f"Old/gold-answer information leaked to GPT: {phrase}"
        )

    print("\n✅ ACTIVE PROMPT CHECK PASSED")

    # ============================================================
    # PART 4
    # SHOW WHICH MODEL REALLY GENERATED THE CODE
    # ============================================================

    print("\n" + "=" * 100)
    print("PART 4 — REAL MODEL AND GENERATED CODE")
    print("=" * 100)

    code_generation = response.get("code_generation") or {}
    model_decision = response.get("model_decision") or {}
    pipeline = response.get("pipeline_audit") or {}

    actual_model = (
        code_generation.get("actual_model")
        or pipeline.get("llm_model_used")
    )

    requested_model = code_generation.get("requested_model")
    provider = code_generation.get("provider")

    print("\nSelected tier:")
    print(model_decision.get("selected_tier"))

    print("\nRouter selected model:")
    print(model_decision.get("selected_model"))

    print("\nRequested generation model:")
    print(requested_model)

    print("\nActual generation model:")
    print(actual_model)

    print("\nProvider:")
    print(provider)

    generated_code = response.get("generated_code")

    print("\nGPT-4.1 GENERATED CODE")
    print("-" * 100)
    print(generated_code)
    print("-" * 100)

    assert generated_code, "No generated Python code returned."

    assert actual_model, "Actual model was not recorded."

    assert "gpt-4.1" in str(actual_model).lower(), (
        f"Expected real GPT-4.1 generation, "
        f"but actual model was: {actual_model}"
    )

    print("\n✅ REAL GPT-4.1 GENERATION CONFIRMED")

    # ============================================================
    # PART 5
    # VERIFY THE NORMAL PIPELINE'S NEW VERIFIER RESULT
    # ============================================================

    print("\n" + "=" * 100)
    print("PART 5 — VERIFY NEW GENERATED CODE VERIFIER")
    print("=" * 100)

    verification = (
        response.get("code_verification")
        or response.get("code_execution")
        or {}
    )

    def get_verification_value(name):

        if name in verification:
            return verification.get(name)

        return pipeline.get(name)

    structure_passed = get_verification_value(
        "structure_validation_passed"
    )

    request_match_passed = get_verification_value(
        "request_match_passed"
    )

    local_execution_passed = get_verification_value(
        "local_execution_passed"
    )

    result_validation_passed = get_verification_value(
        "result_validation_passed"
    )

    fully_correct = get_verification_value(
        "fully_correct"
    )

    print("\nStructure validation:")
    print(structure_passed)

    print("\nRequest match:")
    print(request_match_passed)

    print("\nLocal execution:")
    print(local_execution_passed)

    print("\nResult validation:")
    print(result_validation_passed)

    print("\nFully correct:")
    print(fully_correct)

    print("\nFailure stage:")
    print(
        verification.get("failure_stage")
        or response.get("failure_stage")
    )

    print("\nRequest mismatches:")
    pprint(
        verification.get("request_mismatches")
        or []
    )

    assert structure_passed is True
    assert request_match_passed is True
    assert local_execution_passed is True
    assert result_validation_passed is True
    assert fully_correct is True

    print("\n✅ POSITIVE VERIFIER TEST PASSED")

    # ============================================================
    # PART 6
    # NEGATIVE CONTROL:
    #
    # This code is safe and syntactically runnable,
    # BUT it does NOT answer the user's question.
    #
    # This is the most important test of the new verifier.
    # ============================================================

    print("\n" + "=" * 100)
    print("PART 6 — NEGATIVE TEST: RUNNABLE BUT WRONG CODE")
    print("=" * 100)

    wrong_but_runnable_code = """
result = analysis.table1(
    predictors=["muscular_strength"],
    target="elite_status",
    controls=[["sex"]],
    filters={}
)
""".strip()

    print("\nWrong but runnable Python:")
    print("-" * 100)
    print(wrong_but_runnable_code)
    print("-" * 100)

    wrong_result_obj = verify_and_execute_generated_code(
        wrong_but_runnable_code,
        user_request=USER_REQUEST,
        requested_analysis=REQUESTED_ANALYSIS,
        requested_filters=REQUESTED_FILTERS,
    )

    wrong_result = as_dict(wrong_result_obj)

    print("\nNegative verification result:")

    print(
        "Structure validation:",
        wrong_result.get("structure_validation_passed"),
    )

    print(
        "Request match:",
        wrong_result.get("request_match_passed"),
    )

    print(
        "Local execution:",
        wrong_result.get("local_execution_passed"),
    )

    print(
        "Result validation:",
        wrong_result.get("result_validation_passed"),
    )

    print(
        "Fully correct:",
        wrong_result.get("fully_correct"),
    )

    print(
        "Failure stage:",
        wrong_result.get("failure_stage"),
    )

    print("\nRequest mismatches:")

    pprint(
        wrong_result.get("request_mismatches")
        or []
    )

    # ------------------------------------------------------------
    # Expected behavior:
    #
    # Python structure itself is valid.
    #
    # But:
    # - only 1 predictor instead of 8
    # - only one control specification instead of four
    #
    # Therefore the new verifier MUST reject it BEFORE execution.
    # ------------------------------------------------------------

    assert wrong_result.get(
        "structure_validation_passed"
    ) is True

    assert wrong_result.get(
        "request_match_passed"
    ) is False

    assert wrong_result.get(
        "local_execution_passed"
    ) is False

    assert wrong_result.get(
        "result_validation_passed"
    ) is False

    assert wrong_result.get(
        "fully_correct"
    ) is False

    assert wrong_result.get(
        "failure_stage"
    ) == "request_validation"

    assert wrong_result.get(
        "executed"
    ) is False

    print(
        "\n✅ NEGATIVE VERIFIER TEST PASSED:"
        "\nRunnable but semantically wrong code was rejected "
        "before local execution."
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    print("\n" + "=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    print()
    print("✅ New pool-based prompt is ACTIVE.")
    print("✅ Actual prompt sent to GPT-4.1 was captured.")
    print("✅ All 7 analysis methods were available to GPT-4.1.")
    print("✅ All 8 public domains were available to GPT-4.1.")
    print("✅ Hidden Request Contract was NOT sent to GPT-4.1.")
    print("✅ GPT-4.1 independently generated Restricted Python.")
    print("✅ Correct GPT-4.1 code passed the new verifier.")
    print("✅ Runnable but incorrect code failed Request Match.")
    print("✅ Incorrect code was NOT locally executed.")
    print()
    print("PROMPT + VERIFIER INTEGRATION TEST PASSED ✅")


if __name__ == "__main__":
    main()