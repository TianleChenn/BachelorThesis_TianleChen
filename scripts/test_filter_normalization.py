from llm.generated_code_verifier import verify_and_execute_generated_code
from sports.config import PREDICTORS


controls = [
    [],
    ["sex"],
    ["age"],
    ["sex", "age"],
]

code = f"""
result = analysis.table1(
    predictors={PREDICTORS!r},
    target="elite_status",
    controls={controls!r},
    filters=None
)
"""


print("=" * 80)
print("TEST A: ALL ATHLETES + filters=None")
print("=" * 80)

result_all = verify_and_execute_generated_code(
    code,
    user_request=(
        "Generate the four logistic regression models corresponding to Table 1 "
        "for all athletes using all eight public domains, with elite_status as "
        "the binary target. Use four model specifications: no controls, sex, "
        "age, and both sex and age."
    ),
    requested_analysis="table1",
    requested_filters={},
)

print("Structure:", result_all.structure_validation_passed)
print("Request Match:", result_all.request_match_passed)
print("Execution:", result_all.local_execution_passed)
print("Fully Correct:", result_all.fully_correct)
print("Failure:", result_all.failure_stage)

assert result_all.request_match_passed is True


print()
print("=" * 80)
print("TEST B: JUNIOR ATHLETES + filters=None")
print("=" * 80)

result_junior = verify_and_execute_generated_code(
    code,
    user_request=(
        "Generate the four logistic regression models corresponding to Table 1 "
        "for junior national team athletes using all eight public domains, "
        "with elite_status as the binary target. Use four model specifications: "
        "no controls, sex, age, and both sex and age."
    ),
    requested_analysis="table1",
    requested_filters={"national_team": "Junior"},
)

print("Structure:", result_junior.structure_validation_passed)
print("Request Match:", result_junior.request_match_passed)
print("Execution:", result_junior.local_execution_passed)
print("Fully Correct:", result_junior.fully_correct)
print("Failure:", result_junior.failure_stage)
print("Mismatches:", result_junior.request_mismatches)

assert result_junior.request_match_passed is False
assert result_junior.failure_stage == "request_validation"


print()
print("✅ FILTER NORMALIZATION WORKS CORRECTLY")