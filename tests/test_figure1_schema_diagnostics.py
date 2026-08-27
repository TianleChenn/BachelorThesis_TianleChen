from llm.generated_code_verifier import inspect_generated_code,verify_and_execute_generated_code

VALID_FIGURE1='''result = analysis.figure1(
    variables=[
        "muscular_strength",
        "lower_body_dynamics",
        "muscle_power_genetics",
        "blood_micronutrients",
        "basic_cognitive_function",
        "mental_health",
        "social_support",
        "training_conditions"
    ],
    target="expertise_value",
    group_field="elite_status",
    correlation_threshold=0.15,
    variance_iterations=1000,
    filters={}
)'''
def test_exact_figure1_schema_parses_and_dispatches_safely():
    parsed=inspect_generated_code(VALID_FIGURE1,user_request="figure1",
        requested_analysis="figure1",requested_filters={})
    assert parsed.generated_method=="figure1" and set(parsed.generated_arguments)=={"variables","target","group_field","correlation_threshold","variance_iterations","filters"}
    execution=verify_and_execute_generated_code(VALID_FIGURE1,user_request="figure1",
        requested_analysis="figure1",requested_filters={});assert execution.allowed
    assert hasattr(execution.result.get("figure"),"savefig")
    assert "athlete_id" not in str(execution.result).lower() and "raw_rows" not in execution.result
def test_figure1_schema_rejections():
    invalid=["result = analysis.generate_figure1(variables=[])","figure = analysis.figure1(variables=[])","variables = []\nresult = analysis.figure1(variables=variables)","import matplotlib.pyplot as plt\nresult = analysis.figure1(variables=[])"]
    for code in invalid:assert not inspect_generated_code(code,user_request="figure1",
        requested_analysis="figure1",requested_filters={}).structure_validation_passed
def test_argument_mismatch_is_specific():
    result=inspect_generated_code("result = analysis.figure1(variables=[], iterations=1000)",
        user_request="figure1",requested_analysis="figure1",requested_filters={})
    assert "Keyword 'iterations' is not allowed for figure1." in result.validation_error
