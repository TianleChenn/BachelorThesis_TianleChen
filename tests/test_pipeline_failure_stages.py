from llm.generated_code_verifier import inspect_generated_code


def test_validator_distinguishes_format_and_request_stages():
    invalid = [
        "result = analysis.figure1(",
        'result = analysis.network_analysis(variables=["mental_health"], filters={})',
        'result = analysis.figure1(variables=["strength"], filters={})',
    ]
    stages=[inspect_generated_code(code,user_request="figure1",requested_analysis="figure1",
        requested_filters={}).failure_stage for code in invalid]
    assert stages == ["format_validation","format_validation","request_validation"]


def test_success_flags():
    code='result = analysis.figure1(variables=["muscular_strength"], target="expertise_value", group_field="elite_status", correlation_threshold=0.15, variance_iterations=1000, filters={})'
    result=inspect_generated_code(code,user_request="figure1",requested_analysis="figure1",
        requested_filters={})
    assert result.structure_validation_passed
    assert not result.request_match_passed
