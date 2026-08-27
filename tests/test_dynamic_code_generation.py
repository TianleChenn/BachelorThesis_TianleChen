from llm.analysis_request_contracts import build_request_contract,render_request_contract
from llm.generated_code_verifier import inspect_generated_code,verify_and_execute_generated_code


def test_restricted_shape_and_legacy_rejection():
    code=render_request_contract(build_request_contract("correlation"))
    parsed=inspect_generated_code(code,user_request="correlation",
        requested_analysis="correlation",requested_filters={})
    assert parsed.generated_method=="correlation" and parsed.request_match_passed
    for unsafe in ["result = run_table1()","import os\nresult = analysis.routing_evaluation()",
            "result = analysis.df","result = analysis.correlation(variables=get_variables())"]:
        assert not inspect_generated_code(unsafe,user_request="correlation",
            requested_analysis="correlation",requested_filters={}).structure_validation_passed


def test_dynamic_figure_calls_validate():
    figure1=render_request_contract(build_request_contract("figure1"))
    prompt="Generate Figure 2 showing 20 athletes"
    figure2=render_request_contract(build_request_contract("figure2",{"sport":"table tennis"},prompt))
    assert inspect_generated_code(figure1,user_request="figure1",requested_analysis="figure1",
        requested_filters={}).request_match_passed
    assert inspect_generated_code(figure2,user_request=prompt,requested_analysis="figure2",
        requested_filters={"sport":"table tennis"}).request_match_passed


def test_complete_table1_contract_executes():
    code=render_request_contract(build_request_contract("table1"))
    execution=verify_and_execute_generated_code(code,user_request="table1",
        requested_analysis="table1",requested_filters={})
    assert execution.allowed and execution.executed and execution.result is not None
