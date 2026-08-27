from copy import deepcopy
import inspect
from unittest.mock import MagicMock, patch

import frontend
from sports import service


def _render(response, llm_result=None):
    streamlit=MagicMock()
    streamlit.columns.return_value=[MagicMock(),MagicMock()]
    with patch.object(frontend,"st",streamlit):
        frontend.render_llm_result(llm_result or {
            "privacy_route":"cloud", "router_applicable":True,
            "p_cloud":.4, "threshold":.5, "selected_model":"local_ministral",
        },response=response)
    return streamlit


def _verification_rows(streamlit):
    return streamlit.dataframe.call_args.args[0].to_dict(orient="records")


def test_successful_restricted_call_and_four_verification_stages_are_visible():
    code='result = analysis.correlation(variables=["muscular_strength", "lower_body_dynamics"], filters={})'
    response={"generated_code":code,"code_execution":{"generated_method":"correlation",
        "generated_arguments":{"variables":["muscular_strength","lower_body_dynamics"],"filters":{}}},"pipeline_diagnostics":{
        "structure_validation_passed":True,"request_match_passed":True,
        "local_execution_passed":True,"result_validation_passed":True}}
    streamlit=_render(response)
    generated_rows=streamlit.dataframe.call_args_list[0].args[0].to_dict(orient="records")
    assert generated_rows[0]=={"Field":"Analysis Method","Generated Value":"Correlation"}
    assert [row["Field"] for row in generated_rows]==["Analysis Method","Variables","Filters"]
    assert generated_rows[1]["Generated Value"]=="Muscular Strength, Lower-body Dynamics"
    assert generated_rows[2]["Generated Value"]=="None"
    streamlit.code.assert_not_called()
    assert _verification_rows(streamlit)==[
        {"Verification Step":"Structure Validation","Status":"PASS"},
        {"Verification Step":"Request Match","Status":"PASS"},
        {"Verification Step":"Execution","Status":"PASS"},
        {"Verification Step":"Result Validation","Status":"PASS"}]
    assert "athlete_id" not in code.casefold()


def test_cloud_generated_call_still_shows_local_backend_verification():
    code='result = analysis.table1(filters={})'
    response={"generated_code":code,"code_execution":{"generated_method":"table1",
        "generated_arguments":{"filters":{}}},"pipeline_diagnostics":{
        "structure_validation_passed":True,"request_match_passed":True,
        "local_execution_passed":True,"result_validation_passed":True}}
    streamlit=_render(response,{"privacy_route":"cloud","router_applicable":True,
        "p_cloud":.9,"threshold":.5,"selected_model":"cloud_gemini"})
    assert streamlit.dataframe.call_args_list[0].args[0].iloc[0].to_dict()=={
        "Field":"Analysis Method","Generated Value":"Logistic Regression (Table 1)"}
    assert all(row["Status"]=="PASS" for row in _verification_rows(streamlit))


def test_request_mismatch_keeps_call_error_and_marks_later_stages_not_run():
    code='result = analysis.table2(filters={"sport": "volleyball"})'
    response={"generated_code":code,"code_execution":{"generated_method":"table2",
        "generated_arguments":{"filters":{"sport":"volleyball"}}},
        "sanitized_error":"Expected filters={} but generated volleyball.",
        "pipeline_diagnostics":{"structure_validation_passed":True,
        "request_match_passed":False,"local_execution_passed":False,
        "result_validation_passed":False}}
    streamlit=_render(response)
    generated_rows=streamlit.dataframe.call_args_list[0].args[0].to_dict(orient="records")
    assert generated_rows[1]=={"Field":"Filters","Generated Value":"Sport = Volleyball"}
    assert [row["Status"] for row in _verification_rows(streamlit)]==[
        "PASS","FAIL","NOT RUN","NOT RUN"]
    streamlit.error.assert_any_call("Expected filters={} but generated volleyball.")


def test_model_unavailable_shows_no_fake_call_or_verification_pass():
    response={"generated_code":None,"sanitized_error":"The local code generation model is unavailable.",
        "pipeline_diagnostics":{"structure_validation_passed":False,
        "request_match_passed":False,"local_execution_passed":False,
        "result_validation_passed":False}}
    streamlit=_render(response)
    streamlit.code.assert_not_called()
    streamlit.info.assert_any_call("No Restricted Analysis Call was generated.")
    assert [row["Status"] for row in _verification_rows(streamlit)]==["NOT RUN"]*4
    streamlit.error.assert_any_call("The local code generation model is unavailable.")
    streamlit.markdown.assert_any_call("#### Code Generation Error")


def test_generated_call_failure_keeps_validation_error_heading():
    response={"generated_code":"result = analysis.table2(filters={})",
        "sanitized_error":"Generated call did not match the request.",
        "pipeline_diagnostics":{"structure_validation_passed":True,
        "request_match_passed":False,"local_execution_passed":False,
        "result_validation_passed":False}}
    streamlit=_render(response)
    streamlit.markdown.assert_any_call("#### Validation Error")


def test_restricted_call_humanization_preserves_underlying_arguments():
    arguments={
        "predictors":["muscular_strength","lower_body_dynamics","muscle_power_genetics",
            "blood_micronutrients","basic_cognitive_function","mental_health",
            "social_support","training_conditions"],
        "target":"elite_status",
        "controls":[[],["sex"],["age"],["sex","age"]],
        "filters":{"national_team":"Junior"},
        "visualization":False,
    }
    original=deepcopy(arguments)
    rows=frontend._restricted_call_display_rows("table1",arguments)
    assert arguments==original
    assert sum(row["Field"]=="Analysis Method" for row in rows)==1
    values={row["Field"]:row["Generated Value"] for row in rows}
    assert values["Analysis Method"]=="Logistic Regression (Table 1)"
    assert values["Predictors"].startswith("Muscular Strength, Lower-body Dynamics")
    assert "[" not in values["Predictors"]
    assert values["Target"]=="Elite Status"
    assert values["Controls"]=="Base Model; + Sex; + Age; + Sex & Age"
    assert values["Filters"]=="National Team = Junior"
    assert values["Visualization"]=="No"


def test_empty_filters_and_boolean_values_have_readable_display():
    rows=frontend._restricted_call_display_rows("correlation",{
        "method":"pearson","filters":{},"visualization":True})
    values={row["Field"]:row["Generated Value"] for row in rows}
    assert values=={"Analysis Method":"Correlation","Correlation Method":"Pearson",
        "Filters":"None","Visualization":"Yes"}


def test_technical_details_include_provider_failure_diagnostics_for_all_routes():
    renderer=inspect.getsource(frontend.show_pipeline_response)
    for label in (
        "Requested analysis",
        "Selected generator tier",
        "Requested model",
        "Actual model",
        "Provider",
        "Failure stage",
        "Provider retry used",
        "Generation request ID",
        "Sanitized provider or validation error",
    ):
        assert label in renderer
    diagnostics=inspect.getsource(service.handle_user_request)
    for key in (
        "requested_generator_channel",
        "used_generator_channel",
        "requested_model",
        "actual_model",
        "provider",
        "model_call_success",
        "model_unavailable",
        "provider_retry_used",
    ):
        assert f'"{key}"' in diagnostics
