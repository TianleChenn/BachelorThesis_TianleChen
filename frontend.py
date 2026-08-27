from llm.env import load_local_env

load_local_env()

import streamlit as st
import pandas as pd
import altair as alt
import ast
import uuid
import html
import json
import pprint
from pathlib import Path
from ui.data_generation_helpers import (build_domain_construction_rows,
    format_expertise_range,get_expertise_group_explanation,
    load_generation_report,load_safe_dataset_summary)
from ui.result_renderer import (
    NOISE_DASHBOARD_VERSION,
    get_noise_analysis_presentation,
    render_analysis_result,
    render_noise_utility,
)
from ui.cohort_prompts import build_dashboard_prompt
from ui.theme import apply_theme
from llm.model_config import LOCAL_EDGE_GENERATOR_MODEL
from privacy.numerical_perturbation import NOISE_HIGH, NOISE_LOW
from sports.analysis_noise_utility import EXPERIMENT_VERSION as ALL_ANALYSIS_NOISE_VERSION

NOISE_ENABLED_ANALYSES={"table1","table2","figure1","figure2","correlation","variance_analysis"}

def _has_current_noise_utility(response:dict,expected_analysis_key:str)->bool:
    result=response.get("result") if isinstance(response,dict) else None
    utility=result.get("noise_utility") if isinstance(result,dict) else None
    mean_result=utility.get("mean_perturbed_result") if isinstance(utility,dict) else None
    average_difference=utility.get("average_difference") if isinstance(utility,dict) else None
    has_mean_result=bool(isinstance(mean_result,dict) and (
        isinstance(mean_result.get("table"),(list,tuple,pd.DataFrame))
        or hasattr(mean_result.get("figure"),"savefig")
    ))
    return bool(isinstance(utility,dict) and utility.get("analysis_key")==expected_analysis_key
        and utility.get("noise_range")==[NOISE_LOW,NOISE_HIGH] and utility.get("repetitions")==50
        and utility.get("experiment_version")==ALL_ANALYSIS_NOISE_VERSION
        and has_mean_result
        and isinstance(average_difference,dict) and average_difference.get("value") is not None
        and hasattr(utility.get("stability_figure"),"savefig"))

try:
    from sports.config import SPORTS
except Exception:
    SPORTS = [
        "3x3 basketball",
        "ice hockey",
        "volleyball",
        "artistic gymnastics",
        "trampoline gymnastics",
        "rhythmic gymnastics",
        "table tennis",
        "modern pentathlon",
    ]

try:
    from sports.service import handle_user_request
except Exception:
    handle_user_request = None

try:
    from sports.anonymous_subject import ATHLETE_GROUP_OPTIONS, select_anonymous_subject
except Exception:
    ATHLETE_GROUP_OPTIONS = ("Elite", "Semi-elite", "All athletes")
    select_anonymous_subject = None

ATHLETE_GROUP_LABELS = {
    "Elite": "Elite (Expertise Score ≥ 13)",
    "Semi-elite": "Semi-elite (Expertise Score < 13)",
    "All athletes": "All athletes",
}

PRIVACY_METHOD_COMPARISON_PATH = "artifacts/privacy_methods_frontend60_comparison.json"
PRIVACY_METHOD_DISPLAY_NAMES = {
    "method_a_fixed_4d": "Fixed Rules + Soft Gating",
    "method_b_llm_scalar": "Simple LLM + Soft Gating",
    "method_c_llm_4d_soft_gating": "Privacy Prompt LLM + Soft Gating",
}
PRIVACY_ROUTE_ORDER = ("cloud", "collaboration", "local_edge", "blocked")

st.set_page_config(
    page_title="Private Athlete Analysis Platform",
    page_icon="shield",
    layout="wide",
)

apply_theme()


def show_result_table(result):
    render_analysis_result(result)
    return None


def _friendly_label(value):
    labels = {
        "cloud": "Cloud",
        "collaboration": "Collaboration",
        "local_edge": "Local Edge",
        "blocked": "Blocked",
        "cloud_gemini": "Cloud — Gemini 3.5 Flash",
        "local_ministral": "Local — Ministral-3-8B-Local",
        "none": "None",
    }
    return labels.get(value, str(value or "Unknown").replace("_", " ").title())


def _get_prism_route(response: dict | None) -> str:
    response = response or {}
    result = response.get("prism_privacy_result") or {}
    privacy = response.get("privacy_decision") or {}
    return str(result.get("route") or privacy.get("route") or "unknown")


def display_model_name(model_name: str | None) -> str:
    mapping = {
        "gpt-4.1": "GPT-4.1",
        "gemini-3.5-flash": "Gemini 3.5 Flash",
        "Ministral-3-8B-Local": "Ministral-3-8B-Local",
    }
    if not model_name:
        return "Unknown"
    return mapping.get(str(model_name), str(model_name))


def build_llm_selection_label(
    privacy_route: str,
    selected_tier: str | None,
    selected_model: str | None,
    local_model: str | None,
) -> str:
    """Build the single model-selection label shown in Pipeline Overview."""
    route = str(privacy_route or "").strip().lower()
    tier = str(selected_tier or "").strip().lower()
    if route == "local_edge":
        return f"Local — {display_model_name(local_model or LOCAL_EDGE_GENERATOR_MODEL)}"
    if route == "blocked":
        return "None"
    if route in {"cloud", "collaboration"}:
        if tier == "cloud":
            return f"Cloud — {display_model_name(selected_model or 'gemini-3.5-flash')}"
        if tier == "local":
            return f"Local — {display_model_name(local_model or LOCAL_EDGE_GENERATOR_MODEL)}"
        return "Routing error"
    return "Unknown"


def show_pipeline_overview(response: dict | None = None) -> None:
    response = response or {}
    audit = response.get("pipeline_audit") or {}
    diagnostics = response.get("pipeline_diagnostics") or {}
    model_decision = response.get("model_decision") or {}
    route = _get_prism_route(response)
    tier = audit.get("cost_aware_selected_tier") or model_decision.get("selected_tier")
    selected_model = audit.get("cost_aware_selected_model") or model_decision.get("execution_model")
    local_model = audit.get("actual_generator_model") or model_decision.get("execution_model")
    st.markdown("## Pipeline Overview")
    columns = st.columns(4)
    columns[0].metric("Privacy-aware Router", _friendly_label(route))
    columns[1].metric(
        "Cost-aware Router",
        _friendly_label(tier),
    )
    columns[2].metric(
        "Selected LLM",
        build_llm_selection_label(route, tier, selected_model, local_model),
    )
    columns[3].metric(
        "Local Verification",
        ("NOT RUN" if not response.get("generated_code")
         else "PASS" if diagnostics.get("validation_passed") else "FAIL"),
    )


def _format_probability_percent(value) -> str:
    bounded = max(0.0, min(1.0, float(value)))
    if abs(bounded) < 1e-12:
        bounded = 0.0
    return f"{bounded * 100:.1f}%"


def show_prism_privacy_result(response: dict | None = None) -> None:
    """Display the semantic privacy assessment and final routing result."""
    result = (response or {}).get("prism_privacy_result") or {}
    if not result:
        return
    st.markdown("## Privacy-aware Router Result")
    render_privacy_test((response or {}).get("privacy_test"))


def render_privacy_test(privacy_test: dict | None) -> None:
    if not isinstance(privacy_test, dict):
        st.warning("Privacy-test details are unavailable for this request.")
        return
    assessment = privacy_test.get("llm_generated_json") or {}
    st.markdown("### Privacy Assessor Output")
    st.caption("Values returned by the Cloud LLM Privacy Assessor.")
    if not privacy_test.get("assessment_success", True):
        st.error("LLM privacy assessment failed. The configured model is read from LLM_STRONG_MODEL. No four-dimensional scores were generated, and Soft Gating was skipped.")
        return
    blocked = bool(assessment.get("blocked_request"))
    st.dataframe(pd.DataFrame([{
        "Privacy Risk":f"{float(assessment.get('privacy_risk_score',0)):.2f}",
        "Subject Scope":f"{float(assessment.get('subject_scope',0)):.2f}",
        "Data Sensitivity":f"{float(assessment.get('data_sensitivity',0)):.2f}",
        "Disclosure Level":f"{float(assessment.get('disclosure_level',0)):.2f}",
        "Blocked":"Yes" if blocked else "No",
    }]),use_container_width=True,hide_index=True)
    st.markdown("### Assessment Explanation")
    st.write(assessment.get("explanation", "No explanation available."))
    st.markdown("---")
    st.markdown("### Four-dimensional Soft Gating")
    st.caption("The Soft Gating router reads the four continuous privacy dimensions returned by the Privacy Assessor.")
    st.caption("If Blocked is Yes, the request is stopped before Soft Gating.")
    if blocked:
        st.info("Soft Gating was not executed because the Privacy Assessor blocked this request.")
        return
    feature_labels = ["Privacy Risk", "Subject Scope", "Data Sensitivity", "Disclosure Level"]
    features = privacy_test.get("gating_features")
    st.markdown("#### Soft Gating Input")
    if isinstance(features,list) and len(features)==4:
        st.dataframe(pd.DataFrame([{label:f"{float(value):.2f}" for label,value in zip(feature_labels,features)}]),use_container_width=True,hide_index=True)
    else:
        st.warning("Soft Gating input values are unavailable.")
    probabilities = privacy_test.get("gating_probabilities")
    selected_route = _friendly_label(privacy_test.get("selected_route"))
    st.markdown("#### Soft Gating Output")
    if isinstance(probabilities,dict):
        output_columns=st.columns(4)
        output_columns[0].metric("Cloud",_format_probability_percent(probabilities.get("cloud",0.0)))
        output_columns[1].metric("Collaboration",_format_probability_percent(probabilities.get("collaboration",0.0)))
        output_columns[2].metric("Local Edge",_format_probability_percent(probabilities.get("local_edge",0.0)))
        output_columns[3].metric("Selected Route",selected_route)
    else:
        st.write(f"Selected Privacy Route: {selected_route}")


def _format_llm_decimal(value) -> str:
    try:
        return "Not available" if value is None else f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "Not available"


def _display_llm_model_name(value) -> str:
    mapping = {
        "cloud_gemini": "Cloud — Gemini 3.5 Flash",
        "local_ministral": "Local — Ministral-3-8B-Local",
        "none": "None",
    }
    text = str(value or "").strip()
    return mapping.get(text, display_model_name(text)) if text else "Not available"


def _render_full_value_card(label: str, value: str) -> None:
    """Render metric-like content without Streamlit's single-line truncation."""
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    st.markdown(
        f"""
        <div style="
            min-height: 150px;
            padding: 18px;
            border: 1px solid rgba(120, 145, 175, 0.28);
            border-radius: 16px;
            background: rgba(248, 250, 253, 0.45);
            box-sizing: border-box;
        ">
            <div style="font-size: 0.9rem; color: #43566f; margin-bottom: 6px;">
                {safe_label}
            </div>
            <div style="
                font-size: 1.45rem;
                line-height: 1.3;
                font-weight: 700;
                color: #0f172a;
                white-space: normal;
                overflow: visible;
                overflow-wrap: anywhere;
                word-break: break-word;
            ">
                {safe_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _verification_display_status(value, *, reached: bool) -> str:
    if not reached:
        return "NOT RUN"
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "NOT AVAILABLE"


_RESTRICTED_METHOD_LABELS={
    "table1":"Logistic Regression (Table 1)",
    "table2":"Multiple Linear Regression (Table 2)",
    "figure1":"Network Analysis",
    "figure2":"Athlete Profile Visualization",
    "correlation":"Correlation",
    "variance_analysis":"Variance Analysis",
    "individual_profile":"Individual Profile",
}
_RESTRICTED_PARAMETER_LABELS={
    "predictors":"Predictors","variables":"Variables","target":"Target",
    "controls":"Controls","filters":"Filters","method":"Correlation Method",
    "visualization":"Visualization",
}
_RESTRICTED_VALUE_LABELS={
    "muscular_strength":"Muscular Strength",
    "lower_body_dynamics":"Lower-body Dynamics",
    "muscle_power_genetics":"Muscle-power Genetics",
    "blood_micronutrients":"Blood Micronutrients",
    "basic_cognitive_function":"Basic Cognitive Function",
    "mental_health":"Mental Health","social_support":"Social Support",
    "training_conditions":"Training Conditions","elite_status":"Elite Status",
}


def _humanize_restricted_name(value: object) -> str:
    text=str(value)
    return _RESTRICTED_VALUE_LABELS.get(text,text.replace("_"," ").title())


def _format_restricted_filters(value: object) -> str:
    if value in ({},None):return "None"
    if not isinstance(value,dict):return _format_restricted_value(value)
    return "; ".join(
        f"{_humanize_restricted_name(name)} = {_format_restricted_value(item)}"
        for name,item in value.items())


def _format_restricted_controls(value: object) -> str:
    if value is None:return "None"
    if not isinstance(value,list):return _format_restricted_value(value)
    labels=[]
    for control_set in value:
        if control_set==[]:labels.append("Base Model")
        elif isinstance(control_set,list):
            labels.append("+ "+" & ".join(_humanize_restricted_name(item) for item in control_set))
        else:labels.append(_format_restricted_value(control_set))
    return "; ".join(labels) if labels else "None"


def _format_restricted_value(value: object) -> str:
    if value is None:return "None"
    if isinstance(value,bool):return "Yes" if value else "No"
    if isinstance(value,dict):return _format_restricted_filters(value)
    if isinstance(value,(list,tuple)):
        return ", ".join(_format_restricted_value(item) for item in value) if value else "None"
    if isinstance(value,str):return _humanize_restricted_name(value)
    return str(value)


def _restricted_call_display_rows(method: str, arguments: dict) -> list[dict[str,str]]:
    rows=[{"Field":"Analysis Method","Generated Value":_RESTRICTED_METHOD_LABELS.get(
        method,_humanize_restricted_name(method))}]
    for name,value in arguments.items():
        if name=="filters":display_value=_format_restricted_filters(value)
        elif name=="controls":display_value=_format_restricted_controls(value)
        else:display_value=_format_restricted_value(value)
        rows.append({"Field":_RESTRICTED_PARAMETER_LABELS.get(name,_humanize_restricted_name(name)),
            "Generated Value":display_value})
    return rows


def render_llm_result(llm_result: dict | None, response: dict | None = None) -> None:
    """Render the cost-aware Cloud/Local decision."""
    llm_result = llm_result or {}
    privacy_route = str(llm_result.get("privacy_route") or "")
    applicable = bool(llm_result.get("router_applicable"))
    st.markdown("## Cost-aware Router Result")
    st.caption("This section shows the cost-aware Cloud/Local Router decision.")
    st.caption(f"Privacy Route: {_friendly_label(privacy_route)}")

    if privacy_route == "local_edge":
        local_model = llm_result.get("execution_model") or LOCAL_EDGE_GENERATOR_MODEL
        st.markdown("### Cloud/Local Router")
        _render_full_value_card("Router Status", "Not Applicable")
        st.write(f"**Selected Model:** Local — {display_model_name(local_model)}")
        st.write("**Actual Execution:** Local")
        st.info(
            "Because the privacy route is Local Edge, the Cost-aware Cloud/Local Router is "
            "bypassed completely and no cloud model is called."
        )
    elif applicable:
        routing_columns = st.columns(2)
        with routing_columns[0]:
            st.metric(
                "P_cloud",
                _format_llm_decimal(llm_result.get("p_cloud")),
            )
        with routing_columns[1]:
            st.metric(
                "Decision Threshold",
                _format_llm_decimal(
                    llm_result.get("athlete_router_threshold", llm_result.get("threshold"))
                ),
            )
        selected = _display_llm_model_name(llm_result.get("selected_model"))
        st.markdown("### Selected Model")
        _render_full_value_card("Selected Model", selected)
        st.write(f"**Actual Execution:** {selected}")
    else:
        st.markdown("### Cloud/Local Router")
        _render_full_value_card("Router Status", "Bypassed")
        st.write(f"**Actual Execution:** {_friendly_label(privacy_route)}")
    st.caption("Cloud Model: Gemini 3.5 Flash | Local Model: Ministral-3-8B-Local")
    if llm_result.get("router_error"):
        st.error(f"Router unavailable: {llm_result['router_error']}")

    st.info(
        "P_cloud is produced locally by the Cloud/Local classifier. Its threshold "
        "is selected from five-fold out-of-fold predictions."
    )
    st.caption("The selected LLM generates only a Restricted Analysis Call. The call is verified and executed locally on the protected backend.")
    response=response or {}
    generated_call=response.get("generated_code")
    code_execution=response.get("code_execution") or {}
    diagnostics=response.get("pipeline_diagnostics") or {}
    st.markdown("### Restricted Analysis Call")
    st.caption("Generated by the selected LLM from the current analysis request using the allowed analysis schema.")
    if generated_call:
        generated_method=code_execution.get("generated_method") or diagnostics.get("generated_method")
        generated_arguments=code_execution.get("generated_arguments")
        if generated_method and isinstance(generated_arguments,dict):
            call_rows=_restricted_call_display_rows(generated_method,generated_arguments)
            st.dataframe(pd.DataFrame(call_rows),use_container_width=True,hide_index=True)
        else:
            st.dataframe(pd.DataFrame([{"Field":"Analysis Method",
                "Generated Value":_RESTRICTED_METHOD_LABELS.get(generated_method,"Unavailable")}]),
                use_container_width=True,hide_index=True)
    else:
        st.info("No Restricted Analysis Call was generated.")
    structure_reached=bool(generated_call)
    structure_status=_verification_display_status(
        diagnostics.get("structure_validation_passed"),reached=structure_reached)
    request_reached=structure_status=="PASS"
    request_status=_verification_display_status(
        diagnostics.get("request_match_passed"),reached=request_reached)
    execution_reached=request_status=="PASS"
    execution_status=_verification_display_status(
        diagnostics.get("local_execution_passed"),reached=execution_reached)
    result_reached=execution_status=="PASS"
    result_status=_verification_display_status(
        diagnostics.get("result_validation_passed"),reached=result_reached)
    st.markdown("### Local Verification")
    st.dataframe(pd.DataFrame([
        {"Verification Step":"Structure Validation","Status":structure_status},
        {"Verification Step":"Request Match","Status":request_status},
        {"Verification Step":"Execution","Status":execution_status},
        {"Verification Step":"Result Validation","Status":result_status},
    ]),use_container_width=True,hide_index=True)
    if response.get("sanitized_error"):
        st.markdown("#### Validation Error" if generated_call else "#### Code Generation Error")
        st.error(str(response["sanitized_error"]))


def _render_module_separator() -> None:
    """Draw a clear visual boundary between major pipeline modules."""
    st.markdown(
        '<hr style="border: 0; border-top: 3px solid rgba(110, 120, 135, 0.55); '
        'margin: 2rem 0 1.75rem 0;">',
        unsafe_allow_html=True,
    )


def _render_final_analysis_result(response: dict) -> None:
    st.markdown("### Final Analysis Result")
    result=response.get("result") if response.get("result") is not None else response.get("analysis_result")
    noise_utility=result.get("noise_utility") if isinstance(result,dict) else None
    if isinstance(noise_utility,dict):
        presentation=get_noise_analysis_presentation(str(noise_utility.get("analysis_key") or ""))
        st.caption("PART 1")
        st.markdown(f"### {presentation['original_title']}")
    final_text=(response.get("answer") or response.get("summary")
        or response.get("final_answer") or response.get("message"))
    if final_text:
        if response.get("allowed"):st.success(str(final_text))
        else:st.error(str(final_text))
    if result is not None:
        displayed_result={key:value for key,value in result.items() if key!="noise_utility"} if noise_utility else result
        show_result_table(displayed_result)
        if isinstance(result,dict) and result.get("model_stats"):
            with st.expander("Model diagnostics",expanded=False):
                st.dataframe(pd.DataFrame(result["model_stats"]),use_container_width=True,hide_index=True)
                vif_rows=[]
                for model_stat in result["model_stats"]:
                    for vif_row in model_stat.get("vif_table") or []:vif_rows.append({"model":model_stat.get("model"),**vif_row})
                if vif_rows:
                    st.markdown("#### VIF");st.dataframe(pd.DataFrame(vif_rows),use_container_width=True,hide_index=True)
        if noise_utility:render_noise_utility(noise_utility,baseline_result=displayed_result)


def show_pipeline_response(response):
    """
    Display the full professor-aligned pipeline:
    PRISM decision -> RouteLLM decision -> Generated Python code -> local execution -> result.
    """
    if not isinstance(response, dict):
        render_analysis_result(response)
        return

    show_pipeline_overview(response=response)
    _render_module_separator()
    _render_final_analysis_result(response)

    _render_module_separator()
    show_prism_privacy_result(response=response)

    _render_module_separator()
    render_llm_result(response.get("llm_result"),response=response)

    route=_get_prism_route(response)
    with st.expander("Local execution diagnostics" if route=="local_edge" else "Technical details", expanded=False):
        diagnostics=response.get("pipeline_diagnostics") or {}
        failure_stage=response.get("failure_stage")
        if response.get("generated_code"):
            st.markdown("**Exact Restricted Analysis Call**")
            st.code(str(response["generated_code"]),language="python")
        st.write(f"**Requested analysis:** {diagnostics.get('requested_analysis') or 'Not available'}")
        selected_tier=diagnostics.get("selected_generator_tier")
        st.write(f"**Selected generator tier:** {_friendly_label(selected_tier) if selected_tier else 'Not available'}")
        st.write(f"**Requested model:** {diagnostics.get('requested_model') or 'Not available'}")
        st.write(f"**Actual model:** {diagnostics.get('actual_model') or 'Not available'}")
        st.write(f"**Provider:** {diagnostics.get('provider') or 'Not available'}")
        st.write(f"**Failure stage:** {failure_stage or 'None'}")
        st.write(f"**Provider retry used:** {'Yes' if diagnostics.get('provider_retry_used') else 'No'}")
        if route=="local_edge":
            st.write(f"**Local generator available:** {'Yes' if diagnostics.get('local_generator_available') else 'No'}")
            st.write(f"**First generation non-empty:** {'Yes' if diagnostics.get('first_generation_non_empty') else 'No'}")
            st.write(f"**First validation passed:** {'Yes' if diagnostics.get('first_validation_passed') else 'No'}")
            st.write(f"**Repair attempted:** {'Yes' if diagnostics.get('repair_attempted') else 'No'}")
            st.write(f"**Repair validation passed:** {'Yes' if diagnostics.get('repair_validation_passed') else 'No'}")
            st.write(f"**Final execution completed:** {'Yes' if diagnostics.get('final_execution_completed') else 'No'}")
        else:
            st.markdown("**PRISM Input Prompt**")
            st.code(str((response.get("prism_privacy_result") or {}).get("prism_input_prompt") or "Not available"),language="text")
        st.write(f"**Generated method:** {diagnostics.get('generated_method') or 'Not available'}")
        if failure_stage=="format_validation":
            st.caption("No athlete data were accessed because validation failed before local execution.")
        elif failure_stage=="request_validation":
            st.caption(
                "The generated code was structurally valid but did not represent the "
                "requested analysis, so it was not executed."
            )
        st.write(f"**Generation request ID:** {diagnostics.get('generation_request_id') or 'Not available'}")
        if response.get("sanitized_error"):
            st.write(f"**Sanitized provider or validation error:** {response['sanitized_error']}")


def run_request(prompt: str, use_openai: bool = True, requested_analysis: str | None = None,
                private_local_context: dict | None = None,
                analysis_filters: dict[str, str] | None = None):
    """Run the professor-aligned backend pipeline.

    The frontend does not implement PRISM/RouteLLM itself. It delegates to
    sports.service.handle_user_request(), which performs:
    PRISM routing -> RouteLLM model selection -> schema-only code generation ->
    local whitelist execution.
    """
    st.session_state.pop("privacy_test",None)
    st.session_state.pop("latest_analysis_result", None)
    if handle_user_request is None:
        response = {
            "allowed": False,
            "answer": "Backend service is not available. Please check sports/service.py imports.",
            "privacy_decision": {},
            "model_decision": {},
            "code_generation": {},
            "code_execution": {},
            "result": None,
        }
        return response
    try:
        if "privacy_session_id" not in st.session_state:
            st.session_state.privacy_session_id = str(uuid.uuid4())
        response = handle_user_request(
            prompt,
            use_openai=use_openai,
            session_id=st.session_state.privacy_session_id,
            requested_analysis=requested_analysis,
            private_local_context=private_local_context,
            analysis_filters=analysis_filters,
        )
        if isinstance(response.get("privacy_test"),dict):
            st.session_state["privacy_test"]=response["privacy_test"]
        st.session_state["latest_analysis_result"] = response
        return response
    except Exception as exc:
        return {
            "allowed": False,
            "answer": f"The backend request failed safely: {exc}",
            "privacy_decision": {},
            "model_decision": {},
            "code_generation": {},
            "code_execution": {},
            "result": None,
        }

def _select_primary_page():
    selected = st.session_state.get("primary_navigation")
    if selected:
        st.session_state.active_page = selected
        st.session_state.secondary_navigation = None


def _select_secondary_page():
    selected = st.session_state.get("secondary_navigation")
    if selected:
        st.session_state.active_page = selected
        st.session_state.primary_navigation = None


def render_sidebar():
    st.sidebar.markdown("## Private Athlete Platform")

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Protected analysis dashboard"
    primary_pages = ["Protected analysis dashboard", "Individual athlete analysis"]
    secondary_pages = ["Routing evaluation", "Data Generation & Processing"]
    active_page = st.session_state.active_page
    if active_page in primary_pages:
        st.session_state.primary_navigation = active_page
        st.session_state.secondary_navigation = None
    else:
        st.session_state.primary_navigation = None
        st.session_state.secondary_navigation = active_page
    st.sidebar.radio(
        "Choose function",
        primary_pages,
        index=None,
        key="primary_navigation",
        on_change=_select_primary_page,
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.radio(
        "Evaluation and data",
        secondary_pages,
        index=None,
        key="secondary_navigation",
        on_change=_select_secondary_page,
        label_visibility="collapsed",
    )
    st.sidebar.caption(
        "Raw athlete rows are never shown to users or sent to OpenAI."
    )

    return st.session_state.active_page


def render_header():
    st.markdown(
        """
        <div class="main-header">
            <div>
                <h1>Private Athlete Data Analysis Platform</h1>
                <p>
                Run Nature-style statistical analyses on protected athlete data.
                Users and LLMs can access analysis plans, tables, figures,
                and summaries, but never raw athlete rows.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_dashboard_results() -> None:
    for key in (
        "table1_response","table2_response","figure1_response","figure2_response",
        "correlation_response","variance_response","dashboard_selected_response_key",
        "noise_dashboard_versions",
    ):
        st.session_state.pop(key, None)


def _dashboard_cohort_selector() -> tuple[str, dict[str, str]]:
    cohort_type = st.selectbox(
        "Analysis cohort",
        [
            "All athletes",
            "Expertise group",
            "Sport discipline",
            "Sex",
            "National team level",
            "Age group",
        ],
        key="analysis_cohort_type",
        on_change=_clear_dashboard_results,
        help=(
            "The selected cohort is applied consistently to every dashboard analysis."
        ),
    )

    if cohort_type == "All athletes":
        return "all athletes", {}
    if cohort_type == "Expertise group":
        expertise_group = st.selectbox(
            "Expertise group",
            ["elite", "semi_elite"],
            format_func=lambda value: (
                "Elite (Expertise Score ≥ 13)"
                if value == "elite"
                else "Semi-elite (Expertise Score < 13)"
            ),
            key="analysis_expertise_group",
            on_change=_clear_dashboard_results,
        )
        filters={"expertise_group":expertise_group}
        return ("elite athletes" if expertise_group=="elite" else "semi-elite athletes"),filters
    if cohort_type == "Sport discipline":
        sport = st.selectbox("Sport discipline", SPORTS,key="analysis_sport",on_change=_clear_dashboard_results)
        filters={"sport":sport}
        return f"{sport} athletes",filters
    if cohort_type == "Sex":
        sex = st.selectbox("Sex", ["female", "male"],key="analysis_sex",on_change=_clear_dashboard_results)
        filters={"sex":sex}
        return f"{sex} athletes",filters
    if cohort_type == "National team level":
        team = st.selectbox(
            "National team level",
            ["junior_national_team", "senior_national_team"],
            format_func=lambda value: value.replace("_", " ").title(),
            key="analysis_team",on_change=_clear_dashboard_results,
        )
        filters={"national_team":"Junior" if team.startswith("junior") else "Senior"}
        return ("junior national team athletes" if team.startswith("junior") else "senior national team athletes"),filters

    age_group = st.selectbox(
        "Age group",
        ["under_20", "20_and_above"],
        format_func=lambda value: "Under 20" if value == "under_20" else "20 and above",
        key="analysis_age_group",on_change=_clear_dashboard_results,
    )
    filters={"age_group":age_group}
    return ("athletes under 20" if age_group=="under_20" else "athletes aged 20 and above"),filters


def page_dashboard():
    st.subheader("Protected Nature-style analysis dashboard")
    st.write(
        "This page runs the main analysis types inspired by the Nature paper. "
        "Each button now uses the same PRISM -> RouteLLM -> generated code -> "
        "local backend execution pipeline. Only aggregate tables and figures are returned."
    )

    control_col, note_col = st.columns([1, 2])
    with control_col:
        cohort_group, active_filters = _dashboard_cohort_selector()
    with note_col:
        st.info(
            "Choose one card below. The selected request will generate safe Python code, "
            "validate it locally, and execute it on the protected backend."
        )
        st.caption(
            "The selected cohort is used by all six analyses. Analyses that "
            "mathematically require both elite and semi-elite athletes may be "
            "not applicable to a single-class cohort."
        )

    analyses = [
        {
            "title": "Logistic regression",
            "caption": "Aggregate factors associated with the elite/semi-elite outcome.",
            "button": "Run Table 1",
            "requested_analysis": "table1",
        },
        {
            "title": "Multiple linear regression",
            "caption": "Expertise-value regression for the selected cohort.",
            "button": "Run Table 2",
            "requested_analysis": "table2",
        },
        {
            "title": "Network Analysis",
            "caption": (
                "Regression coefficients, predictor correlations, "
                "and elite vs semi-elite variance."
            ),
            "button": "Generate Network Analysis",
            "requested_analysis": "figure1",
        },
        {
            "title": "Athlete Profile Visualization",
            "caption": "Anonymous standardized z-score profiles for the selected cohort.",
            "button": "Generate Athlete Profile Visualization",
            "requested_analysis": "figure2",
        },
        {
            "title": "Correlation",
            "caption": "Pairwise relationships among predictor variables.",
            "button": "Run correlation",
            "requested_analysis": "correlation",
        },
        {
            "title": "Variance",
            "caption": "Elite vs semi-elite variance comparison.",
            "button": "Run variance",
            "requested_analysis": "variance_analysis",
        },
    ]

    if "dashboard_selected_response_key" not in st.session_state:
        st.session_state.dashboard_selected_response_key = None
    response_keys = {
        "table1": "table1_response",
        "table2": "table2_response",
        "figure1": "figure1_response",
        "figure2": "figure2_response",
        "correlation": "correlation_response",
        "variance_analysis": "variance_response",
    }
    for response_key in response_keys.values():
        if response_key not in st.session_state:
            st.session_state[response_key] = None

    st.markdown("### Select protected analysis")
    for row_start in range(0, len(analyses), 3):
        cols = st.columns(3)
        for col, analysis in zip(cols, analyses[row_start : row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="analysis-card">
                        <div class="analysis-card-title">{analysis["title"]}</div>
                        <div class="analysis-card-caption">{analysis["caption"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if analysis["requested_analysis"] == "figure2":
                    figure2_size_option = st.selectbox(
                        "Number of athletes",
                        options=["20", "50", "80", "All"],
                        index=1,
                        key="figure2_size_option",
                    )
                    analysis_prompt=build_dashboard_prompt("figure2",cohort_group,figure2_size_option)
                else:
                    analysis_prompt=build_dashboard_prompt(analysis["requested_analysis"],cohort_group)

                if st.button(analysis["button"], use_container_width=True):
                    response_key = response_keys[analysis["requested_analysis"]]
                    st.session_state[response_key] = run_request(
                        analysis_prompt,
                        requested_analysis=analysis["requested_analysis"],
                        analysis_filters=active_filters,
                    )
                    if analysis["requested_analysis"] in NOISE_ENABLED_ANALYSES:
                        versions=st.session_state.setdefault("noise_dashboard_versions",{})
                        versions[response_key]=NOISE_DASHBOARD_VERSION
                    st.session_state.dashboard_selected_response_key = response_key

    selected_response_key = st.session_state.dashboard_selected_response_key
    selected_response = st.session_state.get(selected_response_key) if selected_response_key else None
    selected_analysis=next((key for key,value in response_keys.items() if value==selected_response_key),None)
    if selected_response:
        show_pipeline_response(selected_response)
        if (selected_analysis in NOISE_ENABLED_ANALYSES
                and selected_response.get("allowed")
                and isinstance(selected_response.get("result"),dict)
                and not _has_current_noise_utility(selected_response,selected_analysis)):
            st.warning("Controlled perturbation results are unavailable for this analysis. The original analysis result is still shown.")
    else:
        st.info("Select one analysis card above to generate code and run the protected backend pipeline.")


def page_data_generation_processing():
    st.title("Data Generation & Processing")
    st.caption("How the synthetic athlete dataset is created, transformed, validated, and protected.")
    report=load_generation_report("data/synthetic_generation_report.json")
    summary=load_safe_dataset_summary("data/synthetic_athlete_data.csv")
    if not report:st.warning("Generation report not found. Run the synthetic data generator first.")
    elif report.get("status")=="regenerate_with_generator":st.warning("Generation report not found. Run the synthetic data generator first.")
    if not summary:st.warning("Synthetic analysis dataset is not available.")
    st.markdown("## A. Dataset Overview")
    expertise=summary.get("expertise_summary") or (report or {}).get("expertise_summary") or {}
    overview=[("Dataset Type","Correlation-aware synthetic athlete dataset"),("Number of Athletes",summary.get("number_of_athletes",report.get("n_athletes","Not available") if report else "Not available")),("Number of Sports",summary.get("number_of_sports","Not available")),("Number of Final Domains",8),("Mean Expertise Score",f"{expertise['mean']:.2f}" if expertise.get("mean") is not None else "Not available"),("Expertise Score Range",format_expertise_range(expertise.get("min"),expertise.get("max")))]
    for start in range(0,len(overview),4):
        cols=st.columns(min(4,len(overview)-start))
        for col,(label,value) in zip(cols,overview[start:start+4]):
            with col:_render_full_value_card(label,str(value))
    st.info("This dataset is synthetic and is intended for software, privacy, and analysis-pipeline evaluation. It does not reproduce the confidential original athlete records.")
    st.markdown("### Expertise Distribution")
    st.write("Expertise is represented as a continuous score from 2 to 16. Binary groups are created only when required by specific statistical analyses.")
    distribution=expertise.get("distribution_bins") or {}
    if distribution:
        distribution_rows=pd.DataFrame([{"Expertise Score Range":label,"Number of Athletes":count} for label,count in distribution.items()])
        st.bar_chart(distribution_rows,x="Expertise Score Range",y="Number of Athletes")
        st.dataframe(distribution_rows,use_container_width=True,hide_index=True)
    st.info(get_expertise_group_explanation())
    st.dataframe(pd.DataFrame([{"Analysis Use":"Descriptive analysis","Expertise Representation":"Continuous score from 2 to 16"},{"Analysis Use":"Multiple linear regression","Expertise Representation":"Continuous Expertise Score"},{"Analysis Use":"Logistic regression","Expertise Representation":"Higher-expertise group >= 13 vs Comparison Group <= 12"},{"Analysis Use":"Individual profile analysis","Expertise Representation":"Continuous standardized domain scores"}]),use_container_width=True,hide_index=True)
    st.markdown("## B. Generation Pipeline")
    pipeline=[("Published sample structure","Sex, sport, national-team structure, and age distributions."),("Synthetic athlete background","ID, age, sex, sport, national-team group, and age group are created locally."),("Simulated raw diagnostic measurements","Categories include physical tests, genotype dosages, micronutrients, cognitive tests, questionnaires, and satisfaction items; values are never shown here."),("Published score-construction procedures","Simulated measurements are transformed using documented standardization and aggregation rules."),("Eight standardized domains","Only protected standardized domain variables are used by downstream analyses."),("Protected statistical analysis","Analyses run locally; results are returned after local execution.")]
    for index,(name,description) in enumerate(pipeline,1):st.markdown(f"**{index}. {name}**  \n{description}" + ("  \n↓" if index<len(pipeline) else ""))
    st.markdown("## C. Eight-Domain Construction")
    st.dataframe(pd.DataFrame(build_domain_construction_rows()),use_container_width=True,hide_index=True,height=560,column_config={name:st.column_config.TextColumn(width="large") for name in ["Raw Inputs","Processing","Scientific Basis"]})
    st.markdown("## D. Scientific Basis and Simulation Assumptions")
    left,right=st.columns(2)
    with left:
        st.markdown("### Published basis");st.markdown("- Sample structure and age distributions\n- Sport-specific lower-body tests\n- Relative grip-strength construction\n- Sex-specific gene combinations\n- Micronutrient thresholds and formula\n- Cognitive and questionnaire score construction\n- Expertise taxonomy and selected domain correlations")
    with right:
        st.markdown("### Simulation assumptions");st.markdown("- Raw weight and grip distributions\n- Jump, sprint, and tapping distributions\n- Allele frequencies and micronutrient distributions\n- Questionnaire response probabilities\n- Sport effects, truncation bounds, noise, and latent parameters")
    st.info("Published methods define how scores are constructed. Unpublished raw distributions are simulated and explicitly marked as assumptions.")
def _load_json_artifact(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv_artifact(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _load_jsonl_artifact(path: str) -> list[dict]:
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _format_percentage(value) -> str:
    if value is None:
        return "Not available"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "Not available"


def _get_comparison_method_metrics(report: dict, method_key: str) -> tuple[dict, dict]:
    method_payload = ((report.get("methods") or {}).get(method_key) or {})
    common = report.get("common_completed_subset") or {}
    common_methods = common.get("methods") or {}
    common_metrics = common_methods.get(method_key) or common.get(f"{method_key}_metrics")
    if isinstance(common_metrics, dict) and common_metrics:
        return method_payload, common_metrics
    nested = method_payload.get("metrics")
    return method_payload, nested if isinstance(nested, dict) else method_payload


def _format_method_result(example: dict, prefix: str) -> str:
    prediction = example.get(f"{prefix}_prediction")
    route_label = _friendly_label(prediction) if prediction else "Failed"
    indicator = "✓" if example.get(f"{prefix}_correct") else "✗"
    return f"{route_label} {indicator}"


def _method_c_examples_with_local_fallback(report: dict) -> dict:
    examples = report.get("method_c_route_examples") or {}
    complete = all(
        isinstance(examples.get(route), dict)
        and not examples[route].get("missing")
        for route in PRIVACY_ROUTE_ORDER
    )
    if complete:
        return examples
    details_path = Path("artifacts/privacy_methods_frontend60_details.jsonl")
    if not details_path.exists():
        return examples
    try:
        from scripts.evaluate_privacy_methods_frontend60 import (
            build_method_c_route_examples,
        )
        records = _load_jsonl_artifact(str(details_path))
        return build_method_c_route_examples(records)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return examples


def _render_method_c_route_examples(report: dict) -> None:
    st.markdown("#### Representative Examples by Method C Route")
    st.caption(
        "One deterministic comparison example is shown for each Method C route, "
        "prioritizing correct predictions with visible disagreement between methods."
    )
    examples = _method_c_examples_with_local_fallback(report)
    method_specs = (
        ("method_a", "Method A"),
        ("method_b", "Method B"),
        ("method_c", "Method C"),
    )
    for index, route in enumerate(PRIVACY_ROUTE_ORDER, 1):
        example = examples.get(route) or {
            "missing": True,
            "message": "No Method C example found for this route.",
        }
        with st.container(border=True):
            st.markdown(
                f"### Example {index} — Method C predicts {_friendly_label(route)}"
            )
            if example.get("missing"):
                st.info(example.get("message", "No example is available."))
                continue
            st.markdown("**User Query**")
            st.info(str(example.get("user_query", "Not available")))
            st.write(
                f"**Expected Route:** "
                f"{_friendly_label(example.get('ground_truth_route'))}"
            )
            for prefix, label in method_specs:
                st.write(f"**{label}:** {_format_method_result(example, prefix)}")
                score = example.get(f"{prefix}_score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    st.caption(f"Risk Score: {float(score):.2f}")
                st.caption(
                    f"Comparison Summary: "
                    f"{example.get('comparison_summary', 'Not available')}"
                )


def _select_cloud_local_representative_examples(
    per_sample_results: list[dict],
) -> dict[str, dict]:
    """Backward-compatible names for the explanatory router examples."""
    selected = _select_cost_router_explanation_examples(per_sample_results)
    return {
        "correct_cloud": selected.get("cloud_needed"),
        "correct_local": selected.get("local_sufficient"),
        "incorrect": selected.get("routing_error"),
    }


def _select_cost_router_explanation_examples(
    per_sample_results: list[dict],
) -> dict[str, dict]:
    """Select deterministic capability-focused examples from saved evaluation rows."""
    valid = [
        sample for sample in per_sample_results
        if isinstance(sample, dict)
        and str(sample.get("ground_truth") or "").lower() in {"cloud", "local"}
    ]

    def first_match(predicate):
        return next((sample for sample in valid if predicate(sample)), None)

    def fully_correct(sample, model):
        result = sample.get(model) or {}
        return result.get("fully_correct") is True

    cloud_needed = first_match(lambda sample: (
        sample.get("ground_truth") == "cloud"
        and sample.get("prediction") == "cloud"
        and not fully_correct(sample, "local")
        and fully_correct(sample, "cloud")
    ))
    local_sufficient = first_match(lambda sample: (
        sample.get("ground_truth") == "local"
        and sample.get("prediction") == "local"
        and fully_correct(sample, "local")
        and fully_correct(sample, "cloud")
    )) or first_match(lambda sample: (
        sample.get("ground_truth") == "local"
        and sample.get("prediction") == "local"
        and fully_correct(sample, "local")
    ))
    routing_error = first_match(lambda sample: (
        sample.get("ground_truth") == "cloud"
        and sample.get("prediction") == "local"
        and not fully_correct(sample, "local")
        and fully_correct(sample, "cloud")
    )) or first_match(lambda sample: (
        sample.get("prediction") != sample.get("ground_truth")
    ))
    return {
        "cloud_needed": cloud_needed,
        "local_sufficient": local_sufficient,
        "routing_error": routing_error,
    }


def _router_score_caption(example: dict) -> str:
    p_cloud = float(example.get("p_cloud"))
    threshold = float(example.get("threshold"))
    comparison = ">=" if p_cloud >= threshold else "<"
    return (
        f"Router score: P_cloud = {p_cloud:.4f} {comparison} "
        f"threshold = {threshold:.4f}"
    )


def _model_correct_status(example: dict, model: str) -> str:
    value = (example.get(model) or {}).get("fully_correct")
    return "PASS" if value is True else "FAIL" if value is False else "Not available"


def _render_cloud_local_evaluation_tab():
    app_root = Path(__file__).resolve().parent
    training_artifact_path = app_root / "artifacts/athlete_cloud_local_router.json"
    evaluation_artifact_path = app_root / "artifacts/athlete_cloud_local_router_evaluation.json"
    training = _load_json_artifact(str(training_artifact_path))
    independent = _load_json_artifact(str(evaluation_artifact_path))
    st.markdown("### Cost-aware Routing")
    st.caption(
        "This page reads saved offline calibration and independent-evaluation results. "
        "It never trains models or calls external APIs from the web interface."
    )
    if not training or training.get("status") != "trained":
        st.warning(
            "The Cloud/Local Router has not been trained from objective code-validation labels."
        )
        st.code("python scripts/train_athlete_cloud_local_router.py", language="powershell")
        return
    st.success("Saved Cloud/Local Router loaded successfully.")
    calibration_columns = st.columns(3)
    calibration_columns[0].metric("Router", "Cloud/Local Router")
    calibration_columns[1].metric("Training Samples", training.get("training_samples"))
    calibration_columns[2].metric("Threshold", f"{float(training['threshold']):.6f}")
    evaluation_complete = bool(independent and independent.get("status") == "evaluated")
    metrics = independent if evaluation_complete else {}
    accuracy_columns = st.columns(3)
    accuracy_columns[0].metric(
        "Routing Accuracy", _format_percentage(metrics.get("routing_accuracy"))
    )
    routing_metric_columns = st.columns(3)
    routing_metric_columns[0].metric(
        "Cloud Recall", _format_percentage(metrics.get("cloud_recall"))
    )
    routing_metric_columns[1].metric(
        "Local Recall", _format_percentage(metrics.get("local_recall"))
    )
    routing_metric_columns[2].metric(
        "Cloud Usage Rate", _format_percentage(metrics.get("cloud_usage_rate"))
    )
    if not evaluation_complete:
        st.info("The independent 40-request Cloud/Local router evaluation has not been completed.")
    st.caption("Independent benchmark: 40 requests, eight per supported analysis task.")
    _render_cloud_codegen_evaluation(app_root)
    _render_local_codegen_evaluation(app_root)

    st.markdown("## Representative Cost-aware Routing Examples")
    st.caption(
        "The Router predicts from the request text whether the Local Model is sufficient "
        "for Restricted Analysis Call generation. Cloud is selected only when additional "
        "model capability is predicted to be necessary."
    )
    st.caption(
        "PASS means the generated Restricted Analysis Call passed the full objective "
        "code-generation evaluation."
    )
    if not evaluation_complete:
        st.info("Representative examples will be available after the independent evaluation is completed.")
        return
    examples = _select_cost_router_explanation_examples(
        independent.get("per_sample_results") or []
    )
    example_specs = (
        ("cloud_needed", "Example A — Cloud Model Needed"),
        ("local_sufficient", "Example B — Local Model Is Sufficient"),
        ("routing_error", "Example C — Router Prediction Error"),
    )
    for key, title in example_specs:
        example = examples.get(key)
        with st.container(border=True):
            st.markdown(f"### {title}")
            if not isinstance(example, dict):
                st.info("No matching saved independent evaluation example is available.")
                continue
            st.markdown("#### User Request")
            st.write(example.get("prompt", ""))
            ground_truth = str(example.get("ground_truth") or "").lower()
            prediction = str(example.get("prediction") or "").lower()
            local_status = _model_correct_status(example, "local")
            cloud_status = _model_correct_status(example, "cloud")

            if key == "cloud_needed":
                st.dataframe(pd.DataFrame([{
                    "Router Decision": _friendly_label(prediction),
                    "Local Model": local_status,
                    "Cloud Model": cloud_status,
                    "Why": "Local Model could not generate a Fully Correct Restricted Analysis Call, so Cloud capability was required.",
                }]), use_container_width=True, hide_index=True)
                st.success("✓ Correct Router Decision")
            elif key == "local_sufficient":
                explanation = (
                    "Local Model already generates a Fully Correct Restricted Analysis "
                    "Call, so an additional Cloud Model call is unnecessary."
                    if cloud_status == "PASS" else
                    "Local Model generates a Fully Correct Restricted Analysis Call and "
                    "is sufficient for this request."
                )
                st.dataframe(pd.DataFrame([{
                    "Router Decision": _friendly_label(prediction),
                    "Local Model": local_status,
                    "Cloud Model": cloud_status,
                    "Why": explanation,
                }]), use_container_width=True, hide_index=True)
                st.success("✓ Cloud Call Avoided")
            else:
                st.dataframe(pd.DataFrame([{
                    "Router Decision": _friendly_label(prediction),
                    "Actually Needed": _friendly_label(ground_truth),
                    "Local Model": local_status,
                    "Cloud Model": cloud_status,
                }]), use_container_width=True, hide_index=True)
                if ground_truth == "cloud" and prediction == "local":
                    st.write(
                        "The Router underestimated the difficulty of this request and "
                        "selected the Local Model although the Cloud Model was required."
                    )
                else:
                    st.write(
                        "The saved Router prediction did not match the model capability "
                        "required by the objective evaluation."
                    )
                st.warning("✗ Incorrect Router Decision")
            st.caption(_router_score_caption(example))


def _render_cloud_codegen_evaluation(app_root: Path) -> None:
    st.markdown("### Cloud Model Comparison for Restricted Code Generation")
    st.write(
        "The same 40 frontend-realistic requests are sent independently to GPT-4.1, "
        "Gemini 3.5 Flash, and Claude Sonnet 5. Each model receives the same restricted "
        "code-generation prompt. Generated code is verified and executed locally. The "
        "main metric is Fully Correct Code Rate."
    )
    st.write("**Prompt Version:** Pool + Typed Schema V3")
    artifact_path=app_root / "artifacts/cloud_codegen_model_evaluation.json"
    report=_load_json_artifact(str(artifact_path))
    command="python -m scripts.evaluate_cloud_codegen_models"
    if not report:
        st.info("Cloud LLM code-generation evaluation has not been run.")
        st.code(command,language="powershell")
        return
    if report.get("status")!="complete":
        st.warning("The saved cloud code-generation evaluation is incomplete.")
        st.code(command+" --resume",language="powershell")
        return
    rows=[]
    for result in report.get("overall_results") or []:
        latency=result.get("Average Latency Seconds")
        cost=result.get("total_estimated_cost_usd")
        rows.append({
            "Model":result.get("Model"),
            "Fully Correct / 40":f"{result.get('Fully Correct',0)} / 40",
            "Fully Correct Accuracy":_format_percentage(result.get("Fully Correct Accuracy")),
            "Structure Valid Rate":_format_percentage(result.get("Structure Valid Rate")),
            "Request Match Rate":_format_percentage(result.get("Request Match Rate")),
            "Execution Success Rate":_format_percentage(result.get("Execution Success Rate")),
            "Average Latency":f"{float(latency):.2f} s" if latency is not None else "Not available",
            "Estimated Cost":f"${float(cost):.4f}" if cost is not None else "Not available",
        })
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    _render_cloud_codegen_accuracy_charts(report)
    st.markdown("### Representative Cloud Model Evaluation Example")
    example = _select_cloud_codegen_representative_example(
        report.get("per_sample_results") or []
    )
    if example:
        st.markdown("#### User Request")
        st.write(example.get("prompt") or "Not available")
        st.dataframe(
            _codegen_example_table_data(example, _CLOUD_CODEGEN_MODEL_SPECS),
            use_container_width=True,
            hide_index=True,
        )
        _render_representative_codegen_calls(example, _CLOUD_CODEGEN_MODEL_SPECS)
    else:
        st.info("No complete three-model Cloud evaluation example is available.")


def _format_representative_generated_code(code: str) -> str:
    """Pretty-print a restricted call for display without changing its stored value."""
    try:
        module = ast.parse(code)
        assignment = module.body[0]
        call = assignment.value
        if not (
            len(module.body) == 1
            and isinstance(assignment, ast.Assign)
            and isinstance(call, ast.Call)
        ):
            return code

        lines = [f"{ast.unparse(assignment.targets[0])} = {ast.unparse(call.func)}("]
        for argument in call.args:
            lines.append(f"    {ast.unparse(argument)},")
        for keyword in call.keywords:
            try:
                value = ast.literal_eval(keyword.value)
                rendered_value = pprint.pformat(value, width=72, sort_dicts=False)
            except (TypeError, ValueError):
                rendered_value = ast.unparse(keyword.value)
            value_lines = rendered_value.splitlines()
            lines.append(f"    {keyword.arg}={value_lines[0]}")
            lines.extend(f"    {line}" for line in value_lines[1:])
            lines[-1] += ","
        lines.append(")")
        return "\n".join(lines)
    except (AttributeError, IndexError, SyntaxError, TypeError, ValueError):
        return code


_CLOUD_CODEGEN_MODEL_SPECS = (
    ("gpt4_1", "GPT-4.1"),
    ("gemini", "Gemini 3.5 Flash"),
    ("claude", "Claude Sonnet 5"),
)
_CLOUD_CODEGEN_MODEL_COLORS = {
    "GPT-4.1": "#3b82f6",
    "Gemini 3.5 Flash": "#10b981",
    "Claude Sonnet 5": "#8b5cf6",
}
_CLOUD_CODEGEN_TASK_LABELS = {
    "Table 1": "Logistic Regression",
    "Table 2": "Multiple Linear Regression",
    "Figure 1": "Network Analysis",
}
_MODEL_COMPARISON_CHART_HEIGHT = 350


def _render_cloud_codegen_accuracy_charts(report: dict) -> None:
    overall_rows = []
    for result in report.get("overall_results") or []:
        model = str(result.get("Model") or "")
        value = result.get("Fully Correct Accuracy")
        if model in _CLOUD_CODEGEN_MODEL_COLORS and isinstance(value, (int, float)):
            overall_rows.append({
                "Model": model,
                "Fully Correct Accuracy (%)": round(float(value) * 100.0, 10),
            })
    overall_data = pd.DataFrame(overall_rows)
    color_scale = alt.Scale(
        domain=list(_CLOUD_CODEGEN_MODEL_COLORS),
        range=list(_CLOUD_CODEGEN_MODEL_COLORS.values()),
    )
    value_field = "Fully Correct Accuracy (%)"
    overall_base = alt.Chart(overall_data).encode(
        x=alt.X(
            "Model:N",
            sort=list(_CLOUD_CODEGEN_MODEL_COLORS),
            title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=260),
        ),
        y=alt.Y(
            f"{value_field}:Q",
            title=value_field,
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color("Model:N", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip(f"{value_field}:Q", format=".1f"),
        ],
    )
    overall_chart = (
        overall_base.mark_bar(size=90)
        + overall_base.mark_text(dy=-8, fontSize=13).encode(
            text=alt.Text(f"{value_field}:Q", format=".1f")
        )
    ).properties(
        height=_MODEL_COMPARISON_CHART_HEIGHT,
        title="Cloud LLM Restricted Code Fully Correct Accuracy",
    )
    st.altair_chart(overall_chart, use_container_width=True, theme=None)

    task_rows = []
    for result in report.get("per_task_results") or []:
        model = str(result.get("Model") or "")
        value = result.get("Accuracy")
        if model in _CLOUD_CODEGEN_MODEL_COLORS and isinstance(value, (int, float)):
            task_rows.append({
                "Analysis Task": _CLOUD_CODEGEN_TASK_LABELS.get(
                    str(result.get("Task") or ""),
                    str(result.get("Task") or ""),
                ),
                "Model": model,
                value_field: round(float(value) * 100.0, 10),
            })
    task_data = pd.DataFrame(task_rows)
    task_order = list(dict.fromkeys(task_data.get("Analysis Task", [])))
    task_chart = (
        alt.Chart(task_data)
        .mark_bar(size=24)
        .encode(
            x=alt.X(
                "Analysis Task:N",
                sort=task_order,
                title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=220),
            ),
            xOffset=alt.XOffset("Model:N", sort=list(_CLOUD_CODEGEN_MODEL_COLORS)),
            y=alt.Y(
                f"{value_field}:Q",
                title=value_field,
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "Model:N",
                scale=color_scale,
                legend=alt.Legend(title="Model", orient="top-right"),
            ),
            tooltip=[
                alt.Tooltip("Analysis Task:N"),
                alt.Tooltip("Model:N"),
                alt.Tooltip(f"{value_field}:Q", format=".1f"),
            ],
        )
        .properties(
            height=_MODEL_COMPARISON_CHART_HEIGHT,
            title="Fully Correct Accuracy by Analysis Task",
        )
    )
    st.altair_chart(task_chart, use_container_width=True, theme=None)
_LOCAL_CODEGEN_MODELS = (
    "Ministral-3-8B",
    "Qwen2.5-Coder-7B-Instruct",
    "Llama-3.1-8B-Instruct",
)
_LOCAL_CODEGEN_MODEL_SPECS = (
    ("ministral", "Ministral-3-8B"),
    ("qwen", "Qwen2.5-Coder-7B-Instruct"),
    ("llama", "Llama-3.1-8B-Instruct"),
)
_LOCAL_CODEGEN_MODEL_COLORS = {
    "Ministral-3-8B": "#3b82f6",
    "Qwen2.5-Coder-7B-Instruct": "#10b981",
    "Llama-3.1-8B-Instruct": "#8b5cf6",
}
_LOCAL_CODEGEN_OVERALL_METRICS = (
    ("Structure", "Structure Valid Rate"),
    ("Request Match", "Request Match Rate"),
    ("Execution", "Execution Success Rate"),
    ("Result", "Result Valid Rate"),
    ("Fully Correct", "Fully Correct Accuracy"),
)
_LOCAL_CODEGEN_TASK_LABELS = {
    "Table 1": "Logistic Regression",
    "Table 2": "Multiple Linear Regression",
    "Figure 1": "Network Analysis",
    "Correlation": "Correlation Analysis",
    "Variance Analysis": "Variance Analysis",
}


def _select_codegen_comparison_example(
    per_sample_results: list[dict],
    model_keys: tuple[str, ...],
    *,
    preferred_model: str | None = None,
) -> dict | None:
    """Select a complete same-request comparison in stable saved order."""
    groups = {}
    required = set(model_keys)
    for row in per_sample_results:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or "")
        model_key = str(row.get("model_key") or "")
        if not sample_id or model_key not in required:
            continue
        group = groups.setdefault(
            sample_id,
            {"sample_id": sample_id, "prompt": row.get("prompt") or "", "models": {}},
        )
        group["models"].setdefault(model_key, row)

    complete = [group for group in groups.values() if set(group["models"]) == required]
    for group in complete:
        outcomes = {
            key: group["models"][key].get("fully_correct") is True
            for key in model_keys
        }
        if preferred_model is not None:
            if outcomes.get(preferred_model) and not all(outcomes.values()):
                return group
        elif any(outcomes.values()) and not all(outcomes.values()):
            return group
    return complete[0] if complete else None


def _select_cloud_codegen_representative_example(
    per_sample_results: list[dict],
) -> dict | None:
    return _select_codegen_comparison_example(
        per_sample_results,
        tuple(key for key, _ in _CLOUD_CODEGEN_MODEL_SPECS),
    )


def _local_codegen_overall_chart_data(report: dict) -> pd.DataFrame:
    """Convert saved overall evaluation rates to chart-ready percentages."""
    rows = []
    for result in report.get("overall_results") or []:
        model = str(result.get("Model") or "")
        if model not in _LOCAL_CODEGEN_MODELS:
            continue
        for metric_label, artifact_field in _LOCAL_CODEGEN_OVERALL_METRICS:
            value = result.get(artifact_field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append({
                    "Metric": metric_label,
                    "Model": model,
                    "Accuracy (%)": round(float(value) * 100.0, 10),
                })
    return pd.DataFrame(rows, columns=("Metric", "Model", "Accuracy (%)"))


def _local_codegen_fully_correct_chart_data(report: dict) -> pd.DataFrame:
    """Read only overall Fully Correct Accuracy for the three-model bar chart."""
    rows = []
    results_by_model = {
        str(result.get("Model") or ""): result
        for result in report.get("overall_results") or []
        if isinstance(result, dict)
    }
    for model in _LOCAL_CODEGEN_MODELS:
        value = (results_by_model.get(model) or {}).get("Fully Correct Accuracy")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({
                "Model": model,
                "Fully Correct Accuracy (%)": round(float(value) * 100.0, 10),
            })
    return pd.DataFrame(
        rows,
        columns=("Model", "Fully Correct Accuracy (%)"),
    )


def _local_codegen_overall_table_data(report: dict) -> pd.DataFrame:
    """Build the Cloud-style Local Model summary table from saved results."""
    results_by_model = {
        str(result.get("Model") or ""): result
        for result in report.get("overall_results") or []
        if isinstance(result, dict)
    }
    rows = []
    for model in _LOCAL_CODEGEN_MODELS:
        result = results_by_model.get(model)
        if not result:
            continue
        rows.append({
            "Model": model,
            "Fully Correct / 40": (
                f"{result.get('Fully Correct')} / {result.get('Samples')}"
            ),
            "Fully Correct Accuracy": _format_percentage(
                result.get("Fully Correct Accuracy")
            ),
            "Structure Valid Rate": _format_percentage(
                result.get("Structure Valid Rate")
            ),
            "Request Match Rate": _format_percentage(result.get("Request Match Rate")),
            "Execution Rate": _format_percentage(result.get("Execution Success Rate")),
            "Result Valid Rate": _format_percentage(result.get("Result Valid Rate")),
        })
    return pd.DataFrame(
        rows,
        columns=(
            "Model",
            "Fully Correct / 40",
            "Fully Correct Accuracy",
            "Structure Valid Rate",
            "Request Match Rate",
            "Execution Rate",
            "Result Valid Rate",
        ),
    )


def _select_local_model_comparison_example(
    per_sample_results: list[dict],
) -> dict | None:
    """Select one complete three-model request in stable saved order."""
    return _select_codegen_comparison_example(
        per_sample_results,
        ("ministral", "qwen", "llama"),
        preferred_model="ministral",
    )


def _saved_validation_status(
    result: dict,
    field: str,
    prerequisites: tuple[str, ...] = (),
) -> str:
    if any(result.get(prerequisite) is not True for prerequisite in prerequisites):
        return "NOT RUN"
    value = result.get(field)
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "NOT RUN"


def _codegen_example_table_data(
    example: dict,
    model_specs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    models = example.get("models") or {}
    specifications = (
        ("Structure", "structure_validation_passed", ()),
        ("Request Match", "request_match_passed", ("structure_validation_passed",)),
        ("Execution", "local_execution_passed", (
            "structure_validation_passed", "request_match_passed")),
        ("Result", "result_validation_passed", (
            "structure_validation_passed", "request_match_passed",
            "local_execution_passed")),
        ("Fully Correct", "fully_correct", ()),
    )
    rows = []
    for label, field, prerequisites in specifications:
        row = {"Validation Measure": label}
        for model_key, display_name in model_specs:
            row[display_name] = _saved_validation_status(
                models.get(model_key) or {}, field, prerequisites
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _local_model_example_table_data(example: dict) -> pd.DataFrame:
    return _codegen_example_table_data(example, _LOCAL_CODEGEN_MODEL_SPECS)


def _render_representative_codegen_calls(
    example: dict,
    model_specs: tuple[tuple[str, str], ...],
) -> None:
    st.markdown("#### Representative Generated Code")
    models = example.get("models") or {}
    for model_key, display_name in model_specs:
        result = models.get(model_key) or {}
        with st.container(border=True):
            st.markdown(f"##### {display_name}")
            generated_code = result.get("generated_code")
            if generated_code:
                st.code(
                    _format_representative_generated_code(str(generated_code)),
                    language="python",
                )
            else:
                st.info("No valid Restricted Analysis Call generated.")
            st.write(
                f"**Fully Correct:** {'Yes' if result.get('fully_correct') is True else 'No'}"
            )


def _local_codegen_task_chart_data(report: dict) -> pd.DataFrame:
    """Convert saved per-task accuracy to thesis-facing chart labels."""
    rows = []
    for result in report.get("per_task_results") or []:
        model = str(result.get("Model") or "")
        task = _LOCAL_CODEGEN_TASK_LABELS.get(str(result.get("Task") or ""))
        value = result.get("Accuracy")
        if (
            model in _LOCAL_CODEGEN_MODELS
            and task is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            rows.append({
                "Analysis Task": task,
                "Model": model,
                "Fully Correct Accuracy (%)": round(float(value) * 100.0, 10),
            })
    return pd.DataFrame(
        rows,
        columns=("Analysis Task", "Model", "Fully Correct Accuracy (%)"),
    )


def _render_grouped_accuracy_chart(
    data: pd.DataFrame,
    *,
    category_field: str,
    value_field: str,
    category_order: list[str],
) -> None:
    chart = (
        alt.Chart(data)
        .mark_bar(size=24)
        .encode(
            x=alt.X(
                f"{category_field}:N",
                sort=category_order,
                title=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelOverlap=False,
                    labelLimit=220,
                    labelFontSize=11,
                ),
                scale=alt.Scale(paddingInner=0.28, paddingOuter=0.12),
            ),
            xOffset=alt.XOffset("Model:N", sort=list(_LOCAL_CODEGEN_MODELS)),
            y=alt.Y(
                f"{value_field}:Q",
                title=value_field,
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "Model:N",
                sort=list(_LOCAL_CODEGEN_MODELS),
                scale=alt.Scale(
                    domain=list(_LOCAL_CODEGEN_MODEL_COLORS),
                    range=list(_LOCAL_CODEGEN_MODEL_COLORS.values()),
                ),
                legend=alt.Legend(
                    title="Model",
                    orient="top-right",
                    direction="vertical",
                    fillColor="white",
                    strokeColor="#d0d0d0",
                    padding=8,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{category_field}:N"),
                alt.Tooltip("Model:N"),
                alt.Tooltip(f"{value_field}:Q", format=".1f"),
            ],
        )
        .properties(
            height=_MODEL_COMPARISON_CHART_HEIGHT,
            title=alt.TitleParams(
                text="Fully Correct Accuracy by Analysis Task",
                anchor="middle",
                fontSize=14,
                offset=14,
            ),
        )
        .configure_axis(gridColor="#e5e7eb", gridOpacity=0.8)
        .configure_view(stroke="#666666")
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def _render_simple_fully_correct_chart(data: pd.DataFrame) -> None:
    value_field = "Fully Correct Accuracy (%)"
    base = alt.Chart(data).encode(
        x=alt.X(
            "Model:N",
            sort=list(_LOCAL_CODEGEN_MODELS),
            title=None,
            axis=alt.Axis(
                labelAngle=0,
                labelOverlap=False,
                labelLimit=260,
                labelFontSize=11,
            ),
        ),
        y=alt.Y(
            f"{value_field}:Q",
            title=value_field,
            scale=alt.Scale(domain=[0, 100]),
        ),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip(f"{value_field}:Q", format=".1f"),
        ],
        color=alt.Color(
            "Model:N",
            scale=alt.Scale(
                domain=list(_LOCAL_CODEGEN_MODEL_COLORS),
                range=list(_LOCAL_CODEGEN_MODEL_COLORS.values()),
            ),
            legend=None,
        ),
    )
    bars = base.mark_bar(size=70)
    labels = base.mark_text(dy=-8, fontSize=13).encode(
        text=alt.Text(f"{value_field}:Q", format=".1f")
    )
    st.altair_chart(
        (bars + labels).properties(height=_MODEL_COMPARISON_CHART_HEIGHT),
        use_container_width=True,
    )


def _render_local_codegen_evaluation(app_root: Path) -> None:
    st.markdown("### Local Model Comparison for Restricted Code Generation")
    st.write(
        "The same 40 Restricted Code Generation requests are processed independently "
        "by Ministral-3-8B, Qwen2.5-Coder-7B-Instruct, and Llama-3.1-8B-Instruct. "
        "Each model receives exactly the same Restricted Code Generation Prompt. "
        "Generated calls are verified and executed locally. The main metric is Fully "
        "Correct Code Rate."
    )

    artifact_path = app_root / "artifacts/local_codegen_model_evaluation.json"
    report = _load_json_artifact(str(artifact_path))
    command = "python -m scripts.evaluate_local_codegen_models --resume"
    if not report:
        st.info("Local Model code-generation evaluation has not been run.")
        st.code(command, language="powershell")
        return
    if report.get("status") != "complete":
        st.warning("The saved Local Model code-generation evaluation is incomplete.")
        st.code(command, language="powershell")
        return

    saved_prompt_version = (
        report.get("prompt_version")
        or next(
            (
                row.get("prompt_version")
                for row in report.get("overall_results") or []
                if isinstance(row, dict) and row.get("prompt_version")
            ),
            None,
        )
    )
    prompt_version_label = {
        "pool_typed_schema_v3": "Pool + Typed Schema V3",
    }.get(str(saved_prompt_version), str(saved_prompt_version or "Not available"))
    st.write(f"**Prompt Version:** {prompt_version_label}")
    st.dataframe(
        _local_codegen_overall_table_data(report),
        use_container_width=True,
        hide_index=True,
    )

    fully_correct_data = _local_codegen_fully_correct_chart_data(report)
    st.markdown("#### Local LLM Restricted Code Fully Correct Accuracy")
    _render_simple_fully_correct_chart(fully_correct_data)

    task_data = _local_codegen_task_chart_data(report)
    _render_grouped_accuracy_chart(
        task_data,
        category_field="Analysis Task",
        value_field="Fully Correct Accuracy (%)",
        category_order=list(_LOCAL_CODEGEN_TASK_LABELS.values()),
    )

    st.markdown("#### Representative Local Model Evaluation Example")
    example = _select_local_model_comparison_example(
        report.get("per_sample_results") or []
    )
    if example:
        st.markdown("#### User Request")
        st.write(example.get("prompt") or "Not available")
        st.dataframe(
            _local_model_example_table_data(example),
            use_container_width=True,
            hide_index=True,
        )
        _render_representative_codegen_calls(example, _LOCAL_CODEGEN_MODEL_SPECS)
    else:
        st.info("No complete three-model evaluation example is available.")
    st.info(
        "Ministral-3-8B achieves the highest overall Fully Correct rate among the "
        "three evaluated Local Models and is therefore used as the Local Model in "
        "the Cost-aware Routing pipeline."
    )


_PRIVACY_ASSESSOR_COLORS = {
    "GPT-4.1": "#3b82f6",
    "Gemini 3.5 Flash": "#10b981",
    "Claude Sonnet 5": "#8b5cf6",
}
_PRIVACY_METHOD_COLORS = {
    "Method A": "#3b82f6",
    "Method B": "#f59e0b",
    "Method C": "#10b981",
}
_PRIVACY_ROUTE_LABELS = {
    "cloud": "Cloud",
    "collaboration": "Collaboration",
    "local_edge": "Local Edge",
    "blocked": "Blocked",
}


def _privacy_grouped_chart(
    data: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    series_field: str,
    category_order: list[str],
    colors: dict[str, str],
    show_value_labels: bool = False,
) -> None:
    chart_data = data.copy()
    value_label_field = "_value_label"
    if show_value_labels:
        chart_data[value_label_field] = chart_data[y_field].map(
            lambda value: f"{float(value):.1f}%"
        )
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            f"{x_field}:N",
            sort=category_order,
            title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=220),
        ),
        xOffset=alt.XOffset(f"{series_field}:N", sort=list(colors)),
        y=alt.Y(f"{y_field}:Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            f"{series_field}:N",
            scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        ),
        tooltip=[x_field, series_field, alt.Tooltip(y_field, format=".1f")],
    )
    chart = base.mark_bar(size=25)
    if show_value_labels:
        chart += base.mark_text(dy=-8, fontSize=11).encode(
            text=alt.Text(f"{value_label_field}:N")
        )
    chart = chart.properties(height=330)
    st.altair_chart(chart, use_container_width=True, theme=None)


def _select_privacy_assessor_example(per_sample_results: list[dict]) -> dict | None:
    groups = {}
    required = {"gpt4_1", "gemini", "claude"}
    for row in per_sample_results:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or "")
        model_key = str(row.get("model_key") or "")
        if not sample_id or model_key not in required:
            continue
        group = groups.setdefault(sample_id, {
            "sample_id": sample_id,
            "prompt": row.get("prompt") or "",
            "ground_truth_route": row.get("ground_truth_route"),
            "models": {},
        })
        group["models"].setdefault(model_key, row)
    complete = [group for group in groups.values() if set(group["models"]) == required]
    return next((group for group in complete if len({
        row.get("predicted_route") for row in group["models"].values()
    }) > 1), complete[0] if complete else None)


def _privacy_assessor_example_table(example: dict) -> pd.DataFrame:
    def feature(value):
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "Not available"

    rows = []
    for model_key, display_name in (
        ("gpt4_1", "GPT-4.1"),
        ("gemini", "Gemini 3.5 Flash"),
        ("claude", "Claude Sonnet 5"),
    ):
        result = (example.get("models") or {}).get(model_key) or {}
        rows.append({
            "Privacy Assessor": display_name,
            "Privacy Risk": feature(result.get("privacy_risk_score")),
            "Subject Scope": feature(result.get("subject_scope")),
            "Data Sensitivity": feature(result.get("data_sensitivity")),
            "Disclosure Level": feature(result.get("disclosure_level")),
            "Blocked": "Yes" if result.get("blocked_request") is True else "No",
            "Predicted Route": _friendly_label(result.get("predicted_route")),
        })
    return pd.DataFrame(rows)


def _privacy_assessor_snapshot_frames(
    report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive thesis display values from frozen integer counts."""
    benchmark = report.get("benchmark") or {}
    total = benchmark.get("total_samples")
    ground_truth = benchmark.get("ground_truth_counts") or {}
    route_names = list(_PRIVACY_ROUTE_LABELS.values())
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError("The thesis Privacy Assessor snapshot has an invalid sample count.")
    if any(
        not isinstance(ground_truth.get(route), int)
        or isinstance(ground_truth.get(route), bool)
        or ground_truth[route] <= 0
        for route in route_names
    ) or sum(ground_truth[route] for route in route_names) != total:
        raise ValueError("The thesis Privacy Assessor snapshot has invalid route counts.")

    blocked_total = ground_truth["Blocked"]
    nonblocked_total = total - blocked_total
    models = report.get("models") or {}
    table_rows = []
    overall_rows = []
    route_rows = []
    for model in _PRIVACY_ASSESSOR_COLORS:
        result = models.get(model) or {}
        correct_total = result.get("correct_total")
        correct_nonblocked = result.get("correct_nonblocked")
        route_correct = result.get("route_correct") or {}
        count_limits = [
            (correct_total, total),
            (correct_nonblocked, nonblocked_total),
            *((route_correct.get(route), ground_truth[route]) for route in route_names),
        ]
        if any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= limit
            for count, limit in count_limits
        ):
            raise ValueError(f"The thesis Privacy Assessor snapshot has invalid counts for {model}.")
        if sum(route_correct[route] for route in route_names) != correct_total:
            raise ValueError(f"The route counts for {model} do not match its total correct count.")
        if correct_total - route_correct["Blocked"] != correct_nonblocked:
            raise ValueError(f"The non-blocked count for {model} is inconsistent.")

        exact = correct_total / total
        nonblocked = correct_nonblocked / nonblocked_total
        blocked_recall = route_correct["Blocked"] / blocked_total
        table_rows.append({
            "Model": model,
            "Correct": f"{correct_total} / {total}",
            "Exact": _format_percentage(exact),
            "Non-blocked": _format_percentage(nonblocked),
            "Blocked Recall": _format_percentage(blocked_recall),
        })
        for metric, value in (
            ("Exact Route Accuracy", exact),
            ("Non-blocked Accuracy", nonblocked),
            ("Blocked Recall", blocked_recall),
        ):
            overall_rows.append({
                "Metric": metric,
                "Privacy Assessor": model,
                "Accuracy (%)": value * 100.0,
            })
        for route in route_names:
            route_rows.append({
                "Privacy Route": route,
                "Privacy Assessor": model,
                "Accuracy (%)": route_correct[route] / ground_truth[route] * 100.0,
            })
    return pd.DataFrame(table_rows), pd.DataFrame(overall_rows), pd.DataFrame(route_rows)


def _render_privacy_assessor_evaluation(report: dict) -> None:
    st.markdown("## Privacy Assessor Model Comparison")
    st.write(
        "GPT-4.1, Gemini 3.5 Flash, and Claude Sonnet 5 are evaluated on the same "
        "privacy requests using the same Privacy Assessment Prompt and the same frozen "
        "Four-dimensional Soft Gating model. Only the Privacy Assessor model changes."
    )
    st.caption("Saved thesis evaluation snapshot on the 60-request Independent Benchmark.")
    st.caption("Ground Truth distribution: Cloud 5, Collaboration 35, Local Edge 10, Blocked 10.")
    try:
        table_data, overall_data, route_data = _privacy_assessor_snapshot_frames(report)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.dataframe(table_data, use_container_width=True, hide_index=True)
    st.markdown("#### Overall Privacy Routing Performance")
    _privacy_grouped_chart(overall_data, x_field="Metric", y_field="Accuracy (%)",
        series_field="Privacy Assessor", category_order=["Exact Route Accuracy",
        "Non-blocked Accuracy", "Blocked Recall"], colors=_PRIVACY_ASSESSOR_COLORS,
        show_value_labels=True)

    st.markdown("#### Route Accuracy by Privacy Route")
    _privacy_grouped_chart(route_data, x_field="Privacy Route", y_field="Accuracy (%)",
        series_field="Privacy Assessor", category_order=list(_PRIVACY_ROUTE_LABELS.values()),
        colors=_PRIVACY_ASSESSOR_COLORS, show_value_labels=True)


def _render_method_benchmark_comparison(metrics: dict, independent_data: dict,
                                         controlled_data: dict) -> None:
    st.markdown("## Privacy Routing Method Comparison")
    st.dataframe(pd.DataFrame([
        {"Method": "Method A", "Privacy Feature Generation": "Fixed rules and predefined weights",
         "Blocking Decision": "Rule-based hard blocking", "Downstream Router": "Same 4D Soft Gating"},
        {"Method": "Method B", "Privacy Feature Generation": "LLM with short Privacy Assessment Prompt",
         "Blocking Decision": "LLM-generated block decision", "Downstream Router": "Same 4D Soft Gating"},
        {"Method": "Method C", "Privacy Feature Generation": "LLM with complete Privacy Assessment Prompt",
         "Blocking Decision": "LLM-generated block decision", "Downstream Router": "Same 4D Soft Gating"},
    ]), use_container_width=True, hide_index=True)
    summary_columns = st.columns(2)
    controlled_samples = controlled_data.get("samples") or []
    with summary_columns[0]:
        st.metric(
            "Independent Benchmark",
            independent_data.get("sample_count", "Not available"),
        )
        st.write(
            "Contains varied analysis requests with different tasks, athlete groups, "
            "filters, and privacy conditions."
        )
    with summary_columns[1]:
        st.metric(
            "Controlled Benchmark",
            controlled_data.get("sample_count", len(controlled_samples)),
        )
        st.write(
            "Keeps the analysis task similar while privacy conditions change "
            "systematically from L0 (Cloud) to L3 (Blocked)."
        )
    rows, chart_rows = [], []
    for method in _PRIVACY_METHOD_COLORS:
        independent = (metrics.get("independent") or {}).get(method) or {}
        controlled = (metrics.get("controlled") or {}).get(method) or {}
        rows.append({"Method": method,
            "Independent Benchmark": _format_percentage(independent.get("exact_route_accuracy")),
            "Controlled Benchmark": _format_percentage(controlled.get("exact_route_accuracy"))})
        for benchmark, result in (("Independent", independent), ("Controlled", controlled)):
            value = result.get("exact_route_accuracy")
            if isinstance(value, (int, float)):
                chart_rows.append({"Benchmark": benchmark, "Method": method,
                    "Exact Route Accuracy (%)": float(value) * 100.0})
    st.markdown("#### Overall Exact Route Accuracy")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    _privacy_grouped_chart(pd.DataFrame(chart_rows), x_field="Benchmark",
        y_field="Exact Route Accuracy (%)", series_field="Method",
        category_order=["Independent", "Controlled"], colors=_PRIVACY_METHOD_COLORS)


def _render_controlled_benchmark_analysis(controlled_data: dict,
                                           per_level: pd.DataFrame) -> None:
    st.markdown("## Controlled Benchmark Analysis")
    st.write(
        "The Controlled Benchmark keeps the analysis task approximately constant "
        "within each family while privacy conditions change from L0 to L3."
    )
    examples = {}
    for sample in controlled_data.get("samples") or []:
        examples.setdefault(sample.get("privacy_level"), sample)
    st.dataframe(pd.DataFrame([{
        "Privacy Level": f"L{level}",
        "Ground Truth Route": _friendly_label(sample.get("ground_truth_route")),
        "Example Request": sample.get("prompt"),
    } for level, sample in sorted(examples.items())]), use_container_width=True, hide_index=True)
    st.markdown("#### Exact Route Accuracy by Privacy Level")
    if per_level.empty:
        st.warning("Controlled per-level accuracy artifact is not available.")
        return
    chart_data = per_level.rename(columns={"method": "Method",
        "level_label": "Privacy Level", "accuracy_percent": "Accuracy (%)"})
    _privacy_grouped_chart(chart_data, x_field="Privacy Level", y_field="Accuracy (%)",
        series_field="Method", category_order=["L0 Cloud", "L1 Collaboration",
        "L2 Local Edge", "L3 Blocked"], colors=_PRIVACY_METHOD_COLORS)


def _render_controlled_privacy_feature_chart(rows: list[dict], value_field: str,
                                              y_label: str) -> None:
    data = pd.DataFrame([{"Privacy Level": f"L{int(row['privacy_level'])}",
        "Method": row.get("method"), y_label: row.get(value_field)}
        for row in sorted(rows, key=lambda item: (item.get("method", ""),
                                                   int(item.get("privacy_level", 0))))
        if row.get(value_field) is not None])
    chart = alt.Chart(data).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Privacy Level:N", sort=["L0", "L1", "L2", "L3"]),
        y=alt.Y(f"{y_label}:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Method:N", scale=alt.Scale(domain=list(_PRIVACY_METHOD_COLORS),
            range=list(_PRIVACY_METHOD_COLORS.values()))),
        tooltip=["Privacy Level", "Method", alt.Tooltip(y_label, format=".3f")],
    ).properties(height=330)
    st.altair_chart(chart, use_container_width=True, theme=None)


def _render_controlled_privacy_features(report: dict) -> None:
    st.markdown("## Controlled Privacy Feature Analysis")
    st.write(
        "The four charts show the mean privacy features produced before the final "
        "routing decision. Higher values indicate stronger privacy-related signals."
    )
    rows = report.get("summary") or []
    specs = (
        ("Privacy Risk", "privacy_risk_score_mean", "Mean Privacy Risk"),
        ("Subject Scope", "subject_scope_mean", "Mean Subject Scope"),
        ("Data Sensitivity", "data_sensitivity_mean", "Mean Data Sensitivity"),
        ("Disclosure Level", "disclosure_level_mean", "Mean Disclosure Level"),
    )
    for tab, (_, field, label) in zip(st.tabs([item[0] for item in specs]), specs):
        with tab:
            _render_controlled_privacy_feature_chart(rows, field, label)


def _render_privacy_technical_details(metrics: dict, per_route: pd.DataFrame,
                                      confusion: pd.DataFrame) -> None:
    with st.expander("Technical Details", expanded=False):
        safety_rows = []
        for benchmark in ("independent", "controlled"):
            for method in _PRIVACY_METHOD_COLORS:
                result = (metrics.get(benchmark) or {}).get(method) or {}
                safety_rows.append({"Benchmark": benchmark.title(), "Method": method,
                    "Safety-aware Accuracy": _format_percentage(result.get("safety_aware_accuracy")),
                    "Under-protection Rate": _format_percentage(result.get("under_protection_rate")),
                    "Over-protection Rate": _format_percentage(result.get("over_protection_rate")),
                    "Mean Route Distance": result.get("mean_route_distance")})
        st.markdown("#### Benchmark Safety Metrics")
        st.dataframe(pd.DataFrame(safety_rows), use_container_width=True, hide_index=True)
        st.markdown("#### Per-route Precision / Recall / F1")
        if not per_route.empty:
            display = per_route.copy()
            for column in ("precision", "recall", "f1"):
                display[column] = display[column].map(_format_percentage)
            display["route"] = display["route"].map(
                lambda value: _PRIVACY_ROUTE_LABELS.get(value, value))
            st.dataframe(display, use_container_width=True, hide_index=True)
        if not confusion.empty:
            st.markdown("#### Confusion Matrices")
            predicted = ["predicted_cloud", "predicted_collaboration",
                "predicted_local_edge", "predicted_blocked"]
            for benchmark in ("independent", "controlled"):
                for method in _PRIVACY_METHOD_COLORS:
                    selected = confusion[(confusion["benchmark"] == benchmark)
                        & (confusion["method"] == method)]
                    if selected.empty:
                        continue
                    st.markdown(f"**{benchmark.title()} — {method}**")
                    matrix = selected.set_index("ground_truth_route")[predicted]
                    matrix.index = [_PRIVACY_ROUTE_LABELS.get(value, value)
                        for value in matrix.index]
                    matrix.columns = list(_PRIVACY_ROUTE_LABELS.values())
                    st.dataframe(matrix, use_container_width=True)


def _render_privacy_evaluation_tab() -> None:
    app_root = Path(__file__).resolve().parent
    st.markdown("### Privacy-aware Routing Evaluation")
    st.caption(
        "This page displays saved offline Privacy Evaluation results. It does not "
        "rerun LLM calls, retrain Soft Gating, or modify the evaluation artifacts."
    )
    assessor = _load_json_artifact(str(app_root /
        "artifacts/thesis_evaluation/privacy_cloud_model_evaluation.json"))
    metrics = _load_json_artifact(str(app_root / "artifacts/privacy_benchmark_metrics.json"))
    per_level = _load_csv_artifact(str(app_root / "artifacts/controlled_per_level_accuracy.csv"))
    features = _load_json_artifact(str(
        app_root / "artifacts/controlled_privacy_feature_summary.json"))
    per_route = _load_csv_artifact(str(app_root / "artifacts/privacy_benchmark_per_route.csv"))
    confusion = _load_csv_artifact(str(
        app_root / "artifacts/privacy_benchmark_confusion_matrices.csv"))
    independent_data = _load_json_artifact(str(
        app_root / "evaluation/frontend_realistic_benchmark_60.json"))
    controlled_data = _load_json_artifact(str(
        app_root / "evaluation/privacy_controlled_benchmark.json"))

    if assessor:
        _render_privacy_assessor_evaluation(assessor)
    else:
        st.error("Saved thesis Privacy Assessor evaluation snapshot was not found.")
    st.divider()
    if metrics:
        _render_method_benchmark_comparison(metrics, independent_data, controlled_data)
    else:
        st.warning("Privacy benchmark metrics artifact is not available.")
    st.divider()
    _render_controlled_benchmark_analysis(controlled_data, per_level)
    st.divider()
    if features:
        _render_controlled_privacy_features(features)
    else:
        st.warning("Controlled privacy feature summary artifact is not available.")
    _render_privacy_technical_details(metrics, per_route, confusion)


def _render_privacy_method_comparison_tab():
    """Display only the latest saved formal three-method comparison artifact."""
    report = _load_json_artifact(PRIVACY_METHOD_COMPARISON_PATH)
    st.markdown("### Privacy Routing Method Comparison")
    st.write(
        "This module compares three privacy-routing approaches on the same locked "
        "evaluation dataset. The evaluation is executed locally and this page only "
        "displays the saved result."
    )
    command = "python scripts/evaluate_privacy_methods_frontend60.py --resume"
    if not report:
        st.info("No saved three-method privacy comparison result was found.")
        st.code(command, language="powershell")
        return

    distribution = report.get("route_distribution") or {}
    expected_distribution = {
        "cloud": 5,
        "collaboration": 35,
        "local_edge": 10,
        "blocked": 10,
    }
    is_formal = (
        report.get("status") == "formal_comparison"
        and report.get("sample_count") == 60
        and all(
            distribution.get(route) == count
            for route, count in expected_distribution.items()
        )
    )
    if not is_formal:
        st.warning(
            "The saved comparison is incomplete and cannot be presented as the "
            "formal evaluation result."
        )
        st.write(f"Status: {report.get('status', 'Not available')}")
        st.write(f"Sample count: {report.get('sample_count', 'Not available')}")
        st.code(command, language="powershell")
        return

    st.success("Formal three-method comparison result loaded successfully.")
    common = report.get("common_completed_subset") or {}
    information = st.columns(2)
    information[0].metric("Evaluation Samples", report["sample_count"])
    information[1].metric("Evaluation Dataset", report.get("dataset_name", "Not available"))
    if report.get("dataset_sha256"):
        st.caption(f"Dataset SHA256: {str(report['dataset_sha256'])[:12]}")
    st.caption(
        "The benchmark contains 50 frontend-realistic requests plus 10 additional "
        "privacy stress requests."
    )

    st.caption(
        "All three methods were evaluated using the same questions and ground-truth "
        "routes. Method B thresholds were fixed before this evaluation."
    )

    annotation = report.get("dataset_annotation") or {}
    if annotation:
        with st.expander("Dataset Annotation Information", expanded=False):
            st.write(f"Annotation type: {annotation.get('annotation_type', 'Not available')}")
            st.write(f"Generator model: {annotation.get('generator_model', 'Not available')}")
            st.write(f"Verifier model: {annotation.get('verifier_model', 'Not available')}")

    use_common = bool((common.get("methods") or {}) or any(
        common.get(f"{key}_metrics") for key in PRIVACY_METHOD_DISPLAY_NAMES
    ))
    if use_common:
        st.caption(
            "Metrics below use the common completed subset, containing only samples "
            "successfully processed by all three methods."
        )
    else:
        st.caption("Metrics below use each method's available completed samples.")

    comparison_rows = []
    resolved = {}
    short_names = ("Method A", "Method B", "Method C")
    for short_name, (method_key, display_name) in zip(
        short_names, PRIVACY_METHOD_DISPLAY_NAMES.items()
    ):
        payload, metrics = _get_comparison_method_metrics(report, method_key)
        resolved[method_key] = (payload, metrics)
        completed = payload.get("completed", payload.get("completed_count", metrics.get("completed", metrics.get("completed_count"))))
        failed = payload.get("failed", metrics.get("failed", payload.get("failure_count", metrics.get("failure_count"))))
        comparison_rows.append({
            "Method": short_name,
            "Exact Route Accuracy": _format_percentage(metrics.get("exact_route_accuracy", metrics.get("exact_route_match_rate"))),
            "Safety-aware Accuracy": _format_percentage(metrics.get("safety_aware_accuracy")),
            "Macro F1": _format_percentage(metrics.get("macro_f1")),
            "Overprotection Rate": _format_percentage(metrics.get("overprotection_rate")),
            "Completed": completed if completed is not None else "Not available",
            "Failed": failed if failed is not None else "Not available",
        })

    summary_columns = st.columns(3)
    for column, short_name, (method_key, display_name) in zip(
        summary_columns, short_names, PRIVACY_METHOD_DISPLAY_NAMES.items()
    ):
        _, metrics = resolved[method_key]
        with column:
            st.markdown(f"**{short_name}**")
            st.caption(display_name)
            st.metric("Exact Accuracy", _format_percentage(metrics.get("exact_route_accuracy", metrics.get("exact_route_match_rate"))))
            st.metric("Safety-aware Accuracy", _format_percentage(metrics.get("safety_aware_accuracy")))

    st.markdown("#### Main Comparison")
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Higher Exact and Safety-aware Accuracy are better. Lower Overprotection is better."
    )

    _render_method_c_route_examples(report)

    with st.expander("Per-route Accuracy Details", expanded=False):
        route_rows = []
        for method_key, display_name in PRIVACY_METHOD_DISPLAY_NAMES.items():
            _, metrics = resolved[method_key]
            route_accuracy = metrics.get("route_accuracy") or {}
            route_rows.append({"Method": display_name, **{
                _friendly_label(route): _format_percentage(
                    route_accuracy.get(route, metrics.get(f"{route}_accuracy"))
                ) for route in PRIVACY_ROUTE_ORDER
            }})
        st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

    with st.expander("Confusion Matrices", expanded=False):
        for method_key, display_name in PRIVACY_METHOD_DISPLAY_NAMES.items():
            st.markdown(f"**{display_name}**")
            matrix = resolved[method_key][1].get("confusion_matrix")
            if isinstance(matrix, dict):
                values = [[(matrix.get(actual) or {}).get(predicted, 0) for predicted in PRIVACY_ROUTE_ORDER] for actual in PRIVACY_ROUTE_ORDER]
            elif isinstance(matrix, list) and len(matrix) == len(PRIVACY_ROUTE_ORDER):
                values = matrix
            else:
                st.caption("Confusion matrix not available.")
                continue
            frame = pd.DataFrame(values, index=PRIVACY_ROUTE_ORDER, columns=PRIVACY_ROUTE_ORDER)
            st.dataframe(frame, use_container_width=True)


_PRIVACY_PROMPT_DISPLAY_NAMES = {
    "minimal": "Simple Prompt",
    "defined": "Medium Prompt",
    "full": "Privacy Assessment Prompt",
}
_PRIVACY_PROMPT_BENCHMARK_COLORS = {
    "Controlled": "#7c3aed",
    "Independent": "#f59e0b",
    "Combined": "#0f9f9a",
}
_CODEGEN_PROMPT_DISPLAY_NAMES = {
    "basic_interface": "Simple Prompt",
    "defined": "Medium Prompt",
    "full": "Restricted Code Generation Prompt",
}
_CODEGEN_PROMPT_COLORS = {
    "Simple Prompt": "#3b82f6",
    "Medium Prompt": "#f59e0b",
    "Restricted Code Generation Prompt": "#16a34a",
}
_CODEGEN_PROMPT_STAGES = (
    ("Structure", "structure_validation"),
    ("Request", "request_match"),
    ("Execute", "execution"),
    ("Result", "result_valid"),
    ("Fully Correct", "fully_correct"),
)


def _saved_numeric_value(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _privacy_prompt_chart_data(report: dict) -> tuple[pd.DataFrame, bool]:
    """Build thesis-facing rates solely from the saved Privacy Prompt artifact."""
    benchmarks = report.get("benchmarks") or {}
    rows = []
    incomplete = False
    for prompt_key, prompt_name in _PRIVACY_PROMPT_DISPLAY_NAMES.items():
        results = {
            benchmark: ((benchmarks.get(benchmark.lower()) or {}).get(prompt_key) or {})
            for benchmark in ("Controlled", "Independent")
        }
        correct_counts = {}
        expected_counts = {}
        for benchmark, result in results.items():
            if result.get("complete") is not True:
                incomplete = True
            accuracy = _saved_numeric_value(result.get("exact_route_accuracy"))
            completed = _saved_numeric_value(result.get("completed_samples"))
            expected = _saved_numeric_value(result.get("expected_samples"))
            if accuracy is not None:
                rows.append({
                    "Prompt": prompt_name,
                    "Benchmark": benchmark,
                    "Exact Route Accuracy (%)": accuracy * 100.0,
                })
            if accuracy is not None and completed is not None:
                correct_counts[benchmark] = round(accuracy * completed)
            if expected is not None:
                expected_counts[benchmark] = expected
        if set(correct_counts) == {"Controlled", "Independent"} and set(expected_counts) == {
            "Controlled", "Independent",
        }:
            denominator = expected_counts["Controlled"] + expected_counts["Independent"]
            if denominator > 0:
                rows.append({
                    "Prompt": prompt_name,
                    "Benchmark": "Combined",
                    "Exact Route Accuracy (%)": (
                        correct_counts["Controlled"] + correct_counts["Independent"]
                    ) / denominator * 100.0,
                })
    return pd.DataFrame(rows, columns=(
        "Prompt", "Benchmark", "Exact Route Accuracy (%)",
    )), incomplete


def _codegen_prompt_chart_data(report: dict) -> pd.DataFrame:
    """Read the five saved validation rates without exposing detailed results."""
    prompts = report.get("prompts") or {}
    rows = []
    for prompt_key, prompt_name in _CODEGEN_PROMPT_DISPLAY_NAMES.items():
        result = prompts.get(prompt_key) or {}
        for stage, field in _CODEGEN_PROMPT_STAGES:
            value = _saved_numeric_value(result.get(field))
            if value is not None:
                rows.append({
                    "Stage": stage,
                    "Prompt": prompt_name,
                    "Accuracy (%)": value * 100.0,
                })
    return pd.DataFrame(rows, columns=("Stage", "Prompt", "Accuracy (%)"))


def _render_prompt_design_line_chart(
    data: pd.DataFrame,
    *,
    x_field: str,
    series_field: str,
    value_field: str,
    x_order: list[str],
    series_order: list[str],
    colors: dict[str, str],
) -> None:
    if data.empty:
        st.warning("The saved evaluation artifact does not contain chart-ready results.")
        return
    chart_data = data.copy()
    chart_data["Percentage Label"] = chart_data[value_field].map(
        lambda value: f"{float(value):.1f}%"
    )
    color_scale = alt.Scale(
        domain=series_order,
        range=[colors[name] for name in series_order],
    )
    base = alt.Chart(chart_data).encode(
        x=alt.X(
            f"{x_field}:N",
            sort=x_order,
            title=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=260),
        ),
        y=alt.Y(
            f"{value_field}:Q",
            title=value_field,
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            f"{series_field}:N",
            sort=series_order,
            scale=color_scale,
            legend=alt.Legend(title=None, orient="top"),
        ),
        detail=alt.Detail(f"{series_field}:N"),
        tooltip=[
            alt.Tooltip(f"{x_field}:N"),
            alt.Tooltip(f"{series_field}:N"),
            alt.Tooltip(f"{value_field}:Q", format=".1f"),
        ],
    )
    chart = (
        base.mark_line(point=True, strokeWidth=3)
        + base.mark_text(dy=-12, fontSize=11).encode(
            text=alt.Text("Percentage Label:N")
        )
    ).properties(height=340)
    st.altair_chart(chart, use_container_width=True, theme=None)


def _render_prompt_evaluation_tab() -> None:
    app_root = Path(__file__).resolve().parent
    privacy_path = (
        app_root / "evaluation/results/prompt_ablation/privacy_prompt_ablation_summary.json"
    )
    cloud_path = (
        app_root / "evaluation/results/prompt_design_v2/codegen_prompt_design_v2_summary.json"
    )
    local_path = (
        app_root
        / "evaluation/results/local_prompt_design_v2/local_codegen_prompt_design_v2_summary.json"
    )

    st.markdown("### Prompt Design Evaluation")
    st.caption(
        "This page displays saved offline Prompt Evaluation results. It does not rerun "
        "LLM calls, change prompts, retrain routing models, or modify evaluation artifacts."
    )

    st.markdown("## Privacy Assessment Prompt Design Evaluation")
    st.write(
        "Three Privacy Assessment Prompt variants are compared while keeping the "
        "Privacy Assessor model and the frozen Four-dimensional Soft Gating model unchanged."
    )
    privacy_report = _load_json_artifact(str(privacy_path))
    if not privacy_report:
        st.warning("The saved Privacy Assessment Prompt evaluation artifact is unavailable.")
        st.code(
            "python scripts/evaluate_privacy_prompt_ablation.py --resume",
            language="powershell",
        )
    else:
        privacy_data, privacy_incomplete = _privacy_prompt_chart_data(privacy_report)
        if privacy_incomplete:
            st.warning(
                "One saved Privacy Assessment Prompt evaluation run is incomplete; "
                "the page will not rerun it automatically."
            )
        _render_prompt_design_line_chart(
            privacy_data,
            x_field="Prompt",
            series_field="Benchmark",
            value_field="Exact Route Accuracy (%)",
            x_order=list(_PRIVACY_PROMPT_DISPLAY_NAMES.values()),
            series_order=["Controlled", "Independent", "Combined"],
            colors=_PRIVACY_PROMPT_BENCHMARK_COLORS,
        )

    st.divider()
    st.markdown("## Restricted Code Generation Prompt Design Evaluation")
    st.write(
        "The same 40 Restricted Code Generation benchmark requests are evaluated with "
        "three Prompt variants. Saved results are shown for the Cloud Model and Local Model."
    )

    st.markdown("#### Cloud Model (Gemini 3.5 Flash)")
    cloud_report = _load_json_artifact(str(cloud_path))
    if not cloud_report:
        st.warning("The saved Cloud Model Prompt evaluation artifact is unavailable.")
        st.code(
            "python scripts/evaluate_codegen_prompt_design_v2.py --resume",
            language="powershell",
        )
    else:
        _render_prompt_design_line_chart(
            _codegen_prompt_chart_data(cloud_report),
            x_field="Stage",
            series_field="Prompt",
            value_field="Accuracy (%)",
            x_order=[stage for stage, _ in _CODEGEN_PROMPT_STAGES],
            series_order=list(_CODEGEN_PROMPT_DISPLAY_NAMES.values()),
            colors=_CODEGEN_PROMPT_COLORS,
        )

    st.markdown("#### Local Model (Ministral-3-8B)")
    local_report = _load_json_artifact(str(local_path))
    if not local_report:
        st.warning("The saved Local Model Prompt evaluation artifact is unavailable.")
        st.code(
            "python scripts/evaluate_local_codegen_prompt_design_v2.py --resume",
            language="powershell",
        )
    else:
        _render_prompt_design_line_chart(
            _codegen_prompt_chart_data(local_report),
            x_field="Stage",
            series_field="Prompt",
            value_field="Accuracy (%)",
            x_order=[stage for stage, _ in _CODEGEN_PROMPT_STAGES],
            series_order=list(_CODEGEN_PROMPT_DISPLAY_NAMES.values()),
            colors=_CODEGEN_PROMPT_COLORS,
        )


def page_routing_evaluation():
    st.subheader("Evaluation")
    route_tab, privacy_tab, prompt_tab = st.tabs(
        ["Cloud/Local Evaluation", "Privacy Evaluation", "Prompt Evaluation"]
    )
    with route_tab:
        _render_cloud_local_evaluation_tab()
    with privacy_tab:
        _render_privacy_evaluation_tab()
    with prompt_tab:
        _render_prompt_evaluation_tab()


def _clear_individual_analysis_on_group_change() -> None:
    for key in (
        "individual_analysis_response",
        "last_anonymous_athlete_id", "last_anonymous_athlete_group",
    ):
        st.session_state.pop(key, None)


def page_individual():
    st.subheader("Anonymous Athlete Profile")
    st.write(
        "This page returns only a standardized descriptive profile. "
        "It does not predict whether an athlete is elite and does not show raw measurements."
    )

    selected_group = st.selectbox(
        "Athlete group",
        list(ATHLETE_GROUP_OPTIONS),
        format_func=lambda group: ATHLETE_GROUP_LABELS.get(group, group),
        key="individual_athlete_group",
        on_change=_clear_individual_analysis_on_group_change,
    )

    if "individual_analysis_response" not in st.session_state:
        st.session_state.individual_analysis_response = None

    if st.button("Analyze anonymous athlete", use_container_width=True):
        st.session_state.pop("individual_analysis_response", None)
        if select_anonymous_subject is None:
            st.error("Anonymous athlete selection is unavailable.")
            return
        previous_id = (
            st.session_state.get("last_anonymous_athlete_id")
            if st.session_state.get("last_anonymous_athlete_group") == selected_group
            else None
        )
        try:
            subject_reference = select_anonymous_subject(selected_group, previous_id=previous_id)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["last_anonymous_athlete_id"] = subject_reference
        st.session_state["last_anonymous_athlete_group"] = selected_group
        prompt = (
            "Generate a protected standardized individual athlete profile for CURRENT_SUBJECT. "
            "Include the eight standardized domains, "
            "strongest and weakest domains, the profile figure, and whether "
            "the profile matches the paper's three-domain group-level pattern. "
            "Do not expose the athlete identifier or raw measurements in the final result. "
            "Do not produce individual status or future performance forecasts."
        )
        response = run_request(
            prompt,
            requested_analysis="individual_profile",
            private_local_context={"CURRENT_SUBJECT": subject_reference},
        )
        if response.get("allowed") and response.get("result") is not None:
            response["answer"] = "Anonymous standardized profile generated locally."
        st.session_state.individual_analysis_response = response

    if st.session_state.individual_analysis_response:
        show_pipeline_response(st.session_state.individual_analysis_response)


def main():
    render_header()
    page = render_sidebar()

    if page == "Protected analysis dashboard":
        page_dashboard()
    elif page == "Data Generation & Processing":
        page_data_generation_processing()
    elif page == "Routing evaluation":
        page_routing_evaluation()
    elif page == "Individual athlete analysis":
        page_individual()


if __name__ == "__main__":
    main();  # Semicolon intentionally suppresses Streamlit magic rendering.
