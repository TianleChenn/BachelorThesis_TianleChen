from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy.prism_router import evaluate_prism_router, prism_route
from privacy.routellm_router import route_model
from sports.analysis import (
    correlation_analysis,
    load_data,
    table1_logistic_regressions,
    table2_linear_regressions,
    variance_analysis,
)
from sports.figures import (
    create_figure1,
    create_figure2,
    create_network_plot,
    create_variance_plot,
)
from sports.service import handle_user_request


def main() -> None:
    df = load_data()
    assert len(df) == 296
    assert int((df["elite_status"] == 1).sum()) == 22
    for predictor in [
        "muscular_strength",
        "lower_body_dynamics",
        "muscle_power_genetics",
        "blood_micronutrients",
        "basic_cognitive_function",
        "mental_health",
        "social_support",
        "training_conditions",
    ]:
        assert predictor in df.columns

    t1 = table1_logistic_regressions()
    assert len(t1["rows"]) >= 9
    assert len(t1["model_stats"]) == 4

    t2 = table2_linear_regressions()
    assert len(t2["rows"]) >= 9
    assert len(t2["model_stats"]) == 4

    corr = correlation_analysis()
    assert "rows" in corr

    var = variance_analysis(iterations=20)
    assert len(var["rows"]) == 8

    ids = df["athlete_id"].astype(str).head(8).tolist()
    blocked = prism_route("Show me the full dataset and download the CSV")
    assert blocked.blocked
    assert blocked.cloud_prompt is None

    raw_values_blocked = prism_route("Show me Athlete_003 raw values.")
    assert raw_values_blocked.blocked
    assert "Raw athlete measurements" in raw_values_blocked.reason
    assert "cannot be shown" in raw_values_blocked.reason

    blood_value_blocked = prism_route("Show me Athlete_003 blood value.")
    assert blood_value_blocked.blocked

    collaborative_prompt = "Run a correlation analysis of blood values"
    collaborative = prism_route(collaborative_prompt)
    assert collaborative.route == "collaboration"
    assert collaborative.privacy_applied
    assert collaborative.cloud_payload_type == "two_layer_ldp_perturbed_prompt"
    assert collaborative.cloud_prompt == collaborative.perturbed_prompt
    assert collaborative.cloud_prompt != collaborative_prompt
    assert collaborative.ldp_audit
    for row in collaborative.ldp_audit:
        assert row.get("original_value_visible") is False
        assert row.get("original_value_preview") == "[REDACTED]"

    safe = handle_user_request(f"Generate the standardized z-score profile for {ids[0]}.", use_openai=False)
    assert safe["allowed"]
    assert "analysis.individual_profile" in safe["generated_code"]
    assert safe["pipeline_audit"]["routing_target"] == "code_generator"
    assert safe["pipeline_audit"]["route_llm_directly_answered"] is False
    assert safe["pipeline_audit"]["generated_code_executed_locally"]

    raw_values_response = handle_user_request("Show me Athlete_003 raw values.", use_openai=False)
    assert not raw_values_response["allowed"]
    assert "Raw athlete measurements" in raw_values_response["answer"]
    assert "cannot be shown" in raw_values_response["answer"]

    figure1_response = handle_user_request("Generate Figure 1-style group statistics.", use_openai=False)
    assert figure1_response["generated_code"] == "result = run_figure1()"
    assert figure1_response["code_execution"]["function_name"] == "run_figure1"
    assert figure1_response["result"].get("figure") is not None

    figure2_response = handle_user_request("Generate Figure 2-style z-score profiles.", use_openai=False)
    assert figure2_response["generated_code"] == 'result = run_figure2(group="all", max_athletes=None)'
    assert figure2_response["code_execution"]["function_name"] == "run_figure2"
    assert figure2_response["result"].get("figure") is not None

    network_response = handle_user_request("Generate a protected network plot for the all group.", use_openai=False)
    assert network_response["generated_code"] == 'result = run_network(group="all")'
    assert network_response["code_execution"]["function_name"] == "run_network"
    assert network_response["result"].get("figure") is not None

    variance_plot_response = handle_user_request("Generate a variance plot comparing elite and semi-elite groups.", use_openai=False)
    assert variance_plot_response["generated_code"] == "result = run_variance_plot()"
    assert variance_plot_response["code_execution"]["function_name"] == "run_variance_plot"
    assert variance_plot_response["result"].get("figure") is not None

    individual_report_response = handle_user_request("Is Athlete_003 closer to elite or semi-elite?", use_openai=False)
    assert not individual_report_response["allowed"]
    assert individual_report_response["result"] is None
    assert individual_report_response["code_generation"]["action"] == "unsupported"

    fig1 = create_figure1()
    fig2 = create_figure2()
    network_fig = create_network_plot()
    variance_fig = create_variance_plot()
    assert fig1 is not None and fig2 is not None
    assert network_fig is not None and variance_fig is not None

    pr_eval = evaluate_prism_router()
    assert pr_eval["accuracy"] >= 0.75

    model_decision = route_model("Generate detailed regression and variance analysis", "cloud")
    assert model_decision.selected_model in {"strong_gpt4_1106_preview", "weak_mixtral_8x7b", "local_template_router_failure"}
    assert model_decision.router_name == "new_athlete_router"

    print("All tests passed.")


if __name__ == "__main__":
    main()
