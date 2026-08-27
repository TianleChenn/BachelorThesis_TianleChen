from pathlib import Path


FRONTEND = Path("frontend.py").read_text(encoding="utf-8")
SERVICE = Path("sports/service.py").read_text(encoding="utf-8")
RESTRICTED_API = Path("sports/restricted_analysis_api.py").read_text(encoding="utf-8")


def test_retired_cohort_protection_removed_from_runtime():
    combined = FRONTEND + SERVICE + RESTRICTED_API
    retired_tokens = (
        "minimum_" + "group_size_blocked",
        "output_" + "protection_status",
        "MIN_" + "GROUP_SIZE",
        "Small" + "GroupError",
        "privacy." + "output_protection",
        "minimum release " + "size",
    )

    assert all(token not in combined for token in retired_tokens)


def test_frontend_does_not_render_output_protection_section():
    assert "def render_output_protection" not in FRONTEND
    assert "def _has_current_output_protection" not in FRONTEND
    assert 'st.subheader("Output Protection")' not in FRONTEND
    assert "Output protection completed successfully" not in FRONTEND
    assert "Minimum Group Size Protection" not in FRONTEND


def test_filter_change_still_clears_dashboard_responses():
    callback_start = FRONTEND.index("def _clear_dashboard_results")
    callback_end = FRONTEND.index("\ndef _dashboard_cohort_selector", callback_start)
    callback = FRONTEND[callback_start:callback_end]
    assert '"dashboard_selected_response_key"' in callback
    assert "on_change=_clear_dashboard_results" in FRONTEND


def test_missing_noise_utility_preserves_and_renders_the_main_response():
    start = FRONTEND.index("def page_dashboard")
    end = FRONTEND.index("\ndef page_data_generation_processing", start)
    dashboard = FRONTEND[start:end]
    selected = dashboard.index("selected_response_key =")
    renderer = dashboard.index("show_pipeline_response(selected_response)", selected)
    warning = dashboard.index("Controlled perturbation results are unavailable", renderer)
    assert renderer < warning
    assert 'selected_response.get("allowed")' in dashboard[renderer:warning]
    assert 'isinstance(selected_response.get("result"),dict)' in dashboard[renderer:warning]
    assert "st.session_state[selected_response_key] = None" not in dashboard
    assert "st.session_state.dashboard_selected_response_key = None" not in dashboard[selected:]
    assert "The saved analysis used an older noise configuration" not in FRONTEND


def test_failed_pipeline_response_is_not_reclassified_as_a_noise_warning():
    start = FRONTEND.index("def page_dashboard")
    end = FRONTEND.index("\ndef page_data_generation_processing", start)
    dashboard = FRONTEND[start:end]
    condition_start = dashboard.index("if (selected_analysis in NOISE_ENABLED_ANALYSES")
    warning_condition = dashboard[condition_start:dashboard.index("st.warning", condition_start)]
    assert 'selected_response.get("allowed")' in warning_condition
    assert "show_pipeline_response(selected_response)" in dashboard
