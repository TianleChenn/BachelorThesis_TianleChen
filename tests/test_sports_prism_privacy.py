def test_complete_athlete_row_is_blocked():
    from privacy.prism_router import prism_route

    decision = prism_route("Print the complete athlete row.")
    assert decision.route == "blocked"
    assert decision.blocked is True


def test_athlete_raw_blood_value_is_blocked():
    from privacy.prism_router import prism_route

    decision = prism_route("What is Athlete_003 vitamin B12 value?")
    assert decision.route == "blocked"
    assert decision.blocked is True


def test_athlete_protected_profile_is_local_edge():
    from privacy.prism_router import prism_route

    decision = prism_route("Generate protected profile for Athlete_003.")
    assert decision.route == "local_edge"
    assert decision.blocked is False


def test_aggregate_figure1_is_direct_low_risk_cloud_prompt():
    from privacy.prism_router import prism_route

    decision = prism_route("Generate Figure 1-style group statistics.")
    assert decision.route == "cloud"
    assert decision.cloud_prompt is not None
    assert decision.cloud_payload_type == "original_prompt"
    assert "Athlete_003" not in decision.cloud_prompt


def test_sensitive_aggregate_goes_to_collaboration():
    from privacy.prism_router import prism_route

    decision = prism_route("Explain blood micronutrients in the aggregate regression.")
    assert decision.route == "collaboration"
    assert decision.cloud_prompt is not None
    assert decision.cloud_payload_type == "two_layer_ldp_perturbed_prompt"


def test_ldp_audit_does_not_expose_original_values():
    from privacy.prism_router import prism_route

    decision = prism_route("Explain blood micronutrients in the aggregate regression.")
    for row in decision.ldp_audit or []:
        assert row.get("original_value_visible") is False
        assert row.get("original_value_preview") == "[REDACTED]"
