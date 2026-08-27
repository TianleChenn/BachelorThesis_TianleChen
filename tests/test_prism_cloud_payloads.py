from privacy.prism_router import detect_entities, prism_route, two_layer_ldp


def test_low_risk_prompt_is_sent_directly_to_cloud():
    prompt = "What analyses are available?"
    decision = prism_route(prompt)
    assert decision.route == "cloud"
    assert decision.cloud_prompt == prompt
    assert decision.cloud_payload_type == "original_prompt"
    assert decision.privacy_method == "llm_privacy_assessment"
    assert decision.privacy_applied is False
    assert decision.semantic_sketch is None


def test_collaboration_sends_ldp_prompt_not_schema_sketch():
    decision = prism_route("Explain blood micronutrients in the aggregate regression")
    assert decision.route == "collaboration"
    assert decision.privacy_applied is True
    assert decision.cloud_payload_type == "two_layer_ldp_perturbed_prompt"
    assert decision.cloud_prompt == decision.perturbed_prompt
    assert decision.cloud_prompt != decision.semantic_sketch
    assert decision.ldp_audit
    assert all(not row["original_value_visible"] for row in decision.ldp_audit)


def test_two_layer_ldp_with_test_seed():
    prompt = "My blood and genetic information should be analyzed"
    entities = detect_entities(prompt)
    first_prompt, first_audit = two_layer_ldp(prompt, entities, seed=7)
    second_prompt, second_audit = two_layer_ldp(prompt, entities, seed=7)
    assert first_prompt == second_prompt
    assert first_audit == second_audit
    assert any(row["replacement_applied"] for row in first_audit)
    assert all("epsilon_1_category" in row for row in first_audit)
    assert all("epsilon_2_value" in row for row in first_audit)


def test_high_risk_prompt_has_no_cloud_payload():
    decision = prism_route("Analyze Athlete_003 using a protected profile")
    assert decision.route == "local_edge"
    assert decision.cloud_prompt is None
