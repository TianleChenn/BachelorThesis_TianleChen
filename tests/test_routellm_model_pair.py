from llm.model_config import LOCAL_EDGE_GENERATOR_MODEL, ROUTELLM_STRONG_MODEL, ROUTELLM_WEAK_MODEL


def test_route_model_pair_uses_current_weak_default_and_keeps_local_edge_distinct():
    assert ROUTELLM_STRONG_MODEL == "gpt-4.1"
    assert ROUTELLM_WEAK_MODEL == "Ministral-3-8B"
    assert LOCAL_EDGE_GENERATOR_MODEL == "Ministral-3-8B-Local"
    assert ROUTELLM_WEAK_MODEL != LOCAL_EDGE_GENERATOR_MODEL
