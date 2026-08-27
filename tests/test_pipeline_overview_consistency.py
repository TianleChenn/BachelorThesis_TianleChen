from frontend import _get_prism_route


def test_pipeline_uses_prism_result_as_primary_source():
    response = {
        "prism_privacy_result": {"route": "collaboration"},
        "privacy_decision": {"route": "cloud"},
    }
    assert _get_prism_route(response) == "collaboration"


def test_pipeline_falls_back_to_privacy_decision():
    response = {"privacy_decision": {"route": "local_edge"}}
    assert _get_prism_route(response) == "local_edge"
