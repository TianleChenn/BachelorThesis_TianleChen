from privacy.routellm_router import route_model


class _Router:
    def __init__(self, tier="strong"): self.tier = tier
    def predict(self, *args, **kwargs):
        return {"p_strong": .7 if self.tier == "strong" else .2, "threshold": .6,
                "selected_tier": self.tier, "model_version": "test"}


def test_runtime_uses_classifier_metadata_threshold():
    decision = route_model("safe", "cloud", threshold=.11593, athlete_router=_Router("strong"))
    assert decision.threshold == .6 and decision.selected_tier == "strong"


def test_legacy_explicit_threshold_does_not_control_and_local_routes_skip_classifier():
    assert route_model("safe", "cloud", threshold=.99, athlete_router=_Router("weak")).selected_tier == "weak"
    assert route_model("private", "local_edge", athlete_router=_Router()).threshold is None
    assert route_model("blocked", "blocked", athlete_router=_Router()).threshold is None

