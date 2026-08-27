import hashlib
import json
import joblib

from llm.athlete_router_features import build_classifier_pipeline
from llm.athlete_strong_weak_router import AthleteStrongWeakRouter
from privacy.routellm_router import route_model


class FakeRouter:
    def __init__(self, tier="strong"):
        self.tier, self.calls = tier, []

    def predict(self, prompt, requested_analysis, difficulty, privacy_route, filters, requires_code, source):
        self.calls.append((prompt, privacy_route, source))
        return {
            "p_strong": .7 if self.tier == "strong" else .2, "threshold": .6,
            "selected_tier": self.tier,
            "selected_model": "strong_gpt4_1106_preview" if self.tier == "strong" else "weak_mixtral_8x7b",
            "execution_model": "model", "router_name": "athlete_specific_logistic_router",
            "router_prompt_source": source, "model_version": "abc",
        }


def test_runtime_loads_joblib_and_verifies_hash(tmp_path, monkeypatch):
    prompts = ["simple aggregate athlete summary", "complex regression athlete analysis"] * 3
    model = build_classifier_pipeline().fit(prompts, [0, 1] * 3)
    model_path = tmp_path / "router.joblib"
    joblib.dump(model, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "router.json"
    metadata_path.write_text(json.dumps({"status": "trained", "router_type": "project_specific_preference_router",
        "uses_official_mf_score": False, "threshold": .4, "model_sha256": digest}), encoding="utf-8")
    result = AthleteStrongWeakRouter(model_path, metadata_path).predict(
        "question", "table1", "easy", "cloud", {}, True, "original_prompt"
    )
    assert result["router_name"] == "new_athlete_router"
    assert result["strong_model_probability"] == result["p_strong"]
    assert "official_mf_score" not in result


def test_evaluation_prediction_uses_saved_text_router_without_enabling_local_execution(
    tmp_path,
):
    prompts = ["simple aggregate athlete summary", "complex regression athlete analysis"] * 3
    model = build_classifier_pipeline().fit(prompts, [0, 1] * 3)
    model_path = tmp_path / "router.joblib"
    joblib.dump(model, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "router.json"
    metadata_path.write_text(json.dumps({
        "status": "trained", "router_type": "project_specific_preference_router",
        "uses_official_mf_score": False, "threshold": .4, "model_sha256": digest,
    }), encoding="utf-8")
    router = AthleteStrongWeakRouter(model_path, metadata_path)

    evaluation = router.predict_for_evaluation("protected individual athlete profile")

    assert evaluation["evaluation_only"] is True
    assert evaluation["threshold"] == .4
    assert evaluation["hypothetical_model"] in {"GPT-4.1", "Ministral-3-8B"}
    assert evaluation["router_artifact_path"] == str(model_path.resolve())
    try:
        router.predict("same prompt", privacy_route="local_edge")
    except ValueError:
        pass
    else:
        raise AssertionError("Local Edge must remain invalid for actual Strong/Weak routing")


def test_cloud_and_collaboration_use_classifier_and_local_blocked_skip_it():
    fake = FakeRouter("strong")
    cloud = route_model("original", "cloud", athlete_router=fake)
    collaboration = route_model("approved", "collaboration", privacy_prompt="approved",
                                router_prompt_source="prism_cloud_payload", athlete_router=fake)
    assert cloud.router_name == collaboration.router_name == "new_athlete_router"
    assert fake.calls[1] == ("approved", "collaboration", "prism_cloud_payload")
    calls = len(fake.calls)
    assert route_model("private", "local_edge", athlete_router=fake).selected_tier == "none"
    assert route_model("blocked", "blocked", athlete_router=fake).selected_tier == "none"
    assert len(fake.calls) == calls
    assert cloud.threshold == .6 and cloud.threshold != .11593
