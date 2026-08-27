"""Validated runtime loader for the athlete-specific Strong/Weak classifier."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib

from llm.athlete_router_features import normalize_prompt
from llm.env import load_local_env
from llm.model_config import ROUTELLM_STRONG_MODEL, ROUTELLM_WEAK_MODEL


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "athlete_strong_weak_router.joblib"
DEFAULT_METADATA = ROOT / "artifacts" / "athlete_strong_weak_router.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AthleteStrongWeakRouter:
    def __init__(self, model_path: Path = DEFAULT_MODEL, metadata_path: Path = DEFAULT_METADATA):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self._pipeline = None
        self._metadata = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline, self._metadata
        if not self.model_path.exists() or not self.metadata_path.exists():
            missing = self.model_path if not self.model_path.exists() else self.metadata_path
            raise RuntimeError(f"Router unavailable: {missing.name} not found")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "trained":
            raise RuntimeError("Router unavailable: metadata is not marked trained")
        if metadata.get("router_type") != "project_specific_preference_router":
            raise RuntimeError("Router unavailable: incompatible legacy router artifact")
        if metadata.get("uses_official_mf_score") is not False:
            raise RuntimeError("Router unavailable: artifact still depends on Official MF features")
        actual_hash = _sha256(self.model_path)
        if metadata.get("model_sha256") != actual_hash:
            raise RuntimeError("Router unavailable: model hash does not match metadata")
        try:
            pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Router unavailable: {type(exc).__name__}: {exc}") from exc
        if not hasattr(pipeline, "predict_proba"):
            raise RuntimeError("Router unavailable: saved classifier does not support predict_proba")
        self._pipeline, self._metadata = pipeline, metadata
        return pipeline, metadata

    def predict(
        self,
        prompt: str,
        requested_analysis: str | None = None,
        difficulty: str | None = None,
        privacy_route: str = "cloud",
        filters: dict | None = None,
        requires_code: bool = True,
        router_prompt_source: str = "original_prompt",
    ) -> dict:
        """Predict preference from request text before either execution model runs.

        ``strong_model_probability`` is the classifier probability for the
        Strong-preference class. It is neither router accuracy nor GPT-4.1
        answer quality.
        """
        if privacy_route not in {"cloud", "collaboration"}:
            raise ValueError("Athlete-specific Strong/Weak routing only accepts Cloud or Collaboration")
        result = self._predict_from_text(prompt, router_prompt_source)
        load_local_env()
        result["execution_model"] = (
            os.getenv("LLM_STRONG_MODEL", ROUTELLM_STRONG_MODEL)
            if result["selected_tier"] == "strong"
            else os.getenv("LLM_WEAK_MODEL", ROUTELLM_WEAK_MODEL)
        )
        return result

    def _predict_from_text(self, prompt: str, router_prompt_source: str) -> dict:
        """Run the saved text classifier without deciding whether execution is allowed."""
        pipeline, metadata = self._load()
        try:
            probability = float(pipeline.predict_proba([normalize_prompt(prompt)])[0, 1])
        except Exception as exc:
            raise RuntimeError(f"Router unavailable: text preprocessing mismatch: {exc}") from exc
        threshold = float(metadata["threshold"])
        tier = "strong" if probability >= threshold else "weak"
        return {
            "p_strong": probability,
            "strong_model_probability": probability,
            "threshold": threshold,
            "decision": tier,
            "selected_tier": tier,
            "selected_model": (
                "strong_gpt4_1106_preview" if tier == "strong" else "weak_mixtral_8x7b"
            ),
            "router_name": "new_athlete_router",
            "router_prompt_source": router_prompt_source,
            "model_version": metadata.get("model_sha256", "")[:12],
        }

    def predict_for_evaluation(self, prompt: str) -> dict:
        """Score text for display only; this method never selects an execution path."""
        result = self._predict_from_text(prompt, "individual_athlete_evaluation_only")
        return {
            **result,
            "evaluation_only": True,
            "hypothetical_model": (
                "GPT-4.1" if result["selected_tier"] == "strong" else "Ministral-3-8B"
            ),
            "router_artifact_path": str(self.model_path.resolve()),
            "router_metadata_path": str(self.metadata_path.resolve()),
        }


def predict_athlete_router(
    user_query: str,
    *,
    requested_analysis: str | None = None,
    difficulty: str | None = None,
    privacy_route: str = "cloud",
    filters: dict | None = None,
    requires_code: bool = True,
    router_prompt_source: str = "original_prompt",
    router: AthleteStrongWeakRouter | None = None,
) -> dict:
    """Central runtime entry point using the trained preprocessing pipeline and threshold."""
    result = (router or AthleteStrongWeakRouter()).predict(
        user_query, requested_analysis, difficulty, privacy_route, filters or {},
        requires_code, router_prompt_source,
    )
    return {
        **result,
        "strong_probability": float(result["p_strong"]),
        "selected_model_label": str(result["selected_tier"]).title(),
        "router_display_name": "New Athlete Router",
    }
