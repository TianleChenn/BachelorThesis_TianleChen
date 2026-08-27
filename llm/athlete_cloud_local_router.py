"""Validated runtime loader for the project-specific Cloud/Local classifier."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib

from llm.athlete_router_features import normalize_prompt
from llm.env import load_local_env

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "athlete_cloud_local_router.joblib"
DEFAULT_METADATA = ROOT / "artifacts" / "athlete_cloud_local_router.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AthleteCloudLocalRouter:
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
            raise RuntimeError(f"Cloud/Local router unavailable: {missing.name} not found")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "trained" or metadata.get("router_type") != "project_specific_cloud_local_router":
            raise RuntimeError("Cloud/Local router artifact is incompatible or not trained")
        if metadata.get("model_sha256") != _sha256(self.model_path):
            raise RuntimeError("Cloud/Local router model hash does not match metadata")
        pipeline = joblib.load(self.model_path)
        if not hasattr(pipeline, "predict_proba"):
            raise RuntimeError("Cloud/Local router classifier does not support predict_proba")
        self._pipeline, self._metadata = pipeline, metadata
        return pipeline, metadata

    def predict(self, prompt: str, *, router_prompt_source: str = "original_prompt") -> dict:
        pipeline, metadata = self._load()
        classifier = getattr(pipeline, "named_steps", {}).get("classifier")
        classes = getattr(pipeline, "classes_", None)
        if classes is None and classifier is not None:
            classes = getattr(classifier, "classes_", None)
        if classes is not None and list(classes) != [0, 1]:
            raise ValueError(
                "Cloud/Local router expected classifier classes [0, 1] "
                "for Local=0 and Cloud=1."
            )
        probabilities = pipeline.predict_proba([normalize_prompt(prompt)])
        if len(probabilities) != 1:
            raise ValueError("Cloud/Local router expected exactly one probability row.")
        row = probabilities[0]
        if len(row) < 2:
            raise ValueError("Cloud/Local router expected binary class probabilities.")
        probability = float(row[1])
        threshold = float(metadata["threshold"])
        tier = "cloud" if probability >= threshold else "local"
        load_local_env()
        execution_model = (
            os.getenv("LLM_GEMINI_MODEL", "gemini-3.5-flash")
            if tier == "cloud" else os.getenv("LLM_LOCAL_MODEL", "Ministral-3-8B-Local")
        )
        return {
            "p_cloud": probability,
            "cloud_model_probability": probability,
            "threshold": threshold,
            "selected_tier": tier,
            "selected_model": "cloud_gemini" if tier == "cloud" else "local_ministral",
            "execution_model": execution_model,
            "router_name": "athlete_cloud_local_router",
            "router_prompt_source": router_prompt_source,
            "model_version": metadata.get("model_sha256", "")[:12],
        }
