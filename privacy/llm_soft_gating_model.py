"""Linear four-dimensional Soft Gating for validated LLM privacy assessments."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

ROUTE_TO_INDEX = {"cloud": 0, "collaboration": 1, "local_edge": 2}
INDEX_TO_ROUTE = {0: "cloud", 1: "collaboration", 2: "local_edge"}
FEATURE_NAMES = [
    "privacy_risk_score",
    "subject_scope",
    "data_sensitivity",
    "disclosure_level",
]
INPUT_DIM = 4
NUM_ROUTES = 3
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "prism_soft_gater_4d_llm_hard.pt"
)


class LLMPrivacySoftGater(nn.Module):
    """A deliberately small linear gater for the simulation-stage dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, NUM_ROUTES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(
                f"Expected {INPUT_DIM} features, received {features.shape[-1]}"
            )
        return self.linear(features)


def build_llm_gating_features(assessment) -> list[float]:
    """Return the four continuous LLM values without mapping or quantization."""
    features = [
        float(assessment.privacy_risk_score),
        float(assessment.subject_scope),
        float(assessment.data_sensitivity),
        float(assessment.disclosure_level),
    ]
    if len(features) != INPUT_DIM:
        raise ValueError(f"Expected exactly {INPUT_DIM} Soft Gating features")
    if any(not math.isfinite(value) for value in features):
        raise ValueError("Soft Gating features must be finite")
    if any(value < 0.0 or value > 1.0 for value in features):
        raise ValueError("Soft Gating features must be in [0, 1]")
    return features


def softmax_probabilities(logits: torch.Tensor) -> dict[str, float]:
    """Convert one three-route logit vector to named probabilities."""
    probabilities = torch.softmax(logits, dim=-1)
    if probabilities.ndim > 1:
        if probabilities.shape[0] != 1:
            raise ValueError("Prediction expects exactly one sample")
        probabilities = probabilities[0]
    if probabilities.shape[-1] != NUM_ROUTES:
        raise ValueError(f"Expected {NUM_ROUTES} route logits")
    return {
        INDEX_TO_ROUTE[index]: float(probabilities[index].item())
        for index in range(NUM_ROUTES)
    }


def load_llm_privacy_soft_gater(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> LLMPrivacySoftGater:
    """Load only a compatible 4D checkpoint; never fall back to the 6D model."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"LLM-based 4D Soft Gating checkpoint not found: {path}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("input_dim") != INPUT_DIM:
        raise RuntimeError("The checkpoint input dimension is not 4")
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("The checkpoint feature names do not match the 4D schema")
    model = LLMPrivacySoftGater()
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError("The 4D checkpoint state is incompatible") from exc
    model.eval()
    return model


def predict_llm_privacy_route(
    features: list[float],
    model: LLMPrivacySoftGater | None = None,
) -> tuple[str, dict[str, float]]:
    """Predict a route and return all three Soft Gating probabilities."""
    if len(features) != INPUT_DIM:
        raise ValueError(f"Expected exactly {INPUT_DIM} Soft Gating features")
    values = [float(value) for value in features]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Soft Gating features must be finite values in [0, 1]")
    active_model = model or load_llm_privacy_soft_gater()
    with torch.no_grad():
        probabilities = softmax_probabilities(
            active_model(torch.tensor([values], dtype=torch.float32))
        )
    return max(probabilities, key=probabilities.get), probabilities
