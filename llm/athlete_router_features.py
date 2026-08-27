"""Text-only features for the project-specific Cloud/Local preference router."""
from __future__ import annotations

from collections.abc import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def normalize_prompt(prompt: object) -> str:
    """Return the original request in the stable form used for training and inference."""
    return " ".join(str(prompt or "").split())


def build_prompt_texts(samples: Iterable[dict]) -> list[str]:
    return [normalize_prompt(sample.get("prompt", "")) for sample in samples]


def build_classifier_pipeline() -> Pipeline:
    """Build a RouteLLM-inspired word/character preference classifier."""
    features = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=5000,
        )),
        ("character_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
        )),
    ])
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=2026,
    )
    return Pipeline([("features", features), ("classifier", classifier)])
