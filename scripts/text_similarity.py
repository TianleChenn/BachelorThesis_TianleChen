"""Text normalization and similarity helpers for athlete-router datasets."""
from __future__ import annotations

import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_prompt(text: str) -> str:
    lowered = " ".join(str(text).lower().split())
    return " ".join(
        lowered.translate(str.maketrans("", "", string.punctuation)).split()
    )


def maximum_similarities(prompts: list[str], references: list[str]) -> list[float]:
    if not prompts:
        return []
    if not references:
        return [0.0] * len(prompts)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(references + prompts)
    similarities = cosine_similarity(matrix[len(references):], matrix[:len(references)])
    return [float(row.max()) if row.size else 0.0 for row in similarities]
