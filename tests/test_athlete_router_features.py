from llm.athlete_router_features import (
    build_classifier_pipeline,
    build_prompt_texts,
    normalize_prompt,
)


def test_router_features_are_only_word_and_character_prompt_tfidf():
    assert normalize_prompt("  Compare   athlete domains ") == "Compare athlete domains"
    assert build_prompt_texts([{"prompt": "same", "official_mf_score": .99}]) == ["same"]
    pipeline = build_classifier_pipeline()
    names = {name for name, _ in pipeline["features"].transformer_list}
    assert names == {"word_tfidf", "character_tfidf"}
    assert pipeline["classifier"].max_iter == 5000
    assert pipeline["classifier"].class_weight == "balanced"


def test_metadata_and_mf_values_cannot_change_router_input():
    texts = build_prompt_texts([
        {"prompt": "same", "difficulty": "easy", "official_mf_score": .01},
        {"prompt": "same", "difficulty": "hard", "official_mf_score": .99},
    ])
    assert texts == ["same", "same"]
