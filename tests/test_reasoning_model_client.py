from llm.model_clients import is_reasoning_model

def test_reasoning_model_family_detection():
    assert is_reasoning_model("o1-preview")
    assert is_reasoning_model("o3-mini")
    assert is_reasoning_model("o4-mini-2025-04-16")
    assert not is_reasoning_model("gpt-4.1-mini")

