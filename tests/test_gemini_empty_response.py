from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm.model_clients import ModelCallResult, _call, call_gemini_cloud_model


def _result(
    content,
    *,
    success,
    actual_model="gemini-3.5-flash",
    finish_reason=None,
    error=None,
):
    return ModelCallResult(
        content=content,
        requested_model="gemini-3.5-flash",
        actual_model=actual_model,
        provider="openai_compatible",
        success=success,
        unavailable=False,
        fallback_used=False,
        error=error,
        endpoint="chat.completions",
        finish_reason=finish_reason,
        content_state="non-empty string" if content else "None",
    )


def test_gemini_retries_once_after_empty_token_exhaustion(monkeypatch):
    monkeypatch.setattr("llm.model_clients.load_local_env", lambda: None)
    monkeypatch.setenv("LLM_GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("LLM_GEMINI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_GEMINI_API_KEY", "TEST_SECRET")
    monkeypatch.setenv("LLM_GEMINI_BASE_URL", "https://example.invalid/v1/")
    messages = [{"role": "user", "content": "same request"}]
    exhausted = _result(
        None,
        success=False,
        finish_reason="MAX_TOKENS",
        error="empty",
    )
    completed = _result("OK", success=True, finish_reason="stop")

    with patch("llm.model_clients._call", side_effect=[exhausted, completed]) as call:
        result = call_gemini_cloud_model(messages, max_tokens=4096)

    assert result is completed
    assert call.call_count == 2
    assert call.call_args_list[0].args[-1] == 4096
    assert call.call_args_list[1].args[-1] == 16384
    assert call.call_args_list[0].args[0] is messages
    assert call.call_args_list[1].args[0] is messages


def test_gemini_does_not_retry_authentication_or_safety_failures(monkeypatch):
    monkeypatch.setattr("llm.model_clients.load_local_env", lambda: None)
    failures = (
        _result(None, success=False, actual_model=None, error="HTTP 401"),
        _result(None, success=False, finish_reason="SAFETY", error="empty"),
        _result(None, success=False, finish_reason="stop", error="empty"),
    )

    for failure in failures:
        with patch("llm.model_clients._call", return_value=failure) as call:
            assert call_gemini_cloud_model([], max_tokens=4096) is failure
        call.assert_called_once()


def test_call_reports_safe_metadata_for_none_content():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )
        ],
        model="gemini-3.5-flash",
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=4096,
            total_tokens=4100,
        ),
    )

    with patch("openai.OpenAI", return_value=client):
        result = _call(
            [{"role": "user", "content": "Reply only with OK"}],
            "gemini-3.5-flash",
            "openai_compatible",
            "TEST_SECRET",
            "https://example.invalid/v1/",
            None,
            4096,
        )

    assert result.success is False
    assert result.content_state == "None"
    assert result.finish_reason == "length"
    assert result.input_tokens == 4
    assert result.output_tokens == 4096
    assert result.total_tokens == 4100
    assert "requested_model=gemini-3.5-flash" in result.error
    assert "actual_model=gemini-3.5-flash" in result.error
    assert "endpoint=chat.completions" in result.error
    assert "content_state=None" in result.error
    assert "TEST_SECRET" not in result.error


def test_call_extracts_visible_text_parts_without_reasoning_content():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "reasoning", "text": "PRIVATE_REASONING"},
                        {"type": "text", "text": "OK"},
                    ]
                ),
                finish_reason="stop",
            )
        ],
        model="gemini-3.5-flash",
        usage=None,
    )

    with patch("openai.OpenAI", return_value=client):
        result = _call(
            [{"role": "user", "content": "Reply only with OK"}],
            "gemini-3.5-flash",
            "openai_compatible",
            "TEST_SECRET",
            "https://example.invalid/v1/",
            None,
            4096,
        )

    assert result.success is True
    assert result.content == "OK"
    assert "PRIVATE_REASONING" not in result.content
    assert result.content_state == "text content parts"
