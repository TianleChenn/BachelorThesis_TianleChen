from pathlib import Path

from llm.model_clients import ModelCallResult
from scripts import check_gemini_runtime


def test_gemini_runtime_check_prints_safe_metadata(monkeypatch, capsys):
    secret = "TEST_GEMINI_SECRET_MUST_NOT_PRINT"
    monkeypatch.setenv("LLM_GEMINI_API_KEY", secret)
    monkeypatch.setenv("LLM_GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr(check_gemini_runtime, "load_local_env", lambda: Path(".env"))
    calls = []
    result = ModelCallResult(
        content=None,
        requested_model="gemini-3.5-flash",
        actual_model="gemini-3.5-flash",
        provider="openai_compatible",
        success=False,
        unavailable=False,
        fallback_used=False,
        error=f"HTTP 429 api_key={secret}",
        endpoint="chat.completions",
        input_tokens=4,
        output_tokens=4096,
        total_tokens=4100,
        finish_reason="MAX_TOKENS",
        content_state="None",
    )

    def call(messages, **kwargs):
        calls.append((messages, kwargs))
        return result

    monkeypatch.setattr(check_gemini_runtime, "call_gemini_cloud_model", call)

    assert check_gemini_runtime.main() == 1
    output = capsys.readouterr().out

    assert "Gemini:\n" in output
    assert "api_key: LOADED" in output
    assert "model: gemini-3.5-flash" in output
    assert "endpoint: chat.completions" in output
    assert "success: False" in output
    assert "finish_reason: MAX_TOKENS" in output
    assert "input_tokens: 4" in output
    assert "output_tokens: 4096" in output
    assert "total_tokens: 4100" in output
    assert "content_state: None" in output
    assert "HTTP 429" in output
    assert secret not in output
    assert calls == [
        (
            [{"role": "user", "content": "Reply only with OK"}],
            {"max_tokens": 4096},
        )
    ]
