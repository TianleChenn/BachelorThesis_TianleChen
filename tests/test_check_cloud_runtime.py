from pathlib import Path
from types import SimpleNamespace

from scripts import check_cloud_runtime


def test_cloud_runtime_diagnostic_is_credential_safe(monkeypatch, capsys):
    openai_secret = "TEST_OPENAI_SECRET_MUST_NOT_PRINT"
    gemini_secret = "TEST_GEMINI_SECRET_MUST_NOT_PRINT"
    monkeypatch.setenv("OPENAI_API_KEY", openai_secret)
    monkeypatch.setenv("LLM_GEMINI_API_KEY", gemini_secret)
    monkeypatch.setenv("LLM_STRONG_MODEL", "gpt-test")
    monkeypatch.setenv("LLM_GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(check_cloud_runtime, "load_local_env", lambda: Path(".env"))
    calls = []

    def privacy_call(messages, **kwargs):
        calls.append(("privacy", messages, kwargs))
        return SimpleNamespace(success=True, error=None)

    def gemini_call(messages, **kwargs):
        calls.append(("gemini", messages, kwargs))
        return SimpleNamespace(
            success=False,
            error=f"HTTP 401 api_key={gemini_secret}; bearer {openai_secret}",
        )

    monkeypatch.setattr(check_cloud_runtime, "call_privacy_risk_model", privacy_call)
    monkeypatch.setattr(check_cloud_runtime, "call_gemini_cloud_model", gemini_call)

    assert check_cloud_runtime.main() == 1
    output = capsys.readouterr().out

    assert "Environment file: .env" in output
    assert "OPENAI_API_KEY: LOADED" in output
    assert "LLM_GEMINI_API_KEY: LOADED" in output
    assert "GPT-4.1 model: gpt-test" in output
    assert "Gemini model: gemini-test" in output
    assert "GPT-4.1 privacy-assessor call: SUCCESS" in output
    assert "Gemini call: FAILURE" in output
    assert "Gemini:\n" in output
    assert "finish_reason: None" in output
    assert "content_state: None" in output
    assert "HTTP 401" in output
    assert openai_secret not in output
    assert gemini_secret not in output
    assert "safe_local_fallback / local_edge" in output
    assert calls == [
        (
            "privacy",
            [{"role": "user", "content": "Reply only with OK"}],
            {"max_tokens": 8},
        ),
        (
            "gemini",
            [{"role": "user", "content": "Reply only with OK"}],
            {"max_tokens": 4096},
        ),
    ]


def test_cloud_runtime_diagnostic_reports_blank_keys_as_not_set(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("LLM_GEMINI_API_KEY", "")
    monkeypatch.setattr(check_cloud_runtime, "load_local_env", lambda: None)
    unavailable = SimpleNamespace(success=False, error="credentials unavailable")
    monkeypatch.setattr(
        check_cloud_runtime, "call_privacy_risk_model", lambda *args, **kwargs: unavailable
    )
    monkeypatch.setattr(
        check_cloud_runtime, "call_gemini_cloud_model", lambda *args, **kwargs: unavailable
    )

    assert check_cloud_runtime.main() == 1
    output = capsys.readouterr().out

    assert "Environment file: Not found" in output
    assert "OPENAI_API_KEY: NOT SET" in output
    assert "LLM_GEMINI_API_KEY: NOT SET" in output
