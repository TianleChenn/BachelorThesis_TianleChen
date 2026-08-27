from __future__ import annotations

import os
from pathlib import Path

import pytest

import llm.env as env_module
import llm.model_clients as model_clients
from llm.local_model_provider import LocalModelProvider
from scripts import check_api_env


CONFIG_VARIABLES = (
    "OPENAI_API_KEY",
    "LLM_STRONG_MODEL",
    "LLM_GEMINI_API_KEY",
    "LLM_GEMINI_MODEL",
    "LLM_GEMINI_BASE_URL",
    "LLM_GEMINI_PROVIDER",
    "LLM_CLAUDE_API_KEY",
    "LLM_CLAUDE_MODEL",
    "LLM_CLAUDE_BASE_URL",
    "LLM_LOCAL_API_KEY",
    "LLM_LOCAL_MODEL",
    "LLM_LOCAL_BASE_URL",
    "LLM_LOCAL_PROVIDER",
)


def _use_project_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(env_module, "PROJECT_ROOT", root)
    monkeypatch.delenv("THESIS_ENV_FILE", raising=False)


def _clear_config(monkeypatch) -> None:
    for variable in CONFIG_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_default_selects_dot_env_when_present(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("TEST_SETTING=from-default\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("TEST_SETTING=from-example\n", encoding="utf-8")

    assert env_module.get_env_file_path() == tmp_path / ".env"


def test_explicit_example_file_selection(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("TEST_SETTING=from-default\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("TEST_SETTING=from-example\n", encoding="utf-8")
    monkeypatch.setenv("THESIS_ENV_FILE", ".env.example")

    assert env_module.get_env_file_path() == tmp_path / ".env.example"


def test_default_falls_back_to_example_when_dot_env_is_absent(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env.example").write_text("TEST_SETTING=from-example\n", encoding="utf-8")

    assert env_module.get_env_file_path() == tmp_path / ".env.example"


def test_missing_explicit_file_raises_clear_error(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.setenv("THESIS_ENV_FILE", "missing.env")

    with pytest.raises(RuntimeError, match="Configured environment file does not exist: missing.env"):
        env_module.get_env_file_path()


def test_existing_process_environment_has_precedence(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("THESIS_TEST_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.setenv("THESIS_TEST_VALUE", "from-process")

    env_module.load_local_env()

    assert os.environ["THESIS_TEST_VALUE"] == "from-process"


def test_check_script_never_prints_secret_values(tmp_path, monkeypatch, capsys):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    monkeypatch.setenv("THESIS_ENV_FILE", ".env.example")
    secret_markers = {
        "OPENAI_API_KEY": "TEST_OPENAI_SECRET_DO_NOT_PRINT",
        "LLM_GEMINI_API_KEY": "TEST_GEMINI_SECRET_DO_NOT_PRINT",
        "LLM_CLAUDE_API_KEY": "TEST_CLAUDE_SECRET_DO_NOT_PRINT",
        "LLM_LOCAL_API_KEY": "TEST_LOCAL_SECRET_DO_NOT_PRINT",
    }
    lines = [f"{name}={value}" for name, value in secret_markers.items()]
    lines.extend((
        "LLM_STRONG_MODEL=gpt-test",
        "LLM_GEMINI_MODEL=gemini-test",
        "LLM_CLAUDE_MODEL=claude-test",
        "LLM_LOCAL_MODEL=local-test",
    ))
    (tmp_path / ".env.example").write_text("\n".join(lines), encoding="utf-8-sig")

    assert check_api_env.main() == 0
    output = capsys.readouterr().out

    assert str(tmp_path / ".env.example") in output
    assert output.count("LOADED") == 4
    assert all(secret not in output for secret in secret_markers.values())
    assert "No network calls performed: YES" in output


def test_local_model_provider_uses_selected_central_file(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    monkeypatch.setenv("THESIS_ENV_FILE", ".env.example")
    (tmp_path / ".env.example").write_text(
        "\n".join((
            "LLM_LOCAL_PROVIDER=openai_compatible",
            "LLM_LOCAL_MODEL=Selected-Local-Model",
            "LLM_LOCAL_BASE_URL=http://127.0.0.1:9090/v1",
            "LLM_LOCAL_API_KEY=none",
        )),
        encoding="utf-8",
    )
    LocalModelProvider._instance = None

    try:
        provider = LocalModelProvider()
        assert provider.provider == "openai_compatible"
        assert provider.model_id == "Selected-Local-Model"
        assert provider.base_url == "http://127.0.0.1:9090/v1"
        assert provider.api_key == "none"
    finally:
        LocalModelProvider._instance = None


def test_active_cloud_clients_use_selected_central_file(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    monkeypatch.setenv("THESIS_ENV_FILE", ".env.example")
    (tmp_path / ".env.example").write_text(
        "\n".join((
            "OPENAI_API_KEY=OPENAI_TEST_CREDENTIAL",
            "LLM_STRONG_MODEL=gpt-central-test",
            "LLM_GEMINI_API_KEY=GEMINI_TEST_CREDENTIAL",
            "LLM_GEMINI_MODEL=gemini-central-test",
            "LLM_GEMINI_BASE_URL=https://gemini.invalid/v1/",
            "LLM_GEMINI_PROVIDER=openai_compatible",
            "LLM_CLAUDE_API_KEY=CLAUDE_TEST_CREDENTIAL",
            "LLM_CLAUDE_MODEL=claude-central-test",
            "LLM_CLAUDE_BASE_URL=https://claude.invalid/v1/",
        )),
        encoding="utf-8",
    )
    calls = []

    def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(model_clients, "_call", fake_call)

    model_clients.call_privacy_risk_model([])
    model_clients.call_gemini_cloud_model([])
    model_clients.call_cloud_codegen_evaluation_model("claude", [])

    assert calls[0][0][1:5] == (
        "gpt-central-test", "openai_privacy_assessor", "OPENAI_TEST_CREDENTIAL", None,
    )
    assert calls[1][0][1:5] == (
        "gemini-central-test", "openai_compatible", "GEMINI_TEST_CREDENTIAL",
        "https://gemini.invalid/v1/",
    )
    assert calls[2][0][1:5] == (
        "claude-central-test", "anthropic", "CLAUDE_TEST_CREDENTIAL",
        "https://claude.invalid/v1/",
    )
