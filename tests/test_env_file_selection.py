from __future__ import annotations

import os
from pathlib import Path

import llm.env as env_module
import llm.model_clients as model_clients
from llm.local_model_provider import LocalModelProvider
from scripts import check_api_env, check_environment_config


CONFIG_VARIABLES = tuple(
    dict.fromkeys(
        check_environment_config.ACTIVE_ENVIRONMENT_VARIABLES
        + check_environment_config.LEGACY_OPTIONAL_ENVIRONMENT_VARIABLES
    )
)


def _use_project_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(env_module, "PROJECT_ROOT", root)


def _clear_config(monkeypatch) -> None:
    for variable in CONFIG_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_dot_env_is_preferred_and_dot_env_example_is_not_read(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_VALUE", raising=False)
    (tmp_path / ".env").write_text("TEST_VALUE=from_env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "TEST_VALUE=from_example\n", encoding="utf-8"
    )
    reads = []
    original_read_text = Path.read_text

    def tracked_read_text(path, *args, **kwargs):
        reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)

    selected_path = env_module.load_local_env()

    assert selected_path == tmp_path / ".env"
    assert os.environ["TEST_VALUE"] == "from_env"
    assert reads == [tmp_path / ".env"]


def test_dot_env_example_does_not_fill_missing_dot_env_values(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_SECRET", raising=False)
    monkeypatch.delenv("TEST_MODEL", raising=False)
    (tmp_path / ".env").write_text("TEST_SECRET=my_secret\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "TEST_MODEL=model-name\n", encoding="utf-8"
    )

    env_module.load_local_env()

    assert os.environ["TEST_SECRET"] == "my_secret"
    assert "TEST_MODEL" not in os.environ


def test_dot_env_overrides_stale_process_environment(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("TEST_VALUE=from_env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "TEST_VALUE=from_example\n", encoding="utf-8"
    )
    monkeypatch.setenv("TEST_VALUE", "from_system")

    env_module.load_local_env()

    assert os.environ["TEST_VALUE"] == "from_env"


def test_only_example_exists_and_blank_api_key_is_not_configured(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_CLAUDE_API_KEY", raising=False)
    (tmp_path / ".env.example").write_text(
        (
            "TEST_MODEL=model-name\n"
            "OPENAI_API_KEY=\n"
            "LLM_GEMINI_API_KEY=\"\"\n"
            "LLM_CLAUDE_API_KEY='   '\n"
        ),
        encoding="utf-8-sig",
    )

    selected_path = env_module.load_local_env()

    assert selected_path == tmp_path / ".env.example"
    assert env_module.get_env_file_path() == tmp_path / ".env.example"
    assert os.environ["TEST_MODEL"] == "model-name"
    assert "OPENAI_API_KEY" not in os.environ
    assert "LLM_GEMINI_API_KEY" not in os.environ
    assert "LLM_CLAUDE_API_KEY" not in os.environ


def test_example_fallback_preserves_process_environment(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.setenv("TEST_VALUE", "from_system")
    (tmp_path / ".env.example").write_text(
        "TEST_VALUE=from_example\n", encoding="utf-8"
    )

    env_module.load_local_env()

    assert os.environ["TEST_VALUE"] == "from_system"


def test_unreadable_dot_env_uses_example_fallback(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_VALUE", raising=False)
    (tmp_path / ".env").write_text("TEST_VALUE=from_env\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(
        "TEST_VALUE=from_example\n", encoding="utf-8"
    )
    original_load_env_file = env_module._load_env_file

    def fail_primary(path, **kwargs):
        if path == tmp_path / ".env":
            raise PermissionError("test simulates an unreadable .env")
        return original_load_env_file(path, **kwargs)

    monkeypatch.setattr(env_module, "_load_env_file", fail_primary)

    selected_path = env_module.load_local_env()

    assert selected_path == tmp_path / ".env.example"
    assert os.environ["TEST_VALUE"] == "from_example"


def test_real_dot_env_is_git_ignored():
    gitignore = (env_module.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    active_lines = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".env" in active_lines


def test_check_script_never_prints_secret_values(tmp_path, monkeypatch, capsys):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    secret_markers = {
        "OPENAI_API_KEY": "TEST_OPENAI_SECRET_DO_NOT_PRINT",
        "LLM_GEMINI_API_KEY": "TEST_GEMINI_SECRET_DO_NOT_PRINT",
        "LLM_CLAUDE_API_KEY": "TEST_CLAUDE_SECRET_DO_NOT_PRINT",
        "LLM_LOCAL_API_KEY": "TEST_LOCAL_SECRET_DO_NOT_PRINT",
    }
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}={value}" for name, value in secret_markers.items()),
        encoding="utf-8",
    )
    template_values = {
        name: "" for name in check_environment_config.ACTIVE_ENVIRONMENT_VARIABLES
    }
    template_values.update(
        {
            "LLM_STRONG_MODEL": "gpt-test",
            "LLM_GEMINI_MODEL": "gemini-test",
            "LLM_CLAUDE_MODEL": "claude-test",
            "LLM_LOCAL_MODEL": "local-test",
        }
    )
    (tmp_path / ".env.example").write_text(
        "\n".join(f"{name}={value}" for name, value in template_values.items()),
        encoding="utf-8-sig",
    )

    assert check_environment_config.main() == 0
    output = capsys.readouterr().out

    assert "Environment structure: PASS" in output
    assert "Cloud credentials: PASS" in output
    assert all(secret not in output for secret in secret_markers.values())


def test_check_api_env_prints_status_but_never_secret_values(
    tmp_path, monkeypatch, capsys
):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    secret_markers = {
        "OPENAI_API_KEY": "TEST_OPENAI_SECRET_DO_NOT_PRINT",
        "LLM_GEMINI_API_KEY": "TEST_GEMINI_SECRET_DO_NOT_PRINT",
        "LLM_CLAUDE_API_KEY": "TEST_CLAUDE_SECRET_DO_NOT_PRINT",
        "LLM_LOCAL_API_KEY": "TEST_LOCAL_SECRET_DO_NOT_PRINT",
    }
    (tmp_path / ".env").write_text(
        "\n".join(f"{name}={value}" for name, value in secret_markers.items()),
        encoding="utf-8",
    )

    assert check_api_env.main() == 0
    output = capsys.readouterr().out

    assert "Environment file:\n  .env" in output
    assert output.count("LOADED") == len(secret_markers)
    assert all(secret not in output for secret in secret_markers.values())


def test_check_script_rejects_nonempty_secret_in_public_template(
    tmp_path, monkeypatch, capsys
):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    template_values = {
        name: "" for name in check_environment_config.ACTIVE_ENVIRONMENT_VARIABLES
    }
    template_values["OPENAI_API_KEY"] = "PUBLIC_TEMPLATE_MUST_NOT_CONTAIN_THIS"
    template_values["LLM_LOCAL_API_KEY"] = "none"
    (tmp_path / ".env.example").write_text(
        "\n".join(f"{name}={value}" for name, value in template_values.items()),
        encoding="utf-8",
    )

    assert check_environment_config.main() == 1
    output = capsys.readouterr().out

    assert "Unsafe non-empty secret fields in .env.example: 1" in output
    assert "PUBLIC_TEMPLATE_MUST_NOT_CONTAIN_THIS" not in output


def test_local_model_provider_uses_central_example_fallback(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    (tmp_path / ".env.example").write_text(
        "\n".join(
            (
                "LLM_LOCAL_PROVIDER=openai_compatible",
                "LLM_LOCAL_MODEL=Fallback-Local-Model",
                "LLM_LOCAL_BASE_URL=http://127.0.0.1:9090/v1",
                "LLM_LOCAL_API_KEY=none",
            )
        ),
        encoding="utf-8",
    )
    LocalModelProvider._instance = None

    try:
        provider = LocalModelProvider()
        assert provider.provider == "openai_compatible"
        assert provider.model_id == "Fallback-Local-Model"
        assert provider.base_url == "http://127.0.0.1:9090/v1"
        assert provider.api_key == "none"
    finally:
        LocalModelProvider._instance = None


def test_active_cloud_clients_use_central_example_fallback(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    assert not (tmp_path / ".env").exists()
    (tmp_path / ".env.example").write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=OPENAI_TEST_CREDENTIAL",
                "LLM_STRONG_MODEL=gpt-central-test",
                "LLM_GEMINI_API_KEY=GEMINI_TEST_CREDENTIAL",
                "LLM_GEMINI_MODEL=gemini-central-test",
                "LLM_GEMINI_BASE_URL=https://gemini.invalid/v1/",
                "LLM_GEMINI_PROVIDER=openai_compatible",
                "LLM_CLAUDE_API_KEY=CLAUDE_TEST_CREDENTIAL",
                "LLM_CLAUDE_MODEL=claude-central-test",
                "LLM_CLAUDE_BASE_URL=https://claude.invalid/v1/",
            )
        ),
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
        "gpt-central-test",
        "openai_privacy_assessor",
        "OPENAI_TEST_CREDENTIAL",
        None,
    )
    assert calls[1][0][1:5] == (
        "gemini-central-test",
        "openai_compatible",
        "GEMINI_TEST_CREDENTIAL",
        "https://gemini.invalid/v1/",
    )
    assert calls[2][0][1:5] == (
        "claude-central-test",
        "anthropic",
        "CLAUDE_TEST_CREDENTIAL",
        "https://claude.invalid/v1/",
    )


def test_blank_cloud_credential_reports_unavailable(tmp_path, monkeypatch):
    _use_project_root(monkeypatch, tmp_path)
    _clear_config(monkeypatch)
    (tmp_path / ".env.example").write_text(
        "\n".join(
            (
                "LLM_GEMINI_API_KEY=",
                "LLM_GEMINI_MODEL=gemini-test",
                "LLM_GEMINI_BASE_URL=https://gemini.invalid/v1/",
                "LLM_GEMINI_PROVIDER=openai_compatible",
            )
        ),
        encoding="utf-8",
    )

    result = model_clients.call_gemini_cloud_model([])

    assert result.success is False
    assert "credentials unavailable" in (result.error or "")
