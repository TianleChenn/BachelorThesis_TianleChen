from pathlib import Path


SCRIPTS = Path("scripts")


def _source(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_ensure_local_model_is_localhost_only_and_managed():
    source = _source("ensure_local_model.ps1")
    assert '"http://127.0.0.1:8080/v1/models"' in source
    assert '"--host", "127.0.0.1"' in source
    assert '"--port", "8080"' in source
    assert '"--alias", $ModelAlias' in source
    assert "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M" in source
    assert "local_model_server.pid" in source
    assert "local_model_server_stdout.log" in source
    assert "local_model_server_stderr.log" in source
    assert "0.0.0.0" not in source
    assert "gemini" not in source.casefold()


def test_project_startup_delegates_to_single_ensure_script():
    source = _source("start_project.ps1")
    assert 'Join-Path $ScriptDir "ensure_local_model.ps1"' in source
    assert "Start-Process" not in source
    assert "streamlit run frontend.py" in source


def test_real_verification_wrapper_starts_local_model_and_forwards_arguments():
    source = _source("verify_local_edge.ps1")
    assert 'Join-Path $ScriptDir "ensure_local_model.ps1"' in source
    assert 'Join-Path $ScriptDir "verify_local_edge_analyses.py"' in source
    assert "@AnalysisArguments" in source
    assert "No Cloud fallback was attempted" in source
