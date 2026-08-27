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


def test_ready_local_model_recovers_verified_port_owner_pid():
    source = _source("ensure_local_model.ps1")
    ready_branch = source[source.index("if (Test-LocalModelReady)") :]

    assert "Get-NetTCPConnection" in source
    assert '$listener.OwningProcess' in source
    assert "$readyProcess = Get-LocalModelPortOwner" in ready_branch
    assert "$readyProcess.ProcessName -notmatch '^llama-server$'" in ready_branch
    assert "Save-ManagedLocalModelPid $readyProcess" in ready_branch
    assert ready_branch.index("Save-ManagedLocalModelPid $readyProcess") < ready_branch.index(
        'Write-Host "Local Ministral: READY"'
    )


def test_stop_script_recovers_stale_pid_from_port_without_killing_unrelated_processes():
    source = _source("stop_local_model.ps1")

    assert "local_model_server.pid" in source
    assert "Remove-StalePidFile" in source
    assert "$portOwner = Get-LocalModelPortOwner" in source
    assert "$portOwner.ProcessName -notmatch '^llama-server$'" in source
    assert "Port 8080 is owned by a non-llama-server process; it was not stopped." in source
    assert "$Process.ProcessName -notmatch '^llama-server$'" in source
    assert "Stop-Process -Id $Process.Id" in source
    assert "Get-Process |" not in source


def test_project_startup_delegates_to_main_lifecycle():
    source = _source("start_project.ps1")
    assert 'Join-Path $ProjectRoot "main.py"' in source
    assert "ensure_local_model.ps1" not in source
    assert "Start-Process" not in source
    assert "streamlit run frontend.py" not in source


def test_real_verification_wrapper_starts_local_model_and_forwards_arguments():
    source = _source("verify_local_edge.ps1")
    assert 'Join-Path $ScriptDir "ensure_local_model.ps1"' in source
    assert 'Join-Path $ScriptDir "verify_local_edge_analyses.py"' in source
    assert "@AnalysisArguments" in source
    assert "No Cloud fallback was attempted" in source
