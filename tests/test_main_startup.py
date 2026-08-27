from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import main


def _completed(returncode: int) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=returncode)


def test_ready_local_model_is_reused_before_streamlit_launch():
    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main.subprocess, "run", side_effect=[_completed(0), _completed(0)]) as run:
        assert main.main() == 0
    assert run.call_count == 2
    ensure_command = run.call_args_list[0].args[0]
    assert ensure_command[:5] == [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    assert Path(ensure_command[5]).name == "ensure_local_model.ps1"
    assert run.call_args_list[0].kwargs["check"] is False
    assert "stdout" not in run.call_args_list[0].kwargs
    assert "stderr" not in run.call_args_list[0].kwargs
    assert run.call_args_list[1].args[0][1:4] == ["-m", "streamlit", "run"]


def test_ensure_success_launches_frontend_after_local_model_startup():
    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main.subprocess, "run", side_effect=[_completed(0), _completed(7)]) as run:
        assert main.main() == 7
    assert Path(run.call_args_list[0].args[0][-1]).name == "ensure_local_model.ps1"
    assert Path(run.call_args_list[1].args[0][-1]).name == "frontend.py"


def test_local_model_startup_failure_prevents_streamlit_launch(capsys):
    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main.subprocess, "run", return_value=_completed(23)) as run:
        assert main.main() == 23
    assert run.call_count == 1
    assert "ensure_local_model.ps1" in str(run.call_args.args[0][-1])
    assert "Streamlit was not launched" in capsys.readouterr().err


def test_non_windows_keeps_direct_streamlit_startup():
    with patch.object(main.sys, "platform", "linux"), \
         patch.object(main.subprocess, "run", return_value=_completed(0)) as run:
        assert main.main() == 0
    run.assert_called_once()
    assert run.call_args.args[0][1:4] == ["-m", "streamlit", "run"]
