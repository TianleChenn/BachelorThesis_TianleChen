from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import MagicMock, patch

import main


def _completed(returncode: int) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=returncode)


def _exited_frontend(returncode: int = 0) -> MagicMock:
    process = MagicMock()
    process.wait.return_value = returncode
    process.poll.return_value = returncode
    process.pid = 4242
    return process


def test_ensure_local_model_invokes_the_managed_powershell_script():
    with patch.object(main.subprocess, "run", return_value=_completed(0)) as run:
        assert main.ensure_local_model(Path.cwd()) == 0

    ensure_command = run.call_args.args[0]
    assert ensure_command[:5] == [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"
    ]
    assert Path(ensure_command[5]).name == "ensure_local_model.ps1"
    assert run.call_args.kwargs["check"] is False
    assert "stdout" not in run.call_args.kwargs
    assert "stderr" not in run.call_args.kwargs


def test_environment_loads_before_local_model_and_streamlit_startup():
    events = []
    process = _exited_frontend(7)

    def launch(*args, **kwargs):
        events.append("streamlit")
        return process

    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main, "load_local_env", side_effect=lambda: events.append("env")), \
         patch.object(main, "ensure_local_model", side_effect=lambda root: events.append("local") or 0), \
         patch.object(main.subprocess, "Popen", side_effect=launch) as popen, \
         patch.object(main, "stop_local_model", side_effect=lambda root: events.append("stop")):
        assert main.main() == 7

    assert events == ["env", "local", "streamlit", "stop"]
    assert popen.call_args.args[0][1:4] == ["-m", "streamlit", "run"]
    assert Path(popen.call_args.args[0][-1]).name == "frontend.py"


def test_local_model_startup_failure_prevents_streamlit_and_still_cleans_up(capsys):
    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main, "ensure_local_model", return_value=23), \
         patch.object(main.subprocess, "Popen") as popen, \
         patch.object(main, "stop_local_model") as stop:
        assert main.main() == 23

    popen.assert_not_called()
    stop.assert_called_once()
    assert "Shutdown complete." in capsys.readouterr().out


def test_non_windows_launches_streamlit_without_local_model_management():
    process = _exited_frontend()
    with patch.object(main.sys, "platform", "linux"), \
         patch.object(main.subprocess, "Popen", return_value=process) as popen, \
         patch.object(main, "ensure_local_model") as ensure, \
         patch.object(main, "stop_local_model") as stop:
        assert main.main() == 0

    ensure.assert_not_called()
    stop.assert_not_called()
    assert popen.call_args.args[0][1:4] == ["-m", "streamlit", "run"]


def test_keyboard_interrupt_terminates_frontend_and_stops_local_model(capsys):
    process = MagicMock()
    process.pid = 4242
    process.wait.side_effect = [KeyboardInterrupt(), 0]
    process.poll.return_value = None

    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main, "ensure_local_model", return_value=0), \
         patch.object(main.subprocess, "Popen", return_value=process), \
         patch.object(main, "stop_local_model") as stop:
        assert main.main() == 130

    process.terminate.assert_called_once_with()
    process.wait.assert_called_with(timeout=main.FRONTEND_STOP_TIMEOUT_SECONDS)
    stop.assert_called_once()
    output = capsys.readouterr().out
    assert "Stopping Frontend..." in output
    assert "Shutdown complete." in output


def test_windows_timeout_force_kills_only_the_streamlit_child_tree():
    process = MagicMock()
    process.pid = 4242
    process.poll.return_value = None
    process.wait.side_effect = [TimeoutExpired(cmd="streamlit", timeout=5), 0]

    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main.subprocess, "run", return_value=_completed(0)) as run:
        main._terminate_frontend_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()
    assert run.call_args.args[0] == ["taskkill", "/PID", "4242", "/T", "/F"]


def test_streamlit_launch_error_still_runs_windows_shutdown(capsys):
    with patch.object(main.sys, "platform", "win32"), \
         patch.object(main, "ensure_local_model", return_value=0), \
         patch.object(main.subprocess, "Popen", side_effect=OSError("launch failed")), \
         patch.object(main, "stop_local_model") as stop:
        assert main.main() == 1

    stop.assert_called_once()
    assert "Could not launch Streamlit" in capsys.readouterr().err
