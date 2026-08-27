"""Start the managed Local Ministral service, then launch Streamlit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llm.env import load_local_env


FRONTEND_STOP_TIMEOUT_SECONDS = 5
LOCAL_MODEL_STOP_TIMEOUT_SECONDS = 15


def _print_banner(title: str) -> None:
    print("=" * 60, flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)


def ensure_local_model(project_root: Path) -> int:
    """Delegate Windows Local Ministral lifecycle management to PowerShell."""
    script_path = project_root / "scripts" / "ensure_local_model.ps1"
    if not script_path.is_file():
        print(f"ERROR: Local model startup script was not found: {script_path}", file=sys.stderr)
        return 1
    print("Checking Local Ministral...", flush=True)
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script_path)],
            cwd=project_root,
            check=False,
        )
    except OSError as exc:
        print(f"ERROR: Could not run Local Ministral startup script: {exc}", file=sys.stderr)
        return 1
    if completed.returncode != 0:
        print(
            f"ERROR: Local Ministral startup failed with exit code {completed.returncode}. "
            "Streamlit was not launched.",
            file=sys.stderr,
        )
    return completed.returncode


def stop_local_model(project_root: Path) -> None:
    """Stop only the Windows Local Ministral process managed by the project."""
    script_path = project_root / "scripts" / "stop_local_model.ps1"
    if not script_path.is_file():
        print(f"WARNING: Local model stop script was not found: {script_path}", file=sys.stderr)
        return
    print("Stopping Local Ministral...", flush=True)
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script_path)],
            cwd=project_root,
            check=False,
            timeout=LOCAL_MODEL_STOP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print("WARNING: Local Ministral shutdown timed out.", file=sys.stderr)
    except OSError as exc:
        print(f"WARNING: Could not run Local Ministral stop script: {exc}", file=sys.stderr)
    else:
        if completed.returncode != 0:
            print(
                f"WARNING: Local Ministral stop script exited with code "
                f"{completed.returncode}.",
                file=sys.stderr,
            )


def _terminate_frontend_process(process: subprocess.Popen) -> None:
    """Terminate only the Streamlit child (and its Windows tree if necessary)."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=FRONTEND_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=FRONTEND_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print("WARNING: Frontend process did not exit after forced shutdown.", file=sys.stderr)


def main() -> int:
    load_local_env()
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "frontend.py"
    frontend_process: subprocess.Popen | None = None
    windows = sys.platform == "win32"
    exit_code = 1
    _print_banner("Athlete LLM Project Startup")
    try:
        if windows:
            startup_code = ensure_local_model(project_root)
            if startup_code != 0:
                return startup_code
            print("Local Ministral is ready.", flush=True)
        _print_banner("Starting Frontend")
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=project_root,
        )
        exit_code = frontend_process.wait()
    except KeyboardInterrupt:
        print("Stopping Frontend...", flush=True)
        exit_code = 130
    except OSError as exc:
        print(f"ERROR: Could not launch Streamlit: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            if frontend_process is not None:
                _terminate_frontend_process(frontend_process)
        except OSError as exc:
            print(f"WARNING: Could not terminate Frontend cleanly: {exc}", file=sys.stderr)
        finally:
            if windows:
                stop_local_model(project_root)
            print("Shutdown complete.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
