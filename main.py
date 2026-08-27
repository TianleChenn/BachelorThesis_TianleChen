"""Start the managed Local Ministral service, then launch Streamlit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "frontend.py"
    _print_banner("Athlete LLM Project Startup")
    if sys.platform == "win32":
        startup_code = ensure_local_model(project_root)
        if startup_code != 0:
            return startup_code
        print("Local Ministral is ready.", flush=True)
    _print_banner("Starting Frontend")
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=project_root,
            check=False,
        )
    except OSError as exc:
        print(f"ERROR: Could not launch Streamlit: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
