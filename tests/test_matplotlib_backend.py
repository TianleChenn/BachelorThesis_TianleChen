from __future__ import annotations

import matplotlib


def test_pytest_uses_non_interactive_matplotlib_backend():
    backend = str(matplotlib.get_backend()).lower()
    assert "agg" in backend
    assert "tk" not in backend
