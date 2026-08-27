from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    app = AppTest.from_file(str(ROOT / "frontend.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Protected analysis dashboard").run()
    table1_button = next(button for button in app.button if button.label == "Run Table 1")
    table1_button.click().run()

    metric_labels = {metric.label for metric in app.metric}
    expected_metrics = {
        "Privacy-aware Router",
        "Cost-aware Router",
        "Selected LLM",
        "Local Verification",
        "Cloud",
        "Collaboration",
        "Local Edge",
        "Selected Route",
        "P_cloud",
        "Decision Threshold",
    }
    assert expected_metrics.issubset(metric_labels)
    assert not app.exception
    assert any("finished locally" in item.value for item in app.success)

    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "Cost-aware Router Result" in rendered_markdown
    assert "PGR" not in metric_labels
    assert "CPT" not in metric_labels

    button_labels = {button.label for button in app.button}
    assert "Generate network plot" not in button_labels
    assert "Plot variance" not in button_labels

    print("Frontend pipeline UI checks passed.")


if __name__ == "__main__":
    main()
