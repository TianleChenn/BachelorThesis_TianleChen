from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_figure1_has_no_overlapping_nodes(tmp_path):
    from sports.analysis import correlation_analysis
    from sports.figures import (
        _correlation_mds_positions,
        _relax_node_positions,
        create_figure1,
    )

    result = correlation_analysis(group="all")
    matrix = result["correlation_matrix"]
    initial = _correlation_mds_positions(matrix)

    assert len(initial) == 8

    radii = {name: 0.45 for name in initial}
    positions = _relax_node_positions(
        initial,
        radii,
        minimum_gap=0.30,
    )

    names = list(positions)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            dx = positions[first][0] - positions[second][0]
            dy = positions[first][1] - positions[second][1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            assert distance >= radii[first] + radii[second] + 0.25

    figure = create_figure1()
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_xlabel()
    assert axis.get_ylabel()
    figure_text = {text.get_text() for text in axis.texts}
    assert "x" in figure_text
    assert "y" in figure_text

    destination = tmp_path / "figure1.png"
    figure.savefig(destination, dpi=120, bbox_inches="tight")

    assert destination.exists()
    assert destination.stat().st_size > 10_000
