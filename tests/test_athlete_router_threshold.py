import numpy as np
import pytest

from scripts.train_athlete_strong_weak_router import (
    label_from_scores,
    select_preference_threshold,
    threshold_candidates,
)


def test_existing_quality_aware_label_helper_is_preserved():
    assert label_from_scores(True, True, 8.5, 8.0) == 0
    assert label_from_scores(True, True, 8.51, 8.0) == 1


def test_threshold_candidates_are_the_locked_991_point_grid():
    candidates = threshold_candidates()
    assert len(candidates) == 991
    assert candidates[0] == pytest.approx(.01)
    assert candidates[-1] == pytest.approx(.99)


def test_threshold_selection_maximizes_balanced_accuracy():
    report = select_preference_threshold(
        np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1])
    )
    assert report["accuracy"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["strong_recall"] == 1.0
    assert report["weak_recall"] == 1.0
