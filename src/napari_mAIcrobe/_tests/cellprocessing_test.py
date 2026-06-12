import numpy as np

from napari_mAIcrobe.mAIcrobe.cellprocessing import (
    bound_rectangle,
    bounded_point,
    bounded_value,
    rotation_matrices,
    stats_format,
)


def test_bounded_value_and_point_clamp_to_limits():
    assert bounded_value(1, 3, 0) == 1
    assert bounded_value(1, 3, 4) == 3
    assert bounded_value(1, 3, 2) == 2

    assert bounded_point(0, 10, 5, 8, (-1, 9)) == (0, 8)
    assert bounded_point(0, 10, 5, 8, (3, 6)) == (3, 6)


def test_bound_rectangle_returns_min_max_and_short_width():
    points = np.array([[3, 5], [1, 9], [6, 7]])

    assert bound_rectangle(points) == (1, 5, 6, 9, 4)


def test_rotation_matrices_respect_step_and_identity_first():
    matrices = rotation_matrices(45)

    assert len(matrices) == 4
    np.testing.assert_allclose(matrices[0], np.eye(2))
    np.testing.assert_allclose(
        matrices[2], np.array([[0, 1], [-1, 0]]), atol=1e-7
    )


def test_stats_format_toggles_optional_columns():
    base = {
        "find_septum": False,
        "find_openseptum": False,
        "classify_cell_cycle": False,
    }

    labels = [label for label, _digits in stats_format(base)]
    assert "frame" not in labels
    assert "Septum Median" not in labels
    assert "Cell Cycle Phase" not in labels

    extended = {
        **base,
        "include_frame": True,
        "find_septum": True,
        "classify_cell_cycle": True,
    }
    labels = [label for label, _digits in stats_format(extended)]
    assert labels[0] == "frame"
    assert "Septum Median" in labels
    assert "Cell Cycle Phase" in labels
