from types import SimpleNamespace

import numpy as np
import pytest

from napari_mAIcrobe.mAIcrobe.cellaverager import CellAverager


def test_calculate_cell_outline_removes_eroded_interior():
    binary = np.zeros((5, 5), dtype=int)
    binary[1:4, 1:4] = 1

    outline = CellAverager.calculate_cell_outline(binary)

    assert outline.sum() == 8
    assert outline[2, 2] == 0


def test_calculate_major_axis_returns_two_points_for_outline():
    outline = np.zeros((6, 6), dtype=int)
    outline[1:5, 2] = 1

    axis = CellAverager.calculate_major_axis(outline)

    assert len(axis) == 2
    assert len(axis[0]) == 2
    assert axis[0][0] < axis[1][0]


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ([[0, 0], [1, 0]], 0.0),
        ([[0, 0], [0, 1]], 90.0),
        ([[0, 0], [1, 1]], 135.0),
        ([[1, 0], [0, 1]], 45.0),
    ],
)
def test_calculate_axis_angle_branches(axis, expected):
    assert CellAverager.calculate_axis_angle(axis) == pytest.approx(expected)


def test_align_adds_rotated_mask_and_average_builds_model():
    fluor = np.ones((8, 8), dtype=float)
    cell_mask = np.zeros((5, 5))
    cell_mask[1:4, 1:4] = 1
    cell = SimpleNamespace(
        cell_mask=cell_mask,
        image_box=lambda image: image[2:7, 2:7],
    )
    averager = CellAverager(fluor)

    averager.align(cell)
    averager.average()

    assert len(averager.aligned_fluor_masks) == 1
    assert averager.model is not None
    assert averager.model.ndim == 2
