import numpy as np
import pytest

from napari_mAIcrobe.mAIcrobe.cells import Cell


def _cell():
    cell = Cell.__new__(Cell)
    cell.box_margin = 1
    cell.box = (0, 0, 4, 4)
    cell.cell_mask = np.zeros((5, 5), dtype=float)
    cell.cell_mask[1:4, 1:4] = 1
    cell.fluor_mask = np.arange(25, dtype=float).reshape(5, 5)
    cell.short_axis = np.array([[0, 2], [4, 2]])
    cell.stats = {"Baseline": 0}
    return cell


def test_compute_perim_mask_returns_boundary_pixels():
    cell = _cell()

    perim = cell.compute_perim_mask(2)

    assert perim.shape == cell.cell_mask.shape
    assert perim.sum() > 0
    assert perim.sum() <= cell.cell_mask.sum()


def test_compute_sept_mask_box_currently_calls_with_wrong_signature():
    cell = _cell()

    with pytest.raises(TypeError):
        cell.compute_sept_mask(2, "Box")


def test_compute_opensept_mask_isodata_currently_calls_wrong_signature():
    cell = _cell()

    with pytest.raises(TypeError):
        cell.compute_opensept_mask(2, "Isodata")


def test_compute_sept_mask_invalid_algorithm_returns_none(capsys):
    cell = _cell()

    assert cell.compute_sept_mask(2, "Missing") is None
    assert "valid algorithm" in capsys.readouterr().out


def test_compute_sept_box_draws_short_axis_inside_cell_mask():
    cell = _cell()

    sept = cell.compute_sept_box(2)

    assert sept.shape == (5, 5)
    assert sept[:, 2].sum() > 0
    assert np.all(sept <= cell.cell_mask)


def test_get_outline_points_handles_edges_and_interior():
    cell = _cell()
    data = np.ones((3, 3), dtype=int)

    outline = cell.get_outline_points(data)

    assert set(outline) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }


def test_compute_sept_box_fix_clamps_outline_box_to_mask_shape():
    cell = _cell()
    outline = [(0, 0), (2, 3), (4, 4)]

    assert cell.compute_sept_box_fix(outline, (5, 5)) == (0, 0, 4, 4)


def test_measure_fluor_handles_full_fraction_top_fraction_and_missing_roi():
    cell = _cell()
    fluor = np.array([[1, 2], [3, 4]], dtype=float)
    roi = np.array([[1, 0], [1, 1]], dtype=float)

    assert cell.measure_fluor(fluor, roi) == pytest.approx(3)
    assert cell.measure_fluor(fluor, roi, fraction=0.5) == pytest.approx(4)
    assert cell.measure_fluor(fluor, roi, fraction=0.1) == 0
    assert cell.measure_fluor(fluor, None) == 0


def test_compute_fluor_baseline_can_store_nan_without_background():
    cell = _cell()
    cell.box = (1, 1, 3, 3)
    mask = np.zeros((7, 7), dtype=int)
    mask[2:5, 2:5] = 1
    fluor = np.arange(49, dtype=float).reshape(7, 7)

    cell.compute_fluor_baseline(mask, fluor, margin=1)

    assert np.isnan(cell.stats["Baseline"])


def test_set_image_uses_zero_optional_channel_when_missing():
    cell = _cell()
    cell.params = {"find_septum": False, "find_openseptum": False}
    cell.perim_mask = np.ones((5, 5))
    cell.cyto_mask = np.ones((5, 5))
    cell.sept_mask = None
    fluor = np.ones((5, 5))

    cell.set_image(fluor, optional=None)

    assert cell.image.shape == (5, 35)
    assert cell.image[:, 10:15].sum() == 0
