from types import SimpleNamespace

import numpy as np
import pytest

from napari_mAIcrobe.mAIcrobe.cells import CellManager


def _params(**overrides):
    params = {
        "pixel_size": 1.0,
        "inner_mask_thickness": 4,
        "septum_algorithm": "Isodata",
        "baseline_margin": 30,
        "find_septum": False,
        "find_openseptum": False,
        "classify_cell_cycle": False,
        "model": "S.aureus Membrane Epi",
        "custom_model_path": "",
        "custom_model_input": "Membrane",
        "custom_model_maxsize": 50,
        "generate_report": False,
        "report_path": "",
        "cell_averager": False,
        "coloc": False,
    }
    params.update(overrides)
    return params


def test_model_requires_dna_for_prebuilt_and_custom_models():
    manager = CellManager(np.zeros((2, 2)), np.zeros((2, 2)), None, _params())
    assert manager._model_requires_dna() is False

    manager.params["model"] = "S.aureus DNA Epi"
    assert manager._model_requires_dna() is True

    manager.params["model"] = "custom"
    manager.params["custom_model_input"] = "Membrane+DNA"
    assert manager._model_requires_dna() is True


def test_compute_dna_threshold_returns_nan_without_signal():
    labels = np.ones((3, 3), dtype=int)

    assert np.isnan(CellManager._compute_dna_threshold(labels, None))
    assert np.isnan(CellManager._compute_dna_threshold(labels, np.zeros((3, 3))))


def test_frame_data_returns_2d_or_requested_stack_frame():
    labels = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    fluor = labels + 10
    optional = labels + 20
    manager = CellManager(labels, fluor, optional, _params())

    frame_labels, frame_fluor, frame_optional = manager._frame_data(1)

    np.testing.assert_array_equal(frame_labels, np.ones((2, 2)))
    np.testing.assert_array_equal(frame_fluor, np.full((2, 2), 11))
    np.testing.assert_array_equal(frame_optional, np.full((2, 2), 21))


def test_rows_to_properties_converts_lists_to_arrays():
    properties = CellManager._rows_to_properties({"label": [1, 2], "Area": [3, 4]})

    assert all(isinstance(value, np.ndarray) for value in properties.values())
    np.testing.assert_array_equal(properties["label"], np.array([1, 2]))


def test_compute_cell_properties_rejects_mismatched_shapes():
    manager = CellManager(
        np.zeros((3, 3)),
        np.zeros((4, 3)),
        None,
        _params(),
    )

    with pytest.raises(ValueError, match="same shape"):
        manager.compute_cell_properties()


def test_compute_cell_properties_rejects_missing_required_dna():
    manager = CellManager(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        None,
        _params(
            classify_cell_cycle=True,
            model="S.aureus DNA Epi",
        ),
    )

    with pytest.raises(ValueError, match="requires DNA image"):
        manager.compute_cell_properties()


def test_calculate_dna_ratio_uses_cell_box_and_mask():
    cell = SimpleNamespace(
        box=(1, 1, 3, 3),
        cell_mask=np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
            ]
        ),
    )
    dna = np.zeros((5, 5), dtype=float)
    dna[1:4, 1:4] = np.array(
        [
            [2, 0, 5],
            [3, 9, 1],
            [0, 4, 6],
        ]
    )

    ratio = CellManager.calculate_DNARatio(cell, dna, thresh=2.5)

    assert ratio == pytest.approx(3 / 6)


def test_calculate_dna_ratio_returns_nan_without_dna_or_threshold():
    cell = SimpleNamespace(box=(0, 0, 1, 1), cell_mask=np.ones((2, 2)))

    assert np.isnan(CellManager.calculate_DNARatio(cell, None, thresh=1))
    assert np.isnan(
        CellManager.calculate_DNARatio(cell, np.ones((2, 2)), thresh=np.nan)
    )
