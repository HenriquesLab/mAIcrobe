import numpy as np

from napari_mAIcrobe.mAIcrobe.segments import SegmentsManager


def _params(**overrides):
    params = {
        "peak_min_distance_from_edge": 1,
        "peak_min_distance": 2,
        "peak_min_height": 1,
        "max_peaks": 10,
    }
    params.update(overrides)
    return params


def test_clear_all_resets_computed_state():
    manager = SegmentsManager()
    manager.features = np.ones((2, 2))
    manager.labels = np.ones((2, 2))
    manager.base_w_features = np.ones((2, 2))
    manager.fluor_w_features = np.ones((2, 2))

    manager.clear_all()

    assert manager.features is None
    assert manager.labels is None
    assert manager.base_w_features is None
    assert manager.fluor_w_features is None


def test_compute_distance_peaks_filters_by_margin_and_sorts_low_to_high():
    mask = np.zeros((12, 12), dtype=int)
    mask[2:5, 2:5] = 1
    mask[6:11, 6:11] = 1

    peaks = SegmentsManager.compute_distance_peaks(mask, _params())

    assert peaks[0] == (3, 3)
    assert peaks[-1] == (8, 8)


def test_compute_features_normalizes_minimum_margin_in_params():
    manager = SegmentsManager()
    params = _params(peak_min_distance_from_edge=0)
    mask = np.zeros((9, 9), dtype=int)
    mask[2:7, 2:7] = 1

    manager.compute_features(params, mask)

    assert params["peak_min_distance_from_edge"] == 1
    assert manager.features.shape == mask.shape
    assert manager.features.max() > 0


def test_compute_segments_populates_feature_overlay_and_labels():
    manager = SegmentsManager()
    mask = np.zeros((15, 15), dtype=int)
    mask[2:6, 2:6] = 1
    mask[9:13, 9:13] = 1

    manager.compute_segments(_params(), mask)

    assert manager.features is not None
    assert manager.base_w_features is not None
    assert manager.labels is not None
    assert set(np.unique(manager.labels)) >= {0, 1, 2}
