import numpy as np

from napari_mAIcrobe.mAIcrobe.mask import mask_computation
from napari_mAIcrobe.mAIcrobe.segments import SegmentsManager


def test_segmentation_isodata(phase_example):
    """Test segmentation using isodata method."""
    # Test data
    phase_image = phase_example

    # Run isodata
    mask = mask_computation(phase_image, algorithm="Isodata")

    # Assert results
    assert mask is not None
    assert np.any(mask)  # Ensure that the mask is not empty

    # Run watershed
    pars = {
        "peak_min_distance_from_edge": 10,
        "peak_min_distance": 5,
        "peak_min_height": 5,
        "max_peaks": 100000,
    }
    seg_man = SegmentsManager()
    seg_man.compute_segments(pars, mask)
    labels = seg_man.labels

    # Assert labels
    assert labels is not None
    assert labels.max() > 0  # Should detect some cells
    assert len(np.unique(labels)) > 1  # Multiple cell labels


def test_segmentation_local_average(phase_example):
    """Test segmentation using local average method."""
    # Test data
    phase_image = phase_example

    # Run segmentation
    mask = mask_computation(phase_image, algorithm="Local Average")

    # Assert results
    assert mask is not None
    assert np.any(mask)  # Ensure that the mask is not empty

    # Run watershed
    pars = {
        "peak_min_distance_from_edge": 10,
        "peak_min_distance": 5,
        "peak_min_height": 5,
        "max_peaks": 100000,
    }
    seg_man = SegmentsManager()
    seg_man.compute_segments(pars, mask)
    labels = seg_man.labels

    # Assert labels
    assert labels is not None
    assert labels.max() > 0  # Should detect some cells
    assert len(np.unique(labels)) > 1  # Multiple cell labels


# TODO add tests for UNet segmentation, Stardist segmentation, and Cellpose segmentation
