import numpy as np
import pytest

from napari_mAIcrobe.mAIcrobe.mask import mask_alignment, mask_computation


def test_mask_computation_local_average_even_blocksize_is_supported():
    image = np.ones((21, 21), dtype=float)
    image[7:14, 7:14] = 0

    mask = mask_computation(
        image,
        algorithm="Local Average",
        blocksize=10,
        closing=0,
        dilation=0,
        fillholes=False,
    )

    assert mask.shape == image.shape
    assert mask.dtype.kind in {"b", "i", "u"}
    assert mask[10, 10] == 1


def test_mask_computation_can_fill_holes():
    image = np.ones((20, 20), dtype=float)
    image[4:16, 4:16] = 0
    image[8:12, 8:12] = 1

    unfilled = mask_computation(image, closing=0, fillholes=False)
    filled = mask_computation(image, closing=0, fillholes=True)

    assert unfilled[10, 10] == 0
    assert filled[10, 10]


def test_mask_computation_invalid_algorithm_raises_unboundlocalerror_today():
    with pytest.raises(UnboundLocalError):
        mask_computation(np.zeros((4, 4)), algorithm="Missing")


def test_mask_alignment_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        mask_alignment(np.zeros((5, 5)), np.zeros((4, 5)))


def test_mask_alignment_preserves_shape_and_intensity_range():
    mask = np.zeros((12, 12), dtype=float)
    fluor = np.zeros_like(mask)
    mask[3:7, 4:8] = 1
    fluor[3:7, 4:8] = 2

    aligned = mask_alignment(mask, fluor)

    assert aligned.shape == fluor.shape
    assert aligned.max() <= 2.0
    assert aligned.min() >= 0.0
