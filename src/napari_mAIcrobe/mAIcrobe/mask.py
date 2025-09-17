"""
Module that contains the logic for mask computation
"""

import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_isodata, threshold_local
from skimage.morphology import binary_closing, binary_dilation
from skimage.transform import EuclideanTransform, warp


def mask_computation(
    base_image: np.ndarray,
    algorithm: str = "Isodata",
    blocksize: int = 151,
    offset: float = 0.02,
    closing: int = 1,
    dilation: int = 0,
    fillholes: bool = False,
):
    """Compute a binary mask from an image using Isodata or Local Average
    thresholding. Optionally, apply binary closing, dilation and fill
    holes operations to the binary mask.

    Parameters
    ----------
    base_image : numpy.ndarray
        Input 2D image.
    algorithm : {"Isodata", "Local Average"}, optional
        Thresholding strategy, by default "Isodata".
    blocksize : int, optional
        Neighborhood size for Local Average; must be odd, by default 151.
    offset : float, optional
        Offset for Local Average threshold, by default 0.02.
    closing : int, optional
        Size of binary closing kernel; if >0 applied, by default 1.
    dilation : int, optional
        Number of dilation iterations, by default 0.
    fillholes : bool, optional
        Whether to fill holes after morphology, by default False.

    Returns
    -------
    numpy.ndarray
        Binary mask with non-zero values inside objects.
    """

    # Binarization
    if algorithm == "Isodata":
        mask = base_image > threshold_isodata(base_image)
        mask = mask.astype(int)
        mask = 1 - mask
    elif algorithm == "Local Average":
        if blocksize % 2 == 0:
            blocksize += 1
        mask = base_image > threshold_local(
            base_image, block_size=blocksize, method="gaussian", offset=offset
        )
        mask = mask.astype(int)
        mask = 1 - mask

    # remove spots (both white and dark)
    if closing > 0:
        # removes small white spots and then small dark spots
        closing_matrix = np.ones((int(closing), int(closing)))
        mask = binary_closing(mask, closing_matrix)
        mask = 1 - binary_closing(1 - mask, closing_matrix)

    # dilation
    for f in range(dilation):
        mask = binary_dilation(mask, np.ones((3, 3)))

    # binary fill holes
    if fillholes:
        mask = ndimage.binary_fill_holes(mask)

    return mask


def mask_alignment(mask: np.ndarray, fluor_image: np.ndarray):
    """Align a fluorescence image to a mask via phase correlation.

    Uses FFT-based cross-correlation to estimate translation and
    translates the fluorescence image accordingly.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask image.
    fluor_image : numpy.ndarray
        Fluorescence image to align.

    Returns
    -------
    numpy.ndarray
        Aligned fluorescence image (same shape as input).
    """

    corr = signal.fftconvolve(mask, fluor_image[::-1, ::-1])
    deviation = np.unravel_index(np.argmax(corr), corr.shape)
    cm = ndimage.center_of_mass(np.ones(corr.shape))

    dy, dx = np.subtract(deviation, cm)
    matrix = EuclideanTransform(rotation=0, translation=(dx, dy))

    aligned_fluor = warp(
        fluor_image, matrix.inverse, preserve_range=True
    )  # TODO triple check if fluor intensity values stay the same

    return aligned_fluor
