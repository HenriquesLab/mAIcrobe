"""
UNet-based segmentation utilities.

Includes percentile normalization helpers (adapted from ZeroCostDL4Mic
UNet notebook), tiled prediction, and label computation with
watershed.
"""

import os
from math import ceil

import numpy as np
import requests
from scipy import ndimage
from scipy.ndimage import label as lbl
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)
from skimage.segmentation import watershed
from tensorflow.keras.models import load_model


############################################################################
## THIS FUNCTION ARE COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def normalizePercentile(
    x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32
):
    """Percentile-based image normalization.

    This function is adapted from Martin Weigert and copied from the
    ZeroCostDL4Mic UNet notebook.

    Parameters
    ----------
    x : numpy.ndarray
        Input image.
    pmin : float, optional
        Lower percentile, by default 1.
    pmax : float, optional
        Upper percentile, by default 99.8.
    axis : int or tuple of int, optional
        Percentile computation axis, by default None.
    clip : bool, optional
        Clip output to [0, 1], by default False.
    eps : float, optional
        Epsilon to avoid division by zero, by default 1e-20.
    dtype : numpy.dtype, optional
        Output dtype, by default numpy.float32.

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


############################################################################
## THIS FUNCTION ARE COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """Normalize by explicit min/max values.

    This function is adapted from Martin Weigert and copied from the
    ZeroCostDL4Mic UNet notebook.

    Parameters
    ----------
    x : numpy.ndarray
        Input image.
    mi : float or numpy.ndarray
        Minimum value(s).
    ma : float or numpy.ndarray
        Maximum value(s).
    clip : bool, optional
        Clip output to [0, 1], by default False.
    eps : float, optional
        Epsilon to avoid division by zero, by default 1e-20.
    dtype : numpy.dtype, optional
        Output dtype, by default numpy.float32.

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


############################################################################
## THIS FUNCTION IS COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def predict_as_tiles(img, model):
    """Run tiled prediction with a Keras model over a 2D image.

    Pads the image to at least the model's input patch size, iterates
    over tiles, and stitches the argmax class predictions.

    THIS FUNCTION IS COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER
    NOTEBOOK.

    Parameters
    ----------
    img : numpy.ndarray
        2D input image.
    model : keras.Model
        Loaded Keras model with classification output per pixel.

    Returns
    -------
    numpy.ndarray
        2D array of predicted class indices.
    """

    # Read the data in and normalize
    Image_raw = normalizePercentile(img)

    # Get the patch size from the input layer of the model
    patch_size = model.layers[0].output_shape[0][1:3]

    # Pad the image with zeros if any of its dimensions is smaller than the patch size
    if (
        Image_raw.shape[0] < patch_size[0]
        or Image_raw.shape[1] < patch_size[1]
    ):
        Image = np.zeros(
            (
                max(Image_raw.shape[0], patch_size[0]),
                max(Image_raw.shape[1], patch_size[1]),
            )
        )
        Image[0 : Image_raw.shape[0], 0 : Image_raw.shape[1]] = Image_raw
    else:
        Image = Image_raw

    # Calculate the number of patches in each dimension
    n_patch_in_width = ceil(Image.shape[0] / patch_size[0])
    n_patch_in_height = ceil(Image.shape[1] / patch_size[1])

    prediction = np.zeros(Image.shape, dtype="uint8")

    for x in range(n_patch_in_width):
        for y in range(n_patch_in_height):
            xi = patch_size[0] * x
            yi = patch_size[1] * y

            # If the patch exceeds the edge of the image shift it back
            if xi + patch_size[0] >= Image.shape[0]:
                xi = Image.shape[0] - patch_size[0]

            if yi + patch_size[1] >= Image.shape[1]:
                yi = Image.shape[1] - patch_size[1]

            # Extract and reshape the patch
            patch = Image[xi : xi + patch_size[0], yi : yi + patch_size[1]]
            patch = np.reshape(patch, patch.shape + (1,))
            patch = np.reshape(patch, (1,) + patch.shape)

            # Get the prediction from the patch and paste it in the prediction in the right place
            predicted_patch = model.predict(patch, batch_size=1)

            prediction[xi : xi + patch_size[0], yi : yi + patch_size[1]] = (
                np.argmax(np.squeeze(predicted_patch), axis=-1)
            ).astype(np.uint8)

    return prediction[0 : Image_raw.shape[0], 0 : Image_raw.shape[1]]


def computelabel_unet(path2model, base_image, closing, dilation, fillholes):
    """Compute mask and labels using a UNet model and watershed. The U-Net
    model outputs an image with 3 classes: background, edges, insides.
    Background = 0, edges = 1, insides = 2. A binary mask is created by
    the binary union of the edges and insides. To generate the final
    label image, we use the insides as markers and run a watershed on
    the inverse of the binary mask. Optionally, this function applies
    some morphological operations to clean up the binary mask.
    Specifically closing, dilation, and hole filling.

    Parameters
    ----------
    path2model : str or os.PathLike
        Path to a saved Keras model.
    base_image : numpy.ndarray
        Input 2D image to segment.
    closing : int
        Size of binary closing kernel; if >0 applied to remove small spots.
    dilation : int
        Number of binary dilation iterations.
    fillholes : bool
        Whether to binary fill holes after morphology.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (mask, labels) binary mask and integer labels.
    """

    model = load_model(path2model)
    prediction = predict_as_tiles(base_image, model)

    mask = prediction > 0

    mask = binary_opening(mask, np.ones((3, 3)))

    # edges = prediction==1
    insides = prediction == 2
    for _ in range(0):  # TODO
        insides = binary_erosion(insides)
    insides = insides.astype(np.uint16)
    insides, _ = lbl(insides)

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

    labels = watershed(~mask, markers=insides, mask=mask)

    return mask, labels


def download_github_file_raw(filename, cachepath, branch="main"):
    """Download a file from the mAIcrobe GitHub repository.
    If the file already exists at cachepath, does nothing.
    Used to download pretrained models.
    Parameters
    ----------
    filename : str
        Name of the file to download (e.g. 'model.h5').
    cachepath : str or os.PathLike
        Path where to save the downloaded file.
    branch : str, optional
        GitHub branch to download from, by default 'main'.
    """

    # substitute / in filename with os.path.join
    if "/" in filename:
        filename_path = os.path.join(*filename.split("/"))

    if os.path.exists(os.path.join(cachepath, filename_path)):
        return os.path.join(cachepath, filename_path)

    url = f"https://raw.githubusercontent.com/HenriquesLab/mAIcrobe/{branch}/docs/{filename}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(os.path.join(cachepath, filename_path), "wb") as f:
        f.write(r.content)
    return os.path.join(cachepath, filename_path)
