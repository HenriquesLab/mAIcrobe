"""
Module responsible for instance segmentation.
Contains ONLY high level functions related to the generation of single cell instances.
Its made to be the entry point to the GUI for the segmentation process, and to be used by the GUI to generate the labels and masks that will be used for the visualization and later processing of the data.
"""

import os

import numpy as np
import tensorflow as tf
from cellpose import models
from stardist.models import StarDist2D

from .mask import mask_computation
from .segments import SegmentsManager
from .unet import (
    computelabel_unet,
    download_github_file_raw,
    normalizePercentile,
)

# force classification to happen on CPU to avoid CUDA problems
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Remove some extraneous log outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


tf.config.set_visible_devices([], "GPU")


__home_folder__ = os.path.expanduser("~")
__cachemodel_folder__ = os.path.join(__home_folder__, ".maicrobecache")
if not os.path.exists(__cachemodel_folder__):
    os.makedirs(__cachemodel_folder__)
if not os.path.exists(
    os.path.join(__cachemodel_folder__, "SegmentationModels")
):
    os.makedirs(os.path.join(__cachemodel_folder__, "SegmentationModels"))


def unet_segmentation(
    img: np.ndarray,
    pretrained: bool,
    pretrained_name: str,
    path2model: str,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
) -> tuple[np.ndarray, np.ndarray]:

    if len(img.shape) == 3:
        img = img[0, :, :]

    # if pretrained, check if model file exists in cache, if not download it
    if pretrained == "Pretrained":
        if pretrained_name == "Ph.C. S. pneumo":
            model_filename = "UNet4strep_20250922.hdf5"
        elif pretrained_name == "WF FtsZ B. subtilis":
            model_filename = "UNet4bsub_20250922.hdf5"
        elif pretrained_name == "Unet S. aureus":
            model_filename = "UNet4staph_20250922.hdf5"

        _path2unet = download_github_file_raw(
            "SegmentationModels/" + model_filename,
            __cachemodel_folder__,
        )
    else:
        _path2unet = path2model

    mask, labels = computelabel_unet(
        path2model=_path2unet,
        base_image=img,
        closing=binary_closing,
        dilation=binary_dilation,
        fillholes=binary_fillholes,
    )

    return mask, labels


def batch_unet_segmentation(
    img: np.ndarray,
    pretrained: bool,
    pretrained_name: str,
    path2model: str,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
) -> tuple[np.ndarray, np.ndarray]:

    for i in range(img.shape[0]):

        mask, label = unet_segmentation(
            img[i, :, :],
            pretrained,
            pretrained_name,
            path2model,
            binary_closing,
            binary_dilation,
            binary_fillholes,
        )

        if i == 0:
            masks = np.zeros((img.shape[0], *mask.shape), dtype=mask.dtype)
            labels = np.zeros((img.shape[0], *label.shape), dtype=label.dtype)
        masks[i] = mask
        labels[i] = label

    return masks, labels


def stardist_segmentation(
    img: np.ndarray, pretrained: bool, pretrained_name: str, path2model: str
) -> tuple[np.ndarray, np.ndarray]:

    if len(img.shape) == 3:
        img = img[0, :, :]

    # if pretrained, check if model dir exists in cache, if not download it
    # be careful, stardist needs a folder with config.json, weights_best.h5 and thresholds.json not a single model file like U-Net
    if pretrained == "Pretrained":
        if pretrained_name == "StarDist S. aureus":
            model_dirname = os.path.join(
                "SegmentationModels", "StarDistSaureus_20250922"
            )
            if not os.path.exists(
                os.path.join(__cachemodel_folder__, model_dirname)
            ):
                os.makedirs(os.path.join(__cachemodel_folder__, model_dirname))
            # download files if they don't exist
            download_github_file_raw(
                "SegmentationModels"
                + "/"
                + "StarDistSaureus_20250922"
                + "/"
                + "config.json",
                __cachemodel_folder__,
            )
            download_github_file_raw(
                "SegmentationModels"
                + "/"
                + "StarDistSaureus_20250922"
                + "/"
                + "weights_best.h5",
                __cachemodel_folder__,
            )
            download_github_file_raw(
                "SegmentationModels"
                + "/"
                + "StarDistSaureus_20250922"
                + "/"
                + "thresholds.json",
                __cachemodel_folder__,
            )

            _path2stardist = os.path.join(__cachemodel_folder__, model_dirname)
    else:
        _path2stardist = path2model

    basedir, name = os.path.split(_path2stardist)
    model = StarDist2D(None, name=name, basedir=basedir)

    labels, _ = model.predict_instances(normalizePercentile(img))
    mask = labels > 0
    mask = mask.astype("uint16")

    return mask, labels


def batch_stardist_segmentation(
    img: np.ndarray, pretrained: bool, pretrained_name: str, path2model: str
) -> tuple[np.ndarray, np.ndarray]:

    for i in range(img.shape[0]):

        mask, label = stardist_segmentation(
            img[i, :, :], pretrained, pretrained_name, path2model
        )

        if i == 0:
            masks = np.zeros((img.shape[0], *mask.shape), dtype=mask.dtype)
            labels = np.zeros((img.shape[0], *label.shape), dtype=label.dtype)
        masks[i] = mask
        labels[i] = label

    return masks, labels


def cellpose_segmentation(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    if len(img.shape) == 3:
        img = img[0, :, :]

    model = models.Cellpose(gpu=True, model_type="cyto3")
    labels, flows, styles, diams = model.eval(img, diameter=None)
    mask = labels > 0
    mask = mask.astype("uint16")

    return mask, labels


def batch_cellpose_segmentation(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    for i in range(img.shape[0]):

        mask, label = cellpose_segmentation(img[i, :, :])

        if i == 0:
            masks = np.zeros((img.shape[0], *mask.shape), dtype=mask.dtype)
            labels = np.zeros((img.shape[0], *label.shape), dtype=label.dtype)
        masks[i] = mask
        labels[i] = label

    return masks, labels


def classical_segmentation(
    img: np.ndarray,
    algorithm: str,
    LAblocksize: int,
    LAoffset: int,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
    pars: dict,
) -> tuple[np.ndarray, np.ndarray]:

    if len(img.shape) == 3:
        img = img[0, :, :]

    mask = mask_computation(
        base_image=img,
        algorithm=algorithm,
        blocksize=LAblocksize,
        offset=LAoffset,
        closing=binary_closing,
        dilation=binary_dilation,
        fillholes=binary_fillholes,
    )

    seg_man = SegmentsManager()
    seg_man.compute_segments(pars, mask)

    labels = seg_man.labels

    return mask, labels


def batch_classical_segmentation(
    img: np.ndarray,
    algorithm: str,
    LAblocksize: int,
    LAoffset: int,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
    pars: dict,
) -> tuple[np.ndarray, np.ndarray]:

    for i in range(img.shape[0]):

        mask, label = classical_segmentation(
            img[i, :, :],
            algorithm,
            LAblocksize,
            LAoffset,
            binary_closing,
            binary_dilation,
            binary_fillholes,
            pars,
        )

        if i == 0:
            masks = np.zeros((img.shape[0], *mask.shape), dtype=mask.dtype)
            labels = np.zeros((img.shape[0], *label.shape), dtype=label.dtype)
        masks[i] = mask
        labels[i] = label

    return masks, labels
