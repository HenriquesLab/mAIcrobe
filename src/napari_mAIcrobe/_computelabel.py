"""
Module responsible for GUI to do label computation and channel alignment.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import napari

import os

import tensorflow as tf
from cellpose import models
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    PushButton,
    RadioButtons,
    SpinBox,
    create_widget,
)
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from stardist.models import StarDist2D

from .mAIcrobe.mask import mask_alignment, mask_computation
from .mAIcrobe.segments import SegmentsManager
from .mAIcrobe.unet import (
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


class compute_label(Container):
    """
    Widget for label computation and optional channel alignment.

    Allows selecting input images, choosing a mask algorithm (Isodata, Local
    Average, Unet, StarDist, CellPose cyto3), tuning parameters, and running
    segmentation. Adds "Mask" and "Labels" layers to the viewer; optionally
    aligns auxiliary channels to the mask and performs binary operations
    like dilation, erosion and fill holes.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The active napari viewer.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Build the UI and connect handlers.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The active napari viewer instance.
        """

        self._viewer = viewer

        # IMAGE INPUTS
        self._baseimg_combo = cast(
            ComboBox,
            create_widget(
                annotation="napari.layers.Image", label="Base Image"
            ),
        )
        self._fluor1_combo = cast(
            ComboBox,
            create_widget(annotation="napari.layers.Image", label="Fluor 1"),
        )
        self._fluor2_combo = cast(
            ComboBox,
            create_widget(annotation="napari.layers.Image", label="Fluor 2"),
        )

        self._closinginput = SpinBox(
            min=0, max=5, step=1, value=0, label="Binary Closing"
        )
        self._dilationinput = SpinBox(
            min=0, max=5, step=1, value=0, label="Binary Dilation"
        )
        self._fillholesinput = CheckBox(label="Fill Holes")
        self._autoaligninput = CheckBox(label="Auto Align")

        # MASK ALGORITHM
        self._algorithm_combo = cast(
            ComboBox,
            create_widget(
                options={
                    "choices": [
                        "Isodata",
                        "Local Average",
                        "Unet",
                        "StarDist",
                        "CellPose cyto3",
                    ]
                },
                label="Mask algorithm",
                value="Isodata",
            ),
        )
        self._algorithm_combo.changed.connect(self._on_algorithm_changed)

        self._titlemasklabel = Label(value="Parameters for Mask computation")
        self._titlemasklabel.native.setAlignment(Qt.AlignCenter)
        self._titlemasklabel.native.setStyleSheet(
            "background-color: rgb(037, 041, 049); border: 1px solid rgb(059, 068, 077);"
        )
        self._titlemasklabel.native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        self._placeholder = Label(value="...")
        self._placeholder.native.setAlignment(Qt.AlignCenter)
        self._placeholder.native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        self._blocksizeinput = SpinBox(
            min=0,
            max=1000,
            step=1,
            value=151,
            label="Blocksize",
            visible=False,
        )
        self._offsetinput = SpinBox(
            min=0, max=1, step=0.001, value=0.02, label="Offset", visible=False
        )

        # UNET MODEL TYPE
        self._unetradio = RadioButtons(
            choices=["Pretrained", "Custom"],
            label="Unet Model Type",
            value="Pretrained",
            visible=False,
        )
        self._unetradio.changed.connect(self._on_pretrainedunet_changed)
        self._unetpretrained = ComboBox(
            choices=[
                "Ph.C. S. pneumo",
                "WF FtsZ B. subtilis",
                "Unet S. aureus",
            ],
            label="Pretrained Unet Model",
            value="Ph.C. S. pneumo",
            visible=False,
        )
        self._path2unet = FileEdit(
            mode="r", label="Path to UnetModel", visible=False
        )

        # STARDIST MODEL
        self._stardistradio = RadioButtons(
            choices=["Pretrained", "Custom"],
            label="StarDist Model Type",
            value="Pretrained",
            visible=False,
        )
        self._stardistradio.changed.connect(
            self._on_pretrainedstardist_changed
        )
        self._stardistpretrained = ComboBox(
            choices=["StarDist S. aureus"],
            label="Pretrained StarDist Model",
            value="StarDist S. aureus",
            visible=False,
        )
        self._path2stardist = FileEdit(
            mode="d", label="Path to StarDistModel", visible=False
        )

        # WATERSHED ALGORITHM
        self._titlewatershedlabel = Label(
            value="Parameters for Watershed Algorithm"
        )
        self._titlewatershedlabel.native.setStyleSheet(
            "background-color: rgb(037, 041, 049); border: 1px solid rgb(059, 068, 077);"
        )
        self._titlewatershedlabel.native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        self._titlewatershedlabel.native.setAlignment(Qt.AlignCenter)
        self._peak_min_distance_from_edge = SpinBox(
            min=0,
            max=50,
            step=1,
            value=10,
            label="Peak Min Distance From Edge",
        )
        self._peak_min_distance = SpinBox(
            min=0, max=50, step=1, value=5, label="Peak Min Distance"
        )
        self._peak_min_height = SpinBox(
            min=0, max=50, step=1, value=5, label="Peak Min Height"
        )
        self._max_peaks = SpinBox(
            min=0, max=100000, step=100, value=100000, label="Max Peaks"
        )

        # RUN
        self._run_button = PushButton(label="Run")
        self._run_button.clicked.connect(self.compute)

        super().__init__(
            widgets=[
                self._baseimg_combo,  # 0
                self._fluor1_combo,  # 1
                self._fluor2_combo,  # 2
                self._closinginput,  # 3
                self._dilationinput,  # 4
                self._fillholesinput,  # 5
                self._autoaligninput,  # 6
                self._algorithm_combo,  # 7
                self._titlemasklabel,  # 8
                self._placeholder,  # 9
                self._blocksizeinput,  # 10
                self._offsetinput,  # 11
                self._unetradio,  # 12
                self._path2unet,  # 13
                self._unetpretrained,  # 14
                self._stardistradio,  # 15
                self._path2stardist,  # 16
                self._stardistpretrained,  # 17
                self._titlewatershedlabel,  # 18
                self._peak_min_distance_from_edge,  # 19
                self._peak_min_distance,  # 20
                self._peak_min_height,  # 21
                self._max_peaks,  # 22
                self._run_button,  # 23
            ],
            labels=True,
        )
        # Initialize visibility according to the current algorithm selection
        self._on_algorithm_changed(self._algorithm_combo.value)

    def _on_algorithm_changed(self, new_algorithm: str):
        """Toggle parameter widgets according to algorithm choice.

        Parameters
        ----------
        new_algorithm : str
            One of {"Isodata", "Local Average", "Unet", "StarDist",
            "CellPose cyto3"}.
        """

        # Mask post-processing controls
        show_basic_ops = new_algorithm in {"Isodata", "Local Average", "Unet"}
        self._closinginput.visible = show_basic_ops
        self._dilationinput.visible = show_basic_ops
        self._fillholesinput.visible = show_basic_ops

        # Mask parameter title and per-algorithm params
        self._titlemasklabel.visible = new_algorithm in {
            "Isodata",
            "Local Average",
            "Unet",
            "StarDist",
        }
        self._placeholder.visible = new_algorithm == "Isodata"
        self._blocksizeinput.visible = new_algorithm == "Local Average"
        self._offsetinput.visible = new_algorithm == "Local Average"

        # Unet: show radio + corresponding input
        is_unet = new_algorithm == "Unet"
        self._unetradio.visible = is_unet
        if is_unet:
            self._unetpretrained.visible = (
                self._unetradio.value == "Pretrained"
            )
            self._path2unet.visible = self._unetradio.value == "Custom"
        else:
            self._unetpretrained.visible = False
            self._path2unet.visible = False

        # StarDist: show radio + corresponding input
        is_stardist = new_algorithm == "StarDist"
        self._stardistradio.visible = is_stardist
        if is_stardist:
            self._stardistpretrained.visible = (
                self._stardistradio.value == "Pretrained"
            )
            self._path2stardist.visible = self._stardistradio.value == "Custom"
        else:
            self._stardistpretrained.visible = False
            self._path2stardist.visible = False

        # Watershed params only for Isodata/Local Average
        show_ws = new_algorithm in {"Isodata", "Local Average"}
        self._titlewatershedlabel.visible = show_ws
        self._peak_min_distance_from_edge.visible = show_ws
        self._peak_min_distance.visible = show_ws
        self._peak_min_height.visible = show_ws
        self._max_peaks.visible = show_ws

        return

    def _on_pretrainedunet_changed(self, new_value: str):
        """Toggle Unet model path/pretrained selection.

        Parameters
        ----------
        new_value : str
            One of {"Pretrained", "Custom"}.
        """
        # make sure unet is selected
        if self._algorithm_combo.value != "Unet":
            return

        if new_value == "Pretrained":
            self._unetpretrained.visible = True
            self._path2unet.visible = False
        else:
            self._unetpretrained.visible = False
            self._path2unet.visible = True

    def _on_pretrainedstardist_changed(self, new_value: str):
        """Toggle StarDist model path/pretrained selection.

        Parameters
        ----------
        new_value : str
            One of {"Pretrained", "Custom"}.
        """
        # make sure stardist is selected
        if self._algorithm_combo.value != "StarDist":
            return

        if new_value == "Pretrained":
            self._stardistpretrained.visible = True
            self._path2stardist.visible = False
        else:
            self._stardistpretrained.visible = False
            self._path2stardist.visible = True

    def compute(self):
        """Run mask/label computation, optional channel alignment and
        binary operations.

        Notes
        -----
        - Unet uses `computelabel_unet` imported from mAIcrobe.
        - StarDist uses the StarDist python package with a model
          directory selected via `_path2stardist`.
        - CellPose uses the CellPose python package to download to cache
          and subsequently use the `cyto3` model for inference.
        - Other algorithms use `mask_computation` imported from mAIcrobe
          + watershed via `SegmentsManager`.

        Side Effects
        ------------
        Adds "Mask" and "Labels" Layers to the viewer. If Auto Align is
        enabled, updates fluor channels with aligned images.
        """

        # TODO some code should be moved to mAIcrobe folder to isolate logic

        _algorithm = self._algorithm_combo.value

        _baseimg = self._baseimg_combo.value
        _fluor1 = self._fluor1_combo.value
        _fluor2 = self._fluor2_combo.value

        _binary_closing = self._closinginput.value
        _binary_dilation = self._dilationinput.value
        _binary_fillholes = self._fillholesinput.value
        _autoalign = self._autoaligninput.value

        _LAblocksize = self._blocksizeinput.value
        _LAoffset = self._offsetinput.value

        _pars = {
            "peak_min_distance_from_edge": self._peak_min_distance_from_edge.value,
            "peak_min_distance": self._peak_min_distance.value,
            "peak_min_height": self._peak_min_height.value,
            "max_peaks": self._max_peaks.value,
        }

        if _algorithm == "Unet":
            # if pretrained, check if model file exists in cache, if not download it
            if self._unetradio.value == "Pretrained":
                if self._unetpretrained.value == "Ph.C. S. pneumo":
                    model_filename = "UNet4strep_20250922.hdf5"
                elif self._unetpretrained.value == "WF FtsZ B. subtilis":
                    model_filename = "UNet4bsub_20250922.hdf5"
                elif self._unetpretrained.value == "Unet S. aureus":
                    model_filename = "UNet4staph_20250922.hdf5"

                _path2unet = download_github_file_raw(
                    "SegmentationModels/" + model_filename,
                    __cachemodel_folder__,
                )
            else:
                _path2unet = self._path2unet.value

            mask, labels = computelabel_unet(
                path2model=_path2unet,
                base_image=_baseimg.data,
                closing=_binary_closing,
                dilation=_binary_dilation,
                fillholes=_binary_fillholes,
            )

        elif _algorithm == "StarDist":
            # if pretrained, check if model dir exists in cache, if not download it
            # be careful, stardist needs a folder with config.json, weights_best.h5 and thresholds.json not a single model file like U-Net
            if self._stardistradio.value == "Pretrained":
                if self._stardistpretrained.value == "StarDist S. aureus":
                    model_dirname = os.path.join(
                        "SegmentationModels", "StarDistSaureus_20250922"
                    )
                    if not os.path.exists(
                        os.path.join(__cachemodel_folder__, model_dirname)
                    ):
                        os.makedirs(
                            os.path.join(__cachemodel_folder__, model_dirname)
                        )
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

                    _path2stardist = os.path.join(
                        __cachemodel_folder__, model_dirname
                    )
            else:
                _path2stardist = self._path2stardist.value

            basedir, name = os.path.split(_path2stardist)
            model = StarDist2D(None, name=name, basedir=basedir)

            labels, _ = model.predict_instances(
                normalizePercentile(_baseimg.data)
            )
            mask = labels > 0
            mask = mask.astype("uint16")

        elif _algorithm == "CellPose cyto3":
            model = models.Cellpose(gpu=True, model_type="cyto3")
            labels, flows, styles, diams = model.eval(
                _baseimg.data, diameter=None
            )
            mask = labels > 0
            mask = mask.astype("uint16")

        else:
            mask = mask_computation(
                base_image=_baseimg.data,
                algorithm=_algorithm,
                blocksize=_LAblocksize,
                offset=_LAoffset,
                closing=_binary_closing,
                dilation=_binary_dilation,
                fillholes=_binary_fillholes,
            )

            seg_man = SegmentsManager()
            seg_man.compute_segments(_pars, mask)

            labels = seg_man.labels

        # add mask to viewer
        self._viewer.add_labels(mask, name="Mask")
        # add labelimg to viewer
        self._viewer.add_labels(labels, name="Labels")

        if _autoalign:
            aligned_fluor_1 = mask_alignment(mask, _fluor1.data)
            aligned_fluor_2 = mask_alignment(mask, _fluor2.data)

            self._viewer.layers[_fluor1.name].data = aligned_fluor_1
            self._viewer.layers[_fluor2.name].data = aligned_fluor_2
