"""
Module responsible for GUI to do label computation and channel alignment.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import napari

import os

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

from .mAIcrobe.mask import mask_alignment
from .mAIcrobe.segmentation import (
    batch_cellpose_segmentation,
    batch_classical_segmentation,
    batch_stardist_segmentation,
    batch_unet_segmentation,
    cellpose_segmentation,
    classical_segmentation,
    stardist_segmentation,
    unet_segmentation,
)

# force classification to happen on CPU to avoid CUDA problems
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Remove some extraneous log outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        self._baseimg_combo.changed.connect(self._on_baseimg_changed)

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

        # TIME LAPSE
        self._timelapse = CheckBox(label="Run analysis for all time points")

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
                self._timelapse,  # 23
                self._run_button,  # 24
            ],
            labels=True,
        )
        # Initialize visibility according to the current algorithm selection
        self._on_algorithm_changed(self._algorithm_combo.value)

        # Initialize visibility of timelapse checkbox according to number of time points in base image (only show if more than 1 time point)
        self._on_baseimg_changed(self._baseimg_combo.value)

    def _on_baseimg_changed(self, new_baseimg):
        """Toggle timelapse checkbox visibility according to number of time points in base image.

        Parameters
        ----------
        new_baseimg : napari.layers.Image
            The newly selected base image layer.
        """
        if new_baseimg is None:
            self._timelapse.visible = False
            return

        if len(new_baseimg.data.shape) == 3:
            self._timelapse.visible = True
        else:
            self._timelapse.visible = False

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

        _timelapse = self._timelapse.value and len(_baseimg.data.shape) == 3

        if _algorithm == "Unet":
            if _timelapse:
                mask, labels = batch_unet_segmentation(
                    _baseimg.data,
                    self._unetradio.value,
                    self._unetpretrained.value,
                    self._path2unet.value,
                    _binary_closing,
                    _binary_dilation,
                    _binary_fillholes,
                )
            else:
                mask, labels = unet_segmentation(
                    _baseimg.data,
                    self._unetradio.value,
                    self._unetpretrained.value,
                    self._path2unet.value,
                    _binary_closing,
                    _binary_dilation,
                    _binary_fillholes,
                )

        elif _algorithm == "StarDist":
            if _timelapse:
                mask, labels = batch_stardist_segmentation(
                    _baseimg.data,
                    self._stardistradio.value,
                    self._stardistpretrained.value,
                    self._path2stardist.value,
                )
            else:
                mask, labels = stardist_segmentation(
                    _baseimg.data,
                    self._stardistradio.value,
                    self._stardistpretrained.value,
                    self._path2stardist.value,
                )

        elif _algorithm == "CellPose cyto3":
            if _timelapse:
                mask, labels = batch_cellpose_segmentation(_baseimg.data)
            else:
                mask, labels = cellpose_segmentation(_baseimg.data)

        else:
            if _timelapse:
                mask, labels = batch_classical_segmentation(
                    _baseimg.data,
                    _algorithm,
                    _LAblocksize,
                    _LAoffset,
                    _binary_closing,
                    _binary_dilation,
                    _binary_fillholes,
                    _pars,
                )
            else:
                mask, labels = classical_segmentation(
                    _baseimg.data,
                    _algorithm,
                    _LAblocksize,
                    _LAoffset,
                    _binary_closing,
                    _binary_dilation,
                    _binary_fillholes,
                    _pars,
                )

        # add mask to viewer
        self._viewer.add_labels(mask, name="Mask")
        # add labelimg to viewer
        self._viewer.add_labels(labels, name="Labels")

        if _autoalign and (not _timelapse):
            aligned_fluor_1 = mask_alignment(mask, _fluor1.data)
            aligned_fluor_2 = mask_alignment(mask, _fluor2.data)

            self._viewer.layers[_fluor1.name].data = aligned_fluor_1
            self._viewer.layers[_fluor2.name].data = aligned_fluor_2

        elif _autoalign and _timelapse:
            for i in range(mask.shape[0]):
                aligned_fluor_1 = mask_alignment(
                    mask[i, :, :], _fluor1.data[i, :, :]
                )
                aligned_fluor_2 = mask_alignment(
                    mask[i, :, :], _fluor2.data[i, :, :]
                )

                self._viewer.layers[_fluor1.name].data[
                    i, :, :
                ] = aligned_fluor_1
                self._viewer.layers[_fluor2.name].data[
                    i, :, :
                ] = aligned_fluor_2
