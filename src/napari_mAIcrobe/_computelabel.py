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
    SpinBox,
    create_widget,
)
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from stardist.models import StarDist2D

from .mAIcrobe.mask import mask_alignment, mask_computation
from .mAIcrobe.segments import SegmentsManager
from .mAIcrobe.unet import computelabel_unet, normalizePercentile

# force classification to happen on CPU to avoid CUDA problems
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Remove some extraneous log outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


tf.config.set_visible_devices([], "GPU")


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
        self._path2unet = FileEdit(
            mode="r", label="Path to UnetModel", visible=False
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
                self._baseimg_combo,
                self._fluor1_combo,
                self._fluor2_combo,
                self._closinginput,
                self._dilationinput,
                self._fillholesinput,
                self._autoaligninput,
                self._algorithm_combo,
                self._titlemasklabel,
                self._placeholder,
                self._blocksizeinput,
                self._offsetinput,
                self._path2unet,
                self._path2stardist,
                self._titlewatershedlabel,
                self._peak_min_distance_from_edge,
                self._peak_min_distance,
                self._peak_min_height,
                self._max_peaks,
                self._run_button,
            ],
            labels=True,
        )

    def _on_algorithm_changed(self, new_algorithm: str):
        """Toggle parameter widgets according to algorithm choice.

        #TODO make this cleaner.
        #TODO watershed parameters should be hidden for Unet, StarDist
        # and CellPose.
        #TODO binary operations should be hidden for StarDist and
        # CellPose.

        Parameters
        ----------
        new_algorithm : str
            One of {"Isodata", "Local Average", "Unet", "StarDist",
            "CellPose cyto3"}.
        """

        if new_algorithm == "Isodata":
            self[9].visible = True
            self[10].visible = False
            self[11].visible = False
            self[12].visible = False
            self[13].visible = False
        elif new_algorithm == "Local Average":
            self[9].visible = False
            self[10].visible = True
            self[11].visible = True
            self[12].visible = False
            self[13].visible = False
        elif new_algorithm == "Unet":
            self[9].visible = False
            self[10].visible = False
            self[11].visible = False
            self[12].visible = True
            self[13].visible = False
        elif new_algorithm == "StarDist":
            self[9].visible = False
            self[10].visible = False
            self[11].visible = False
            self[12].visible = False
            self[13].visible = True
        elif new_algorithm == "CellPose cyto3":
            self[9].visible = False
            self[10].visible = False
            self[11].visible = False
            self[12].visible = False
            self[13].visible = False

        return

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

        if _algorithm == "Unet":

            mask, labels = computelabel_unet(
                path2model=self._path2unet.value,
                base_image=_baseimg.data,
                closing=_binary_closing,
                dilation=_binary_dilation,
                fillholes=_binary_fillholes,
            )

        elif _algorithm == "StarDist":

            basedir, name = os.path.split(self._path2stardist.value)
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
