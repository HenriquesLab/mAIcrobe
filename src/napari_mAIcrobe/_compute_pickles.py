"""
Module to save annotated cell data as pickles to be used as
training data for the classification model.
"""

import pickle
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import napari

import os

import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    RadioButtons,
    create_widget,
)
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from skimage.transform import resize
from skimage.util import img_as_float


class compute_pickles(Container):
    """
    Widget to export per-cell crops and class targets as pickle files for model training.

    This widget expects:
    - A Labels layer with segmented cells.
    - A Points layer whose name is a positive integer (the class id).
      Points should lie inside labeled regions and one point per cell is considered.
    - One or two fluorescence image layers used to extract the cell crops.

    For each point, the underlying label is identified, the cell bounding box (with an
    extra margin) is computed, and fluorescence crops are:
    - Masked by the cell label region
    - Padded to a square shape
    - Resized to 100×100 pixels
    - Concatenated side-by-side if two channels are selected

    Two pickles are saved:
    - Class_<id>_source.p: list of ndarray crops (float64; shape 100×100 or 100×200)
    - Class_<id>_target.p: list of integers (class id), same length as source

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        Active napari viewer instance.

    Attributes
    ----------
    _viewer : napari.viewer.Viewer
        Stored viewer.
    _label_combo : magicgui.widgets.ComboBox
        Selector for a Labels layer.
    _points_combo : magicgui.widgets.ComboBox
        Selector for a Points layer (its name defines the class id).
    _channel_radio : magicgui.widgets.RadioButtons
        Selector for one or two channels.
    channelone_combo : magicgui.widgets.ComboBox
        Selector for the first Image layer.
    channeltwo_combo : magicgui.widgets.ComboBox
        Selector for the second Image layer (hidden when one channel is selected).
    _path2save : magicgui.widgets.FileEdit
        Directory where pickle files are saved.
    _run_button : magicgui.widgets.PushButton
        Triggers the export.
    box_margin : int
        Extra pixels added around the bounding box before cropping (default: 5).

    Notes
    -----
    - Points outside labeled regions or duplicated labels are skipped.
    - The Points layer name must be a positive integer (e.g., "1", "2", ...).
    - Crops are rescaled to [0, 1] using skimage.exposure.rescale_intensity.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Build the widget UI and connect signals.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            Active napari viewer instance.

        Notes
        -----
        The widget contains:
        - Layer selectors for labels, points, and one/two image channels
        - A folder selector to choose the output directory
        - A button to run the export
        """

        self._viewer = viewer

        self._label_combo = cast(
            ComboBox,
            create_widget(
                annotation="napari.layers.Labels", label="Labels layer"
            ),
        )

        self._points_combo = cast(
            ComboBox,
            create_widget(
                annotation="napari.layers.Points", label="Points layer"
            ),
        )

        self._channel_radio = RadioButtons(
            choices=["One Channel", "Two Channels"],
            label="Number of channels",
            value="One Channel",
        )
        self._channel_radio.changed.connect(self._on_channel_change)

        self.channelone_combo = cast(
            ComboBox,
            create_widget(annotation="napari.layers.Image", label="Channel 1"),
        )
        self.channeltwo_combo = cast(
            ComboBox,
            create_widget(annotation="napari.layers.Image", label="Channel 2"),
        )
        self.channeltwo_combo.visible = False

        self._path2save = FileEdit(
            value="", mode="d", label="Path to save pickles"
        )

        self._run_button = PushButton(label="Save Pickle")
        self._run_button.clicked.connect(self._on_run)

        self.box_margin = 5  # pixels around the bounding box

        super().__init__(
            widgets=[
                self._label_combo,
                self._points_combo,
                self._channel_radio,
                self.channelone_combo,
                self.channeltwo_combo,
                self._path2save,
                self._run_button,
            ],
            labels=True,
        )

    def _on_channel_change(self):
        """
        Toggle visibility of the second channel selector.

        Notes
        -----
        Shows `channeltwo_combo` when "Two Channels" is selected; hides it otherwise.

        Returns
        -------
        None
        """
        if self._channel_radio.value == "One Channel":
            self.channeltwo_combo.visible = False
        else:
            self.channeltwo_combo.visible = True

    def _on_run(self):
        """
        Export per-cell crops and class ids to pickle files.

        1. Validate inputs (layers, output path, points layer name as class id).
        2. For each point:
           - Retrieve label id; skip if 0 or already assigned.
           - Compute bounding box with `box_margin`.
           - Mask fluorescence crops by the label region.
           - Pad to square, then resize to 100×100.
           - Concatenate channels horizontally if two channels.
        3. Save:
           - `Class_<classid>_source.p` with list of crops.
           - `Class_<classid>_target.p` with list of class ids.

        Requirements
        ------------
        - Labels layer (ndarray)
        - Points layer (ndarray) named as a positive integer class id
        - One or two Image layers
        - Valid output directory

        Returns
        -------
        None
        """

        label_layer = self._label_combo.value
        points_layer = self._points_combo.value
        path2save = self._path2save.value
        maxrow, maxcol = label_layer.data.shape

        if not os.path.exists(path2save):
            os.makedirs(path2save)

        if label_layer is None or points_layer is None:
            print("Please select both a labels layer and a points layer")
            return

        if not isinstance(label_layer.data, np.ndarray):
            print("Please select a valid labels layer")
            return

        if not isinstance(points_layer.data, np.ndarray):
            print("Please select a valid points layer")
            return

        if self._channel_radio.value == "One Channel":
            if self.channelone_combo.value is None:
                print("Please select a valid image layer for channel 1")
                return
            membimg = self.channelone_combo.value.data
            dnaimg = None
        else:
            if (
                self.channelone_combo.value is None
                or self.channeltwo_combo.value is None
            ):
                print("Please select valid image layers for both channels")
                return
            membimg = self.channelone_combo.value.data
            dnaimg = self.channeltwo_combo.value.data

        # check the name of the points layer
        name = points_layer.name
        try:
            class_int = int(name)
        except ValueError:
            print(
                "Please name the points layer with a positive integer corresponding to the class (e.g. 1, 2, ...)"
            )
            return

        if class_int < 1:
            print(
                "Please name the points layer with a positive integer corresponding to the class (e.g. 1, 2, ...)"
            )
            return

        points = points_layer.data
        labels_assigned = []

        props = pd.DataFrame(
            regionprops_table(label_layer.data, properties=["label", "bbox"])
        )

        combined_crops_list = []

        for p in points:
            row, col = p[0], p[1]
            label = label_layer.data[int(np.rint(row)), int(np.rint(col))]

            if label == 0:
                print(
                    f"Point at ({row}, {col}) is not inside a labeled region. Skipping."
                )
                continue

            if label in labels_assigned:
                print(
                    f"Label {label} already assigned to another point. Skipping."
                )
                continue

            labels_assigned.append(label)

            bbox = (
                props[props["label"] == label]["bbox-0"].item(),
                props[props["label"] == label]["bbox-1"].item(),
                props[props["label"] == label]["bbox-2"].item(),
                props[props["label"] == label]["bbox-3"].item(),
            )  # (min_row, min_col, max_row, max_col)
            bbox = (
                max(bbox[0] - self.box_margin, 0),
                max(bbox[1] - self.box_margin, 0),
                min(bbox[2] + self.box_margin, maxrow - 1),
                min(bbox[3] + self.box_margin, maxcol - 1),
            )

            cell_label = (
                label_layer.data[bbox[0] : bbox[2], bbox[1] : bbox[3]] == label
            )
            cell_channel1 = rescale_intensity(
                img_as_float(
                    membimg[bbox[0] : bbox[2], bbox[1] : bbox[3]] * cell_label
                )
            )
            if dnaimg is not None:
                cell_channel2 = rescale_intensity(
                    img_as_float(
                        dnaimg[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                        * cell_label
                    )
                )
            else:
                cell_channel2 = None

            cropnrow, cropncol = cell_channel1.shape
            # Symmetric padding to make the crop square
            if cropnrow > cropncol:
                pad = cropnrow - cropncol
                padleft = pad // 2
                padright = pad - padleft
                cell_channel1 = np.pad(
                    cell_channel1,
                    ((0, 0), (padleft, padright)),
                    mode="constant",
                    constant_values=0,
                )
                if cell_channel2 is not None:
                    cell_channel2 = np.pad(
                        cell_channel2,
                        ((0, 0), (padleft, padright)),
                        mode="constant",
                        constant_values=0,
                    )
            elif cropncol > cropnrow:
                pad = cropncol - cropnrow
                padtop = pad // 2
                padbottom = pad - padtop
                cell_channel1 = np.pad(
                    cell_channel1,
                    ((padtop, padbottom), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                if cell_channel2 is not None:
                    cell_channel2 = np.pad(
                        cell_channel2,
                        ((padtop, padbottom), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )

            # resize to 100x100
            cell_channel1 = resize(
                cell_channel1, (100, 100), anti_aliasing=False
            )
            if cell_channel2 is not None:
                cell_channel2 = resize(
                    cell_channel2, (100, 100), anti_aliasing=False
                )

            # side by side if needed
            if cell_channel2 is not None:
                combined_crops = np.concatenate(
                    (cell_channel1, cell_channel2), axis=1
                )
            else:
                combined_crops = cell_channel1

            combined_crops_list.append(combined_crops)

        combined_classes_list = [class_int] * len(combined_crops_list)

        with open(
            os.path.join(path2save, f"Class_{class_int}_source.p"), "wb"
        ) as f:
            pickle.dump(combined_crops_list, f)
        with open(
            os.path.join(path2save, f"Class_{class_int}_target.p"), "wb"
        ) as f:
            pickle.dump(combined_classes_list, f)

        print(
            f"Saved {len(combined_crops_list)} crops for class {class_int} in {path2save}"
        )

        return
