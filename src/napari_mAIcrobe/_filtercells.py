"""
Module responsible for fitering cells based on the calculated statistics
"""

import inspect
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import napari

import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatRangeSlider,
    PushButton,
    create_widget,
)
from psygnal import Signal
from qtpy.QtWidgets import QGridLayout, QWidget


class filter_cells(Container):
    """
    Interactive cell filtering widget.

    Provides a Labels layer selector and dynamic property-based filters
    using range sliders. Emits a `changed` signal when filtering updates
    and writes results to a "Filtered Cells" Labels layer.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The active napari viewer.
    """

    changed = Signal(object)

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Create the filter UI and attach to the provided viewer.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The active napari viewer instance.
        """

        self._viewer = viewer

        self._lbl_combo = cast(
            ComboBox, create_widget(annotation="napari.layers.Labels")
        )
        self._lbl_combo.changed.connect(self._on_label_layer_changed)

        self._add_button = PushButton(label="+")
        self._add_button.clicked.connect(self._on_plus_clicked)

        super().__init__(
            widgets=[self._lbl_combo, self._add_button], labels=False
        )

        self.changed.connect(self._on_changed)

        self._current_layer = viewer.layers.selection.active.data
        self._current_layer_properties = (
            viewer.layers.selection.active.properties
        )
        self._viewer.add_labels(self._current_layer, name="Filtered Cells")

    def _on_label_layer_changed(self, new_layer: "napari.layers.Labels"):
        """Handle change of the selected Labels layer. When the layer
        changes, update cached properties and data and generate a
        duplicate label image called "Filtered Cells". If "Filtered
        Cells" already exists, it is removed first. The properties dict
        is essential, as it provides all the filterable properties and
        the values for each labeled cell.

        Parameters
        ----------
        new_layer : napari.layers.Labels
            Newly selected labels layer providing `.data` and
            `.properties`.
        """
        while self.__len__() > 2:
            self.pop()

        self._current_layer = new_layer.data
        self._current_layer_properties = new_layer.properties
        try:
            self._viewer.layers.remove("Filtered Cells")
        except ValueError:
            pass
        self._viewer.add_labels(self._current_layer, name="Filtered Cells")

    def _on_plus_clicked(
        self,
    ):
        """Append a new unit filter to the UI. See `unit_filter` class
        for more info."""
        filter = unit_filter(self)
        self.append(filter)

    def _on_changed(self, obj):
        """Update the Filtered Cells Labels layer by removing labels
        that do not match the current filters. Function is triggered by
        the `changed` signal. The changed signal is given by the
        unit_filter instances when their sliders or property selectors
        change.

        Parameters
        ----------
        obj : filter_cells
            The container instance emitting the change.
        """

        if obj.__len__() > 2:

            # get all indices
            i = 2
            filtered_labels = []
            while i < obj.__len__():
                filtered_labels = [
                    *filtered_labels,
                    *obj.__getitem__(i)._filtered_labels,
                ]
                i += 1

            filtered_labels = list(set(filtered_labels))
            labelimg = obj._current_layer.copy()

            for l in filtered_labels:
                labelimg[labelimg == l] = 0

            obj._viewer.layers["Filtered Cells"].data = labelimg.astype(int)
        else:
            obj._viewer.layers["Filtered Cells"].data = obj._current_layer

        obj._viewer.layers["Filtered Cells"].refresh()


class unit_filter(QWidget):
    """
    Single property filter unit.

    Lets users choose a property from the Labels layer properties dict
    and filter labels by a numeric range using a slider. Updates parent
    container via `parent.changed` signal which triggers a refresh of
    the Filtered Cells Labels layer.

    Parameters
    ----------
    parent : filter_cells
        Parent container providing viewer, current layer data and
        properties.
    """

    def __init__(self, parent):
        """Initialize the unit filter UI.

        Parameters
        ----------
        parent : filter_cells
            Parent filtering container.
        """

        super().__init__(None)
        self.setLayout(QGridLayout())

        self._parent = parent

        self._viewer = self._parent._viewer
        self._layer = self._parent._current_layer
        self._layer_properties = self._parent._current_layer_properties

        try:
            self._properties = list(self._layer_properties.keys())
            self._labels = np.array(self._layer_properties["label"], dtype=int)
            self.current_prop = self._properties[0]
            self.current_prop_arr = np.array(
                self._layer_properties[self.current_prop], dtype=np.float32
            )
            self._filtered_labels = [
                0,
            ]
        except (AttributeError, KeyError):
            self._properties = [
                "",
            ]
            self._labels = np.zeros(1, dtype=int)
            self.current_prop = " "
            self.current_prop_arr = np.zeros(1, dtype=np.float32)
            self._filtered_labels = [
                0,
            ]

        self._close_button = PushButton(label="X")
        self._close_button.clicked.connect(self._close_click)

        self._prop_combo = ComboBox(choices=self._properties, label="Property")
        self._prop_combo.changed.connect(self._on_prop_changed)

        self._slider_range = FloatRangeSlider(
            min=np.min(self.current_prop_arr),
            max=np.max(self.current_prop_arr),
            value=(
                np.min(self.current_prop_arr),
                np.max(self.current_prop_arr),
            ),
            tracking=True,
            step=(
                np.max(self.current_prop_arr) - np.min(self.current_prop_arr)
            )
            / 100,
        )
        self._slider_range.changed.connect(self._slider_change)

        self.layout().addWidget(self._close_button.native, 0, 0)
        self.layout().addWidget(self._prop_combo.native, 0, 1)
        self.layout().addWidget(self._slider_range.native, 1, 0, 1, 2)

        # hack to make sure that magicgui understands that instances of this widget are part of the container
        self.param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        self.name = "property_filter"
        self.native = self
        self.gui_only = False
        self.annotation = None
        self.options = {"enabled": True, "visible": True}

        self._parent.changed.emit(self._parent)

    def _on_prop_changed(self, new_prop):
        """Update slider bounds and reset filtered labels on property change.

        Parameters
        ----------
        new_prop : str
            Property name present in the Labels layer properties.
        """

        self._filtered_labels = [
            0,
        ]

        self.current_prop = new_prop
        self.current_prop_arr = np.array(
            self._layer_properties[self.current_prop], dtype=np.float32
        )

        # to avoid divisions by zero because the slider does not update instantly
        self._slider_range.max = 1e12

        self._slider_range.min = np.min(self.current_prop_arr)
        self._slider_range.max = np.max(self.current_prop_arr)
        self._slider_range.value = (
            self._slider_range.min,
            self._slider_range.max,
        )
        self._slider_range.step = (
            self._slider_range.max - self._slider_range.min
        ) / 100

        self._parent.changed.emit(self._parent)

    def _slider_change(self, new_values):
        """Recompute filtered labels from slider range.

        Parameters
        ----------
        new_values : tuple[float, float]
            (min, max) threshold for the selected property.
        """

        _prop_array = self.current_prop_arr
        _indexes = np.nonzero(
            np.logical_or(
                _prop_array > new_values[1], _prop_array < new_values[0]
            )
        )[0]
        self._filtered_labels = self._labels[_indexes]
        self._parent.changed.emit(self._parent)

    def _close_click(
        self,
    ):
        """Remove this unit filter from the UI and reset its effect."""
        self._filtered_labels = [
            0,
        ]
        self._parent.changed.emit(self._parent)
        self.close()
