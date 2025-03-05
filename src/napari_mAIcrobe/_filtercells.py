"""
Module responsible for fitering cells based on the calculated statistics
"""

import typing
from typing import TYPE_CHECKING, Sequence
from typing import Optional, cast, Literal
import inspect

if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QWidget, QGridLayout
import numpy as np
from magicgui.widgets import Container, create_widget, PushButton, ComboBox, FloatRangeSlider
from psygnal import Signal

class filter_cells(Container):

    changed = Signal(object)

    def __init__(self, viewer: "napari.viewer.Viewer"):

        self._viewer = viewer
        
        self._lbl_combo = cast(ComboBox, create_widget(annotation="napari.layers.Labels"))
        self._lbl_combo.changed.connect(self._on_label_layer_changed)

        self._add_button = PushButton(label="+")
        self._add_button.clicked.connect(self._on_plus_clicked)

        super().__init__(widgets=[self._lbl_combo, self._add_button], labels=False)

        self.changed.connect(self._on_changed)

        self._current_layer = viewer.layers.selection.active.data
        self._current_layer_properties = viewer.layers.selection.active.properties
        self._viewer.add_labels(self._current_layer, name="Filtered Cells")

    def _on_label_layer_changed(self, new_layer: "napari.layers.Labels"):
        while self.__len__() > 2:
            self.pop()

        self._current_layer = new_layer.data
        self._current_layer_properties = new_layer.properties
        try:
            self._viewer.layers.remove("Filtered Cells")
        except ValueError:
            pass
        self._viewer.add_labels(self._current_layer, name="Filtered Cells")

    def _on_plus_clicked(self,):
        filter = unit_filter(self)
        self.append(filter)

    def _on_changed(self, obj):

        if obj.__len__() > 2:

            # get all indices
            i = 2
            filtered_labels = []
            while i < obj.__len__():
                filtered_labels = [*filtered_labels, *obj.__getitem__(i)._filtered_labels]
                i+=1

            filtered_labels = list(set(filtered_labels))
            labelimg = obj._current_layer.copy()

            for l in filtered_labels:
                labelimg[labelimg==l] = 0

            obj._viewer.layers['Filtered Cells'].data = labelimg.astype(int)
        else:
            obj._viewer.layers['Filtered Cells'].data = obj._current_layer

        obj._viewer.layers['Filtered Cells'].refresh()

class unit_filter(QWidget):

    def __init__(self, parent):

        super().__init__(None)
        self.setLayout(QGridLayout())

        self._parent = parent

        self._viewer = self._parent._viewer
        self._layer = self._parent._current_layer
        self._layer_properties = self._parent._current_layer_properties

        try:
            self._properties = list(self._layer_properties.keys())
            self._labels = np.array(self._layer_properties['label'], dtype=int)
            self.current_prop = self._properties[0]
            self.current_prop_arr = np.array(self._layer_properties[self.current_prop], dtype=np.float32) 
            self._filtered_labels = [0,]
        except (AttributeError, KeyError):
            self._properties = ['',]
            self._labels = np.zeros(1,dtype=int)
            self.current_prop = ' '
            self.current_prop_arr = np.zeros(1,dtype=np.float32)
            self._filtered_labels = [0,]

        self._close_button = PushButton(label="X")
        self._close_button.clicked.connect(self._close_click)

        self._prop_combo = ComboBox(choices=self._properties, label="Property")
        self._prop_combo.changed.connect(self._on_prop_changed)

        self._slider_range = FloatRangeSlider(min=np.min(self.current_prop_arr), max=np.max(self.current_prop_arr), 
                                              value=(np.min(self.current_prop_arr), np.max(self.current_prop_arr)), 
                                              tracking=True, step=(np.max(self.current_prop_arr) - np.min(self.current_prop_arr)) / 100)
        self._slider_range.changed.connect(self._slider_change)

        self.layout().addWidget(self._close_button.native, 0, 0)
        self.layout().addWidget(self._prop_combo.native, 0, 1)
        self.layout().addWidget(self._slider_range.native, 1,0,1,2)

        # hack to make sure that magicgui understands that instances of this widget are part of the container
        self.param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        self.name = "property_filter"
        self.native = self
        self.gui_only = False
        self.annotation = None
        self.options = {"enabled": True, "visible": True}

        self._parent.changed.emit(self._parent)
    
    def _on_prop_changed(self, new_prop):
        
        self._filtered_labels = [0,]

        self.current_prop = new_prop
        self.current_prop_arr = np.array(self._layer_properties[self.current_prop], dtype=np.float32) 

        # to avoid divisions by zero because the slider does not update instantly 
        self._slider_range.max = 1e12

        self._slider_range.min = np.min(self.current_prop_arr)
        self._slider_range.max = np.max(self.current_prop_arr)
        self._slider_range.value = (self._slider_range.min, self._slider_range.max)
        self._slider_range.step = (self._slider_range.max - self._slider_range.min) / 100   

        self._parent.changed.emit(self._parent)

    def _slider_change(self, new_values):

        _prop_array = self.current_prop_arr
        _indexes = np.nonzero(np.logical_or(_prop_array > new_values[1], _prop_array < new_values[0]))[0]
        self._filtered_labels = self._labels[_indexes]
        self._parent.changed.emit(self._parent)

    def _close_click(self,):
        self._filtered_labels = [0,]
        self._parent.changed.emit(self._parent)
        self.close()