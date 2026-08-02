"""
Module responsible for preparing arrays for processing by the mAIcrobe plugin. This includes:
- Squeezing arrays to remove singleton dimensions.
- Converting xarray DataArrays to raw NumPy arrays if needed.
- Creating new layers in the viewer for the processed arrays without overwriting the originals.
"""

from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory
from napari.layers import Image

if TYPE_CHECKING:
    import napari


def squeeze_all_layers(viewer, suffix=" squeezed NumPy"):
    """
    Creates new squeezed NumPy versions of all Image layers in the viewer.
    Does not overwrite the original layers.
    """
    new_layers = []

    for layer in list(viewer.layers):
        data = layer.data

        # Convert xarray DataArray to raw NumPy values if needed
        if hasattr(data, "values"):
            data = data.values

        arr = np.asarray(data)
        arr_squeezed = np.squeeze(arr)

        new_name = layer.name + suffix

        # Add back as the same general layer type
        if isinstance(layer, Image):
            new_layer = viewer.add_image(
                arr_squeezed,
                name=new_name,
                colormap=layer.colormap,
                blending=layer.blending,
                opacity=layer.opacity,
                visible=layer.visible,
            )

            layer.visible = False  # Hide the original layer

        else:
            print(f"Skipping {layer.name}: not an Image layer")
            continue

        new_layers.append(new_layer)

    return new_layers


@magic_factory(
    call_button="Prepare layers",
)
def prepare_layers_before_cell_detection(
    Viewer: "napari.Viewer",
):
    """
    Prepare currently open napari layers before running Compute Cells.
    Creates squeezed NumPy copies of Image and Labels layers.
    """
    squeeze_all_layers(Viewer)


def _init_compute_cells_widget(widget):
    widget.Shape_Fit_Type.visible = False
    widget.Phase_Contrast_Image.visible = False
