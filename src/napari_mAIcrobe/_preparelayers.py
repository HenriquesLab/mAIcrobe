"""
Module responsible for preparing arrays for processing by the mAIcrobe plugin. This includes:
- Squeezing arrays to remove singleton dimensions.
- Converting xarray DataArrays to raw NumPy arrays if needed.
- Creating new layers in the viewer for the processed arrays without overwriting the originals.
"""

from typing import TYPE_CHECKING, Annotated

import numpy as np
from magicgui import magic_factory
from napari.layers import Image

if TYPE_CHECKING:
    import napari


def squeeze_all_layers(viewer, suffix=" squeezed NumPy", hide_original=True):

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

            if hide_original:
                layer.visible = False  # Hide the original layer

        else:
            print(f"Skipping {layer.name}: not an Image layer")
            continue

        new_layers.append(new_layer)

    return new_layers


@magic_factory(
    call_button="Prepare layers",
    layout="vertical",
)
def prepare_layers_before_cell_detection(
    Viewer: "napari.Viewer",
    suffix: Annotated[
        str, {"tooltip": "Suffix appended to new layer names"}
    ] = " squeezed NumPy",
    hide_original: Annotated[
        bool,
        {
            "label": "Hide originals",
            "tooltip": "Hide the original layers after creating squeezed copies",
        },
    ] = True,
):
    """
    Prepare layers for downstream processing by squeezing singleton dimensions
    and converting xarray DataArray objects to NumPy arrays.

    Parameters
    - Viewer: the active napari Viewer
    - suffix: suffix appended to newly created layer names
    - hide_original: whether to hide the original layers after creating copies
    """
    squeeze_all_layers(Viewer, suffix=suffix, hide_original=hide_original)
