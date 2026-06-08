"""
Module responsible for computing per cell statistics
"""
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui import magic_factory
from napari_skimage_regionprops import add_table

from .mAIcrobe.cells import CellManager


@magic_factory(
    Septum_algorithm={"choices": ["Isodata", "Box"]},
    Model={
        "choices": [
            "S.aureus DNA+Membrane Epi",
            "S.aureus DNA+Membrane SIM",
            "S.aureus DNA Epi",
            "S.aureus DNA SIM",
            "S.aureus Membrane Epi",
            "S.aureus Membrane SIM",
            "E.coli DNA+Membrane AB phenotyping",
            "custom",
        ]
    },
    Custom_model_path={"widget_type": "FileEdit", "mode": "r"},
    Custom_model_input={"choices": ["Membrane", "DNA", "Membrane+DNA"]},
    Report_path={"widget_type": "FileEdit", "mode": "d"},
)
def compute_cells(
    Viewer: "napari.Viewer",
    Label_Image: "napari.layers.Labels",
    Membrane_Image: "napari.layers.Image",
    DNA_Image: "napari.layers.Image" = None,
    Pixel_size: float = 1,
    Inner_mask_thickness: int = 4,
    Septum_algorithm="Isodata",
    Baseline_margin: int = 30,
    Find_septum: bool = False,
    Find_open_septum: bool = False,
    Classify_cell_cycle: bool = False,
    Model="S.aureus DNA+Membrane Epi",
    Custom_model_path: os.PathLike = "",
    Custom_model_input="Membrane",
    Custom_model_MaxSize: int = 50,
    Compute_Colocalization: bool = False,
    Generate_Report: bool = False,
    Report_path: os.PathLike = "",
    Compute_Heatmap: bool = False,
):
    """Compute per-cell morphological features, classification and optional reports from 2D images or
    timelapse 2D+t data. Additionally supports optional heatmap generation for 2D inputs.

    Supports 2D inputs `(Y, X)` and timelapse inputs `(T, Y, X)`. In
    timelapse mode, each frame is analyzed independently (NO TRACKING),
    and results are aggregated with a `frame` column.

    Parameters
    ----------
    Viewer : napari.Viewer
        Napari viewer to which results (table, images) are added.
    Label_Image : napari.layers.Labels
        Labels layer with segmented cells.
    Membrane_Image : napari.layers.Image
        Primary fluorescence image (e.g., membrane).
    DNA_Image : napari.layers.Image, optional
        Optional secondary fluorescence image (e.g., DNA). If omitted,
        DNA-dependent metrics are NaN, colocalization is
        skipped and classification is limited to one channel.
    Pixel_size : float, optional
        Pixel size passed to analysis (if used downstream), by default 1.
    Inner_mask_thickness : int, optional
        Thickness for inner membrane erosion, by default 4.
    Septum_algorithm : {"Isodata", "Box"}, optional
        Algorithm to detect septum, by default "Isodata".
    Baseline_margin : int, optional
        Margin (pixels) around cell to compute background baseline, by
        default 30.
    Find_septum : bool, optional
        Enable septum detection, by default False.
    Find_open_septum : bool, optional
        Enable open septum detection, by default False.
    Classify_cell_cycle : bool, optional
        Enable cell cycle classification, by default False.
    Model : str, optional
        Prebuilt or custom model selector, by default "S.aureus
        DNA+Membrane Epi".
    Custom_model_path : os.PathLike, optional
        Path to custom Keras model, by default "".
    Custom_model_input : {"Membrane","DNA","Membrane+DNA"}, optional
        Input channels for custom model, by default "Membrane".
    Custom_model_MaxSize : int, optional
        Max dimension for classifier preprocessing, by default 50.
    Compute_Colocalization : bool, optional
        Compute per cell Pearson correlation coefficients between
        channels, by default False.
    Generate_Report : bool, optional
        Generate HTML and CSV report, by default False.
    Report_path : os.PathLike, optional
        Output directory for reports, by default "".
    Compute_Heatmap : bool, optional
        Build average heatmap from aligned cells, by default False.

    Notes
    -----
        - In 2D mode, updates `Label_Image.properties` and opens a
            properties table.
        - In timelapse mode, skips table attachment and processes all
            frames into one combined output.
    - Adds "Cell Averager" image if heatmap is computed (2D mode only).
    - Saves report files if requested and path is valid.
        - Colocalization requires two channels and is skipped when
            `DNA_Image` is not provided.
    - Custom model requires a valid Keras model file (.keras)
    """

    params = {
        "pixel_size": Pixel_size,
        "inner_mask_thickness": Inner_mask_thickness,
        "septum_algorithm": Septum_algorithm,
        "baseline_margin": Baseline_margin,
        "find_septum": Find_septum,
        "find_openseptum": Find_open_septum,
        "classify_cell_cycle": Classify_cell_cycle,
        "model": Model,
        "custom_model_path": Custom_model_path,
        "custom_model_input": Custom_model_input,
        "custom_model_maxsize": Custom_model_MaxSize,
        "generate_report": Generate_Report,
        "report_path": str(Report_path),
        "cell_averager": Compute_Heatmap,
        "coloc": Compute_Colocalization,
    }

    label_data = Label_Image.data
    membrane_data = Membrane_Image.data
    dna_data = DNA_Image.data if DNA_Image is not None else None

    if label_data.ndim not in (2, 3):
        raise ValueError("Label image must be 2D or 3D (T, Y, X).")

    if membrane_data.ndim != label_data.ndim:
        raise ValueError(
            "Label and membrane images must have matching dimensions."
        )

    if membrane_data.shape != label_data.shape:
        raise ValueError(
            "Label and membrane images must have matching shapes."
        )

    if dna_data is not None:
        if dna_data.ndim != label_data.ndim:
            raise ValueError(
                "Optional image must have matching dimensions with label image."
            )
        if dna_data.shape != label_data.shape:
            raise ValueError(
                "Optional image must have matching shape with label image."
            )

    cell_man = CellManager(
        label_img=label_data,
        fluor=membrane_data,
        optional=dna_data,
        params=params,
    )
    cell_man.compute_cell_properties()

    if label_data.ndim == 2:
        Label_Image.properties = cell_man.properties
        add_table(Label_Image, Viewer)
    else:
        print(
            "Timelapse mode detected: skipping napari table attachment; "
            "combined results are available in reports/output properties."
        )

    if Compute_Heatmap:
        Viewer.add_image(cell_man.heatmap_model, name="Cell Averager")
