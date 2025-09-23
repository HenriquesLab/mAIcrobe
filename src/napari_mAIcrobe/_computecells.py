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
    DNA_Image: "napari.layers.Image",
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
    """Compute per-cell features, generate reports. Optionally build
    average heatmap, classification and colocalization.

    #TODO check parameter order in the GUI and in the docstring. It
    should make sense to the user.

    Parameters
    ----------
    Viewer : napari.Viewer
        Napari viewer to which results (table, images) are added.
    Label_Image : napari.layers.Labels
        Labels layer with segmented cells.
    Membrane_Image : napari.layers.Image
        Primary fluorescence image (e.g., membrane).
    DNA_Image : napari.layers.Image
        Optional fluorescence image (e.g., DNA).
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
    - Updates `Label_Image.properties` and opens a properties table.
    - Adds "Cell Averager" image if heatmap is computed.
    - Saves report files if requested and path is valid.
    - Colocalization requires two channels.
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

    cell_man = CellManager(
        label_img=Label_Image.data,
        fluor=Membrane_Image.data,
        optional=DNA_Image.data,
        params=params,
    )
    cell_man.compute_cell_properties()

    Label_Image.properties = cell_man.properties

    add_table(Label_Image, Viewer)

    if Compute_Heatmap:
        Viewer.add_image(cell_man.heatmap_model, name="Cell Averager")
