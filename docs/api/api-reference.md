# API Reference

Programmatic usage of napari-mAIcrobe for automation, batch processing,
and integration into custom analysis pipelines.

- Widget APIs: interactive GUI components for segmentation, analysis, and filtering.
- Core library: masks, labels, cells, colocalization, averaging, reports, utilities.
- Sample data: quick-start images for testing.

## Table of Contents
- Overview
- Module Structure
- Widgets
  - _computelabel
  - _computecells
  - _filtercells
  - _compute_pickles
- Sample Data
- Core Library
  - mask
  - segments
  - unet
  - cells (Cell, CellManager)
  - cellaverager
  - colocmanager
  - reports
  - cellprocessing
  - cellcycleclassifier
- Further Resources

---

## Overview

napari-mAIcrobe provides:
- Interactive widgets (segment, analyse, filter).
- Python-callable functions and classes for reproducible workflows.
- Exportable reports and optional colocalization/classification helpers.

## Module Structure

```
napari_mAIcrobe/
├── _computelabel.py       # Segmentation widget
├── _computecells.py       # Cell analysis widget
├── _filtercells.py        # Labels property-based filtering widget
├── _compute_pickles.py    # Export per-cell crops/targets as pickles for training
├── _sample_data.py        # Sample data providers
└── mAIcrobe/              # Core analysis library
    ├── mask.py            # Thresholding based masks
    ├── segments.py        # Watershed segmentation and labeling for thresholding-based masks
    ├── unet.py            # UNet helpers and prediction
    ├── cells.py           # Cell base class and manager
    ├── cellaverager.py    # Per-cell alignment and averaging of fluorescence
    ├── colocmanager.py    # Pearson correlation coefficient per cell
    ├── reports.py         # HTML/CSV report generation
    ├── cellprocessing.py  # Geometry/utilities
    └── cellcycleclassifier.py  # CNN classifier helpers
```

---

## Widgets

These are classes that handle user interaction and GUI elements in napari. Should not be used directly in scripts.

<details>
<summary><code>napari_mAIcrobe._computelabel</code></summary>

Source: [../../src/napari_mAIcrobe/_computelabel.py](../../src/napari_mAIcrobe/_computelabel.py)

```python
compute_label(viewer)
```
  - Segmentation and optional channel alignment widget.
  - Mask algorithms: "Isodata", "Local Average", "Unet", "StarDist", "CellPose cyto3".
  - Methods:
    - _on_algorithm_changed(new_algorithm: str)
      - Toggles parameter widgets according to algorithm.
    - compute()
      - Computes mask and labels (watershed or model-based).
      - Optional: align auxiliary channels, apply closing/dilation/fill holes.
  - Side effects: adds "Mask" and "Labels" layers; aligns fluor images if enabled.

</details>

<details>
<summary><code>napari_mAIcrobe._computecells</code></summary>

Source: [../../src/napari_mAIcrobe/_computecells.py](../../src/napari_mAIcrobe/_computecells.py)

```python
compute_cells(
  Viewer, Label_Image, Membrane_Image, DNA_Image,
  Pixel_size=1, Inner_mask_thickness=4, Septum_algorithm="Isodata",
  Baseline_margin=30, Find_septum=False, Find_open_septum=False,
  Classify_cell_cycle=False, Model="S.aureus DNA+Membrane Epi",
  Custom_model_path="", Custom_model_input="Membrane",
  Custom_model_MaxSize=50, Compute_Colocalization=False,
  Generate_Report=False, Report_path="", Compute_Heatmap=False
)
```
  - Computes per-cell features; optional colocalization, classification,
    averaged heatmap, and report generation.
  - Notes:
    - Updates Label_Image.properties and opens a properties table.
    - Adds "Cell Averager" image if enabled.
    - **Parameters**:
      - `Viewer`: napari Viewer instance.
      - `Label_Image`: Labels layer with segmented cells.
      - `Membrane_Image`: Membrane fluorescence image layer.
      - `DNA_Image`: DNA fluorescence image layer.
      - `Pixel_size`: Physical pixel size (e.g., 0.065 μm/pixel) (optional).
      - `Inner_mask_thickness`: Thickness for cytoplasmic mask (default: 4).
      - `Septum_algorithm`: "Isodata" or "Box" thresholding for septa.
      - `Baseline_margin`: Background margin size (default: 30).
      - `Find_septum`: Enable septum detection.
      - `Find_open_septum`: Enable open septum detection.
      - `Classify_cell_cycle`: Enable cell cycle classification.
      - `Model`: Pre-trained model name or "Custom".
      - `Custom_model_path`: Path to custom model (.keras) if selected.
      - `Custom_model_input`: Input type for custom model ("Membrane", "DNA", "Membrane+DNA").
      - `Custom_model_MaxSize`: Max cell size for custom model (default: 50 pixels).
      - `Compute_Colocalization`: Enable Pearson correlation computation.
      - `Generate_Report`: Enable HTML report generation.
      - `Report_path`: Directory to save the report.
      - `Compute_Heatmap`: Enable averaged heatmap generation.

</details>

<details>
<summary><code>napari_mAIcrobe._filtercells</code></summary>

Source: [../../src/napari_mAIcrobe/_filtercells.py](../../src/napari_mAIcrobe/_filtercells.py)

```python
from napari_mAIcrobe._filtercells import filter_cells

filter_cells(viewer)
```
  - Interactive filtering container for Labels properties.
  - Signal: changed(object). Writes to "Filtered Cells" Labels layer.
  - **Parameters**:
    - viewer: napari Viewer instance containing a Labels layer with cell properties.

```python
from napari_mAIcrobe._filtercells import unit_filter
unit_filter(parent)
```
  - Single numeric property filter with a range slider.
  - Emits parent.changed on updates.
  - **Parameters**:
    - parent: instance of FilterCells containing this filter.


</details>

<details>
<summary><code>napari_mAIcrobe._compute_pickles</code></summary>

Source: [../../src/napari_mAIcrobe/_compute_pickles.py](../../src/napari_mAIcrobe/_compute_pickles.py)

```python
from napari_mAIcrobe._compute_pickles import compute_pickles
compute_pickles(viewer)
```
  - Export per-cell training data as pickles from annotated layers.
  - Inputs:
    - Labels layer (cells)
    - Points layer named with a positive integer (class id), one point per cell
    - One or two Image layers (channel 1 required, channel 2 optional)
    - Output directory
  - Processing:
    - Crop by label bounding box (+margin 5px), mask by cell, pad to square, resize to 100×100
    - If two channels, concatenate crops side-by-side (shape 100×200)
    - Intensity rescaled to [0, 1]
  - Outputs:
    - `Class_<id>_source.p`: list of ndarray crops
    - `Class_<id>_target.p`: list of class ids (same length as source)
  - Methods:
    - `_on_channel_change()`: toggle second-channel selector visibility
    - `_on_run()`: validate inputs, build crops, and write pickle files

</details>

---

## Sample Data

Source: [../../src/napari_mAIcrobe/_sample_data.py](../../src/napari_mAIcrobe/_sample_data.py)

```python
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example

phase_example()
membrane_example()
dna_example()
```
  - Loads a sample S. aureus phase-contrast, membrane-labeled, and DNA-labeled image respectively.
  - Returns: list[(ndarray, dict, str)] suitable for napari sample data.


---

## Core Library

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.mask</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/mask.py](../../src/napari_mAIcrobe/mAIcrobe/mask.py)

```python
from napari_mAIcrobe.mAIcrobe.mask import mask_computation

mask_computation(base_image, algorithm="Isodata", blocksize=151,
                offset=0.02, closing=1, dilation=0, fillholes=False)
```
  - Build a binary mask using Isodata or Local Average. Applies optional closing, dilation, and hole filling.
  - **Parameters**:
    - `base_image`: 2D ndarray input image.
    - `algorithm`: "Isodata" or "Local Average".
    - `blocksize`: Odd integer for local average window (default: 151).
    - `offset`: Float offset for thresholding (default: 0.02).
    - `closing`: Integer for morphological closing iterations (default: 1).
    - `dilation`: Integer for morphological dilation iterations (default: 0).
    - `fillholes`: Boolean to fill holes in mask (default: False).
  - **Returns**:
    - 2D binary ndarray mask.

```python
from napari_mAIcrobe.mAIcrobe.mask import mask_alignment
mask_alignment(mask, fluor_image)
```
  - Align a binary mask to a fluorescence image via phase correlation.
  - **Parameters**:
    - `mask`: 2D binary ndarray mask.
    - `fluor_image`: 2D ndarray fluorescence image.
  - **Returns**:
    - Aligned image.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.segments</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/segments.py](../../src/napari_mAIcrobe/mAIcrobe/segments.py)


### class SegmentsManager

```python
from napari_mAIcrobe.mAIcrobe.segments import SegmentsManager
SegmentsManager()
```

  - Marker detection via euclidean distance transform and subsequent watershed segmentation.
  - **Attributes**:
    - `features` (ndarray|None)
    - `labels` (ndarray|None).
  - **Methods**:
    - `clear_all()`
        - Resets internal attributes to None.
    - `compute_distance_peaks(mask, params) -> list[(int, int)]`
        - **Parameters**:
            - `mask`: Binary mask used for distance peak computation (non-zero inside cell regions).
            - `params`: Dictionary containing parameters for peak detection.
                - Keys must include:
                    - `peak_min_distance_from_edge`: Minimum distance from edge to consider a peak.
                    - `peak_min_distance`: Minimum distance between peaks.
                    - `peak_min_height`: Minimum height for peak detection.
                    - `max_peaks`: Maximum number of peaks to detect.
        - **Returns**:
            - A list of tuples representing the coordinates of detected peaks.

    - `compute_features(params, mask)`
        - Generates marker features from peak coordinates to be used for watershed segmentation.
        - **Parameters**:
            - `params`: Dictionary containing parameters for peak detection.
            - `mask`: Binary mask.
        - **Returns**:
            - None. Updates internal attributes (features) with computed features.

    - `overlay_features(mask) [deprecated]`

    - `compute_labels(mask)`
        - Runs watershed segmentation. Uses the features attribute computed in compute_features as markers for the watershed algorithm.
        - **Parameters**:
            - `mask`: Binary mask used for label computation.
        - **Returns**:
            - None. Updates internal attributes (labels) with computed labels.

    - `compute_segments(params, mask)`
        - **Parameters**:
            - `params`: Dictionary containing peak detection parameters.
                - Keys must include:
                    - `peak_min_distance_from_edge`: Minimum distance from edge to consider a peak.
                    - `peak_min_distance`: Minimum distance between peaks.
                    - `peak_min_height`: Minimum height for peak detection.
                    - `max_peaks`: Maximum number of peaks to detect.
            - `mask`: Binary mask used for segmentation.
        - **Returns**:
            - None. Updates internal attributes (features, label).


</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.unet</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/unet.py](../../src/napari_mAIcrobe/mAIcrobe/unet.py)

```python
from napari_mAIcrobe.mAIcrobe.unet import normalizePercentile

normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=float32)
```
- Percentile-based normalization of an image.
- This is adapted from Martin Weigert and copied from the ZeroCostDL4Mic UNet notebook.
- **Parameters**:
    - `x`: Input ndarray image.
    - `pmin`: Lower percentile (default: 1).
    - `pmax`: Upper percentile (default: 99.8).
    - `axis`: Axis or axes along which to compute percentiles (default: None).
    - `clip`: Boolean to clip values to [0,1] (default: False).
    - `eps`: Small value to avoid division by zero (default: 1e-20).
    - `dtype`: Data type for output (default: float32).
- **Returns**:
    - Normalized ndarray image.

```python
from napari_mAIcrobe.mAIcrobe.unet import normalize_mi_ma

normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=float32)
```
- Min-max normalization of an image.
- This is adapted from Martin Weigert and copied from the ZeroCostDL4Mic UNet notebook.
- **Parameters**:
    - `x`: Input ndarray image.
    - `mi`: Minimum value for normalization.
    - `ma`: Maximum value for normalization.
    - `clip`: Boolean to clip values to [0,1] (default: False).
    - `eps`: Small value to avoid division by zero (default: 1e-20).
    - `dtype`: Data type for output (default: float32).
- **Returns**:
    - Normalized ndarray image.

```python
from napari_mAIcrobe.mAIcrobe.unet import predict_as_tiles

predict_as_tiles(img, model)
```
- Predicts a large image by tiling it into smaller patches.
- Patch size is taken from the model input shape.
- Patches are normalized using percentile normalization (1%-99.8%).
- This is copied from the ZeroCostDL4Mic UNet notebook.
- **Parameters**:
    - `img`: 2D ndarray input image to predict.
    - `model`: Keras model for prediction (keras.Model).
- **Returns**:
    - 2D ndarray prediction.

```python
from napari_mAIcrobe.mAIcrobe.unet import computelabel_unet

computelabel_unet(path2model, base_image, closing, dilation, fillholes)
```
- Compute mask and labels using a UNet model and watershed:
    - UNet model outputs an image with 3 classes:
        - Background = 0
        - Edges = 1
        - Insides = 2
    - For optimal compatibility, train your UNet using ZeroCostDL4Mic.
    - Binary mask is created by combining edges and insides.
    - Final label image is generated by:
        - Using insides as markers.
        - Running watershed on the inverse of the binary mask.
    - Optional morphological operations can clean up the binary mask (closing, dilation, hole filling).
- **Parameters**:
    - `path2model`: Path to the tensorflow UNet model (.hdf5 file).
    - `base_image`: 2D ndarray input image to segment.
    - `closing`: Size of binary closing kernel; if >0 applied to remove small spots.
    - `dilation`: Number of binary dilation iterations.
    - `fillholes`: Boolean to fill holes in mask.
- **Returns**:
    - Tuple of (mask, labels) as 2D ndarrays.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.cells</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/cells.py](../../src/napari_mAIcrobe/mAIcrobe/cells.py)

### class Cell

```python
from napari_mAIcrobe.mAIcrobe.cells import Cell
Cell(label, regionmask, properties, intensity, params, optional)
```

- Single-cell object that stores and manages per-cell properties and masks.
- **Parameters**:
    - `label`: Integer label ID of the cell.
    - `regionmask`: 2D binary ndarray mask of the cell region.
    - `properties`: pandas DataFrame row of region properties from skimage.regionprops_table that corresponds to this cell.
    - `intensity`: 2D ndarray of the primary fluorescence image.
    - `params`: Dictionary of analysis parameters. Must contain the following keys:
        - `find_septum`: Boolean to enable septum detection.
        - `find_open_septum`: Boolean to enable open septum detection.
        - `septum_algorithm`: String, either "Isodata" or "Box", for septum detection algorithm.
        - `inner_mask_thickness`: Integer thickness for membrane mask (determines cytoplasmic mask).
        - `baseline_margin`: Integer margin size for fluorescence baseline calculation.
    - `optional`: 2D ndarray of an optional secondary fluorescence image (e.g., DNA).
- **Attributes**:
    - `box` : tuple[int, int, int, int]:
        Bounding box (min_row, min_col, max_row, max_col) with padding.
    - `long_axis` : numpy.ndarray
        Two endpoints defining the long axis, integer indices.
    - `short_axis` : numpy.ndarray
        Two endpoints defining the short axis, integer indices.
    - `cell_mask` : numpy.ndarray
        Cell region mask (cropped to bounding box).
    - `perim_mask` : numpy.ndarray or None
        Membrane/perimeter mask (cropped).
    - `sept_mask` : numpy.ndarray or None
        Septum mask (cropped), if computed.
    - `cyto_mask` : numpy.ndarray or None
        Cytoplasm mask (cropped)
    - `membsept_mask` : numpy.ndarray or None
        Union mask of membrane and septum (cropped), if computed.
    - `stats` : dict
        Per-cell fluorescence and morphology statistics.
    - `image` : numpy.ndarray or None
        Image mosaic of fluorescence and masks for visualization. Used
        for reports.
- **Methods**:
  - `image_box(image) -> ndarray | None`
    - Return an image crop corresponding to the cell bounding box.
    - Parameters:
      - `image`: ndarray or None. Full image to crop.
    - Returns: Cropped ndarray or None.

  - `compute_perim_mask(thick: int) -> ndarray`
    - Compute membrane/perimeter mask by eroding the cell mask.
    - Parameters:
      - `thick`: int. Inner mask thickness parameter controlling erosion.
    - Returns: Binary perimeter mask (float array with 0 and 1).

  - `compute_sept_mask(thick: int, algorithm: {"Isodata","Box"}) -> ndarray`
    - Compute septum mask using the specified algorithm.
    - Parameters:
      - `thick`: int. Inner mask thickness parameter.
      - `algorithm`: "Isodata" or "Box".
    - Returns: Binary septum mask.

  - `compute_opensept_mask(thick: int, algorithm: {"Isodata","Box"}) -> ndarray`
    - Compute open-septum mask using the specified algorithm.
    - Parameters:
      - `thick`: int. Inner mask thickness parameter.
      - `algorithm`: "Isodata" or "Box".
    - Returns: Binary open-septum mask.

  - `compute_sept_isodata(thick: int) -> ndarray`
    - Create septum mask using isodata thresholding on the inner region.
    - Parameters:
      - `thick`: int. Inner mask thickness parameter.
    - Returns: Binary septum mask.

  - `compute_opensept_isodata(thick: int) -> ndarray`
    - Create open-septum mask via isodata (largest one or two components).
    - Parameters:
      - `thick`: int. Inner mask thickness parameter.
    - Returns: Binary mask for one or two largest septal components.

  - `compute_sept_box(thick: int) -> ndarray`
    - Create a septum mask by dilating the short axis within the cell box.
    - Parameters:
      - `thick`: int. Dilation kernel size, typically inner mask thickness.
    - Returns: Binary septum estimate mask.

  - `get_outline_points(data: ndarray) -> list[tuple[int,int]]`
    - Extract outline pixel coordinates from a binary mask (e.g., septum).
    - Parameters:
      - `data`: ndarray. Binary mask.
    - Returns: Binary line mask used to subtract from membrane.

  - `compute_sept_box_fix(outline: list[tuple[int,int]], maskshape: tuple[int,int]) -> tuple[int,int,int,int]`
    - Compute bounding box around the septum outline (with padding from box_margin attribute which is set as 5 pixels).
    - Parameters:
      - `outline`: list of (x, y) points.
      - `maskshape`: (height, width). Shape to clamp coordinates.
    - Returns: Septum bounding box (x0, y0, x1, y1).

  - `remove_sept_from_membrane(maskshape: tuple[int,int]) -> ndarray`
    - Build a line mask along the septum axis to subtract from membrane. Based on the septum bounding box.
    - Parameters:
      - `maskshape`: (height, width). Shape of septum/membrane masks.
    - Returns: Binary line mask used to subtract from membrane.

  - `recursive_compute_sept(inner_mask_thickness: int, algorithm: {"Isodata","Box"}) -> None`
    - Compute septum mask, reducing thickness on failure (fallbacks to "Box" if needed).

  - `recursive_compute_opensept(inner_mask_thickness: int, algorithm: {"Isodata","Box"}) -> None`
    - Compute open-septum mask, reducing thickness on failure (fallbacks to "Box" if needed).

  - `compute_regions(params: dict) -> None`
    - Compute cell, membrane, septum (optional), and cytoplasm masks based on params.
    - Parameters:
      - `params`: dict. Must contain `find_septum`, `find_openseptum`, `inner_mask_thickness`, `septum_algorithm`.
    - Side effects: sets `self.cell_mask`, `self.perim_mask`, `self.sept_mask`, `self.cyto_mask`, `self.membsept_mask`.
    - Returns: None.

  - `compute_fluor_baseline(mask: ndarray, fluor: ndarray, margin: int) -> None`
    - Compute baseline fluorescence around the cell and update `self.stats["Baseline"]`.
    - Parameters:
      - `mask`: ndarray. Global mask (0 at cells, 1 outside).
      - `fluor`: ndarray. Full-field fluorescence image.
      - `margin`: int. Expansion margin for baseline region, corresponds to baseline_margin parameter.
    - Side effects: updates `self.stats["Baseline"]`.
    - Returns: None.

  - `measure_fluor(fluorbox: ndarray, roi: ndarray, fraction: float = 1.0) -> float`
    - Median fluorescence in ROI, optionally from brightest fraction.
    - Parameters:
      - `fluorbox`: ndarray. Cropped fluorescence image (cell box).
      - `roi`: ndarray. Binary ROI mask (same shape as fluorbox).
      - `fraction`: 0 < float <= 1.0. Fraction of brightest pixels.
    - Returns: Median fluorescence value.

  - `compute_fluor_stats(params: dict, mask: ndarray, fluor: ndarray) -> None`
    - Compute per-region fluorescence stats and ratios; updates `self.stats`.
    - Parameters:
      - `params`: dict. Must contain `baseline_margin` and `find_septum`.
      - `mask`: ndarray. Global mask (0 at cells, 1 outside).
      - `fluor`: ndarray. Full-field fluorescence image.
    - Side effects: updates `self.stats` with fluorescence statistics.
    - Fluoresence stats computed:
      - `Cell Median`, `Membrane Median`, `Cytoplasm Median`, `Septum Median` (if `find_septum` is True)
      - `Fluor Ratio`, `Fluor Ratio 75%`, `Fluor Ratio 25%`, `Fluor Ratio 10%` (if `find_septum` is True).
    - Returns: None.

  - `set_image(fluor: ndarray, optional: ndarray) -> None`
    - Compose a 7-panel per-cell visualization image and set `self.image`.
    - Panels: fluor, fluor×cell, optional, optional×cell, fluor×perim, fluor×cyto, fluor×septum (if available).
    - Parameters:
      - `fluor`: ndarray. Full-field fluorescence image.
      - `optional`: ndarray. Full-field optional fluorescence image (e.g., DNA).
    - Side effects: sets `self.image`.
    - Returns: None.

### Class CellManager

```python
from napari_mAIcrobe.mAIcrobe.cells import CellManager
CellManager(label_img, fluor, optional, params)
```

- Parameters:
  - `label_img`: ndarray
    - Labeled image where each cell is represented by a unique integer.
  - `fluor`: ndarray
    - Fluorescence image corresponding to the labeled image.
  - `optional`: ndarray
    - Optional image used for additional calculations (e.g., DNA content).
  - `params`: dict
    - Dictionary controlling behavior. Keys include:
      - `"classify_cell_cycle"`: bool — classify cell cycle phase.
      - `"model"`: str — model type for classification.
      - `"custom_model_path"`: str — path to custom model.
      - `"custom_model_input"`: int — input size for the custom model.
      - `"custom_model_maxsize"`: int — maximum size for the custom model.
      - `"cell_averager"`: bool — perform cell averaging.
      - `"coloc"`: bool — compute colocalization metrics.
      - `"generate_report"`: bool — generate an output report.
      - `"report_path"`: str — path to save the report.
      - `"report_id"`: str, optional — report identifier.
      - `"find_septum"`: bool — enable septum detection.
      - `"find_open_septum"`: bool — enable open septum detection.
      - `"septum_algorithm"`: str — algorithm for septum detection ("Isodata" or "Box").
      - `"inner_mask_thickness"`: int — thickness for membrane mask.
      - `"baseline_margin"`: int — margin size for fluorescence baseline calculation.

- Attributes:
  - `label_img`: ndarray
    - Stored labeled image.
  - `fluor_img`: ndarray
    - Stored fluorescence image.
  - `optional_img`: ndarray
    - Stored optional image.
  - `params`: dict
    - Stored parameters dictionary.
  - `properties`: dict | None
    - Computed per-cell properties with keys:
      - `"label"`, `"Area"`, `"Perimeter"`, `"Eccentricity"`, `"Baseline"`,
        `"Cell Median"`, `"Membrane Median"`, `"Septum Median"`,
        `"Cytoplasm Median"`, `"DNA Ratio"`
      - If `find_septum` is True: `"Fluor Ratio"`, `"Fluor Ratio 75%"`, `"Fluor Ratio 25%"`, `"Fluor Ratio 10%"`
  - `heatmap_model`: ndarray | None
    - Heatmap model from the cell averager (if computed).
  - `all_cells`: list | None
    - List of per-cell visualization mosaics (used in reports).

- Methods:
  - `compute_cell_properties() -> None`
    - Compute morphology and fluorescence properties for each labeled cell; optionally performs cell cycle classification, cell averaging, and colocalization analysis.
    - Parameters: None.
    - Side effects:
      - Updates `self.properties` with arrays for:
        `label`, `Area`, `Perimeter`, `Eccentricity`, `Baseline`,
        `Cell Median`, `Membrane Median`, `Septum Median`,
        `Cytoplasm Median`, `Fluor Ratio`, `Fluor Ratio 75%`,
        `Fluor Ratio 25%`, `Fluor Ratio 10%`, `Cell Cycle Phase`,
        `DNA Ratio`.
      - May set `self.heatmap_model` (cell averager).
      - May populate `self.all_cells` and generate reports to `report_path`.
      - May save colocalization report if enabled.
    - Returns: None.

  - `calculate_DNARatio(cell_object: Cell, dna_fov: ndarray, thresh: float) -> float`
    - Static method.
    - Calculate the ratio of area with discernable DNA signal within a cell.
    - Parameters:
      - `cell_object`: Cell instance.
      - `dna_fov`: Full-field DNA image.
      - `thresh`: Threshold for discernable DNA signal.
    - Returns: Fraction of DNA-positive pixels within the cell mask.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.cellaverager</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/cellaverager.py](../../src/napari_mAIcrobe/mAIcrobe/cellaverager.py)

### class CellAverager

```python
from napari_mAIcrobe.mAIcrobe.cellaverager import CellAverager
CellAverager(fluor)
```

- Parameters:
  - `fluor`: ndarray
    - Fluorescence field-of-view image used for per-cell crops and averaging.

- Attributes:
  - `fluor`: ndarray
    - Stored fluorescence field-of-view image.
  - `model`: ndarray | None
    - Averaged heatmap model computed by `average()`.
  - `aligned_fluor_masks`: list[ndarray]
    - List of rotated per-cell fluorescence crops aligned to a common orientation.

- Methods:
  - `align(cell) -> None`
    - Align a cell crop to a common reference and append it to `aligned_fluor_masks`.
    - Parameters:
      - `cell`: napari_mAIcrobe.mAIcrobe.cells.Cell. Must provide `image_box(fluor)` and `cell_mask`.
    - Returns: None.

  - `average() -> None`
    - Compute the average heatmap by resizing aligned crops to a median shape and averaging.
    - Side effects: sets `self.model`.
    - Returns: None.

  - `calculate_rotation_angle(cell) -> float`
    - Estimate the rotation angle (degrees) to align the cell’s major axis vertically.
    - Parameters:
      - `cell`: napari_mAIcrobe.mAIcrobe.cells.Cell.
    - Returns: Rotation angle in degrees.

  - `calculate_cell_outline(binary: ndarray) -> ndarray`
    - Static method.
    - Compute the outline of a binary object. Used for cell outlines.
    - Parameters:
      - `binary`: ndarray. Binary image (non-zero indicates object).
    - Returns: Binary outline image.

  - `calculate_major_axis(outline: ndarray) -> list[list[float]]`
    - Static method.
    - Compute major axis endpoints using PCA on outline coordinates.
    - Parameters:
      - `outline`: ndarray. Binary outline image.
    - Returns: Two endpoints [[x0, y0], [x1, y1]] of the major axis.

  - `calculate_axis_angle(major_axis: list[list[float]]) -> float`
    - Static method.
    - Compute rotation angle (degrees) from major axis endpoints.
    - Parameters:
      - `major_axis`: [[x0, y0], [x1, y1]] endpoints.
    - Returns: Rotation angle in degrees.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.colocmanager</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/colocmanager.py](../../src/napari_mAIcrobe/mAIcrobe/colocmanager.py)

### class ColocManager

```python
from napari_mAIcrobe.mAIcrobe.colocmanager import ColocManager
ColocManager()
```

- Parameters:
  - None.

- Attributes:
  - `report`: dict
    - Mapping of cell label (as str) to a dictionary of computed PCC's over various regions.

- Methods:
  - `save_report(reportID: str, sept: bool = False) -> None`
    - Write a CSV report with the per cell computed Pearson metrics stored in `self.report`.
    - Parameters:
      - `reportID`: Base output directory for the report CSV.
      - `sept`: Include septum metrics if available.
    - Side effects: creates `<reportID>/_pcc_report.csv`.

  - `pearsons_score(channel_1: ndarray, channel_2: ndarray, mask: ndarray) -> tuple[float, float]`
    - Compute Pearson correlation within a masked region after removing zeros.
    - Parameters:
      - `channel_1`: First channel crop.
      - `channel_2`: Second channel crop.
      - `mask`: Binary mask selecting pixels of interest.
    - Returns: `(r, pvalue)` from `scipy.stats.pearsonr`.

  - `computes_cell_pcc(fluor_image: ndarray, optional_image: ndarray, cell: Cell, parameters: dict) -> None`
    - Compute and store Pearson metrics for a single cell.
    - Parameters:
      - `fluor_image`: Full-field fluorescence image (channel 1).
      - `optional_image`: Full-field optional image (channel 2).
      - `cell`: Cell object with region masks and bounding box.
      - `parameters`: Analysis parameters including `find_septum`.
    - Side effects:
      - Adds entries to `self.report[key]`, where key is the cell label, for:
        `Whole Cell`, `Membrane`, `Cytoplasm`, and if `find_septum`: `Septum`, `MembSept`.
      - Stores cropped channels as `Channel 1` and `Channel 2`.
    - Notes: On ValueError (e.g., insufficient data), the cell entry is removed.

  - `compute_pcc(fluor_image: ndarray, optional_image: ndarray, cells: list[Cell], parameters: dict, reportID: str) -> None`
    - DEPRECATED. Use `computes_cell_pcc` instead.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.reports</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/reports.py](../../src/napari_mAIcrobe/mAIcrobe/reports.py)

### class ReportManager(parameters, properties, allcells)

```python
from napari_mAIcrobe.mAIcrobe.reports import ReportManager
ReportManager(parameters, properties, allcells)
```

- Parameters:
  - `parameters`: dict
    - Analysis parameters dictionary.
  - `properties`: dict
    - Per-cell properties dictionary (e.g., Label, Area, etc.).
  - `allcells`: list[ndarray]
    - List of per-cell montage images for visualization.

- Attributes:
  - `cells`: list[ndarray]
    - Padded per-cell mosaic images for uniform display.
  - `max_shape`: tuple[int, int]
    - Maximum shape across cell mosaics for padding reference.
  - `properties`: dict
    - Properties passed at initialization.
  - `params`: dict
    - Parameters passed at initialization.
  - `keys`: list[tuple[str, int]]
    - Property labels from params with display precision from `stats_format`.
  - `cell_data_filename`: str | None
    - Base path of the generated report directory.

- Methods:
  - `html_report(filename: str) -> None`
    - Write an HTML report composing cell thumbnails and stats.
    - Parameters:
      - `filename`: Output directory path for the HTML report and images.
    - Side effects: writes `html_report_.html` and `_images/all_cells.png` in `filename`.

  - `check_filename(filename: str) -> str`
    - Ensure a unique report directory by appending an index if the path exists.
    - Parameters:
      - `filename`: Base filename (without extension).
    - Returns: Available filename not colliding with existing path.

  - `generate_report(path: str, report_id: str | None = None) -> None`
    - Generate HTML report and CSV with properties.
    - Parameters:
      - `path`: Output directory.
      - `report_id`: Optional report identifier appended to directory name.
    - Side effects:
      - Creates directory structure (including `_images`), writes HTML and `Analysis.csv`,
        and sets `self.cell_data_filename`.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.cellprocessing</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/cellprocessing.py](../../src/napari_mAIcrobe/mAIcrobe/cellprocessing.py)

```python
from napari_mAIcrobe.mAIcrobe.cellprocessing import rotation_matrices

rotation_matrices(step)
```
- Generate rotation matrices (0 to <180 deg, transposed).
- Parameters:
  - `step`: int. Angular step in degrees.
- Returns:
  - list[numpy.matrix]: 2x2 rotation matrices (transposed).

```python
from napari_mAIcrobe.mAIcrobe.cellprocessing import bounded_value

bounded_value(minval, maxval, currval)
```
- Clamp a value within [minval, maxval].
- Parameters:
  - `minval`: float. Lower bound.
  - `maxval`: float. Upper bound.
  - `currval`: float. Value to clamp.
- Returns:
  - float: Clamped value.

```python
from napari_mAIcrobe.mAIcrobe.cellprocessing import bounded_point

bounded_point(x0, x1, y0, y1, p)
```
- Clamp a 2D point within a rectangular box.
- Parameters:
  - `x0`: float. Min x.
  - `x1`: float. Max x.
  - `y0`: float. Min y.
  - `y1`: float. Max y.
  - `p`: tuple[float, float]. Point (x, y).
- Returns:
  - tuple[float, float]: Clamped point coordinates.

```python
from napari_mAIcrobe.mAIcrobe.cellprocessing import bound_rectangle

bound_rectangle(points)
```
- Compute bounding rectangle from a list of points.
- Parameters:
  - `points`: ndarray. N x 2 array of (x, y) coordinates.
- Returns:
  - tuple[float, float, float, float, float]: (x0, y0, x1, y1, width) where width is min(x1-x0, y1-y0).

```python
from napari_mAIcrobe.mAIcrobe.cellprocessing import stats_format

stats_format(params)
```
- Select stats to include in reports based on parameters, and stores display precision.
- Parameters:
  - `params`: dict. Flags indicating optional computations (e.g., septum, cell cycle).
- Returns:
  - list[tuple[str, int]]: Pairs of (label, decimals) to include in report.

</details>

<details>
<summary><code>napari_mAIcrobe.mAIcrobe.cellcycleclassifier</code></summary>

Source: [../../src/napari_mAIcrobe/mAIcrobe/cellcycleclassifier.py](../../src/napari_mAIcrobe/mAIcrobe/cellcycleclassifier.py)

### class CellCycleClassifier(fluor_fov, optional_fov, model, model_path, model_input, max_dim)

```python
from napari_mAIcrobe.mAIcrobe.cellcycleclassifier import CellCycleClassifier
CellCycleClassifier(fluor_fov, optional_fov, model, model_path, model_input, max_dim)
```

- Parameters:
  - `fluor_fov`: ndarray
    - Primary fluorescence image (full field).
  - `optional_fov`: ndarray
    - Optional fluorescence image (full field).
  - `model`: str
    - Prebuilt model selector or "custom".
    - Options:
      - "S.aureus DNA+Membrane Epi" (default)
      - "S.aureus DNA+Membrane SIM"
      - "S.aureus DNA Epi"
      - "S.aureus DNA SIM"
      - "S.aureus Membrane Epi"
      - "S.aureus Membrane SIM"
      - "custom": Load a custom model from `model_path`.
  - `model_path`: str
    - Path to custom model when `model == "custom"`.
  - `model_input`: "Membrane" | "DNA" | "Membrane+DNA"
    - Which channels are used as input.
  - `max_dim`: int
    - Maximum dimension used to pad/crop per-cell images.

- Attributes:
  - `model`: keras.Model
    - Loaded classifier model.
  - `max_dim`: int
    - Preprocessing target size for per-cell crops.
  - `model_input`: str
    - Model input type.
  - `custom`: bool
    - Whether a custom model was loaded.
  - `fluor_fov`: ndarray
    - Stored primary fluorescence field-of-view.
  - `optional_fov`: ndarray
    - Stored optional field-of-view.

- Methods:
  - `preprocess_image(image: ndarray) -> ndarray`
    - Pad/crop and reshape an image to (max_dim, max_dim, 1).
    - Parameters:
      - `image`: 2D ndarray to preprocess.
    - Returns: float ndarray of shape (max_dim, max_dim, 1).

  - `classify_cell(cell_object) -> int`
    - Predict cell-cycle phase from per-cell crops.
    - Parameters:
      - `cell_object`: napari_mAIcrobe.mAIcrobe.cells.Cell with `box` and `cell_mask`.
    - Returns: Predicted phase index starting at 1.

</details>

---

## Further Resources

- User Guide: ../user-guide/getting-started.md
- Tutorials: ../tutorials/basic-workflow.md
- GitHub: https://github.com/HenriquesLab/napari-mAIcrobe
- Discussions (support): https://github.com/HenriquesLab/napari-mAIcrobe/discussions

Dependencies:
- napari: https://napari.org/
- TensorFlow: https://www.tensorflow.org/
- scikit-image: https://scikit-image.org/
- StarDist: https://github.com/stardist/stardist
- Cellpose: https://github.com/MouseLand/cellpose
