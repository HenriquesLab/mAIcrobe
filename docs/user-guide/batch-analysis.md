# Batch Analysis Workflow

This tutorial explains how to run mAIcrobe batch analysis over multiple fields of view (FoVs) using a folder-based workflow.

## Goal

Process many FoVs in one run and generate:

- Per-FoV segmentation/analysis outputs
- One merged table with all detected cells
- One error table listing failed FoVs

## Expected input folder structure

Create one root folder where each direct subfolder is one FoV.

```text
input_root/
  FoV_001/
    FileName1.tif
    FileName2.tif
    FileName3.tif
  FoV_002/
    FileName1.tif
    FileName2.tif
    FileName3.tif
  FoV_003/
    FileName1.tif
    FileName2.tif
    FileName3.tif
  FoV_004/
    ...
```

Notes:

- FoV discovery is based on direct child folders containing TIFF files
- Channel files are resolved by filename patterns configured in the widget
- The amount of fluorescence channels can be optional depending on your analysis setup
- Pattern matching is case-insensitive and must match exactly one file per required channel in each FoV

## Step-by-step in napari

1. Open `Plugins > mAIcrobe > Batch analysis`
2. Set the main parameters:
  - `Input_root`: parent folder that contains one direct subfolder per FoV.
  - `Output_root`: destination where all batch outputs will be saved.
  - `Advanced mode`: toggle to show additional controls for segmentation, analysis, and output behavior.
  - `Base pattern`: pattern used to find the segmentation input image inside each FoV folder (example: `*phase*.tif*`).
  - `Membrane pattern`: pattern used to find the membrane channel or other fluorescent channel inside each FoV folder (example: `*mem*.tif*`).
  - `DNA pattern`: pattern used to find the DNA channel or other fluorescent channel inside each FoV folder (example: `*dna*.tif*`). Can be left unmatched if your workflow does not require it.
  - `Segmentation algorithm`: model/method used for all FoVs (`Isodata`, `Local Average`, `Unet`, `StarDist`, or `CellPose cyto3`). Additional controls will appear according to the selected model.
  - `Auto_align`: align fluorescence channels to the computed mask before cell analysis.
  - `Classify cell cycle`: enable to run classification with a pretrained or custom model.
  - `Compute colocalization`: enable to compute per-cell colocalization metrics.
3. Click **Run**


## Segmentation behavior

- `Binary_closing`: morphological closing iterations applied after segmentation (when supported).
- `Binary_dilation`: dilation iterations to expand segmented regions (when supported).
- `Binary_fillholes`: fill holes inside binary masks.
- `Auto_align`: align fluorescence channels to the computed mask before cell analysis.

### Local Average and watershed controls

These are relevant for `Local Average`.

- `LA_blocksize`: odd window size used by local thresholding.
- `LA_offset`: threshold offset applied in local thresholding.

These are relevant for `Isodata` or `Local Average` and other methods and employ watershed workflows.

- `Peak_min_distance_from_edge`: minimum distance from image edge for candidate peaks.
- `Peak_min_distance`: minimum distance between watershed seed peaks.
- `Peak_min_height`: minimum peak intensity/height for watershed seeds.
- `Max_peaks`: hard cap on the number of detected peaks.

### UNet and StarDist model selection

- `Unet_model_type`: choose `Pretrained` or `Custom` for UNet segmentation.
- `Unet_pretrained`: select which bundled UNet model to use.
- `Unet_model_path`: path to a custom UNet model file.


- `StarDist_model_type`: choose `Pretrained` or `Custom` for StarDist segmentation.
- `StarDist_pretrained`: select which bundled StarDist model to use.
- `StarDist_model_path`: path to a custom StarDist model directory.

## Classification parameters

Visible when `Classify cell cycle` is enabled.

- `Model`: pretrained classifier name or `custom`.
- `Custom_model_path`: path to custom classifier model file.
- `Custom_model_input`: expected input modality (`Membrane`, `DNA`, or `Membrane+DNA`).
- `Custom_model_MaxSize`: max cell size used in custom-classifier preprocessing.

## Advanced Mode parameters

When `Advanced mode` is enabled, the widget exposes extra controls.

### Analysis behavior

- `Pixel_size`: Physical pixel size (e.g., 0.065 μm/pixel) (optional)
- `Inner_mask_thickness`:  Membrane thickness for cytoplasmic measurements (default: 4)
- `Septum_algorithm`: "Isodata" or "Box" thresholding
- `Find_septum`: Whether to detect division septa
- `Find_open_septum`: also detect incomplete/open septa.
- `Baseline_margin`: margin used to estimate local background (default: 30).

### Output and failure-handling behavior

- `Generate_per_fov_report`: generate one report per FoV (enabled by default).
- `Save_segmentation_tifs`: save `mask.tif` and `labels.tif` for each FoV.
- `Save_merged_csv`: write `batch_merged_analysis.csv`.
- `Continue_on_error`: continue processing remaining FoVs when one FoV fails.


## Output files

After completion, the output root contains:

- `batch_merged_analysis.csv`: merged per-cell table across successful FoVs
- `batch_errors.csv`: one row per failed FoV with error message
- one output folder per FoV, typically containing:
  - `mask.tif`
  - `labels.tif`
  - optional HTML/CSV report files (advanced mode option, true by default)

## Quality control recommendations

- Test on 2-3 FoVs first to validate file patterns
- Inspect `batch_errors.csv` after each large run
- Keep consistent channel naming across FoVs

## Troubleshooting tips

- **No FoVs found**: ensure FoV folders are direct children of `Input_root`
- **No TIFF files found**: verify file extensions are `.tif` or `.tiff`
- **Channel mismatch**: adjust filename patterns for base/membrane/dna

## Related pages

- [Cell Analysis](./cell-analysis.md)
- [Timelapse Analysis](./timelapse-analysis.md)
- [Troubleshooting](./troubleshooting.md)
