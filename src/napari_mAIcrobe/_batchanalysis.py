"""Batch FoV segmentation and analysis widget and utilities."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from magicgui import magic_factory
from skimage.io import imread, imsave

from .mAIcrobe.cells import CellManager
from .mAIcrobe.segmentation import (
    cellpose_segmentation,
    classical_segmentation,
    stardist_segmentation,
    unet_segmentation,
)

if TYPE_CHECKING:
    import napari


@dataclass
class FoVMapping:
    """Resolved channel files for one FoV directory."""

    name: str
    folder: Path
    base_file: Path
    membrane_file: Path
    dna_file: Path | None


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _list_tiff_files(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and _is_tiff(p)),
        key=lambda p: p.name.lower(),
    )


def discover_fov_directories(input_root: os.PathLike | str) -> list[Path]:
    """Return direct child folders containing at least one TIFF file."""

    root = Path(input_root)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Input root does not exist or is not a dir: {root}")

    fov_dirs = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if len(_list_tiff_files(child)) > 0:
            fov_dirs.append(child)

    return fov_dirs


def _single_pattern_match(
    files: list[Path],
    pattern: str,
    role_name: str,
    required: bool,
) -> Path | None:
    if pattern.strip() == "":
        if required:
            raise ValueError(f"Pattern for {role_name} cannot be empty")
        return None

    lowered_pattern = pattern.lower()
    matches = [
        file_path
        for file_path in files
        if fnmatch(file_path.name.lower(), lowered_pattern)
    ]

    if len(matches) == 0:
        if required:
            raise ValueError(
                f"No file matched {role_name} pattern '{pattern}'"
            )
        return None

    if len(matches) > 1:
        match_names = ", ".join(m.name for m in matches)
        raise ValueError(
            f"Ambiguous {role_name} pattern '{pattern}'. Matches: {match_names}"
        )

    return matches[0]


def map_fov_files(
    fov_dir: os.PathLike | str,
    base_pattern: str,
    membrane_pattern: str,
    dna_pattern: str,
) -> FoVMapping:
    """Resolve base/membrane/dna files in one FoV folder by glob patterns."""

    folder = Path(fov_dir)
    tif_files = _list_tiff_files(folder)

    if len(tif_files) == 0:
        raise ValueError(f"No TIFF files found in {folder}")

    base_file = _single_pattern_match(
        tif_files,
        pattern=base_pattern,
        role_name="base",
        required=True,
    )
    membrane_file = _single_pattern_match(
        tif_files,
        pattern=membrane_pattern,
        role_name="membrane",
        required=True,
    )
    dna_file = _single_pattern_match(
        tif_files,
        pattern=dna_pattern,
        role_name="dna",
        required=False,
    )

    return FoVMapping(
        name=folder.name,
        folder=folder,
        base_file=base_file,
        membrane_file=membrane_file,
        dna_file=dna_file,
    )


def _safe_report_id(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    return sanitized if sanitized else "fov"


def _validate_2d(name: str, image) -> None:
    if image.ndim != 2:
        raise ValueError(f"{name} image must be 2D. Got shape {image.shape}")


def _segment_single_fov(
    base_image,
    segmentation_algorithm: str,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
    la_blocksize: int,
    la_offset: float,
    watershed_pars: dict,
    unet_model_type: str,
    unet_pretrained: str,
    unet_model_path: os.PathLike | str,
    stardist_model_type: str,
    stardist_pretrained: str,
    stardist_model_path: os.PathLike | str,
):
    if segmentation_algorithm == "Unet":
        return unet_segmentation(
            base_image,
            unet_model_type,
            unet_pretrained,
            str(unet_model_path),
            binary_closing,
            binary_dilation,
            binary_fillholes,
        )

    if segmentation_algorithm == "StarDist":
        return stardist_segmentation(
            base_image,
            stardist_model_type,
            stardist_pretrained,
            str(stardist_model_path),
        )

    if segmentation_algorithm == "CellPose cyto3":
        return cellpose_segmentation(base_image)

    return classical_segmentation(
        base_image,
        segmentation_algorithm,
        la_blocksize,
        la_offset,
        binary_closing,
        binary_dilation,
        binary_fillholes,
        watershed_pars,
    )


def _cellmanager_params(
    pixel_size: float,
    inner_mask_thickness: int,
    septum_algorithm: str,
    baseline_margin: int,
    find_septum: bool,
    find_open_septum: bool,
    classify_cell_cycle: bool,
    model: str,
    custom_model_path: os.PathLike | str,
    custom_model_input: str,
    custom_model_maxsize: int,
    compute_colocalization: bool,
    generate_report: bool,
    report_path: Path,
    report_id: str,
):
    return {
        "pixel_size": pixel_size,
        "inner_mask_thickness": inner_mask_thickness,
        "septum_algorithm": septum_algorithm,
        "baseline_margin": baseline_margin,
        "find_septum": find_septum,
        "find_openseptum": find_open_septum,
        "classify_cell_cycle": classify_cell_cycle,
        "model": model,
        "custom_model_path": str(custom_model_path),
        "custom_model_input": custom_model_input,
        "custom_model_maxsize": custom_model_maxsize,
        "generate_report": generate_report,
        "report_path": str(report_path),
        "report_id": report_id,
        "cell_averager": False,
        "coloc": compute_colocalization,
    }


def run_batch_analysis(
    input_root: os.PathLike | str,
    output_root: os.PathLike | str,
    base_pattern: str,
    membrane_pattern: str,
    dna_pattern: str,
    segmentation_algorithm: str,
    binary_closing: int,
    binary_dilation: int,
    binary_fillholes: bool,
    la_blocksize: int,
    la_offset: float,
    peak_min_distance_from_edge: int,
    peak_min_distance: int,
    peak_min_height: int,
    max_peaks: int,
    unet_model_type: str,
    unet_pretrained: str,
    unet_model_path: os.PathLike | str,
    stardist_model_type: str,
    stardist_pretrained: str,
    stardist_model_path: os.PathLike | str,
    pixel_size: float,
    inner_mask_thickness: int,
    septum_algorithm: str,
    baseline_margin: int,
    find_septum: bool,
    find_open_septum: bool,
    classify_cell_cycle: bool,
    model: str,
    custom_model_path: os.PathLike | str,
    custom_model_input: str,
    custom_model_maxsize: int,
    compute_colocalization: bool,
    generate_per_fov_report: bool,
    save_segmentation_tifs: bool,
    save_merged_csv: bool,
    continue_on_error: bool,
) -> dict:
    """Run segmentation and per-cell analysis for all FoV folders."""

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    watershed_pars = {
        "peak_min_distance_from_edge": peak_min_distance_from_edge,
        "peak_min_distance": peak_min_distance,
        "peak_min_height": peak_min_height,
        "max_peaks": max_peaks,
    }

    fov_dirs = discover_fov_directories(input_root)
    if len(fov_dirs) == 0:
        raise ValueError("No FoV folders with TIFF files were found")

    merged_rows = []
    errors = []
    success_count = 0

    last_mask = None
    last_labels = None

    for index, fov_dir in enumerate(fov_dirs, start=1):
        fov_name = fov_dir.name
        print(f"[{index}/{len(fov_dirs)}] Processing {fov_name}")
        fov_output = output_root_path / fov_name
        fov_output.mkdir(parents=True, exist_ok=True)

        try:
            mapping = map_fov_files(
                fov_dir,
                base_pattern=base_pattern,
                membrane_pattern=membrane_pattern,
                dna_pattern=dna_pattern,
            )

            base_image = imread(str(mapping.base_file))
            membrane_image = imread(str(mapping.membrane_file))
            dna_image = (
                imread(str(mapping.dna_file))
                if mapping.dna_file is not None
                else None
            )

            _validate_2d("Base", base_image)
            _validate_2d("Membrane", membrane_image)
            if dna_image is not None:
                _validate_2d("DNA", dna_image)

            mask, labels = _segment_single_fov(
                base_image=base_image,
                segmentation_algorithm=segmentation_algorithm,
                binary_closing=binary_closing,
                binary_dilation=binary_dilation,
                binary_fillholes=binary_fillholes,
                la_blocksize=la_blocksize,
                la_offset=la_offset,
                watershed_pars=watershed_pars,
                unet_model_type=unet_model_type,
                unet_pretrained=unet_pretrained,
                unet_model_path=unet_model_path,
                stardist_model_type=stardist_model_type,
                stardist_pretrained=stardist_pretrained,
                stardist_model_path=stardist_model_path,
            )

            if save_segmentation_tifs:
                imsave(
                    str(fov_output / "mask.tif"),
                    mask.astype("uint16"),
                    check_contrast=False,
                )
                imsave(
                    str(fov_output / "labels.tif"),
                    labels.astype("uint16"),
                    check_contrast=False,
                )

            params = _cellmanager_params(
                pixel_size=pixel_size,
                inner_mask_thickness=inner_mask_thickness,
                septum_algorithm=septum_algorithm,
                baseline_margin=baseline_margin,
                find_septum=find_septum,
                find_open_septum=find_open_septum,
                classify_cell_cycle=classify_cell_cycle,
                model=model,
                custom_model_path=custom_model_path,
                custom_model_input=custom_model_input,
                custom_model_maxsize=custom_model_maxsize,
                compute_colocalization=compute_colocalization,
                generate_report=generate_per_fov_report,
                report_path=fov_output,
                report_id=_safe_report_id(fov_name),
            )

            cell_man = CellManager(
                label_img=labels,
                fluor=membrane_image,
                optional=dna_image,
                params=params,
            )
            cell_man.compute_cell_properties()

            fov_df = pd.DataFrame(cell_man.properties)
            fov_df.insert(0, "fov_path", str(fov_dir))
            fov_df.insert(0, "fov_name", fov_name)
            merged_rows.append(fov_df)

            success_count += 1
            last_mask = mask
            last_labels = labels

        except Exception as exc:
            errors.append(
                {
                    "fov_name": fov_name,
                    "fov_path": str(fov_dir),
                    "error": str(exc),
                }
            )
            print(f"Failed {fov_name}: {exc}")
            if not continue_on_error:
                raise

    merged_csv_path = output_root_path / "batch_merged_analysis.csv"
    if save_merged_csv:
        if len(merged_rows) > 0:
            pd.concat(merged_rows, ignore_index=True).to_csv(
                merged_csv_path, index=False
            )
        else:
            pd.DataFrame(columns=["fov_name", "fov_path"]).to_csv(
                merged_csv_path, index=False
            )

    errors_csv_path = output_root_path / "batch_errors.csv"
    pd.DataFrame(errors).to_csv(errors_csv_path, index=False)

    summary = {
        "total_fovs": len(fov_dirs),
        "success_fovs": success_count,
        "failed_fovs": len(errors),
        "merged_csv": str(merged_csv_path),
        "errors_csv": str(errors_csv_path),
        "last_mask": last_mask,
        "last_labels": last_labels,
    }
    return summary


def _update_segmentation_visibility(gui) -> None:
    """Show only controls needed for the selected segmentation algorithm."""

    algorithm = gui.Segmentation_algorithm.value

    is_unet = algorithm == "Unet"
    is_stardist = algorithm == "StarDist"
    is_classical = algorithm in {"Isodata", "Local Average"}

    show_binary_ops = algorithm in {
        "Isodata",
        "Local Average",
        "Unet",
    }
    gui.Binary_closing.visible = show_binary_ops
    gui.Binary_dilation.visible = show_binary_ops
    gui.Binary_fillholes.visible = show_binary_ops

    gui.LA_blocksize.visible = algorithm == "Local Average"
    gui.LA_offset.visible = algorithm == "Local Average"

    gui.Peak_min_distance_from_edge.visible = is_classical
    gui.Peak_min_distance.visible = is_classical
    gui.Peak_min_height.visible = is_classical
    gui.Max_peaks.visible = is_classical

    gui.Unet_model_type.visible = is_unet
    gui.Unet_pretrained.visible = (
        is_unet and gui.Unet_model_type.value == "Pretrained"
    )
    gui.Unet_model_path.visible = (
        is_unet and gui.Unet_model_type.value == "Custom"
    )

    gui.StarDist_model_type.visible = is_stardist
    gui.StarDist_pretrained.visible = (
        is_stardist and gui.StarDist_model_type.value == "Pretrained"
    )
    gui.StarDist_model_path.visible = (
        is_stardist and gui.StarDist_model_type.value == "Custom"
    )


def _update_model_visibility(gui) -> None:
    """Show analysis/classifier controls according to Advanced mode."""

    advanced_mode = gui.Advanced_mode.value
    classify_cell_cycle = gui.Classify_cell_cycle.value

    gui.Pixel_size.visible = advanced_mode
    gui.Inner_mask_thickness.visible = advanced_mode
    gui.Septum_algorithm.visible = advanced_mode
    gui.Baseline_margin.visible = advanced_mode
    gui.Find_septum.visible = advanced_mode
    gui.Find_open_septum.visible = advanced_mode

    gui.Compute_Colocalization.visible = True
    gui.Generate_per_fov_report.visible = advanced_mode
    gui.Save_segmentation_tifs.visible = advanced_mode
    gui.Save_merged_csv.visible = advanced_mode
    gui.Continue_on_error.visible = advanced_mode

    gui.Model.visible = classify_cell_cycle
    is_custom_model = gui.Model.value == "custom"
    gui.Custom_model_path.visible = classify_cell_cycle and is_custom_model
    gui.Custom_model_input.visible = classify_cell_cycle and is_custom_model
    gui.Custom_model_MaxSize.visible = classify_cell_cycle and is_custom_model


def _init_batch_widget(gui) -> None:
    """Connect UI signals and initialize field visibility."""

    gui.Advanced_mode.changed.connect(
        lambda _value: _update_segmentation_visibility(gui)
    )
    gui.Advanced_mode.changed.connect(
        lambda _value: _update_model_visibility(gui)
    )
    gui.Segmentation_algorithm.changed.connect(
        lambda _value: _update_segmentation_visibility(gui)
    )
    gui.Unet_model_type.changed.connect(
        lambda _value: _update_segmentation_visibility(gui)
    )
    gui.StarDist_model_type.changed.connect(
        lambda _value: _update_segmentation_visibility(gui)
    )
    gui.Classify_cell_cycle.changed.connect(
        lambda _value: _update_model_visibility(gui)
    )
    gui.Model.changed.connect(lambda _value: _update_model_visibility(gui))

    _update_segmentation_visibility(gui)
    _update_model_visibility(gui)


@magic_factory(
    widget_init=_init_batch_widget,
    Input_root={"widget_type": "FileEdit", "mode": "d"},
    Output_root={"widget_type": "FileEdit", "mode": "d"},
    Segmentation_algorithm={
        "choices": [
            "Isodata",
            "Local Average",
            "Unet",
            "StarDist",
            "CellPose cyto3",
        ]
    },
    Unet_model_type={"choices": ["Pretrained", "Custom"]},
    Unet_pretrained={
        "choices": [
            "Ph.C. S. pneumo",
            "WF FtsZ B. subtilis",
            "Unet S. aureus",
        ]
    },
    Unet_model_path={"widget_type": "FileEdit", "mode": "r"},
    StarDist_model_type={"choices": ["Pretrained", "Custom"]},
    StarDist_pretrained={"choices": ["StarDist S. aureus"]},
    StarDist_model_path={"widget_type": "FileEdit", "mode": "d"},
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
)
def batch_analysis(
    Viewer: napari.Viewer,
    Input_root: os.PathLike = "",
    Output_root: os.PathLike = "",
    Advanced_mode: bool = False,
    Base_pattern: str = "*phase*.tif*",
    Membrane_pattern: str = "*mem*.tif*",
    DNA_pattern: str = "*dna*.tif*",
    Segmentation_algorithm: str = "Isodata",
    Binary_closing: int = 0,
    Binary_dilation: int = 0,
    Binary_fillholes: bool = False,
    LA_blocksize: int = 151,
    LA_offset: float = 0.02,
    Peak_min_distance_from_edge: int = 10,
    Peak_min_distance: int = 5,
    Peak_min_height: int = 5,
    Max_peaks: int = 100000,
    Unet_model_type: str = "Pretrained",
    Unet_pretrained: str = "Ph.C. S. pneumo",
    Unet_model_path: os.PathLike = "",
    StarDist_model_type: str = "Pretrained",
    StarDist_pretrained: str = "StarDist S. aureus",
    StarDist_model_path: os.PathLike = "",
    Pixel_size: float = 1.0,
    Inner_mask_thickness: int = 4,
    Septum_algorithm: str = "Isodata",
    Baseline_margin: int = 30,
    Find_septum: bool = False,
    Find_open_septum: bool = False,
    Classify_cell_cycle: bool = False,
    Model: str = "S.aureus DNA+Membrane Epi",
    Custom_model_path: os.PathLike = "",
    Custom_model_input: str = "Membrane",
    Custom_model_MaxSize: int = 50,
    Compute_Colocalization: bool = False,
    Generate_per_fov_report: bool = True,
    Save_segmentation_tifs: bool = True,
    Save_merged_csv: bool = True,
    Continue_on_error: bool = True,
):
    """Batch analyze one root folder with one subfolder per FoV."""

    summary = run_batch_analysis(
        input_root=Input_root,
        output_root=Output_root,
        base_pattern=Base_pattern,
        membrane_pattern=Membrane_pattern,
        dna_pattern=DNA_pattern,
        segmentation_algorithm=Segmentation_algorithm,
        binary_closing=Binary_closing,
        binary_dilation=Binary_dilation,
        binary_fillholes=Binary_fillholes,
        la_blocksize=LA_blocksize,
        la_offset=LA_offset,
        peak_min_distance_from_edge=Peak_min_distance_from_edge,
        peak_min_distance=Peak_min_distance,
        peak_min_height=Peak_min_height,
        max_peaks=Max_peaks,
        unet_model_type=Unet_model_type,
        unet_pretrained=Unet_pretrained,
        unet_model_path=Unet_model_path,
        stardist_model_type=StarDist_model_type,
        stardist_pretrained=StarDist_pretrained,
        stardist_model_path=StarDist_model_path,
        pixel_size=Pixel_size,
        inner_mask_thickness=Inner_mask_thickness,
        septum_algorithm=Septum_algorithm,
        baseline_margin=Baseline_margin,
        find_septum=Find_septum,
        find_open_septum=Find_open_septum,
        classify_cell_cycle=Classify_cell_cycle,
        model=Model,
        custom_model_path=Custom_model_path,
        custom_model_input=Custom_model_input,
        custom_model_maxsize=Custom_model_MaxSize,
        compute_colocalization=Compute_Colocalization,
        generate_per_fov_report=Generate_per_fov_report,
        save_segmentation_tifs=Save_segmentation_tifs,
        save_merged_csv=Save_merged_csv,
        continue_on_error=Continue_on_error,
    )

    print(
        "Batch finished. "
        f"Total={summary['total_fovs']} "
        f"Success={summary['success_fovs']} "
        f"Failed={summary['failed_fovs']}"
    )
    print(f"Merged CSV: {summary['merged_csv']}")
    print(f"Errors CSV: {summary['errors_csv']}")
