from pathlib import Path

import numpy as np
from skimage.io import imsave

from napari_mAIcrobe._batchanalysis import (
    discover_fov_directories,
    map_fov_files,
    run_batch_analysis,
)


def _make_test_image(shape=(64, 64)):
    img = np.zeros(shape, dtype=np.uint16)
    img[12:26, 12:26] = 180
    img[36:52, 38:54] = 240
    return img


def _write_tif(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(path), data, check_contrast=False)


def test_discovery_and_mapping(tmp_path):
    root = tmp_path / "input"
    fov_a = root / "fov_a"
    fov_b = root / "fov_b"
    fov_empty = root / "fov_empty"

    image = _make_test_image()
    _write_tif(fov_a / "sample_phase.tif", image)
    _write_tif(fov_a / "sample_mem.tif", image)
    _write_tif(fov_a / "sample_dna.tif", image)

    _write_tif(fov_b / "x_phase.tif", image)
    _write_tif(fov_b / "x_mem_a.tif", image)
    _write_tif(fov_b / "x_mem_b.tif", image)

    fov_empty.mkdir(parents=True, exist_ok=True)

    discovered = discover_fov_directories(root)
    assert [d.name for d in discovered] == ["fov_a", "fov_b"]

    mapped = map_fov_files(
        fov_a,
        base_pattern="*phase*.tif",
        membrane_pattern="*mem*.tif",
        dna_pattern="*dna*.tif",
    )
    assert mapped.base_file.name == "sample_phase.tif"
    assert mapped.membrane_file.name == "sample_mem.tif"
    assert mapped.dna_file.name == "sample_dna.tif"

    try:
        map_fov_files(
            fov_b,
            base_pattern="*phase*.tif",
            membrane_pattern="*mem*.tif",
            dna_pattern="*dna*.tif",
        )
    except ValueError as exc:
        assert "Ambiguous membrane pattern" in str(exc)
    else:
        raise AssertionError("Expected ambiguous membrane match to fail")


def test_run_batch_analysis_outputs(tmp_path):
    input_root = tmp_path / "batch_input"
    output_root = tmp_path / "batch_output"

    fov_a = input_root / "fov_a"
    fov_b = input_root / "fov_b"
    fov_bad = input_root / "fov_bad"

    base = _make_test_image()
    membrane = _make_test_image()
    dna = _make_test_image()

    _write_tif(fov_a / "phase.tif", base)
    _write_tif(fov_a / "mem.tif", membrane)
    _write_tif(fov_a / "dna.tif", dna)

    _write_tif(fov_b / "phase.tif", base)
    _write_tif(fov_b / "mem.tif", membrane)

    _write_tif(fov_bad / "only_mem.tif", membrane)

    summary = run_batch_analysis(
        input_root=input_root,
        output_root=output_root,
        base_pattern="*phase*.tif",
        membrane_pattern="*mem*.tif",
        dna_pattern="*dna*.tif",
        segmentation_algorithm="Isodata",
        binary_closing=0,
        binary_dilation=0,
        binary_fillholes=False,
        la_blocksize=151,
        la_offset=0.02,
        peak_min_distance_from_edge=10,
        peak_min_distance=5,
        peak_min_height=5,
        max_peaks=100000,
        unet_model_type="Pretrained",
        unet_pretrained="Ph.C. S. pneumo",
        unet_model_path="",
        stardist_model_type="Pretrained",
        stardist_pretrained="StarDist S. aureus",
        stardist_model_path="",
        pixel_size=1.0,
        inner_mask_thickness=4,
        septum_algorithm="Isodata",
        baseline_margin=30,
        find_septum=False,
        find_open_septum=False,
        classify_cell_cycle=False,
        model="S.aureus Membrane Epi",
        custom_model_path="",
        custom_model_input="Membrane",
        custom_model_maxsize=50,
        compute_colocalization=False,
        generate_per_fov_report=True,
        save_segmentation_tifs=True,
        save_merged_csv=True,
        continue_on_error=True,
    )

    assert summary["total_fovs"] == 3
    assert summary["success_fovs"] == 2
    assert summary["failed_fovs"] == 1

    assert (output_root / "fov_a" / "mask.tif").exists()
    assert (output_root / "fov_a" / "labels.tif").exists()
    assert (output_root / "fov_b" / "mask.tif").exists()
    assert (output_root / "fov_b" / "labels.tif").exists()

    assert (output_root / "fov_a" / "Report_fov_a_1" / "Analysis.csv").exists()
    assert (output_root / "fov_b" / "Report_fov_b_1" / "Analysis.csv").exists()

    assert (output_root / "batch_merged_analysis.csv").exists()
    assert (output_root / "batch_errors.csv").exists()
