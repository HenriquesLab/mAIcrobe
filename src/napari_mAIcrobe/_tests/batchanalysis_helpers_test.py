from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_mAIcrobe import _batchanalysis as batch


def test_discover_fov_directories_rejects_missing_root(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        batch.discover_fov_directories(tmp_path / "missing")


def test_map_fov_files_rejects_empty_required_pattern(tmp_path):
    fov = tmp_path / "fov"
    fov.mkdir()
    (fov / "phase.tif").write_bytes(b"not really a tif")

    with pytest.raises(ValueError, match="cannot be empty"):
        batch.map_fov_files(fov, "", "*mem*.tif", "*dna*.tif")


def test_map_fov_files_allows_missing_optional_dna(tmp_path):
    fov = tmp_path / "fov"
    fov.mkdir()
    (fov / "phase.tif").write_bytes(b"")
    (fov / "mem.tif").write_bytes(b"")

    mapping = batch.map_fov_files(fov, "*phase*.tif", "*mem*.tif", "*dna*.tif")

    assert mapping.name == "fov"
    assert mapping.dna_file is None


def test_safe_report_id_sanitizes_names():
    assert batch._safe_report_id("FoV 01 / test!") == "FoV_01_test"
    assert batch._safe_report_id("!!!") == "fov"


def test_validate_2d_rejects_stack():
    with pytest.raises(ValueError, match="must be 2D"):
        batch._validate_2d("Base", np.zeros((2, 3, 3)))


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("Unet", "unet"),
        ("StarDist", "stardist"),
        ("CellPose cyto3", "cellpose"),
        ("Isodata", "classical"),
    ],
)
def test_segment_single_fov_dispatches(monkeypatch, algorithm, expected):
    calls = []

    monkeypatch.setattr(
        batch,
        "unet_segmentation",
        lambda *args: calls.append("unet") or ("mask", "labels"),
    )
    monkeypatch.setattr(
        batch,
        "stardist_segmentation",
        lambda *args: calls.append("stardist") or ("mask", "labels"),
    )
    monkeypatch.setattr(
        batch,
        "cellpose_segmentation",
        lambda *args: calls.append("cellpose") or ("mask", "labels"),
    )
    monkeypatch.setattr(
        batch,
        "classical_segmentation",
        lambda *args: calls.append("classical") or ("mask", "labels"),
    )

    result = batch._segment_single_fov(
        base_image=np.zeros((2, 2)),
        segmentation_algorithm=algorithm,
        binary_closing=0,
        binary_dilation=0,
        binary_fillholes=False,
        la_blocksize=151,
        la_offset=0.02,
        watershed_pars={},
        unet_model_type="Custom",
        unet_pretrained="",
        unet_model_path=Path("model.h5"),
        stardist_model_type="Custom",
        stardist_pretrained="",
        stardist_model_path=Path("model"),
    )

    assert calls == [expected]
    assert result == ("mask", "labels")


def test_cellmanager_params_maps_gui_values_to_internal_keys(tmp_path):
    params = batch._cellmanager_params(
        pixel_size=0.5,
        inner_mask_thickness=3,
        septum_algorithm="Box",
        baseline_margin=10,
        find_septum=True,
        find_open_septum=False,
        classify_cell_cycle=True,
        model="custom",
        custom_model_path=Path("model.keras"),
        custom_model_input="DNA",
        custom_model_maxsize=40,
        compute_colocalization=True,
        generate_report=True,
        report_path=tmp_path,
        report_id="fov_1",
    )

    assert params["pixel_size"] == 0.5
    assert params["septum_algorithm"] == "Box"
    assert params["custom_model_path"] == "model.keras"
    assert params["report_path"] == str(tmp_path)
    assert params["report_id"] == "fov_1"
    assert params["coloc"] is True


def test_update_visibility_helpers_toggle_expected_fields():
    def widget(value=None):
        return SimpleNamespace(value=value, visible=None)

    gui = SimpleNamespace(
        Segmentation_algorithm=widget("Unet"),
        Binary_closing=widget(),
        Binary_dilation=widget(),
        Binary_fillholes=widget(),
        LA_blocksize=widget(),
        LA_offset=widget(),
        Peak_min_distance_from_edge=widget(),
        Peak_min_distance=widget(),
        Peak_min_height=widget(),
        Max_peaks=widget(),
        Unet_model_type=widget("Pretrained"),
        Unet_pretrained=widget(),
        Unet_model_path=widget(),
        StarDist_model_type=widget("Custom"),
        StarDist_pretrained=widget(),
        StarDist_model_path=widget(),
        Advanced_mode=widget(True),
        Classify_cell_cycle=widget(True),
        Pixel_size=widget(),
        Inner_mask_thickness=widget(),
        Septum_algorithm=widget(),
        Baseline_margin=widget(),
        Find_septum=widget(),
        Find_open_septum=widget(),
        Compute_Colocalization=widget(),
        Generate_per_fov_report=widget(),
        Save_segmentation_tifs=widget(),
        Save_merged_csv=widget(),
        Continue_on_error=widget(),
        Model=widget("custom"),
        Custom_model_path=widget(),
        Custom_model_input=widget(),
        Custom_model_MaxSize=widget(),
    )

    batch._update_segmentation_visibility(gui)
    batch._update_model_visibility(gui)

    assert gui.Unet_pretrained.visible is True
    assert gui.Unet_model_path.visible is False
    assert gui.Peak_min_distance.visible is False
    assert gui.Custom_model_path.visible is True
    assert gui.Generate_per_fov_report.visible is True
