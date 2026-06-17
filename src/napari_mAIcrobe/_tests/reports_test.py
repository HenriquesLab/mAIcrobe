import numpy as np
import pandas as pd
import pytest

from napari_mAIcrobe.mAIcrobe.reports import ReportManager


def _params():
    return {
        "include_frame": False,
        "find_septum": False,
        "find_openseptum": False,
        "classify_cell_cycle": False,
    }


def _properties():
    return {
        "label": np.array([1]),
        "Area": np.array([4.0]),
        "Perimeter": np.array([8.0]),
        "Eccentricity": np.array([0.5]),
        "Baseline": np.array([1.0]),
        "Cell Median": np.array([2.0]),
        "Membrane Median": np.array([3.0]),
        "Cytoplasm Median": np.array([4.0]),
        "Cell Cycle Phase": np.array([2]),
    }


def test_report_manager_pads_cells_to_common_shape():
    cells = [np.zeros((2, 3)), np.ones((4, 2))]

    report = ReportManager(_params(), _properties(), cells)

    assert report.max_shape.tolist() == [4, 3]
    assert [cell.shape for cell in report.cells] == [(4, 3), (4, 3)]
    assert report.cells[0][0, 0] == 1


def test_generate_report_with_no_cells_still_writes_csv_and_html(tmp_path):
    report = ReportManager(_params(), _properties(), [])

    report.generate_report(str(tmp_path))

    report_dir = tmp_path / "Report_1"
    assert (report_dir / "html_report_.html").exists()
    csv_path = report_dir / "Analysis.csv"
    assert csv_path.exists()
    assert pd.read_csv(csv_path)["label"].tolist() == [1]


def test_generate_report_with_cell_writes_image_and_phase_counts(tmp_path):
    params = {**_params(), "classify_cell_cycle": True}
    properties = _properties()
    cell = np.zeros((3, 14), dtype=float)

    report = ReportManager(params, properties, [cell])
    report.generate_report(str(tmp_path), report_id="sample")

    report_dir = tmp_path / "Report_sample_1"
    assert (report_dir / "_images" / "all_cells.png").exists()
    html = (report_dir / "html_report_.html").read_text(encoding="utf-16")
    assert "Total cells: 1" in html
    assert "Phase 2 cells: 1" in html


def test_check_filename_increments_numeric_suffix(tmp_path):
    (tmp_path / "Report_1").mkdir()
    report = ReportManager(_params(), _properties(), [])

    assert report.check_filename(str(tmp_path / "Report_1")).endswith(
        "Report_2"
    )


def test_check_filename_non_numeric_suffix_currently_fails(tmp_path):
    (tmp_path / "Report_sample").mkdir()
    report = ReportManager(_params(), _properties(), [])

    with pytest.raises(ValueError):
        report.check_filename(str(tmp_path / "Report_sample"))
