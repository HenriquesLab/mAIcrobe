from types import SimpleNamespace

import numpy as np

from napari_mAIcrobe.mAIcrobe.colocmanager import ColocManager


def _cell():
    mask = np.ones((3, 3))
    return SimpleNamespace(
        label=7,
        box=(1, 1, 3, 3),
        cell_mask=mask,
        perim_mask=mask,
        cyto_mask=mask,
        sept_mask=mask,
        membsept_mask=mask,
    )


def test_pearsons_score_calculates_masked_correlation():
    manager = ColocManager()
    channel_1 = np.arange(9, dtype=float).reshape(3, 3)
    channel_2 = channel_1 * 2
    mask = np.ones((3, 3))

    score, _pvalue = manager.pearsons_score(channel_1, channel_2, mask)

    assert score > 0.99


def test_pearsons_score_filters_nonzero_pixels_in_pairs():
    manager = ColocManager()
    channel_1 = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    channel_2 = np.array([[0.0, 4.0, 5.0], [0.0, 6.0, 0.0], [0.0, 0.0, 0.0]])

    score, _pvalue = manager.pearsons_score(
        channel_1, channel_2, np.ones((3, 3))
    )

    assert score > 0.99


def test_computes_cell_pcc_records_whole_cell_regions():
    manager = ColocManager()
    fluor = np.arange(25, dtype=float).reshape(5, 5) + 1
    optional = fluor * 3

    manager.computes_cell_pcc(
        fluor,
        optional,
        _cell(),
        {"find_septum": True},
        cell_label="frame0:7",
    )

    report = manager.report["frame0:7"]
    assert report["Whole Cell"] > 0.99
    assert report["Membrane"] > 0.99
    assert report["Cytoplasm"] > 0.99
    assert report["Septum"] > 0.99
    assert report["MembSept"] > 0.99


def test_computes_cell_pcc_drops_cells_with_too_few_pixels():
    manager = ColocManager()
    cell = _cell()
    cell.cell_mask = np.zeros((3, 3))

    manager.computes_cell_pcc(
        np.ones((5, 5)),
        np.ones((5, 5)),
        cell,
        {"find_septum": False},
    )

    assert manager.report == {}


def test_save_report_writes_sorted_semicolon_csv(tmp_path):
    manager = ColocManager()
    manager.report = {
        "2": {"Whole Cell": 0.2, "Membrane": 0.3, "Cytoplasm": 0.4},
        "1": {"Whole Cell": 0.1, "Membrane": 0.2, "Cytoplasm": 0.3},
    }

    manager.save_report(str(tmp_path), sept=False)

    lines = (tmp_path / "_pcc_report.csv").read_text().splitlines()
    assert lines[0] == "Cell ID;Whole Cell;Membrane;Cytoplasm;"
    assert lines[1].startswith("1;")
    assert lines[2].startswith("2;")
