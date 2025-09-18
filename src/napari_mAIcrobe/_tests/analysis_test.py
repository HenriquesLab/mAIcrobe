import numpy as np

from napari_mAIcrobe.mAIcrobe.cells import CellManager


def test_cellmanager_single_cell(membrane_example, dna_example):
    # Build a single rectangular label well inside the image
    h, w = membrane_example.shape
    lbl = np.zeros((h, w), dtype=np.int32)
    x0, y0 = h // 4, w // 4
    x1, y1 = x0 + min(40, h // 3), y0 + min(40, w // 3)
    lbl[x0:x1, y0:y1] = 1

    params = {
        "pixel_size": 1.0,
        "inner_mask_thickness": 4,
        "septum_algorithm": "Isodata",
        "baseline_margin": 30,
        "find_septum": False,
        "find_openseptum": False,
        "classify_cell_cycle": False,
        "model": "S.aureus DNA+Membrane Epi",
        "custom_model_path": "",
        "custom_model_input": "Membrane",
        "custom_model_maxsize": 50,
        "generate_report": False,
        "report_path": "",
        "cell_averager": False,
        "coloc": False,
    }

    cm = CellManager(
        label_img=lbl,
        fluor=membrane_example,
        optional=dna_example,
        params=params,
    )
    cm.compute_cell_properties()

    props = cm.properties
    assert props is not None
    assert "label" in props and len(props["label"]) == 1
    # spot-check a few expected keys
    for k in (
        "Area",
        "Perimeter",
        "Eccentricity",
        "Baseline",
        "Cell Median",
        "Membrane Median",
        "Cytoplasm Median",
        "DNA Ratio",
    ):
        assert k in props and len(props[k]) == 1

    # Check that area matches
    assert lbl.sum() == props["Area"][0]


# Add more tests that cover different parameters and edge cases
