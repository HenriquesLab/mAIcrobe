from pathlib import Path

import pytest
from skimage.io import imread

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"


@pytest.fixture
def phase_example():
    return imread(DOCS_DIR / "test_phase.tif")


@pytest.fixture
def membrane_example():
    return imread(DOCS_DIR / "test_membrane.tif")


@pytest.fixture
def dna_example():
    return imread(DOCS_DIR / "test_dna.tif")


@pytest.fixture
def all_sample_data():
    return (
        imread(DOCS_DIR / "test_phase.tif"),
        imread(DOCS_DIR / "test_membrane.tif"),
        imread(DOCS_DIR / "test_dna.tif"),
    )
