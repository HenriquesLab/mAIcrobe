import pytest
from skimage.io import imread


@pytest.fixture
def phase_example():
    return imread(
        "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_phase.tif"
    )


@pytest.fixture
def membrane_example():
    return imread(
        "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_membrane.tif"
    )


@pytest.fixture
def dna_example():
    return imread(
        "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_dna.tif"
    )


@pytest.fixture
def all_sample_data():
    return (
        imread(
            "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_phase.tif"
        ),
        imread(
            "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_membrane.tif"
        ),
        imread(
            "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_dna.tif"
        ),
    )
