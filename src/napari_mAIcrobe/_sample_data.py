"""
Module resposible for downloading sample data of S. aureus microscopy images
"""
from __future__ import annotations

from skimage.io import imread


def phase_example():
    """Load a sample S. aureus phase-contrast image.

    Returns
    -------
    list[tuple[numpy.ndarray, dict, str]]
        A list with one (data, meta, layer_type) tuple suitable for
        napari's sample data hook.
    """
    return [
        (
            imread(
                "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_phase.tif"
            ),
            {"name": "Example S.aureus phase contrast"},
            "image",
        )
    ]


def membrane_example():
    """Load a sample S. aureus membrane-labeled image.

    Returns
    -------
    list[tuple[numpy.ndarray, dict, str]]
        A list with one (data, meta, layer_type) tuple suitable for
        napari's sample data hook.
    """
    return [
        (
            imread(
                "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_membrane.tif"
            ),
            {"name": "Example S.aureus labeled with membrane dye"},
            "image",
        )
    ]


def dna_example():
    """Load a sample S. aureus DNA-labeled image.

    Returns
    -------
    list[tuple[numpy.ndarray, dict, str]]
        A list with one (data, meta, layer_type) tuple suitable for
        napari's sample data hook.
    """
    return [
        (
            imread(
                "https://github.com/HenriquesLab/mAIcrobe/raw/main/docs/test_dna.tif"
            ),
            {"name": "Example S.aureus labeled with DNA dye"},
            "image",
        )
    ]
