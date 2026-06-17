import numpy as np

from napari_mAIcrobe.mAIcrobe.cellcycleclassifier import CellCycleClassifier


class DummyModel:
    def __init__(self, prediction):
        self.prediction = np.asarray(prediction)
        self.seen_shape = None

    def predict(self, array, verbose=0):
        self.seen_shape = array.shape
        return self.prediction


class DummyCell:
    box = (1, 1, 3, 4)
    cell_mask = np.ones((3, 4))


def _classifier(model_input, prediction):
    classifier = CellCycleClassifier.__new__(CellCycleClassifier)
    classifier.max_dim = 6
    classifier.model_input = model_input
    classifier.custom = False
    classifier.model = DummyModel(prediction)
    classifier.fluor_fov = np.arange(36, dtype=float).reshape(6, 6)
    classifier.optional_fov = np.arange(36, 72, dtype=float).reshape(6, 6)
    return classifier


def test_preprocess_image_pads_to_centered_target_shape():
    classifier = CellCycleClassifier.__new__(CellCycleClassifier)
    classifier.max_dim = 5
    image = np.ones((3, 3))

    processed = classifier.preprocess_image(image)

    assert processed.shape == (5, 5, 1)
    np.testing.assert_array_equal(processed[1:4, 1:4, 0], image)
    assert processed[:, 0, 0].sum() == 0
    assert processed[0, :, 0].sum() == 0


def test_preprocess_image_crops_to_centered_target_shape():
    classifier = CellCycleClassifier.__new__(CellCycleClassifier)
    classifier.max_dim = 3
    image = np.arange(25, dtype=float).reshape(5, 5) / 24

    processed = classifier.preprocess_image(image)

    assert processed.shape == (3, 3, 1)
    np.testing.assert_array_equal(processed[:, :, 0], image[1:4, 1:4])


def test_classify_cell_membrane_uses_single_channel_prediction():
    classifier = _classifier("Membrane", [[0.1, 0.8, 0.1]])

    phase = classifier.classify_cell(DummyCell())

    assert phase == 2
    assert classifier.model.seen_shape == (1, 100, 100, 1)


def test_classify_cell_dna_uses_optional_channel_prediction():
    classifier = _classifier("DNA", [[0.2, 0.2, 0.6]])

    phase = classifier.classify_cell(DummyCell())

    assert phase == 3
    assert classifier.model.seen_shape == (1, 100, 100, 1)


def test_classify_cell_combined_channels_double_width():
    classifier = _classifier("Membrane+DNA", [[0.9, 0.05, 0.05]])

    phase = classifier.classify_cell(DummyCell())

    assert phase == 1
    assert classifier.model.seen_shape == (1, 100, 200, 1)


def test_classify_cell_custom_binary_output_maps_to_two_phases():
    classifier = _classifier("Membrane", [[0.7]])
    classifier.custom = True

    assert classifier.classify_cell(DummyCell()) == 2
