import numpy as np

from napari_mAIcrobe.mAIcrobe import segmentation


def test_unet_segmentation_uses_custom_model_path(monkeypatch):
    calls = {}

    def fake_computelabel_unet(**kwargs):
        calls.update(kwargs)
        return np.ones((3, 3), dtype=np.uint16), np.arange(9).reshape(3, 3)

    monkeypatch.setattr(
        segmentation, "computelabel_unet", fake_computelabel_unet
    )

    mask, labels = segmentation.unet_segmentation(
        np.zeros((2, 3, 3)),
        pretrained="Custom",
        pretrained_name="ignored",
        path2model="/tmp/model.hdf5",
        binary_closing=1,
        binary_dilation=2,
        binary_fillholes=True,
    )

    assert calls["path2model"] == "/tmp/model.hdf5"
    assert calls["base_image"].shape == (3, 3)
    assert calls["closing"] == 1
    assert calls["dilation"] == 2
    assert calls["fillholes"] is True
    assert mask.shape == labels.shape == (3, 3)


def test_batch_unet_segmentation_stacks_frame_results(monkeypatch):
    def fake_unet_segmentation(img, *args):
        return img.astype(np.uint16), (img + 10).astype(np.uint16)

    monkeypatch.setattr(
        segmentation, "unet_segmentation", fake_unet_segmentation
    )
    stack = np.stack([np.ones((2, 2)), np.full((2, 2), 2)])

    masks, labels = segmentation.batch_unet_segmentation(
        stack, "Custom", "ignored", "", 0, 0, False
    )

    assert masks.shape == labels.shape == (2, 2, 2)
    np.testing.assert_array_equal(masks[1], np.full((2, 2), 2))
    np.testing.assert_array_equal(labels[0], np.full((2, 2), 11))


def test_stardist_segmentation_uses_custom_model_and_normalization(
    monkeypatch,
):
    seen = {}

    class FakeStarDist2D:
        def __init__(self, _config, name, basedir):
            seen["name"] = name
            seen["basedir"] = basedir

        def predict_instances(self, image):
            seen["image"] = image
            return np.array([[0, 1], [2, 0]]), None

    monkeypatch.setattr(segmentation, "StarDist2D", FakeStarDist2D)
    monkeypatch.setattr(
        segmentation, "normalizePercentile", lambda image: image + 1
    )

    mask, labels = segmentation.stardist_segmentation(
        np.zeros((2, 2)),
        pretrained="Custom",
        pretrained_name="ignored",
        path2model="/tmp/model_dir",
    )

    assert seen["name"] == "model_dir"
    assert seen["basedir"] == "/tmp"
    np.testing.assert_array_equal(seen["image"], np.ones((2, 2)))
    np.testing.assert_array_equal(labels, np.array([[0, 1], [2, 0]]))
    np.testing.assert_array_equal(
        mask, np.array([[0, 1], [1, 0]], dtype=np.uint16)
    )


def test_cellpose_segmentation_uses_first_frame_for_3d_input(monkeypatch):
    seen = {}

    class FakeCellpose:
        def __init__(self, gpu, model_type):
            seen["gpu"] = gpu
            seen["model_type"] = model_type

        def eval(self, image, diameter=None):
            seen["image"] = image
            return np.array([[0, 4], [0, 5]]), None, None, None

    monkeypatch.setattr(segmentation.models, "Cellpose", FakeCellpose)
    stack = np.stack([np.ones((2, 2)), np.full((2, 2), 2)])

    mask, labels = segmentation.cellpose_segmentation(stack)

    assert seen["gpu"] is True
    assert seen["model_type"] == "cyto3"
    np.testing.assert_array_equal(seen["image"], np.ones((2, 2)))
    np.testing.assert_array_equal(labels, np.array([[0, 4], [0, 5]]))
    np.testing.assert_array_equal(
        mask, np.array([[0, 1], [0, 1]], dtype=np.uint16)
    )


def test_classical_segmentation_delegates_to_mask_and_segments(monkeypatch):
    class FakeSegmentsManager:
        def compute_segments(self, pars, mask):
            self.pars = pars
            self.mask = mask
            self.labels = mask + 3

    monkeypatch.setattr(
        segmentation,
        "mask_computation",
        lambda **kwargs: np.ones((2, 2), dtype=np.uint16),
    )
    monkeypatch.setattr(segmentation, "SegmentsManager", FakeSegmentsManager)

    mask, labels = segmentation.classical_segmentation(
        np.zeros((2, 2)),
        "Isodata",
        151,
        0.02,
        0,
        0,
        False,
        {"peak_min_distance": 1},
    )

    np.testing.assert_array_equal(mask, np.ones((2, 2), dtype=np.uint16))
    np.testing.assert_array_equal(labels, np.full((2, 2), 4, dtype=np.uint16))
