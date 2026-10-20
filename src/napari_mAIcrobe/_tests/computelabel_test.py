from types import SimpleNamespace

import numpy as np

from napari_mAIcrobe import _computelabel


class DummyWidget:
    def __init__(self, value=None):
        self.value = value
        self.visible = None


class DummyViewer:
    def __init__(self):
        self.added = []
        self.layers = {}

    def add_labels(self, data, name):
        self.added.append((name, data))
        self.layers[name] = SimpleNamespace(data=data, name=name)


def _instance(algorithm="Isodata", timelapse=False, autoalign=False):
    base_data = np.ones((2, 4, 4)) if timelapse else np.ones((4, 4))
    fluor_data = np.ones_like(base_data)
    viewer = DummyViewer()
    viewer.layers["fluor1"] = SimpleNamespace(
        data=fluor_data.copy(), name="fluor1"
    )
    viewer.layers["fluor2"] = SimpleNamespace(
        data=fluor_data.copy(), name="fluor2"
    )

    obj = _computelabel.compute_label.__new__(_computelabel.compute_label)
    obj._viewer = viewer
    obj._baseimg_combo = DummyWidget(SimpleNamespace(data=base_data))
    obj._fluor1_combo = DummyWidget(
        SimpleNamespace(data=fluor_data.copy(), name="fluor1")
    )
    obj._fluor2_combo = DummyWidget(
        SimpleNamespace(data=fluor_data.copy(), name="fluor2")
    )
    obj._closinginput = DummyWidget(0)
    obj._dilationinput = DummyWidget(0)
    obj._fillholesinput = DummyWidget(False)
    obj._autoaligninput = DummyWidget(autoalign)
    obj._algorithm_combo = DummyWidget(algorithm)
    obj._titlemasklabel = DummyWidget()
    obj._placeholder = DummyWidget()
    obj._blocksizeinput = DummyWidget(151)
    obj._offsetinput = DummyWidget(0.02)
    obj._unetradio = DummyWidget("Custom")
    obj._path2unet = DummyWidget("model.h5")
    obj._unetpretrained = DummyWidget("Ph.C. S. pneumo")
    obj._stardistradio = DummyWidget("Custom")
    obj._path2stardist = DummyWidget("model_dir")
    obj._stardistpretrained = DummyWidget("StarDist S. aureus")
    obj._omniposeradio = DummyWidget("Custom")
    obj._path2omnipose = DummyWidget("model_dir")
    obj._omniposepretrained = DummyWidget("Bact Phase")
    obj._titlewatershedlabel = DummyWidget()
    obj._peak_min_distance_from_edge = DummyWidget(1)
    obj._peak_min_distance = DummyWidget(1)
    obj._peak_min_height = DummyWidget(1)
    obj._max_peaks = DummyWidget(10)
    obj._timelapse = DummyWidget(timelapse)
    obj._enable_tracking = DummyWidget(True)
    obj._enable_tracking.visible = False
    obj._imgreg = DummyWidget(False)
    obj._reference = DummyWidget(0)
    obj._timeaveraging = DummyWidget(1)
    obj._maxexpecteddrift = DummyWidget(0)
    obj._gaussian = DummyWidget(False)
    obj._gaussian_sigma = DummyWidget(1.5)
    obj._merge = DummyWidget(False)
    obj._merge_pixel_um = DummyWidget(0.065)
    obj._merge_min_solidity = DummyWidget(0.91)
    obj._merge_max_aspect = DummyWidget(1.55)
    obj._merge_contact_min = DummyWidget(0.6)
    obj._merge_contact_max = DummyWidget(1.2)
    obj._merge_max_union = DummyWidget(3.0)
    return obj


def test_base_image_visibility_toggles_timelapse_checkbox():
    obj = _instance()

    obj._on_baseimg_changed(None)
    assert obj._timelapse.visible is False

    obj._on_baseimg_changed(SimpleNamespace(data=np.zeros((2, 3, 3))))
    assert obj._timelapse.visible is True

    obj._on_baseimg_changed(SimpleNamespace(data=np.zeros((3, 3))))
    assert obj._timelapse.visible is False


def test_algorithm_visibility_for_unet_and_local_average():
    obj = _instance()

    obj._on_algorithm_changed("Unet")
    assert obj._unetradio.visible is True
    assert obj._unetpretrained.visible is False
    assert obj._path2unet.visible is True
    assert obj._peak_min_distance.visible is False

    obj._on_algorithm_changed("Local Average")
    assert obj._blocksizeinput.visible is True
    assert obj._peak_min_distance.visible is True
    assert obj._unetradio.visible is False


def test_pretrained_toggles_only_when_matching_algorithm():
    obj = _instance("Unet")
    obj._on_pretrainedunet_changed("Pretrained")
    assert obj._unetpretrained.visible is True
    assert obj._path2unet.visible is False

    obj._algorithm_combo.value = "StarDist"
    obj._on_pretrainedstardist_changed("Custom")
    assert obj._stardistpretrained.visible is False
    assert obj._path2stardist.visible is True


def test_compute_dispatches_classical_and_adds_layers(monkeypatch):
    obj = _instance("Isodata")
    mask = np.ones((4, 4), dtype=np.uint16)
    labels = np.full((4, 4), 2, dtype=np.uint16)

    monkeypatch.setattr(
        _computelabel,
        "classical_segmentation",
        lambda *args: (mask, labels),
    )

    obj.compute()

    assert [name for name, _data in obj._viewer.added] == ["Mask", "Labels"]
    np.testing.assert_array_equal(obj._viewer.layers["Labels"].data, labels)


def test_compute_dispatches_timelapse_unet_and_autoaligns(monkeypatch):
    obj = _instance("Unet", timelapse=True, autoalign=True)
    mask = np.ones((2, 4, 4), dtype=np.uint16)
    labels = np.full((2, 4, 4), 2, dtype=np.uint16)

    monkeypatch.setattr(
        _computelabel,
        "batch_unet_segmentation",
        lambda *args: (mask, labels),
    )
    monkeypatch.setattr(
        _computelabel,
        "mask_alignment",
        lambda mask_frame, fluor_frame: fluor_frame + 5,
    )

    obj.compute()

    assert obj._viewer.layers["fluor1"].data.shape == (2, 4, 4)
    assert np.all(obj._viewer.layers["fluor1"].data == 6)
    assert np.all(obj._viewer.layers["fluor2"].data == 6)


def test_compute_timelapse_relabels_labels_for_tracking(monkeypatch):
    obj = _instance("Unet", timelapse=True, autoalign=False)
    mask = np.ones((2, 4, 4), dtype=np.uint16)
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 1:3] = 9

    monkeypatch.setattr(
        _computelabel,
        "batch_unet_segmentation",
        lambda *args: (mask, labels),
    )

    obj.compute()

    tracked = obj._viewer.layers["Labels"].data
    assert tracked.shape == (2, 4, 4)
    assert set(np.unique(tracked[0]).tolist()) == {0, 1}
    assert set(np.unique(tracked[1]).tolist()) == {0, 1}


def test_tracking_checkbox_visible_for_timelapse():
    obj = _instance(timelapse=False)
    assert obj._enable_tracking.visible is False

    obj._on_baseimg_changed(SimpleNamespace(data=np.zeros((2, 3, 3))))
    assert obj._enable_tracking.visible is True


def test_tracking_disabled_skips_relabeling(monkeypatch):
    obj = _instance("Unet", timelapse=True, autoalign=False)
    obj._enable_tracking.value = False
    mask = np.ones((2, 4, 4), dtype=np.uint16)
    labels = np.zeros((2, 4, 4), dtype=np.uint16)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 1:3] = 9

    monkeypatch.setattr(
        _computelabel,
        "batch_unet_segmentation",
        lambda *args: (mask, labels),
    )

    obj.compute()

    result = obj._viewer.layers["Labels"].data
    assert result.shape == (2, 4, 4)
    assert set(np.unique(result[0]).tolist()) == {0, 1}
    assert set(np.unique(result[1]).tolist()) == {0, 9}


def test_gaussian_and_merge_options_are_applied(monkeypatch):
    obj = _instance("Isodata")
    obj._gaussian.value = True
    obj._gaussian_sigma.value = 2.0
    obj._merge.value = True
    obj._merge_pixel_um.value = 0.1

    mask = np.ones((4, 4), dtype=np.uint16)
    labels = np.full((4, 4), 2, dtype=np.uint16)
    seen = {}

    def fake_segmentation(img, *args):
        seen["img"] = img
        return mask, labels

    def fake_smooth(img, sigma):
        seen["sigma"] = sigma
        return img * 0

    def fake_merge(lab, **kwargs):
        seen["merge"] = kwargs
        return np.full_like(lab, 7), []

    monkeypatch.setattr(_computelabel, "classical_segmentation", fake_segmentation)
    monkeypatch.setattr(_computelabel, "gaussian_smooth", fake_smooth)
    monkeypatch.setattr(_computelabel, "merge_septated_cells", fake_merge)

    obj.compute()

    assert seen["sigma"] == 2.0
    assert not seen["img"].any()
    assert seen["merge"]["pixel_um"] == 0.1
    assert seen["merge"]["contact_width_ratio"] == (0.6, 1.2)
    assert (obj._viewer.layers["Labels"].data == 7).all()


def test_gaussian_and_merge_visibility_toggles():
    obj = _instance()
    obj._on_gaussian_changed(True)
    assert obj._gaussian_sigma.visible is True
    obj._on_merge_changed(True)
    assert obj._merge_pixel_um.visible is True
    obj._on_merge_changed(False)
    assert obj._merge_max_union.visible is False
