import numpy as np
import pytest

from napari_mAIcrobe.mAIcrobe import unet


class FakeLayer:
    output_shape = [(None, 4, 4, 1)]


class FakeModel:
    layers = [FakeLayer()]

    def __init__(self, klass=2):
        self.klass = klass
        self.calls = []

    def predict(self, patch, batch_size=1):
        self.calls.append(patch.shape)
        result = np.zeros((1, 4, 4, 3), dtype=float)
        result[:, :, :, self.klass] = 1
        return result


def test_normalize_mi_ma_supports_clipping_and_dtype():
    result = unet.normalize_mi_ma(
        np.array([-1, 0, 2], dtype=float),
        mi=0,
        ma=1,
        clip=True,
        dtype=np.float32,
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        result, np.array([0, 0, 1], dtype=np.float32)
    )


def test_normalize_percentile_maps_values_between_percentiles():
    image = np.arange(100, dtype=float)

    result = unet.normalizePercentile(image, pmin=0, pmax=100)

    assert result[0] == pytest.approx(0)
    assert result[-1] == pytest.approx(1)


def test_predict_as_tiles_pads_small_images_and_crops_back():
    model = FakeModel(klass=2)

    prediction = unet.predict_as_tiles(np.ones((2, 3)), model)

    assert prediction.shape == (2, 3)
    assert prediction.dtype == np.uint8
    assert np.all(prediction == 2)
    assert model.calls == [(1, 4, 4, 1)]


def test_predict_as_tiles_runs_multiple_tiles_for_larger_image():
    model = FakeModel(klass=1)

    prediction = unet.predict_as_tiles(np.ones((6, 7)), model)

    assert prediction.shape == (6, 7)
    assert np.all(prediction == 1)
    assert len(model.calls) == 4


def test_computelabel_unet_uses_loaded_model_prediction(monkeypatch):
    monkeypatch.setattr(unet, "load_model", lambda path: FakeModel(klass=2))

    mask, labels = unet.computelabel_unet(
        "fake.keras",
        np.ones((4, 4), dtype=float),
        closing=0,
        dilation=0,
        fillholes=False,
    )

    assert mask.shape == labels.shape == (4, 4)
    assert mask.dtype == bool
    assert labels.max() >= 1


def test_download_github_file_raw_returns_cached_path(tmp_path):
    cached = tmp_path / "SegmentationModels" / "model.h5"
    cached.parent.mkdir()
    cached.write_bytes(b"already here")

    result = unet.download_github_file_raw(
        "SegmentationModels/model.h5",
        tmp_path,
    )

    assert result == str(cached)


def test_download_github_file_raw_writes_response_content(
    monkeypatch, tmp_path
):
    calls = {}

    class FakeResponse:
        content = b"model bytes"

        def raise_for_status(self):
            calls["raised"] = True

    def fake_get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(unet.requests, "get", fake_get)

    result = unet.download_github_file_raw("model.h5", tmp_path, branch="dev")

    assert result == str(tmp_path / "model.h5")
    assert (tmp_path / "model.h5").read_bytes() == b"model bytes"
    assert calls == {
        "url": (
            "https://raw.githubusercontent.com/HenriquesLab/mAIcrobe/"
            "dev/docs/model.h5"
        ),
        "timeout": 30,
        "raised": True,
    }
