import pickle
from types import SimpleNamespace

import numpy as np

from napari_mAIcrobe._compute_pickles import compute_pickles


def _widget(tmp_path, channel_mode="One Channel"):
    label_data = np.zeros((20, 20), dtype=int)
    label_data[4:10, 4:12] = 1
    label_data[12:18, 12:18] = 2
    channel_1 = np.arange(400, dtype=float).reshape(20, 20)
    channel_2 = np.flipud(channel_1)

    widget = compute_pickles.__new__(compute_pickles)
    widget.box_margin = 2
    widget._label_combo = SimpleNamespace(
        value=SimpleNamespace(data=label_data)
    )
    widget._points_combo = SimpleNamespace(
        value=SimpleNamespace(
            data=np.array([[5, 5], [13, 13], [0, 0], [5, 5]]),
            name="3",
        )
    )
    widget._channel_radio = SimpleNamespace(value=channel_mode)
    widget.channelone_combo = SimpleNamespace(
        value=SimpleNamespace(data=channel_1),
        visible=None,
    )
    widget.channeltwo_combo = SimpleNamespace(
        value=SimpleNamespace(data=channel_2),
        visible=None,
    )
    widget._path2save = SimpleNamespace(value=str(tmp_path))
    return widget


def test_on_channel_change_toggles_second_channel_visibility():
    widget = compute_pickles.__new__(compute_pickles)
    widget._channel_radio = SimpleNamespace(value="One Channel")
    widget.channeltwo_combo = SimpleNamespace(visible=True)

    widget._on_channel_change()
    assert widget.channeltwo_combo.visible is False

    widget._channel_radio.value = "Two Channels"
    widget._on_channel_change()
    assert widget.channeltwo_combo.visible is True


def test_on_run_exports_one_channel_pickles(tmp_path):
    widget = _widget(tmp_path, channel_mode="One Channel")

    widget._on_run()

    source = pickle.loads((tmp_path / "Class_3_source.p").read_bytes())
    target = pickle.loads((tmp_path / "Class_3_target.p").read_bytes())
    assert len(source) == 2
    assert target == [3, 3]
    assert source[0].shape == (100, 100)


def test_on_run_exports_two_channel_side_by_side_crops(tmp_path):
    widget = _widget(tmp_path, channel_mode="Two Channels")

    widget._on_run()

    source = pickle.loads((tmp_path / "Class_3_source.p").read_bytes())
    target = pickle.loads((tmp_path / "Class_3_target.p").read_bytes())
    assert len(source) == 2
    assert target == [3, 3]
    assert source[0].shape == (100, 200)


def test_on_run_returns_when_points_layer_name_is_not_positive(tmp_path):
    widget = _widget(tmp_path)
    widget._points_combo.value.name = "not-a-class"

    widget._on_run()

    assert not (tmp_path / "Class_3_source.p").exists()


def test_on_run_returns_when_required_channel_is_missing(tmp_path):
    widget = _widget(tmp_path, channel_mode="Two Channels")
    widget.channeltwo_combo.value = None

    widget._on_run()

    assert not (tmp_path / "Class_3_source.p").exists()
