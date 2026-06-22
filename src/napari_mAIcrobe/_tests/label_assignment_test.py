import numpy as np

from napari_mAIcrobe.mAIcrobe.label_assignment import relabel_timelapse_labels


def test_relabel_timelapse_labels_keeps_one_to_one_identity():
    labels = np.zeros((2, 8, 8), dtype=np.int32)
    labels[0, 2:5, 2:5] = 1
    labels[1, 2:5, 2:5] = 7

    tracked = relabel_timelapse_labels(labels)

    assert tracked.shape == labels.shape
    assert np.unique(tracked[0]).tolist() == [0, 1]
    assert np.unique(tracked[1]).tolist() == [0, 1]


def test_relabel_timelapse_labels_split_assigns_new_ids_to_both_children():
    labels = np.zeros((2, 10, 10), dtype=np.int32)
    labels[0, 2:8, 2:8] = 1

    # Split into two children in next frame
    labels[1, 2:8, 2:5] = 2
    labels[1, 2:8, 5:8] = 3

    tracked = relabel_timelapse_labels(labels)

    frame0_ids = set(np.unique(tracked[0]).tolist()) - {0}
    frame1_ids = set(np.unique(tracked[1]).tolist()) - {0}

    assert frame0_ids == {1}
    assert len(frame1_ids) == 2
    assert 1 not in frame1_ids


def test_relabel_timelapse_labels_never_reuses_disappeared_id():
    labels = np.zeros((3, 8, 8), dtype=np.int32)
    labels[0, 1:4, 1:4] = 5
    labels[2, 4:7, 4:7] = 5

    tracked = relabel_timelapse_labels(labels)

    frame0_ids = set(np.unique(tracked[0]).tolist()) - {0}
    frame2_ids = set(np.unique(tracked[2]).tolist()) - {0}

    assert frame0_ids == {5}
    assert len(frame2_ids) == 1
    new_id = next(iter(frame2_ids))
    assert new_id > 5


def test_relabel_timelapse_labels_requires_3d_input():
    labels = np.zeros((8, 8), dtype=np.int32)

    try:
        relabel_timelapse_labels(labels)
        assert False, "Expected ValueError for non-3D input"
    except ValueError:
        pass
