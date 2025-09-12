import numpy as np


def testing_sampledata(all_sample_data):
    """Check if sample data is available for downloading."""

    assert all_sample_data is not None
    assert len(all_sample_data) == 3

    assert isinstance(all_sample_data[0], np.ndarray)
    assert isinstance(all_sample_data[1], np.ndarray)
    assert isinstance(all_sample_data[2], np.ndarray)

    assert all_sample_data[0].shape == (1024, 1376)
    assert all_sample_data[1].shape == (1024, 1376)
    assert all_sample_data[2].shape == (1024, 1376)
