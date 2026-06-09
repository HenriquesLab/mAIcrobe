import numpy as np

from napari_mAIcrobe import _sample_data


def test_sample_data_hooks_return_napari_layer_tuples(monkeypatch):
    calls = []

    def fake_imread(url):
        calls.append(url)
        return np.ones((2, 3))

    monkeypatch.setattr(_sample_data, "imread", fake_imread)

    phase = _sample_data.phase_example()
    membrane = _sample_data.membrane_example()
    dna = _sample_data.dna_example()

    assert phase[0][1]["name"] == "Example S.aureus phase contrast"
    assert membrane[0][1]["name"] == "Example S.aureus labeled with membrane dye"
    assert dna[0][1]["name"] == "Example S.aureus labeled with DNA dye"
    assert phase[0][2] == membrane[0][2] == dna[0][2] == "image"
    assert len(calls) == 3
    assert all(call.startswith("https://github.com/") for call in calls)
