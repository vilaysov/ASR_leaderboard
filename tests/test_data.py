import numpy as np
import pytest
import sys
from types import ModuleType

from asr_eval.data import _audio_array_and_rate


class FakeAudioSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class FakeAudioDecoder:
    def __init__(self, samples):
        self.samples = samples

    def get_all_samples(self):
        return self.samples


def test_audio_array_and_rate_handles_legacy_dictionary() -> None:
    source = {"array": np.array([0.1, -0.1]), "sampling_rate": 16000}

    audio, sample_rate = _audio_array_and_rate(source)

    np.testing.assert_array_equal(audio, source["array"])
    assert sample_rate == 16000


def test_audio_array_and_rate_handles_audio_decoder_samples() -> None:
    source = FakeAudioDecoder(FakeAudioSamples(np.array([[0.1, -0.1]]), 16000))

    audio, sample_rate = _audio_array_and_rate(source)

    np.testing.assert_array_equal(audio, np.array([0.1, -0.1]))
    assert sample_rate == 16000


def test_audio_array_and_rate_downmixes_multiple_channels() -> None:
    source = FakeAudioDecoder(
        FakeAudioSamples(np.array([[1.0, 0.0], [-1.0, 1.0]]), 16000)
    )

    audio, _ = _audio_array_and_rate(source)

    np.testing.assert_array_equal(audio, np.array([0.0, 0.5]))


def test_audio_array_and_rate_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="Unsupported decoded audio"):
        _audio_array_and_rate("not audio")


class FakeDataset:
    features = {"noisy_audio": object(), "transcript": object(), "id": object()}

    def __init__(self, rows):
        self.rows = rows
        self.cast_column_args = None

    def cast_column(self, column, feature):
        self.cast_column_args = (column, feature)
        return self

    def __iter__(self):
        return iter(self.rows)


def _install_fake_datasets(monkeypatch, dataset):
    fake_datasets = ModuleType("datasets")
    fake_datasets.Audio = lambda sampling_rate: ("audio-feature", sampling_rate)
    fake_datasets.load_dataset = lambda *args, **kwargs: dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)


def _urgent_rows(count):
    return [
        {
            "id": "fileid_{}".format(index),
            "transcript": "test transcript {}".format(index),
            "noisy_audio": {
                "array": np.array([0.1, -0.1]),
                "sampling_rate": 16000,
            },
        }
        for index in range(count)
    ]


def test_load_examples_streams_entire_dataset_when_limit_is_none(monkeypatch) -> None:
    dataset = FakeDataset(_urgent_rows(3))
    _install_fake_datasets(monkeypatch, dataset)

    from asr_eval.data import load_examples

    examples = list(
        load_examples(
            dataset_id="urgent-challenge/urgent2024_official",
            dataset_config="test_nonblind",
            split="validation",
            max_samples=None,
            audio_column="noisy_audio",
            text_column="transcript",
            id_column="id",
        )
    )

    assert len(examples) == 3
    assert examples[-1]["sample_id"] == "fileid_2"
    assert examples[-1]["reference"] == "test transcript 2"
    assert dataset.cast_column_args == (
        "noisy_audio",
        ("audio-feature", 16000),
    )


def test_load_examples_respects_sample_limit(monkeypatch) -> None:
    dataset = FakeDataset(_urgent_rows(3))
    _install_fake_datasets(monkeypatch, dataset)

    from asr_eval.data import load_examples

    examples = list(
        load_examples(
            dataset_id="urgent-challenge/urgent2024_official",
            dataset_config="test_nonblind",
            split="validation",
            max_samples=2,
            audio_column="noisy_audio",
            text_column="transcript",
            id_column="id",
        )
    )

    assert len(examples) == 2


def test_load_examples_rejects_missing_reference_labels(monkeypatch) -> None:
    dataset = FakeDataset(
        [
            {
                "id": "fileid_missing",
                "transcript": "<not-available>",
                "noisy_audio": {
                    "array": np.array([0.1, -0.1]),
                    "sampling_rate": 16000,
                },
            }
        ]
    )
    _install_fake_datasets(monkeypatch, dataset)

    from asr_eval.data import load_examples

    with pytest.raises(ValueError, match="no reference transcript"):
        next(
            load_examples(
                dataset_id="urgent-challenge/urgent2024_official",
                dataset_config="test_nonblind",
                split="validation",
                max_samples=None,
                audio_column="noisy_audio",
                text_column="transcript",
                id_column="id",
            )
        )
