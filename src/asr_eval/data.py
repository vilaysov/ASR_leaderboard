"""Hugging Face streaming dataset adapter."""

import logging
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _audio_array_and_rate(audio: Any) -> tuple[Any, int]:
    """Convert Hugging Face audio values to mono samples and their rate."""
    if isinstance(audio, dict):
        if "array" not in audio or "sampling_rate" not in audio:
            raise ValueError("Decoded audio dictionary must contain array and sampling_rate")
        array = audio["array"]
        sampling_rate = int(audio["sampling_rate"])
    elif callable(getattr(audio, "get_all_samples", None)):
        samples = audio.get_all_samples()
        array = samples.data
        if hasattr(array, "detach"):
            array = array.detach()
        if hasattr(array, "cpu"):
            array = array.cpu()
        if hasattr(array, "numpy"):
            array = array.numpy()
        sampling_rate = int(samples.sample_rate)
    else:
        raise ValueError(
            "Unsupported decoded audio value of type {}".format(type(audio).__name__)
        )

    array = np.asarray(array)
    if array.ndim == 2:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = array.mean(axis=0)
    if array.ndim != 1:
        raise ValueError(
            "Expected mono audio samples, got an array with shape {}".format(array.shape)
        )
    return array, sampling_rate


def load_examples(
    dataset_id: str,
    dataset_config: Optional[str],
    split: str,
    max_samples: Optional[int] = 100,
    audio_column: str = "audio",
    text_column: str = "text",
    id_column: Optional[str] = "segment_id",
) -> Iterator[Dict[str, Any]]:
    """Yield decoded examples, optionally limiting the number of rows."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    from datasets import Audio, load_dataset

    logger.info(
        "Opening streaming dataset %s (config=%s, split=%s)",
        dataset_id,
        dataset_config or "default",
        split,
    )
    dataset_args = {"split": split, "streaming": True}
    if dataset_config is None:
        dataset = load_dataset(dataset_id, **dataset_args)
    else:
        dataset = load_dataset(dataset_id, dataset_config, **dataset_args)

    if audio_column not in dataset.features:
        raise ValueError(
            "Dataset {!r} has no audio column {!r}".format(dataset_id, audio_column)
        )
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=16000))

    rows = dataset if max_samples is None else islice(dataset, max_samples)
    for index, row in enumerate(rows):
        if text_column not in row:
            raise ValueError(
                "Dataset {!r} has no text column {!r}".format(dataset_id, text_column)
            )

        sample_id = row.get(id_column) if id_column else None
        if sample_id is None:
            sample_id = str(index)

        audio_array, sampling_rate = _audio_array_and_rate(row[audio_column])

        reference = str(row[text_column])
        if reference.strip().lower() == "<not-available>":
            raise ValueError(
                "Dataset {!r} row {!r} has no reference transcript in column {!r}".format(
                    dataset_id, sample_id, text_column
                )
            )

        yield {
            "sample_id": str(sample_id),
            "audio": audio_array,
            "sampling_rate": sampling_rate,
            "reference": reference,
        }
