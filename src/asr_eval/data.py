"""URGENT24 streaming dataset adapter."""

import logging
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

URGENT24_DATASET_ID = "urgent-challenge/urgent2024_official"
URGENT24_CONFIG = "test_nonblind"
URGENT24_SPLIT = "validation"
DEFAULT_SAMPLE_RATE = 16000


def _audio_array_and_rate(audio: Any) -> tuple[Any, int]:
    """Convert a decoded Hugging Face audio value to mono samples and rate."""
    if isinstance(audio, dict):
        if "array" not in audio or "sampling_rate" not in audio:
            raise ValueError(
                "Decoded audio dictionary must contain array and sampling_rate"
            )
        array = audio["array"]
        sample_rate = int(audio["sampling_rate"])
    elif callable(getattr(audio, "get_all_samples", None)):
        samples = audio.get_all_samples()
        array = samples.data
        if hasattr(array, "detach"):
            array = array.detach()
        if hasattr(array, "cpu"):
            array = array.cpu()
        if hasattr(array, "numpy"):
            array = array.numpy()
        sample_rate = int(samples.sample_rate)
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
            "Expected mono audio samples, got an array with shape {}".format(
                array.shape
            )
        )
    return array, sample_rate


def load_urgent24_asr_examples(
    max_samples: Optional[int] = 100,
) -> Iterator[Dict[str, Any]]:
    """Stream URGENT24 non-blind noisy audio with its reference transcript."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    from datasets import Audio, load_dataset

    logger.info(
        "Opening %s (config=%s, split=%s)",
        URGENT24_DATASET_ID,
        URGENT24_CONFIG,
        URGENT24_SPLIT,
    )
    dataset = load_dataset(
        URGENT24_DATASET_ID,
        URGENT24_CONFIG,
        split=URGENT24_SPLIT,
        streaming=True,
    )
    required_columns = {"id", "noisy_audio", "transcript"}
    missing_columns = required_columns.difference(dataset.features)
    if missing_columns:
        raise ValueError(
            "URGENT24 split is missing required columns: {}".format(
                ", ".join(sorted(missing_columns))
            )
        )
    dataset = dataset.cast_column(
        "noisy_audio", Audio(sampling_rate=DEFAULT_SAMPLE_RATE)
    )
    rows = dataset if max_samples is None else islice(dataset, max_samples)

    for row in rows:
        sample_id = str(row["id"])
        reference = str(row["transcript"])
        if reference.strip().lower() == "<not-available>":
            raise ValueError(
                "URGENT24 sample {!r} has no reference transcript".format(sample_id)
            )
        audio, sample_rate = _audio_array_and_rate(row["noisy_audio"])
        yield {
            "sample_id": sample_id,
            "audio": audio,
            "sampling_rate": sample_rate,
            "reference": reference,
        }
