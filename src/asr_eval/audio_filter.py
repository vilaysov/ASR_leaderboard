"""Audio preprocessing utilities for ASR inference."""

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDPASS_LOW_HZ = 70.0
BANDPASS_HIGH_HZ = 7000.0
BANDPASS_ORDER = 5


@lru_cache(maxsize=8)
def _bandpass_sos(sample_rate: int) -> np.ndarray:
    """Create and cache the Butterworth band-pass coefficients per sample rate."""
    nyquist_hz = sample_rate / 2
    if sample_rate <= 0 or BANDPASS_HIGH_HZ >= nyquist_hz:
        raise ValueError(
            "The 70-7000 Hz band-pass filter requires a sample rate greater "
            "than 14000 Hz; got {} Hz".format(sample_rate)
        )
    return butter(
        BANDPASS_ORDER,
        (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ),
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )


def bandpass_filter_audio(audio: Any, sample_rate: int) -> np.ndarray:
    """Apply a zero-phase 70-7000 Hz Butterworth filter to mono audio."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(
            "Expected mono audio for band-pass filtering; got shape {}".format(
                samples.shape
            )
        )
    if samples.size < 2:
        raise ValueError("At least two audio samples are required for filtering")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples and cannot be filtered")

    sos = _bandpass_sos(int(sample_rate))
    padlen = min(3 * (2 * len(sos) + 1), samples.size - 1)
    filtered = sosfiltfilt(sos, samples, padlen=padlen)
    return np.asarray(filtered, dtype=np.float32)
