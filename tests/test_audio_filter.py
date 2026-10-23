import numpy as np
import pytest

from asr_eval.audio_filter import bandpass_filter_audio


def _tone_amplitude(signal: np.ndarray, frequency_hz: float, sample_rate: int) -> float:
    times = np.arange(signal.size) / sample_rate
    return float(2 * abs(np.dot(signal, np.exp(-2j * np.pi * frequency_hz * times))) / signal.size)


def test_bandpass_filter_suppresses_low_and_high_frequency_tones() -> None:
    sample_rate = 16000
    times = np.arange(sample_rate * 2) / sample_rate
    audio = (
        np.sin(2 * np.pi * 40 * times)
        + np.sin(2 * np.pi * 1000 * times)
        + np.sin(2 * np.pi * 7500 * times)
    ).astype(np.float32)

    filtered = bandpass_filter_audio(audio, sample_rate)

    assert filtered.shape == audio.shape
    assert filtered.dtype == np.float32
    assert _tone_amplitude(filtered, 1000, sample_rate) > 0.9
    assert _tone_amplitude(filtered, 40, sample_rate) < 0.02
    assert _tone_amplitude(filtered, 7500, sample_rate) < 0.02


def test_bandpass_filter_rejects_sample_rate_below_nyquist_requirement() -> None:
    with pytest.raises(ValueError, match="greater than 14000 Hz"):
        bandpass_filter_audio(np.zeros(100, dtype=np.float32), 14000)


def test_bandpass_filter_requires_mono_audio() -> None:
    with pytest.raises(ValueError, match="Expected mono audio"):
        bandpass_filter_audio(np.zeros((2, 100), dtype=np.float32), 16000)


def test_bandpass_filter_requires_more_than_one_audio_sample() -> None:
    with pytest.raises(ValueError, match="At least two audio samples"):
        bandpass_filter_audio(np.zeros(1, dtype=np.float32), 16000)
