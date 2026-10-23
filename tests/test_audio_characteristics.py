import json

import numpy as np
import pytest

from asr_eval.audio_characteristics import (
    DEFAULT_SAMPLE_RATE,
    _decode_with_ffmpeg,
    analyze_audio_pair,
    analyze_urgent24,
    extract_pitch_track,
    median_frame_snr_db,
    pitch_range_estimate,
)


def _tone(frequency_hz: float, seconds: float, sample_rate: int = 16000) -> np.ndarray:
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    return 0.2 * np.sin(2 * np.pi * frequency_hz * times)


def test_extract_pitch_track_estimates_tone_with_short_hop() -> None:
    sample_rate = 16000
    track = extract_pitch_track(_tone(220.0, 1.0), sample_rate)

    voiced_pitch = [value for value in track.f0_hz if value is not None]

    assert len(track.f0_hz) == 97
    assert len(track.frame_times_seconds) == len(track.f0_hz)
    assert track.frame_times_seconds[:3] == [0.0, 0.01, 0.02]
    assert len(voiced_pitch) > 80
    assert np.median(voiced_pitch) == pytest.approx(220.0, abs=2.0)


def test_extract_pitch_track_marks_silence_unvoiced() -> None:
    track = extract_pitch_track(np.zeros(16000), 16000)

    assert track.f0_hz
    assert all(value is None for value in track.f0_hz)


def test_extract_pitch_track_rejects_non_mono_audio() -> None:
    with pytest.raises(ValueError, match="mono audio"):
        extract_pitch_track(np.zeros((2, 1000)), 16000)


def test_median_frame_snr_db_uses_clean_and_noise_residual() -> None:
    sample_rate = 16000
    times = np.arange(sample_rate) / sample_rate
    clean = 0.2 * np.sin(2 * np.pi * 200.0 * times)
    noise = 0.02 * np.sin(2 * np.pi * 800.0 * times)

    measured = median_frame_snr_db(clean, clean + noise, sample_rate)

    assert measured == pytest.approx(20.0, abs=0.1)


def test_median_frame_snr_db_returns_none_for_silence() -> None:
    silence = np.zeros(16000)

    assert median_frame_snr_db(silence, silence, 16000) is None


def test_decode_with_ffmpeg_decodes_raw_payload(monkeypatch) -> None:
    import asr_eval.audio_characteristics as characteristics

    calls = {}
    expected = np.array([0.25, -0.25], dtype="<f4")

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return characteristics.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=expected.tobytes(),
            stderr=b"",
        )

    monkeypatch.setattr(characteristics.subprocess, "run", fake_run)

    samples = _decode_with_ffmpeg(
        {"bytes": b"audio-payload", "path": None},
        "sample-1",
        "clean_audio",
    )

    np.testing.assert_array_equal(samples, expected)
    assert calls["kwargs"]["input"] == b"audio-payload"
    assert calls["command"][-3:] == [
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "pipe:1",
    ]
    assert calls["command"][calls["command"].index("-ac") + 1] == "1"


def test_pitch_range_estimate_is_not_a_gender_label() -> None:
    assert pitch_range_estimate(None) == "insufficient_voiced_frames"
    assert pitch_range_estimate(150.0) == "typically_lower_pitch"
    assert pitch_range_estimate(175.0) == "overlapping_pitch_range"
    assert pitch_range_estimate(200.0) == "typically_higher_pitch"


def test_analyze_audio_pair_includes_pitch_track_and_both_snr_values() -> None:
    sample_rate = 16000
    clean = _tone(180.0, 1.0, sample_rate)
    noise = _tone(700.0, 1.0, sample_rate) * 0.02

    metrics = analyze_audio_pair(
        "example-1",
        clean,
        clean + noise,
        dataset_snr_db=15.5,
        sample_rate=sample_rate,
    )

    assert metrics["sample_id"] == "example-1"
    assert metrics["pitch"]["median_f0_hz"] == pytest.approx(180.0, abs=2.0)
    assert metrics["pitch"]["frame_seconds"] == 0.04
    assert metrics["pitch"]["hop_seconds"] == 0.01
    assert metrics["snr"]["dataset_snr_db"] == 15.5
    assert metrics["snr"]["median_active_frame_snr_db"] == pytest.approx(
        33.98, abs=0.1
    )


def test_analyze_urgent24_writes_per_sample_and_summary(
    monkeypatch, tmp_path
) -> None:
    import asr_eval.audio_characteristics as characteristics

    sample_rate = 16000
    clean = _tone(180.0, 0.5, sample_rate)
    noisy = clean + _tone(700.0, 0.5, sample_rate) * 0.02
    examples = [
        {
            "sample_id": "sample-1",
            "clean_audio": clean,
            "noisy_audio": noisy,
            "sampling_rate": sample_rate,
            "dataset_snr_db": 14.0,
        }
    ]
    monkeypatch.setattr(
        characteristics,
        "load_urgent24_examples",
        lambda max_samples=None: examples,
    )
    output_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = analyze_urgent24(output_path, summary_path)

    record = json.loads(output_path.read_text().splitlines()[0])
    saved_summary = json.loads(summary_path.read_text())
    assert summary == saved_summary
    assert summary["samples"] == 1
    assert record["sample_id"] == "sample-1"
    assert len(record["pitch"]["f0_hz"]) > 0
    assert record["snr"]["dataset_snr_db"] == 14.0
