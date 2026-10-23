"""Audio pitch and SNR measurements for the URGENT24 paired-audio split."""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Optional

import numpy as np

from asr_eval.data import _audio_array_and_rate

logger = logging.getLogger(__name__)

URGENT24_DATASET_ID = "urgent-challenge/urgent2024_official"
URGENT24_CONFIG = "test_nonblind"
URGENT24_SPLIT = "validation"
DEFAULT_SAMPLE_RATE = 16000
PITCH_FRAME_SECONDS = 0.040
PITCH_HOP_SECONDS = 0.010
SNR_FRAME_SECONDS = 0.020
SNR_HOP_SECONDS = 0.010
MIN_PITCH_HZ = 60.0
MAX_PITCH_HZ = 400.0
MIN_VOICING_CONFIDENCE = 0.30
MIN_ACTIVE_RMS = 1e-5
ACTIVE_RMS_RELATIVE_THRESHOLD = 0.01


@dataclass(frozen=True)
class PitchTrack:
    """Frame-level F0 sequence and its extraction settings."""

    frame_times_seconds: list[float]
    f0_hz: list[Optional[float]]
    confidence: list[float]
    frame_seconds: float
    hop_seconds: float
    min_pitch_hz: float
    max_pitch_hz: float


def _mono_float_array(
    audio: Any,
    name: str,
    fallback_sample_rate: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Convert one decoded Hugging Face audio object into finite mono samples."""
    if isinstance(audio, np.ndarray):
        if fallback_sample_rate is None:
            raise ValueError(
                "{} as a raw array requires an explicit sample rate".format(name)
            )
        array, sample_rate = audio, fallback_sample_rate
    else:
        array, sample_rate = _audio_array_and_rate(audio)
    samples = np.asarray(array, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("{} must contain at least one audio sample".format(name))
    if not np.isfinite(samples).all():
        raise ValueError("{} contains non-finite audio samples".format(name))
    if sample_rate <= 0:
        raise ValueError("{} has an invalid sample rate".format(name))
    return samples, sample_rate


def _decode_with_ffmpeg(audio: Any, sample_id: str, column: str) -> np.ndarray:
    """Decode an undecoded Hugging Face audio payload to 16 kHz mono float32."""
    if not isinstance(audio, dict):
        raise ValueError(
            "Expected raw audio payload for {} in sample {!r}".format(
                column, sample_id
            )
        )

    payload = audio.get("bytes")
    if payload is None:
        path = audio.get("path")
        if not path:
            raise ValueError(
                "Audio payload for {} in sample {!r} has neither bytes nor path".format(
                    column, sample_id
                )
            )
        input_source = str(path)
        input_bytes = None
    else:
        input_source = "pipe:0"
        input_bytes = payload

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        input_source,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg is required to decode URGENT24; install it and ensure "
            "the `ffmpeg` executable is on PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "FFmpeg failed to decode {} for sample {!r}: {}".format(
                column, sample_id, detail or "unknown decoding error"
            )
        ) from error

    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError(
            "FFmpeg decoded no samples for {} in sample {!r}".format(
                column, sample_id
            )
        )
    return samples


def extract_pitch_track(
    audio: Any,
    sample_rate: int,
    frame_seconds: float = PITCH_FRAME_SECONDS,
    hop_seconds: float = PITCH_HOP_SECONDS,
    min_pitch_hz: float = MIN_PITCH_HZ,
    max_pitch_hz: float = MAX_PITCH_HZ,
    min_voicing_confidence: float = MIN_VOICING_CONFIDENCE,
) -> PitchTrack:
    """Estimate F0 in overlapping frames using windowed autocorrelation.

    Unvoiced frames have ``None`` for F0. Confidence is the normalized
    autocorrelation peak within the configured pitch lag range.
    """
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("Pitch estimation expects mono audio")
    if not np.isfinite(samples).all():
        raise ValueError("Pitch estimation audio contains non-finite samples")
    if sample_rate <= 0 or frame_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("Sample rate and frame/hop durations must be positive")
    if min_pitch_hz <= 0 or max_pitch_hz <= min_pitch_hz:
        raise ValueError("Pitch limits must be positive and increasing")
    if not 0 <= min_voicing_confidence <= 1:
        raise ValueError("Voicing confidence must be between 0 and 1")

    frame_length = int(round(frame_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    if frame_length < 3 or hop_length < 1:
        raise ValueError("Frame and hop durations are too short for this sample rate")

    lag_min = max(1, int(np.floor(sample_rate / max_pitch_hz)))
    lag_max = min(frame_length - 2, int(np.ceil(sample_rate / min_pitch_hz)))
    if lag_min > lag_max:
        raise ValueError("Frame is too short for the requested pitch range")
    if samples.size < frame_length:
        return PitchTrack(
            frame_times_seconds=[],
            f0_hz=[],
            confidence=[],
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            min_pitch_hz=min_pitch_hz,
            max_pitch_hz=max_pitch_hz,
        )

    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[
        ::hop_length
    ]
    window = np.hanning(frame_length)
    fft_length = 1 << (2 * frame_length - 1).bit_length()
    estimated_pitch: list[Optional[float]] = []
    confidences: list[float] = []
    frame_times: list[float] = []

    for start in range(0, len(frames), 2048):
        frame_batch = np.array(frames[start : start + 2048], copy=True)
        frame_batch -= frame_batch.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(np.square(frame_batch), axis=1))
        spectrum = np.fft.rfft(frame_batch * window, n=fft_length, axis=1)
        autocorrelation = np.fft.irfft(
            spectrum * spectrum.conjugate(), n=fft_length, axis=1
        )[:, :frame_length]
        zero_lag = autocorrelation[:, 0]
        denominator = np.maximum(zero_lag, np.finfo(np.float64).tiny)
        normalized = autocorrelation / denominator[:, None]
        pitches = np.full(len(frame_batch), np.nan, dtype=np.float64)
        peak_confidence = np.zeros(len(frame_batch), dtype=np.float64)

        candidates = normalized[:, lag_min : lag_max + 1]
        best_lags = np.argmax(candidates, axis=1) + lag_min
        row_indices = np.arange(len(frame_batch))
        peak_confidence = normalized[row_indices, best_lags]
        left = normalized[row_indices, best_lags - 1]
        center = peak_confidence
        right = normalized[row_indices, best_lags + 1]
        curvature = left - 2.0 * center + right
        offset = np.zeros_like(curvature)
        valid_curvature = np.abs(curvature) > 1e-12
        offset[valid_curvature] = (
            0.5
            * (left[valid_curvature] - right[valid_curvature])
            / curvature[valid_curvature]
        )
        offset = np.clip(offset, -0.5, 0.5)
        refined_lags = best_lags + offset
        valid_frames = (
            (rms >= max(MIN_ACTIVE_RMS, float(rms.max(initial=0.0)) * 0.01))
            & (peak_confidence >= min_voicing_confidence)
            & (refined_lags > 0)
        )
        pitches[valid_frames] = sample_rate / refined_lags[valid_frames]
        pitches[
            (pitches < min_pitch_hz) | (pitches > max_pitch_hz)
        ] = np.nan

        estimated_pitch.extend(
            None if not np.isfinite(value) else round(float(value), 3)
            for value in pitches
        )
        confidences.extend(round(float(value), 4) for value in peak_confidence)
        frame_times.extend(
            round((start + index) * hop_seconds, 4)
            for index in range(len(frame_batch))
        )

    return PitchTrack(
        frame_times_seconds=frame_times,
        f0_hz=estimated_pitch,
        confidence=confidences,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        min_pitch_hz=min_pitch_hz,
        max_pitch_hz=max_pitch_hz,
    )


def median_frame_snr_db(
    clean_audio: Any,
    noisy_audio: Any,
    sample_rate: int,
    frame_seconds: float = SNR_FRAME_SECONDS,
    hop_seconds: float = SNR_HOP_SECONDS,
) -> Optional[float]:
    """Return median active-frame SNR using ``noise = noisy - clean``."""
    clean = np.asarray(clean_audio, dtype=np.float64)
    noisy = np.asarray(noisy_audio, dtype=np.float64)
    if clean.ndim != 1 or noisy.ndim != 1:
        raise ValueError("SNR estimation expects mono audio")
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("SNR estimation audio contains non-finite samples")
    if sample_rate <= 0 or frame_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("Sample rate and frame/hop durations must be positive")

    sample_count = min(clean.size, noisy.size)
    frame_length = int(round(frame_seconds * sample_rate))
    hop_length = int(round(hop_seconds * sample_rate))
    if frame_length < 1 or hop_length < 1:
        raise ValueError("Frame and hop durations are too short for this sample rate")
    if sample_count < frame_length:
        return None

    clean = clean[:sample_count]
    noise = noisy[:sample_count] - clean
    clean_frames = np.lib.stride_tricks.sliding_window_view(clean, frame_length)[
        ::hop_length
    ]
    noise_frames = np.lib.stride_tricks.sliding_window_view(noise, frame_length)[
        ::hop_length
    ]
    signal_power = np.mean(np.square(clean_frames), axis=1)
    noise_power = np.mean(np.square(noise_frames), axis=1)
    active_threshold = max(
        MIN_ACTIVE_RMS**2,
        float(signal_power.max(initial=0.0)) * ACTIVE_RMS_RELATIVE_THRESHOLD**2,
    )
    active = signal_power >= active_threshold
    if not np.any(active):
        return None

    active_signal_power = signal_power[active]
    active_noise_power = noise_power[active]
    positive_noise = active_noise_power > np.finfo(np.float64).tiny
    if not np.any(positive_noise):
        return None
    snr_values = 10.0 * np.log10(
        active_signal_power[positive_noise] / active_noise_power[positive_noise]
    )
    return round(float(np.median(snr_values)), 4)


def pitch_range_estimate(median_f0_hz: Optional[float]) -> str:
    """Describe pitch range without treating F0 as a gender determination."""
    if median_f0_hz is None:
        return "insufficient_voiced_frames"
    if median_f0_hz < 165.0:
        return "typically_lower_pitch"
    if median_f0_hz > 185.0:
        return "typically_higher_pitch"
    return "overlapping_pitch_range"


def analyze_audio_pair(
    sample_id: str,
    clean_audio: Any,
    noisy_audio: Any,
    dataset_snr_db: Optional[float] = None,
    sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculate frame-level pitch and paired-audio SNR for one dataset row."""
    clean, clean_rate = _mono_float_array(
        clean_audio, "clean_audio", fallback_sample_rate=sample_rate
    )
    noisy, noisy_rate = _mono_float_array(
        noisy_audio, "noisy_audio", fallback_sample_rate=sample_rate
    )
    if clean_rate != noisy_rate:
        raise ValueError(
            "Clean and noisy audio sample rates differ for sample {!r}: {} vs {}".format(
                sample_id, clean_rate, noisy_rate
            )
        )

    track = extract_pitch_track(clean, clean_rate)
    voiced_pitch = [value for value in track.f0_hz if value is not None]
    median_f0 = round(float(median(voiced_pitch)), 3) if voiced_pitch else None
    snr = median_frame_snr_db(clean, noisy, clean_rate)
    return {
        "sample_id": str(sample_id),
        "duration_seconds": round(clean.size / clean_rate, 4),
        "sample_rate_hz": clean_rate,
        "pitch": {
            "median_f0_hz": median_f0,
            "p10_f0_hz": (
                round(float(np.percentile(voiced_pitch, 10)), 3)
                if voiced_pitch
                else None
            ),
            "p90_f0_hz": (
                round(float(np.percentile(voiced_pitch, 90)), 3)
                if voiced_pitch
                else None
            ),
            "voiced_frame_count": len(voiced_pitch),
            "total_frame_count": len(track.f0_hz),
            "voiced_fraction": (
                round(len(voiced_pitch) / len(track.f0_hz), 4)
                if track.f0_hz
                else 0.0
            ),
            "pitch_range_estimate": pitch_range_estimate(median_f0),
            "frame_times_seconds": track.frame_times_seconds,
            "f0_hz": track.f0_hz,
            "confidence": track.confidence,
            "frame_seconds": track.frame_seconds,
            "hop_seconds": track.hop_seconds,
            "min_pitch_hz": track.min_pitch_hz,
            "max_pitch_hz": track.max_pitch_hz,
        },
        "snr": {
            "dataset_snr_db": (
                round(float(dataset_snr_db), 4)
                if dataset_snr_db is not None
                else None
            ),
            "median_active_frame_snr_db": snr,
        },
    }


def load_urgent24_examples(
    max_samples: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """Stream the same labeled URGENT24 non-blind test split used for ASR."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")

    from datasets import Audio, load_dataset

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg is required to decode URGENT24; install it and ensure "
            "the `ffmpeg` executable is on PATH."
        )
    dataset = load_dataset(
        URGENT24_DATASET_ID,
        URGENT24_CONFIG,
        split=URGENT24_SPLIT,
        streaming=True,
    )
    for column in ("clean_audio", "noisy_audio"):
        if column not in dataset.features:
            raise ValueError(
                "URGENT24 split is missing required audio column {!r}".format(column)
            )
        dataset = dataset.cast_column(
            column, Audio(sampling_rate=DEFAULT_SAMPLE_RATE, decode=False)
        )

    count = 0
    for row in dataset:
        if max_samples is not None and count >= max_samples:
            break
        if "id" not in row:
            raise ValueError("URGENT24 row is missing required id field")
        if "snr_dB" not in row:
            raise ValueError("URGENT24 row is missing required snr_dB field")
        clean = _decode_with_ffmpeg(row["clean_audio"], str(row["id"]), "clean_audio")
        noisy = _decode_with_ffmpeg(row["noisy_audio"], str(row["id"]), "noisy_audio")
        yield {
            "sample_id": str(row["id"]),
            "clean_audio": clean,
            "noisy_audio": noisy,
            "sampling_rate": DEFAULT_SAMPLE_RATE,
            "dataset_snr_db": float(row["snr_dB"]),
        }
        count += 1


def analyze_urgent24(
    output_path: Path,
    summary_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream URGENT24 analysis to JSONL and write a dataset-level summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    f0_medians: list[float] = []
    dataset_snrs: list[float] = []
    computed_snrs: list[float] = []
    pitch_ranges: Dict[str, int] = {}
    count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for example in load_urgent24_examples(max_samples=max_samples):
            result = analyze_audio_pair(
                sample_id=example["sample_id"],
                clean_audio=example["clean_audio"],
                noisy_audio=example["noisy_audio"],
                dataset_snr_db=example["dataset_snr_db"],
                sample_rate=example["sampling_rate"],
            )
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            count += 1

            median_f0 = result["pitch"]["median_f0_hz"]
            if median_f0 is not None:
                f0_medians.append(median_f0)
            pitch_range = result["pitch"]["pitch_range_estimate"]
            pitch_ranges[pitch_range] = pitch_ranges.get(pitch_range, 0) + 1
            dataset_snrs.append(result["snr"]["dataset_snr_db"])
            if result["snr"]["median_active_frame_snr_db"] is not None:
                computed_snrs.append(
                    result["snr"]["median_active_frame_snr_db"]
                )
            logger.info("Analyzed sample %d (%s)", count, result["sample_id"])

    if count == 0:
        raise ValueError("The selected URGENT24 split yielded no examples")

    summary = {
        "dataset": URGENT24_DATASET_ID,
        "dataset_config": URGENT24_CONFIG,
        "split": URGENT24_SPLIT,
        "samples": count,
        "pitch": {
            "per_sample_median_f0_hz": {
                "median": round(float(median(f0_medians)), 3) if f0_medians else None,
                "p10": (
                    round(float(np.percentile(f0_medians, 10)), 3)
                    if f0_medians
                    else None
                ),
                "p90": (
                    round(float(np.percentile(f0_medians, 90)), 3)
                    if f0_medians
                    else None
                ),
            },
            "pitch_range_estimate_counts": pitch_ranges,
            "interpretation": (
                "Pitch ranges describe acoustic F0 only and do not determine "
                "speaker gender."
            ),
            "pitch_range_thresholds_hz": {
                "typically_lower_pitch": "median F0 < 165 Hz",
                "overlapping_pitch_range": "median F0 from 165 through 185 Hz",
                "typically_higher_pitch": "median F0 > 185 Hz",
            },
            "frame_seconds": PITCH_FRAME_SECONDS,
            "hop_seconds": PITCH_HOP_SECONDS,
            "min_pitch_hz": MIN_PITCH_HZ,
            "max_pitch_hz": MAX_PITCH_HZ,
        },
        "snr": {
            "dataset_snr_db_median": round(float(median(dataset_snrs)), 4),
            "median_active_frame_snr_db_median": (
                round(float(median(computed_snrs)), 4) if computed_snrs else None
            ),
            "calculation": (
                "Per-frame 10*log10(clean power / (noisy-clean) power), "
                "20 ms windows with 10 ms hop; median over active frames, "
                "then median across examples."
            ),
            "length_handling": (
                "Clean/noisy pairs are truncated to the shorter decoded length."
            ),
            "active_frame_threshold": (
                "clean RMS >= max(1e-5, 1% of the clip's peak frame RMS)"
            ),
        },
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    logger.info("Saved per-sample metrics to %s", output_path)
    logger.info("Saved dataset summary to %s", summary_path)
    return summary
