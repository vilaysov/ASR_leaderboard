"""Plots for pitch- and SNR-stratified WER reports."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _draw_wer_bars(
    axis: Any,
    labels: Sequence[str],
    scores: Sequence[Mapping[str, Any]],
    colors: Sequence[str],
    overall_wer: Optional[float],
) -> None:
    values = [
        float(score["wer"]) * 100 if score["wer"] is not None else float("nan")
        for score in scores
    ]
    bars = axis.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    if overall_wer is not None:
        axis.axhline(
            overall_wer * 100,
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label="Overall WER: {:.1%}".format(overall_wer),
        )
        axis.legend(frameon=False)

    for bar, score, value in zip(bars, scores, values):
        label = "N/A" if score["wer"] is None else "{:.1f}%".format(value)
        axis.annotate(
            "{}\nn={}".format(label, score["samples"]),
            (bar.get_x() + bar.get_width() / 2, 0 if value != value else value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1,
            },
        )

    finite_values = [value for value in values if value == value]
    maximum = max(finite_values + ([overall_wer * 100] if overall_wer else [0.0]))
    axis.set_ylim(0, max(1.0, maximum * 1.3))
    axis.set_ylabel("Word Error Rate (WER, %)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def create_stratified_wer_plots(
    score_result: Mapping[str, Any],
    output_path: Path,
) -> Dict[str, str]:
    """Write English SNR-bin and pitch-range WER plots next to the score report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stratified = score_result["stratified_wer"]
    overall_wer = score_result.get("wer")
    overall_wer = float(overall_wer) if overall_wer is not None else None

    snr_bins = stratified["median_frame_snr_db"]["bins"]
    snr_labels = [entry["snr_db"].replace(" dB", "") for entry in snr_bins]
    snr_scores = [entry["wer"] for entry in snr_bins]
    snr_colors = ["#e87516"] * len(snr_bins)

    pitch_groups = stratified["pitch_range"]["groups"]
    pitch_definitions: Sequence[Tuple[str, str, str]] = (
        (
            "Male pitch-range proxy\n(85–155 Hz)",
            "male_pitch_range_proxy",
            "#4c78a8",
        ),
        (
            "Female pitch-range proxy\n(165–255 Hz)",
            "female_pitch_range_proxy",
            "#e87516",
        ),
        ("Other / unavailable", "other_or_unavailable", "#9a9a9a"),
    )
    pitch_labels = [label for label, _, _ in pitch_definitions]
    pitch_scores = [
        pitch_groups[key]["wer"] for _, key, _ in pitch_definitions
    ]
    pitch_colors = [color for _, _, color in pitch_definitions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snr_path = output_path.with_name(output_path.stem + "-snr-wer.png")
    pitch_path = output_path.with_name(output_path.stem + "-pitch-wer.png")

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    _draw_wer_bars(axis, snr_labels, snr_scores, snr_colors, overall_wer)
    axis.set_title("WER by Median Frame SNR")
    axis.set_xlabel("Median frame SNR bin (dB)")
    figure.savefig(snr_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    _draw_wer_bars(axis, pitch_labels, pitch_scores, pitch_colors, overall_wer)
    axis.set_title("WER by Median F0 Pitch Range")
    axis.set_xlabel("Median F0 pitch-range category")
    axis.tick_params(axis="x", labelsize=9)
    figure.text(
        0.5,
        -0.015,
        "Pitch ranges are acoustic proxies, not verified speaker gender.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.savefig(pitch_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "wer_by_median_frame_snr": str(snr_path),
        "wer_by_median_f0_pitch_range": str(pitch_path),
    }
