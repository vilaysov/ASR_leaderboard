"""Command-line interface for inference and offline scoring."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from asr_eval.audio_characteristics import analyze_urgent24
from asr_eval.data import load_examples
from asr_eval.inference import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    create_transcriber,
    write_predictions,
)
from asr_eval.scoring import score_file, write_score
from asr_eval.stratified_plots import create_stratified_wer_plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and score ASR evaluations.")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="Run a model and save references and hypotheses."
    )
    transcribe.add_argument("--dataset", default="speechcolab/gigaspeech")
    transcribe.add_argument("--dataset-config", default="xs")
    transcribe.add_argument("--split", default="test")
    transcribe.add_argument("--audio-column", default="audio")
    transcribe.add_argument("--text-column", default="text")
    transcribe.add_argument("--id-column", default="segment_id")
    transcribe.add_argument("--model", default=DEFAULT_MODEL_ID)
    transcribe.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    transcribe.add_argument(
        "--bandpass-filter",
        action="store_true",
        help=(
            "Apply a zero-phase 70-7000 Hz Butterworth band-pass filter to audio "
            "before transcription."
        ),
    )
    transcribe.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of examples per inference batch (default: 1).",
    )
    sample_limit = transcribe.add_mutually_exclusive_group()
    sample_limit.add_argument(
        "--max-samples",
        type=int,
        help="Stop after this many rows (default: 100).",
    )
    sample_limit.add_argument(
        "--all-samples",
        action="store_true",
        help="Stream and transcribe every row in the selected dataset split.",
    )
    transcribe.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
    )

    score = commands.add_parser(
        "score", help="Calculate WER from a saved JSONL prediction file."
    )
    score.add_argument("--input", type=Path, required=True)
    score.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/wer.json"),
    )
    score.add_argument(
        "--normalize-reference",
        action="store_true",
        help="Apply the Open ASR leaderboard English normalizer to references.",
    )
    score.add_argument(
        "--normalize-hypothesis",
        action="store_true",
        help="Apply the Open ASR leaderboard English normalizer to hypotheses.",
    )
    score.add_argument(
        "--normalize-repetitions",
        action="store_true",
        help=(
            "On hypotheses, keep one copy of exact adjacent segments repeated "
            "at least 6 times."
        ),
    )
    score.add_argument(
        "--characteristics-file",
        type=Path,
        help=(
            "Optional JSONL pitch/SNR metrics file. When supplied, add WER "
            "breakdowns by pitch range and 5 dB median-frame-SNR bins."
        ),
    )
    score.add_argument(
        "--proper-names-file",
        type=Path,
        help=(
            "Optional CSV proper-name list. When supplied, calculate an "
            "additional proper-name-weighted WER."
        ),
    )
    score.add_argument(
        "--proper-name-weight",
        type=float,
        default=3.0,
        help="Error weight for listed proper-name tokens (default: 3.0).",
    )

    characterize = commands.add_parser(
        "characterize-urgent24",
        help="Extract sub-frame pitch and SNR metrics from URGENT24 audio pairs.",
    )
    characterize.add_argument(
        "--max-samples",
        type=int,
        help="Analyze at most this many rows (default: all 1,000 non-blind test rows).",
    )
    characterize.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/urgent24-test-nonblind-audio-characteristics.jsonl"
        ),
        help="Per-sample metrics JSONL output.",
    )
    characterize.add_argument(
        "--summary-output",
        type=Path,
        help="Dataset summary JSON output (default: OUTPUT with .summary.json suffix).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "characterize-urgent24":
        if args.max_samples is not None and args.max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")
        summary_path = args.summary_output or args.output.with_suffix(".summary.json")
        summary = analyze_urgent24(
            output_path=args.output,
            summary_path=summary_path,
            max_samples=args.max_samples,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "transcribe":
        max_samples = (
            None
            if args.all_samples
            else 100 if args.max_samples is None else args.max_samples
        )
        if max_samples is not None and max_samples <= 0:
            raise ValueError("--max-samples must be a positive integer")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be a positive integer")
        logging.info(
            "Starting transcription of %s samples from %s (config=%s, split=%s)",
            "all" if max_samples is None else "up to {}".format(max_samples),
            args.dataset,
            args.dataset_config or "default",
            args.split,
        )
        examples = load_examples(
            dataset_id=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            max_samples=max_samples,
            audio_column=args.audio_column,
            text_column=args.text_column,
            id_column=args.id_column,
        )
        transcriber = create_transcriber(args.model, revision=args.model_revision)
        count = write_predictions(
            examples,
            transcriber,
            args.dataset,
            args.split,
            args.output,
            dataset_config=args.dataset_config,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            bandpass_filter=args.bandpass_filter,
        )
        logging.info(
            "Done: transcribed %d samples and saved predictions to %s",
            count,
            args.output,
        )
        return

    result = score_file(
        args.input,
        normalize_reference=args.normalize_reference,
        normalize_hypothesis=args.normalize_hypothesis,
        normalize_repetitions=args.normalize_repetitions,
        characteristics_path=args.characteristics_file,
        proper_names_path=args.proper_names_file,
        proper_name_weight=args.proper_name_weight,
    )
    if args.characteristics_file is not None:
        result["stratified_wer"]["figures"] = create_stratified_wer_plots(
            result,
            args.output,
        )
    write_score(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    logging.info("Saved WER report to %s", args.output)
