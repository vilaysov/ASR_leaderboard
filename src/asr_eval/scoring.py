"""Offline WER scoring for saved ASR predictions."""

import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from asr_eval import __version__
from kaldialign import align, edit_distance

from asr_eval._open_asr_normalizer import EnglishTextNormalizer
from asr_eval.weighted_scoring import score_proper_name_weighted_wer
from asr_eval.repetition_normalizer import MAX_SEGMENT_WORDS, MIN_REPETITIONS
from asr_eval.repetition_normalizer import normalize_repetitions as clean_repetitions

TOP_ERROR_LIMIT = 10
ALIGNMENT_EPSILON = ""
PITCH_WER_GROUPS = (
    ("male_pitch_range_proxy", 85.0, 155.0),
    ("female_pitch_range_proxy", 165.0, 255.0),
)
SNR_BIN_WIDTH_DB = 5
SNR_MIN_DB = -20
SNR_MAX_DB = 20


@dataclass(frozen=True)
class ScoreResult:
    """Corpus-level word error rate and edit counts."""

    samples: int
    reference_words: int
    wer: float
    substitutions: int
    deletions: int
    insertions: int


@dataclass(frozen=True)
class WordOutput:
    """Tokenized corpus, Kaldi alignments, and aggregate edit counts."""

    references: List[List[str]]
    hypotheses: List[List[str]]
    alignments: List[List[Tuple[str, str]]]
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int


def _word_output(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> WordOutput:
    """Align word sequences with the leaderboard's Kaldi compound-aware method."""
    if not references:
        raise ValueError("At least one reference/hypothesis pair is required")
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    reference_tokens = [text.split() for text in references]
    hypothesis_tokens = [text.split() for text in hypotheses]
    alignments: List[List[Tuple[str, str]]] = []
    substitutions = deletions = insertions = 0

    for reference, hypothesis in zip(reference_tokens, hypothesis_tokens):
        edits = edit_distance(reference, hypothesis, merge_compounds=True)
        substitutions += edits["sub"]
        deletions += edits["del"]
        insertions += edits["ins"]
        alignments.append(
            align(
                reference,
                hypothesis,
                ALIGNMENT_EPSILON,
                merge_compounds=True,
            )
        )

    return WordOutput(
        references=reference_tokens,
        hypotheses=hypothesis_tokens,
        alignments=alignments,
        reference_words=sum(len(words) for words in reference_tokens),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
    )


def _score_result(word_output: WordOutput) -> ScoreResult:
    """Build aggregate WER metrics from Kaldi's compound-aware edit counts."""
    errors = (
        word_output.substitutions
        + word_output.deletions
        + word_output.insertions
    )
    if word_output.reference_words:
        wer = errors / word_output.reference_words
    else:
        wer = 0.0 if errors == 0 else float("inf")
    return ScoreResult(
        samples=len(word_output.references),
        reference_words=word_output.reference_words,
        wer=wer,
        substitutions=word_output.substitutions,
        deletions=word_output.deletions,
        insertions=word_output.insertions,
    )


def _error_analysis(word_output: WordOutput) -> Dict[str, List[Dict[str, Any]]]:
    """Return the most frequent edits from compound-aware Kaldi alignments."""
    substitutions: Counter[Tuple[str, str]] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()

    for alignment in word_output.alignments:
        for reference, hypothesis in alignment:
            if reference == ALIGNMENT_EPSILON:
                insertions.update(hypothesis.split())
            elif hypothesis == ALIGNMENT_EPSILON:
                deletions.update(reference.split())
            elif reference != hypothesis and (
                reference.replace(" ", "") != hypothesis.replace(" ", "")
            ):
                substitutions.update(zip(reference.split(), hypothesis.split()))

    def most_common(counter: Any, item_names: Tuple[str, ...]) -> List[Dict[str, Any]]:
        entries = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        output = []
        for item, count in entries[:TOP_ERROR_LIMIT]:
            values = item if len(item_names) > 1 else (item,)
            output.append(dict(zip(item_names, values), count=count))
        return output

    return {
        "substitutions": most_common(
            substitutions, ("reference_word", "hypothesis_word")
        ),
        "deletions": most_common(deletions, ("word",)),
        "insertions": most_common(insertions, ("word",)),
    }


def _apply_normalization(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize_reference: bool,
    normalize_hypothesis: bool,
) -> Tuple[Sequence[str], Sequence[str]]:
    """Normalize each scoring side independently when requested."""
    if not normalize_reference and not normalize_hypothesis:
        return references, hypotheses

    normalizer = EnglishTextNormalizer()
    if normalize_reference:
        references = [normalizer(text) for text in references]
    if normalize_hypothesis:
        hypotheses = [normalizer(text) for text in hypotheses]
    return references, hypotheses


def _load_characteristics(
    characteristics_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load pitch and median-frame SNR metrics keyed by dataset sample ID."""
    characteristics: Dict[str, Dict[str, Any]] = {}
    with characteristics_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON on line {} of {}".format(
                        line_number, characteristics_path
                    )
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    "Expected a JSON object on line {} of {}".format(
                        line_number, characteristics_path
                    )
                )
            if "sample_id" not in record:
                raise ValueError(
                    "Line {} of {} must contain sample_id".format(
                        line_number, characteristics_path
                    )
                )
            sample_id = str(record["sample_id"])
            if sample_id in characteristics:
                raise ValueError(
                    "Duplicate sample_id {!r} in {}".format(
                        sample_id, characteristics_path
                    )
                )
            pitch = record.get("pitch")
            snr = record.get("snr")
            if not isinstance(pitch, Mapping) or not isinstance(snr, Mapping):
                raise ValueError(
                    "Line {} of {} must contain pitch and snr objects".format(
                        line_number, characteristics_path
                    )
                )
            characteristics[sample_id] = {
                "median_f0_hz": pitch.get("median_f0_hz"),
                "median_active_frame_snr_db": snr.get(
                    "median_active_frame_snr_db"
                ),
            }
    if not characteristics:
        raise ValueError(
            "No sample characteristics found in {}".format(characteristics_path)
        )
    return characteristics


def _group_score(
    indexes: Sequence[int],
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> Dict[str, Any]:
    """Calculate corpus WER for one selected group of utterances."""
    if not indexes:
        return {
            "samples": 0,
            "reference_words": 0,
            "wer": None,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        }
    selected = _word_output(
        [references[index] for index in indexes],
        [hypotheses[index] for index in indexes],
    )
    return asdict(_score_result(selected))


def _optional_metric_value(
    value: Any,
    sample_id: str,
    metric_name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "Sample {!r} has a non-numeric {}".format(sample_id, metric_name)
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Sample {!r} has a non-numeric {}".format(sample_id, metric_name)
        ) from error
    if not math.isfinite(numeric_value):
        raise ValueError(
            "Sample {!r} has a non-finite {}".format(sample_id, metric_name)
        )
    return numeric_value


def _stratified_wer(
    sample_ids: Sequence[Optional[str]],
    references: Sequence[str],
    hypotheses: Sequence[str],
    characteristics_path: Path,
) -> Dict[str, Any]:
    """Report WER by pitch-range proxy and 5 dB median frame-SNR bins."""
    if any(sample_id is None for sample_id in sample_ids):
        raise ValueError(
            "Prediction JSONL must contain sample_id on every row to calculate "
            "stratified WER"
        )
    ids = [str(sample_id) for sample_id in sample_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Prediction JSONL contains duplicate sample_id values; "
            "stratified WER requires unique sample IDs"
        )

    characteristics = _load_characteristics(characteristics_path)
    missing_ids = [sample_id for sample_id in ids if sample_id not in characteristics]
    if missing_ids:
        raise ValueError(
            "{} prediction sample IDs are missing from {} (for example: {})".format(
                len(missing_ids),
                characteristics_path,
                ", ".join(missing_ids[:5]),
            )
        )

    pitch_indexes: Dict[str, List[int]] = {
        name: [] for name, _, _ in PITCH_WER_GROUPS
    }
    pitch_indexes["other_or_unavailable"] = []
    snr_indexes: Dict[str, List[int]] = {}
    snr_edges = list(range(SNR_MIN_DB, SNR_MAX_DB + SNR_BIN_WIDTH_DB, SNR_BIN_WIDTH_DB))
    for low, high in zip(snr_edges[:-1], snr_edges[1:]):
        upper_bracket = "]" if high == SNR_MAX_DB else ")"
        label = "[{}, {}{} dB".format(low, high, upper_bracket)
        snr_indexes[label] = []
    snr_indexes["below_-20_db"] = []
    snr_indexes["above_20_db"] = []
    snr_indexes["unavailable"] = []

    for index, sample_id in enumerate(ids):
        metrics = characteristics[sample_id]
        f0_value = _optional_metric_value(
            metrics["median_f0_hz"], sample_id, "median_f0_hz"
        )
        pitch_group = None
        for name, low, high in PITCH_WER_GROUPS:
            if f0_value is not None and low <= f0_value <= high:
                pitch_group = name
                break
        if pitch_group is None:
            pitch_indexes["other_or_unavailable"].append(index)
        else:
            pitch_indexes[pitch_group].append(index)

        snr = metrics["median_active_frame_snr_db"]
        snr_value = _optional_metric_value(
            snr, sample_id, "median_active_frame_snr_db"
        )
        if snr_value is None:
            snr_indexes["unavailable"].append(index)
            continue
        if snr_value < SNR_MIN_DB:
            snr_indexes["below_-20_db"].append(index)
        elif snr_value > SNR_MAX_DB:
            snr_indexes["above_20_db"].append(index)
        else:
            bin_index = min(
                int((snr_value - SNR_MIN_DB) // SNR_BIN_WIDTH_DB),
                len(snr_edges) - 2,
            )
            low = snr_edges[bin_index]
            high = snr_edges[bin_index + 1]
            upper_bracket = "]" if high == SNR_MAX_DB else ")"
            label = "[{}, {}{} dB".format(low, high, upper_bracket)
            snr_indexes[label].append(index)

    pitch_results = {
        name: _group_score(indexes, references, hypotheses)
        for name, indexes in pitch_indexes.items()
    }
    snr_results = {
        name: _group_score(indexes, references, hypotheses)
        for name, indexes in snr_indexes.items()
    }
    return {
        "characteristics_file": str(characteristics_path),
        "sample_id_join": "exact",
        "wer_basis": (
            "Same normalization and repetition settings as the overall score; "
            "aggregate edit errors divided by aggregate reference words."
        ),
        "pitch_range": {
            "interpretation": (
                "Pitch ranges are acoustic proxies, not verified speaker gender."
            ),
            "groups": {
                "male_pitch_range_proxy": {
                    "median_f0_hz": [85.0, 155.0],
                    "wer": pitch_results["male_pitch_range_proxy"],
                },
                "female_pitch_range_proxy": {
                    "median_f0_hz": [165.0, 255.0],
                    "wer": pitch_results["female_pitch_range_proxy"],
                },
                "other_or_unavailable": {
                    "description": (
                        "Median F0 outside the two selected ranges or unavailable."
                    ),
                    "wer": pitch_results["other_or_unavailable"],
                },
            },
        },
        "median_frame_snr_db": {
            "bin_width_db": SNR_BIN_WIDTH_DB,
            "range_db": [SNR_MIN_DB, SNR_MAX_DB],
            "interval_convention": (
                "Left-inclusive and right-exclusive, except the final bin includes "
                "20 dB."
            ),
            "bins": [
                {"snr_db": label, "wer": score}
                for label, score in snr_results.items()
                if label not in {"below_-20_db", "above_20_db", "unavailable"}
            ],
            "excluded": {
                "below_-20_db": snr_results["below_-20_db"],
                "above_20_db": snr_results["above_20_db"],
                "unavailable": snr_results["unavailable"],
            },
        },
    }


def score_texts(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalize_reference: bool = False,
    normalize_hypothesis: bool = False,
) -> ScoreResult:
    """Calculate corpus WER, optionally normalizing either side independently."""
    references, hypotheses = _apply_normalization(
        references,
        hypotheses,
        normalize_reference,
        normalize_hypothesis,
    )
    return _score_result(_word_output(references, hypotheses))


def score_file(
    input_path: Path,
    normalize_reference: bool = False,
    normalize_hypothesis: bool = False,
    normalize_repetitions: bool = False,
    characteristics_path: Optional[Path] = None,
    proper_names_path: Optional[Path] = None,
    proper_name_weight: float = 3.0,
) -> Dict[str, Any]:
    """Read predictions, score them, and optionally report a raw WER baseline."""
    references: List[str] = []
    hypotheses: List[str] = []
    sample_ids: List[Optional[str]] = []
    metadata: Dict[str, Any] = {}

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON on line {} of {}".format(line_number, input_path)
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError("Expected a JSON object on line {}".format(line_number))
            if "reference" not in record or "hypothesis" not in record:
                raise ValueError(
                    "Line {} must contain reference and hypothesis".format(line_number)
                )

            references.append(str(record["reference"]))
            hypotheses.append(str(record["hypothesis"]))
            sample_id = record.get("sample_id")
            sample_ids.append(None if sample_id is None else str(sample_id))
            for key in (
                "dataset",
                "dataset_config",
                "split",
                "model",
                "model_revision",
                "batch_size",
                "inference_settings",
                "runtime",
                "audio_filter",
            ):
                if key in record:
                    if key in metadata and metadata[key] != record[key]:
                        raise ValueError(
                            "Inconsistent {!r} metadata on line {} of {}".format(
                                key, line_number, input_path
                            )
                        )
                    metadata[key] = record[key]

    scoring_references, scoring_hypotheses = _apply_normalization(
        references,
        hypotheses,
        normalize_reference,
        normalize_hypothesis,
    )
    if normalize_repetitions:
        cleaned_hypotheses = [
            clean_repetitions(text) for text in scoring_hypotheses
        ]
        metadata["repetition_normalization"] = {
            "enabled": True,
            "applied_to": "hypothesis",
            "minimum_repetitions": MIN_REPETITIONS,
            "maximum_segment_words": MAX_SEGMENT_WORDS,
            "behavior": "keep one copy and remove subsequent exact adjacent copies",
            "samples_changed": sum(
                before != after
                for before, after in zip(scoring_hypotheses, cleaned_hypotheses)
            ),
            "words_removed": sum(
                len(before.split()) - len(after.split())
                for before, after in zip(scoring_hypotheses, cleaned_hypotheses)
            ),
        }
        scoring_hypotheses = cleaned_hypotheses
    word_output = _word_output(scoring_references, scoring_hypotheses)
    result = asdict(_score_result(word_output))
    result["error_analysis"] = _error_analysis(word_output)
    metadata["scoring_method"] = (
        "kaldialign.edit_distance with merge_compounds=True"
    )
    metadata["scoring_runtime"] = {
        "python": platform.python_version(),
        "kaldialign": version("kaldialign"),
        "asr_eval": __version__,
    }
    if normalize_reference or normalize_hypothesis or normalize_repetitions:
        raw_score = asdict(score_texts(references, hypotheses))
        if normalize_reference or normalize_hypothesis:
            metadata["normalization"] = {
                "method": "Hugging Face Open ASR Leaderboard EnglishTextNormalizer",
                "reference": normalize_reference,
                "hypothesis": normalize_hypothesis,
            }
        metadata["raw_score"] = raw_score
        metadata["wer_change"] = result["wer"] - raw_score["wer"]
    if characteristics_path is not None:
        metadata["stratified_wer"] = _stratified_wer(
            sample_ids,
            scoring_references,
            scoring_hypotheses,
            characteristics_path,
        )
    if proper_names_path is not None:
        metadata["proper_name_weighted_wer"] = score_proper_name_weighted_wer(
            scoring_references,
            scoring_hypotheses,
            sample_ids,
            proper_names_path,
            proper_name_weight,
        )
    return dict(metadata, **result)


def write_score(result: Mapping[str, Any], output_path: Path) -> None:
    """Write a machine-readable score report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
