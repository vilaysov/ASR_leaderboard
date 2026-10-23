import json

import pytest

from asr_eval._open_asr_normalizer import EnglishTextNormalizer
from asr_eval.repetition_normalizer import normalize_repetitions
from asr_eval.scoring import score_file, score_texts


def test_leaderboard_english_normalizer_handles_tags_fillers_and_contractions() -> None:
    normalizer = EnglishTextNormalizer()

    assert normalizer("UM, I won't pay $1,200 <COMMA>") == "i will not pay $1200"


def test_score_texts_preserves_case_and_punctuation() -> None:
    result = score_texts(["Hello, world"], ["hello world!"])

    assert result.samples == 1
    assert result.reference_words == 2
    assert result.substitutions == 2
    assert result.wer == 1.0


def test_score_texts_counts_word_edits() -> None:
    result = score_texts(["one two three"], ["one four"])

    assert result.reference_words == 3
    assert result.substitutions == 1
    assert result.deletions == 1
    assert result.insertions == 0
    assert result.wer == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        ("the white paper is good", "the whitepaper is good"),
        ("the whitepaper is good", "the white paper is good"),
    ],
)
def test_score_texts_merges_compound_word_boundaries(
    reference: str, hypothesis: str
) -> None:
    result = score_texts([reference], [hypothesis])

    assert result.reference_words == len(reference.split())
    assert result.substitutions == 0
    assert result.deletions == 0
    assert result.insertions == 0
    assert result.wer == 0.0


def test_score_texts_rejects_empty_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="At least one"):
        score_texts([], [])
    with pytest.raises(ValueError, match="same length"):
        score_texts(["one"], [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no " * 6, "no"),
        ("thank you " * 6, "thank you"),
        ("new york university of " * 7, "new york university of"),
        ("no " * 5, "no no no no no"),
        ("no no yes no no yes", "no no yes no no yes"),
        ("start " + "thank you " * 6 + "finish", "start thank you finish"),
    ],
)
def test_repetition_normalizer_collapses_only_long_exact_runs(
    text: str, expected: str
) -> None:
    assert normalize_repetitions(text) == expected


def test_score_texts_can_normalize_reference_and_hypothesis_independently() -> None:
    raw = score_texts(["HELLO, WORLD!"], ["hello world"])
    reference_normalized = score_texts(
        ["HELLO, WORLD!"], ["hello world"], normalize_reference=True
    )
    hypothesis_normalized = score_texts(
        ["HELLO, WORLD!"], ["hello world"], normalize_hypothesis=True
    )
    both_normalized = score_texts(
        ["HELLO, WORLD!"],
        ["hello world"],
        normalize_reference=True,
        normalize_hypothesis=True,
    )

    assert raw.wer == 1.0
    assert reference_normalized.wer == 0.0
    assert hypothesis_normalized.wer == 1.0
    assert both_normalized.wer == 0.0


def test_score_file_reads_jsonl_and_metadata(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    prediction_file.write_text(
        json.dumps(
            {
                "reference": "one two",
                "hypothesis": "one too",
                "dataset": "example",
                "dataset_config": "config",
                "split": "test",
                "model": "model-id",
                "model_revision": "model-revision",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_file(prediction_file)

    assert result["dataset"] == "example"
    assert result["dataset_config"] == "config"
    assert result["split"] == "test"
    assert result["model"] == "model-id"
    assert result["model_revision"] == "model-revision"
    assert result["scoring_method"] == (
        "kaldialign.edit_distance with merge_compounds=True"
    )
    assert result["scoring_runtime"]["python"]
    assert result["scoring_runtime"]["kaldialign"]
    assert result["scoring_runtime"]["asr_eval"]
    assert result["wer"] == 0.5
    assert result["error_analysis"] == {
        "substitutions": [
            {"reference_word": "two", "hypothesis_word": "too", "count": 1}
        ],
        "deletions": [],
        "insertions": [],
    }


def test_score_file_preserves_reproducibility_metadata_and_rejects_mixed_runs(
    tmp_path,
) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    run_metadata = {
        "batch_size": 1,
        "inference_settings": {"language": "en", "do_sample": False},
        "runtime": {"python": "3.11.0", "torch": "2.5.0"},
        "audio_filter": None,
    }
    records = [
        {"reference": "one two", "hypothesis": "one two", **run_metadata},
        {"reference": "three four", "hypothesis": "three four", **run_metadata},
    ]
    prediction_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = score_file(prediction_file)

    for key, value in run_metadata.items():
        assert result[key] == value

    records[1]["batch_size"] = 4
    prediction_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Inconsistent 'batch_size' metadata"):
        score_file(prediction_file)


def test_score_file_reports_most_common_edit_types(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    records = [
        {"reference": "a cat sleeps", "hypothesis": "a dog"},
        {"reference": "a cat runs", "hypothesis": "a dog runs quickly"},
        {"reference": "birds fly", "hypothesis": "birds fly high"},
    ]
    prediction_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = score_file(prediction_file)

    assert result["error_analysis"] == {
        "substitutions": [
            {"reference_word": "cat", "hypothesis_word": "dog", "count": 2}
        ],
        "deletions": [{"word": "sleeps", "count": 1}],
        "insertions": [
            {"word": "high", "count": 1},
            {"word": "quickly", "count": 1},
        ],
    }


def test_score_file_reports_raw_and_normalized_scores(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    prediction_file.write_text(
        json.dumps({"reference": "HELLO, WORLD!", "hypothesis": "hello world"})
        + "\n",
        encoding="utf-8",
    )

    result = score_file(
        prediction_file,
        normalize_reference=True,
        normalize_hypothesis=True,
    )

    assert result["wer"] == 0.0
    assert result["raw_score"]["wer"] == 1.0
    assert result["wer_change"] == -1.0
    assert result["normalization"] == {
        "method": "Hugging Face Open ASR Leaderboard EnglishTextNormalizer",
        "reference": True,
        "hypothesis": True,
    }
    assert result["error_analysis"] == {
        "substitutions": [],
        "deletions": [],
        "insertions": [],
    }


def test_score_file_can_apply_repetition_cleanup_to_hypothesis(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    characteristics_file = tmp_path / "characteristics.jsonl"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "repeat",
                "reference": "thank you the boy felt ill and terribly alone",
                "hypothesis": "thank you " * 6 + "the boy felt ill and terribly alone",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    characteristics_file.write_text(
        json.dumps(
            {
                "sample_id": "repeat",
                "pitch": {"median_f0_hz": 100.0},
                "snr": {"median_active_frame_snr_db": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = score_file(
        prediction_file,
        normalize_repetitions=True,
        characteristics_path=characteristics_file,
    )

    assert result["wer"] == 0.0
    assert result["raw_score"]["wer"] == pytest.approx(10 / 9)
    assert result["wer_change"] == pytest.approx(-10 / 9)
    assert result["repetition_normalization"] == {
        "enabled": True,
        "applied_to": "hypothesis",
        "minimum_repetitions": 6,
        "maximum_segment_words": 8,
        "behavior": "keep one copy and remove subsequent exact adjacent copies",
        "samples_changed": 1,
        "words_removed": 10,
    }
    assert (
        result["stratified_wer"]["pitch_range"]["groups"][
            "male_pitch_range_proxy"
        ]["wer"]["wer"]
        == result["wer"]
    )


def test_score_file_can_stratify_wer_by_pitch_and_median_frame_snr(
    tmp_path,
) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    characteristics_file = tmp_path / "characteristics.jsonl"
    predictions = [
        {"sample_id": "a", "reference": "one two", "hypothesis": "one two"},
        {"sample_id": "b", "reference": "one two", "hypothesis": "one too"},
        {"sample_id": "c", "reference": "one two", "hypothesis": "one two"},
        {"sample_id": "d", "reference": "one two", "hypothesis": "one too"},
        {"sample_id": "e", "reference": "one two", "hypothesis": "wrong words"},
    ]
    characteristics = [
        {
            "sample_id": "a",
            "pitch": {"median_f0_hz": 100.0},
            "snr": {"median_active_frame_snr_db": -20.0},
        },
        {
            "sample_id": "b",
            "pitch": {"median_f0_hz": 155.0},
            "snr": {"median_active_frame_snr_db": -15.0},
        },
        {
            "sample_id": "c",
            "pitch": {"median_f0_hz": 165.0},
            "snr": {"median_active_frame_snr_db": 15.0},
        },
        {
            "sample_id": "d",
            "pitch": {"median_f0_hz": 255.0},
            "snr": {"median_active_frame_snr_db": 20.0},
        },
        {
            "sample_id": "e",
            "pitch": {"median_f0_hz": 160.0},
            "snr": {"median_active_frame_snr_db": 20.1},
        },
    ]
    prediction_file.write_text(
        "".join(json.dumps(record) + "\n" for record in predictions),
        encoding="utf-8",
    )
    characteristics_file.write_text(
        "".join(json.dumps(record) + "\n" for record in characteristics),
        encoding="utf-8",
    )

    result = score_file(
        prediction_file,
        characteristics_path=characteristics_file,
    )

    stratified = result["stratified_wer"]
    pitch_groups = stratified["pitch_range"]["groups"]
    assert pitch_groups["male_pitch_range_proxy"]["wer"]["samples"] == 2
    assert pitch_groups["male_pitch_range_proxy"]["wer"]["wer"] == pytest.approx(
        0.25
    )
    assert pitch_groups["female_pitch_range_proxy"]["wer"]["samples"] == 2
    assert pitch_groups["female_pitch_range_proxy"]["wer"]["wer"] == pytest.approx(
        0.25
    )
    assert pitch_groups["other_or_unavailable"]["wer"]["samples"] == 1
    assert pitch_groups["other_or_unavailable"]["wer"]["wer"] == 1.0

    snr_bins = {
        entry["snr_db"]: entry["wer"]
        for entry in stratified["median_frame_snr_db"]["bins"]
    }
    assert snr_bins["[-20, -15) dB"]["samples"] == 1
    assert snr_bins["[-15, -10) dB"]["wer"] == pytest.approx(0.5)
    assert snr_bins["[15, 20] dB"]["samples"] == 2
    assert snr_bins["[15, 20] dB"]["wer"] == pytest.approx(0.25)
    assert (
        stratified["median_frame_snr_db"]["excluded"]["above_20_db"]["samples"]
        == 1
    )


def test_score_file_requires_characteristics_for_every_prediction_sample(
    tmp_path,
) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    characteristics_file = tmp_path / "characteristics.jsonl"
    prediction_file.write_text(
        json.dumps(
            {"sample_id": "missing", "reference": "one", "hypothesis": "one"}
        )
        + "\n",
        encoding="utf-8",
    )
    characteristics_file.write_text(
        json.dumps(
            {
                "sample_id": "other",
                "pitch": {"median_f0_hz": 100.0},
                "snr": {"median_active_frame_snr_db": 0.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing from"):
        score_file(
            prediction_file,
            characteristics_path=characteristics_file,
        )


def test_score_file_rejects_duplicate_characteristic_sample_ids(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    characteristics_file = tmp_path / "characteristics.jsonl"
    prediction_file.write_text(
        json.dumps({"sample_id": "a", "reference": "one", "hypothesis": "one"})
        + "\n",
        encoding="utf-8",
    )
    characteristic = {
        "sample_id": "a",
        "pitch": {"median_f0_hz": 100.0},
        "snr": {"median_active_frame_snr_db": 0.0},
    }
    characteristics_file.write_text(
        json.dumps(characteristic) + "\n" + json.dumps(characteristic) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        score_file(
            prediction_file,
            characteristics_path=characteristics_file,
        )


def test_score_file_reports_weighted_wer_for_sample_scoped_proper_names(
    tmp_path,
) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    names_file = tmp_path / "names.csv"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "reference": "Alice met the team",
                "hypothesis": "Alicia met the team",
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "sample-b",
                "reference": "Alice met the team",
                "hypothesis": "Alicia met the team",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    names_file.write_text(
        "name,entity_type,occurrences,sample_ids\n"
        '"Alice","PersonalName","1","sample-a"\n',
        encoding="utf-8",
    )

    result = score_file(prediction_file, proper_names_path=names_file)

    weighted = result["proper_name_weighted_wer"]
    assert result["wer"] == pytest.approx(2 / 8)
    assert weighted["wer"] == pytest.approx(4 / 10)
    assert weighted["proper_name_weight"] == 3.0
    assert weighted["samples"] == 2
    assert weighted["proper_name_reference_tokens"] == 1
    assert weighted["weighted_reference_words"] == 10.0
    assert weighted["weighted_substitution_cost"] == 4.0
    assert weighted["unique_proper_names"] == 1


def test_weighted_wer_weights_inserted_proper_name_tokens(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    names_file = tmp_path / "names.csv"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "reference": "met Bob",
                "hypothesis": "met Alice Bob",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    names_file.write_text(
        "name,sample_ids\nAlice,sample-a\n",
        encoding="utf-8",
    )

    result = score_file(prediction_file, proper_names_path=names_file)

    weighted = result["proper_name_weighted_wer"]
    assert result["wer"] == 0.5
    assert weighted["weighted_insertion_cost"] == 3.0
    assert weighted["weighted_reference_words"] == 2.0
    assert weighted["wer"] == 1.5


def test_weighted_wer_preserves_compound_split_join_matches(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    names_file = tmp_path / "names.csv"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "reference": "I visited New York",
                "hypothesis": "I visited NewYork",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    names_file.write_text(
        "name,sample_ids\nNew York,sample-a\n",
        encoding="utf-8",
    )

    result = score_file(prediction_file, proper_names_path=names_file)

    assert result["wer"] == 0.0
    assert result["proper_name_weighted_wer"]["wer"] == 0.0


def test_weighted_wer_handles_punctuation_only_hypothesis_tokens(tmp_path) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    names_file = tmp_path / "names.csv"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "reference": "Alice arrived",
                "hypothesis": "Alice ... arrived",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    names_file.write_text("name,sample_ids\nAlice,sample-a\n", encoding="utf-8")

    result = score_file(prediction_file, proper_names_path=names_file)

    assert result["proper_name_weighted_wer"]["weighted_insertion_cost"] == 1.0


def test_score_cli_accepts_proper_names_file_and_custom_weight(
    tmp_path, capsys
) -> None:
    from asr_eval.cli import main

    prediction_file = tmp_path / "predictions.jsonl"
    names_file = tmp_path / "names.csv"
    report_file = tmp_path / "wer.json"
    prediction_file.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "reference": "Alice met the team",
                "hypothesis": "Alicia met the team",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    names_file.write_text("name,sample_ids\nAlice,sample-a\n", encoding="utf-8")

    main(
        [
            "score",
            "--input",
            str(prediction_file),
            "--proper-names-file",
            str(names_file),
            "--proper-name-weight",
            "2",
            "--output",
            str(report_file),
        ]
    )
    capsys.readouterr()

    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["wer"] == 0.25
    assert report["proper_name_weighted_wer"]["proper_name_weight"] == 2.0
    assert report["proper_name_weighted_wer"]["wer"] == pytest.approx(2 / 5)
