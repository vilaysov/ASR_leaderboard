import json

from asr_eval.cli import main


def test_score_command_creates_wer_figures_when_characteristics_are_supplied(
    tmp_path, capsys
) -> None:
    prediction_file = tmp_path / "predictions.jsonl"
    characteristics_file = tmp_path / "characteristics.jsonl"
    report_file = tmp_path / "wer.json"
    predictions = [
        {"sample_id": "male", "reference": "one two", "hypothesis": "one too"},
        {
            "sample_id": "female",
            "reference": "one two",
            "hypothesis": "one two",
        },
    ]
    characteristics = [
        {
            "sample_id": "male",
            "pitch": {"median_f0_hz": 100.0},
            "snr": {"median_active_frame_snr_db": 0.0},
        },
        {
            "sample_id": "female",
            "pitch": {"median_f0_hz": 200.0},
            "snr": {"median_active_frame_snr_db": 5.0},
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

    main(
        [
            "score",
            "--input",
            str(prediction_file),
            "--characteristics-file",
            str(characteristics_file),
            "--output",
            str(report_file),
        ]
    )
    capsys.readouterr()

    report = json.loads(report_file.read_text(encoding="utf-8"))
    figures = report["stratified_wer"]["figures"]
    snr_path = tmp_path / "wer-snr-wer.png"
    pitch_path = tmp_path / "wer-pitch-wer.png"
    assert figures == {
        "wer_by_median_frame_snr": str(snr_path),
        "wer_by_median_f0_pitch_range": str(pitch_path),
    }
    for figure_path in (snr_path, pitch_path):
        assert figure_path.is_file()
        assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert figure_path.stat().st_size > 10_000
