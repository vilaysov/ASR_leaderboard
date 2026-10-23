# URGENT24 ASR evaluation

A small Python pipeline for transcribing the URGENT24 non-blind test split,
scoring saved predictions, and analyzing WER by audio characteristics. Inference
and scoring are separate: score files can be regenerated without downloading or
running an ASR model.

## Dataset and models

The pipeline uses the Hugging Face dataset
[`urgent-challenge/urgent2024_official`](https://huggingface.co/datasets/urgent-challenge/urgent2024_official),
config `test_nonblind`, split `validation`. The noisy audio is transcribed and
the dataset `transcript` is used as the reference. The loader resamples audio
to mono 16 kHz. The blind test split is not used because it has no reference
transcripts.

Supported model adapters:

- `openai/whisper-large-v3-turbo`
- `Qwen/Qwen3-ASR-1.7B-hf`
- `ibm-granite/granite-speech-5.0-470m-turboctc-nc`

Granite's non-commercial checkpoint has a non-commercial license; check each
dataset and model card before use.

## Install

Python 3.10 or newer is required. Install all model adapters and the test
dependency with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,qwen,granite]"
```

The models can be run separately; install only the relevant optional extra if
you do not need all three. PyTorch installation and hardware acceleration vary
by platform. On macOS, install FFmpeg (the command-line executable is used by
audio characterization; Hugging Face audio decoding may also require FFmpeg
libraries compatible with TorchCodec).

## Transcribe

The command defaults to the first 100 examples. Use `--all-samples` for the
full 1,000-example split or `--max-samples N` for a smaller run. Each model can
be run with or without the optional band-pass filter:

```bash
python -m asr_eval transcribe \
  --model openai/whisper-large-v3-turbo \
  --max-samples 100 \
  --output outputs/urgent24-whisper-unfiltered.jsonl

python -m asr_eval transcribe \
  --model Qwen/Qwen3-ASR-1.7B-hf \
  --all-samples \
  --output outputs/urgent24-qwen-unfiltered.jsonl

python -m asr_eval transcribe \
  --model ibm-granite/granite-speech-5.0-470m-turboctc-nc \
  --all-samples \
  --output outputs/urgent24-granite-unfiltered.jsonl
```

To evaluate the filtered condition, add `--bandpass-filter` and write to a
different output file. The filter is a fifth-order, zero-phase Butterworth
band-pass from 70 to 7,000 Hz. For example:

```bash
python -m asr_eval transcribe \
  --model Qwen/Qwen3-ASR-1.7B-hf \
  --all-samples \
  --bandpass-filter \
  --output outputs/urgent24-qwen-filtered.jsonl
```

Use the same sample limit and dataset split for filtered and unfiltered runs to
make a paired comparison. `--batch-size` controls inference batch size and
defaults to 1. Use a batch size supported by the available memory.

Each prediction row contains the sample ID, reference, hypothesis, dataset
config/split, model ID and resolved model revision, batch size, decoding
settings, filter settings, and Python/PyTorch/Transformers/Datasets versions.
Pin a checkpoint with `--model-revision COMMIT` to reproduce a particular
model revision. Otherwise the current Hugging Face Hub revision is loaded and
its resolved commit is saved in the output. Different hardware or library
versions can still cause small numerical differences.
For a complete run record, also save the repository commit (`git rev-parse
HEAD`) beside the prediction file. Older prediction files without a resolved
`model_revision` cannot guarantee which exact model checkpoint was used.

## Score saved predictions

Scoring is offline and does not load the dataset or model. The following
example applies the Open ASR Leaderboard English normalizer to references and
hypotheses, removes long exact adjacent repetitions from hypotheses, and
calculates standard and proper-name-weighted WER:

```bash
python -m asr_eval score \
  --input outputs/urgent24-whisper-unfiltered.jsonl \
  --normalize-reference \
  --normalize-hypothesis \
  --normalize-repetitions \
  --proper-names-file outputs/urgent24-test-nonblind-whisper-large-v3-reference-names.csv \
  --proper-name-weight 3 \
  --output outputs/urgent24-whisper-wer.json
```

The normalization flags operate independently. Repetition cleanup is
hypothesis-only and removes an exact adjacent segment when it repeats at least
six times; inspect the original hypothesis before interpreting such a cleanup.
WER uses `kaldialign` edit distance with compound matching enabled. The report
includes the selected WER, edit counts, error examples, and the raw-text score
when normalization or repetition cleanup is enabled.

`--proper-names-file` adds a separate experimental weighted WER; it does not
replace standard WER. The CSV must contain a `name` column and may contain
`sample_ids` separated by semicolons. Listed name tokens receive the selected
weight (3 by default).

## Audio characteristics and stratified WER

Generate per-sample F0 and SNR characteristics for the same URGENT24 split:

```bash
python -m asr_eval characterize-urgent24 \
  --output outputs/urgent24-audio-characteristics.jsonl
```

This writes a JSONL file and a dataset-level summary. F0 is estimated from
clean audio using 40 ms frames and a 10 ms hop. The `dataset_snr_db` field is
the SNR metadata supplied by URGENT24. The separate
`median_active_frame_snr_db` is calculated from clean/noisy audio pairs: it
estimates the noise residual as `noisy_audio - clean_audio`, calculates SNR on
active 20 ms frames with a 10 ms hop, then takes the median per clip. These two
SNR measures are not interchangeable.

Supply the characteristics file when scoring to add WER groups and English
figures for pitch range and **calculated median active-frame SNR**:

```bash
python -m asr_eval score \
  --input outputs/urgent24-whisper-unfiltered.jsonl \
  --characteristics-file outputs/urgent24-audio-characteristics.jsonl \
  --normalize-reference \
  --normalize-hypothesis \
  --output outputs/urgent24-whisper-stratified-wer.json
```

The score report contains corpus WER by 5 dB bins from -20 to 20 dB and the
two figures are saved beside it. These bins use the calculated
`median_active_frame_snr_db`, not the dataset-provided SNR. The scorer requires
unique sample IDs and a matching characteristics row for every prediction.
The 85–155 Hz and 165–255 Hz pitch groups are acoustic proxies only; they do
not identify speaker gender.

For a histogram of **only the dataset-provided SNR**, use
`outputs/urgent24-test-nonblind-dataset-snr-distribution.png`. The tracked
experiment comparison is in
[`outputs/wer-condition-comparison.md`](outputs/wer-condition-comparison.md).

## Tests

Run the unit tests with pytest:

```bash
pytest
```

The tests cover URGENT24 loading and audio characteristics, filtering,
inference adapter settings, scoring and report generation. They use small
synthetic inputs and mocks; they do not download the dataset or model weights.
