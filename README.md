# ASR evaluation

A small pipeline for running Hugging Face ASR models against Hugging Face
speech datasets and scoring saved transcripts independently.

## Setup

Use Python 3.10 or newer and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

An existing virtual environment created with Python 3.9 cannot be upgraded in
place. Create a new environment with an installed Python 3.10+ interpreter,
then reinstall the project dependencies in it.

PyTorch installation can depend on your platform and accelerator. If the
default install does not provide the right build, install the appropriate
PyTorch build for your system before installing this project.

GigaSpeech access may require accepting the dataset's terms on Hugging Face
and authenticating with `hf auth login`.

On macOS, install the FFmpeg 7 shared libraries used by TorchCodec and expose
them to the Python process before running inference:

```bash
brew install ffmpeg@7
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

To make this setting persist in zsh, add the `export` line above to `~/.zshrc`.
Apply the change in the current terminal with:

```bash
source ~/.zshrc
```

The default inference model is `openai/whisper-large-v3-turbo`. By default,
the model is loaded from the current Hugging Face Hub revision, matching the
Open ASR Leaderboard script, which does not pin a model revision. The resolved
Hub commit is recorded in the prediction JSONL. Its script also does not pin a
Transformers package version, so exact library-version matching is not
specified. The adapter
matches that runner's English transcription, greedy decoding, standard
single-window Whisper processing, cleared legacy suppression settings, SDPA
attention, and bfloat16 on CUDA where supported. On Apple Silicon it uses
eager attention and FP16 on MPS because SDPA was returning incomplete
transcriptions in local testing; CPU uses FP32. This is not identical to the
leaderboard's full execution setup: this project processes samples one at a
time rather than in batches of 64, and cannot use its CUDA-only timing and
benchmark path on MPS/CPU. The separate `TheStageAI/thewhisper-large-v3-turbo`
XL TensorRT runner is a different model/runtime and is not used here.

Override the model with `--model`; optionally pin a specific checkpoint using
`--model-revision`. The resolved model revision is recorded in prediction JSONL.

## Run a small GigaSpeech evaluation

Transcribe at most 100 examples from the `xs` GigaSpeech test split:

```bash
python -m asr_eval transcribe \
  --dataset speechcolab/gigaspeech \
  --dataset-config xs \
  --split test \
  --model openai/whisper-large-v3-turbo \
  --max-samples 100 \
  --output outputs/gigaspeech-test-whisper-large-v3.jsonl
```

## Evaluate URGENT24

The official URGENT24 dataset is a speech-enhancement benchmark. Its
`test_nonblind` config has 1,000 examples with transcripts and can be used for
ASR WER: the noisy audio is the model input and `transcript` is the reference.
Use `--all-samples` to stream the entire split rather than the default 100-row
limit:

```bash
python -m asr_eval transcribe \
  --dataset urgent-challenge/urgent2024_official \
  --dataset-config test_nonblind \
  --split validation \
  --audio-column noisy_audio \
  --text-column transcript \
  --id-column id \
  --model openai/whisper-large-v3-turbo \
  --bandpass-filter \
  --all-samples \
  --output outputs/urgent24-test-nonblind-whisper-large-v3.jsonl
```

The optional `--bandpass-filter` flag applies a fifth-order, zero-phase
Butterworth band-pass filter from 70 Hz to 7,000 Hz to each audio sample before
model inference. Its configuration is recorded in each prediction record.
Filtering requires a sample rate above 14 kHz; dataset audio is decoded at
16 kHz by default. Omit the flag to use the original audio unchanged.

Run the same URGENT24 split with Qwen3-ASR:

```bash
python -m pip install -e ".[qwen]"
python -m asr_eval transcribe \
  --dataset urgent-challenge/urgent2024_official \
  --dataset-config test_nonblind \
  --split validation \
  --audio-column noisy_audio \
  --text-column transcript \
  --id-column id \
  --model Qwen/Qwen3-ASR-1.7B-hf \
  --batch-size 1 \
  --all-samples \
  --output outputs/urgent24-test-nonblind-qwen3-asr-1.7b.jsonl
```

The Qwen adapter matches the Qwen leaderboard job's English prompt, 512-token
generation limit, SDPA attention, and bfloat16 on CUDA. The leaderboard job
uses batch size 256 on an H200; choose `--batch-size 256` only on hardware with
enough memory. This project defaults to batch size 1 for local Mac evaluation;
the model and generation settings are otherwise selected independently of the
Whisper adapter.

Run the Granite Speech 5 TurboCTC model with the leaderboard's pinned revision
and batch size:

```bash
python -m pip install -e ".[granite]"
python -m asr_eval transcribe \
  --dataset speechcolab/gigaspeech \
  --dataset-config xs \
  --split test \
  --model ibm-granite/granite-speech-5.0-470m-turboctc-nc \
  --batch-size 128 \
  --max-samples 100 \
  --output outputs/gigaspeech-test-granite-turboctc.jsonl
```

The Granite adapter uses the leaderboard's pinned model revision,
`trust_remote_code` CTC implementation, greedy decode, 32-frame bucketing, and
BF16 with conformer-layer compilation on CUDA. Its leaderboard batch size is
128; reduce `--batch-size` on local hardware if memory is limited. On MPS/CPU,
it uses FP32 and skips CUDA-only compilation. Install Transformers 5.16.0 or
newer with the `granite` extra, which installs the required `torchaudio`
dependency. Keep `torch` and `torchaudio` on compatible versions. The `-nc`
checkpoint is licensed CC-BY-NC-SA-4.0 and is restricted to non-commercial
use; review the model card before use.

The dataset's config is named `test_nonblind`, while its split is named
`validation`. Avoid the `test_blind` config for WER because it has no transcript
column; the `validation` config contains `<not-available>` references. The
loader rejects missing transcript labels rather than generating a misleading
WER. URGENT24 is licensed CC-BY-NC-SA-4.0; review its terms before using the
data or results.

### Characterize URGENT24 pitch and SNR

The analysis command uses the same `test_nonblind` / `validation` split. It
estimates fundamental frequency (F0) from the paired clean audio in overlapping
40 ms frames with a 10 ms hop, and saves the full frame-by-frame pitch track.
It also estimates median active-frame SNR from the clean signal and the
`noisy_audio - clean_audio` residual; the dataset-provided `snr_dB` is retained
alongside the estimate for comparison. This command decodes the source audio
through the `ffmpeg` executable on `PATH`, so it does not require TorchCodec's
shared-library decoding path.

```bash
python -m asr_eval characterize-urgent24 \
  --output outputs/urgent24-test-nonblind-audio-characteristics.jsonl
```

The command processes the entire 1,000-example split by default. Use
`--max-samples 20` for a small test. Per-example metrics are written as JSONL;
a dataset summary is written to the same path with `.summary.json` as its
suffix, or to a custom path supplied with `--summary-output`. Each JSONL record
includes the F0 values, frame times, autocorrelation confidence, median and
10th/90th percentile F0, voiced-frame fraction, dataset SNR, and calculated
median frame SNR. Unvoiced frames have a null F0 value.

The pitch-range description is an acoustic F0 heuristic only. Pitch overlaps
substantially across people and cannot determine a speaker's gender; the output
does not label speakers as women or men. URGENT24's sources include read and
spontaneous speech, so pitch measurements describe each clip, not a stable
speaker identity.

Then score the saved predictions without loading the model or dataset:

```bash
python -m asr_eval score \
  --input outputs/urgent24-test-nonblind-whisper-large-v3.jsonl \
  --characteristics-file outputs/urgent24-test-nonblind-audio-characteristics.jsonl \
  --proper-names-file outputs/urgent24-test-nonblind-whisper-large-v3-reference-names.csv \
  --proper-name-weight 3 \
  --normalize-reference \
  --normalize-hypothesis \
  --normalize-repetitions \
  --output outputs/urgent24-test-nonblind-whisper-large-v3-wer-by-characteristics.json
```

The score flags are independent. Use either flag alone to normalize only the
reference or hypothesis; omit both to score the original text. When either is
enabled, the report includes the selected WER, the raw `raw_score` baseline,
normalization settings, and `wer_change` (normalized WER minus raw WER).
WER uses the leaderboard's Kaldi edit-distance implementation with compound
matching enabled, so a split/join difference such as `white paper` versus
`whitepaper` is not counted as an error.
The score report identifies this as
`kaldialign.edit_distance with merge_compounds=True`.
The optional `--normalize-repetitions` flag is a separate hypothesis-only
cleanup: it keeps one copy of an exact adjacent segment and removes duplicates
when the segment occurs at least 6 times consecutively. Segments may contain
up to 8 words, matching the diagnostic experiment. It does not change saved
prediction JSONL and should be reported as a separate normalized score because
it can remove genuine repeated speech as well as transcription loops.
To add a proper-name-weighted WER, pass the names CSV with
`--proper-names-file`. Listed name tokens receive weight 3 by default; adjust
that multiplier with `--proper-name-weight`. The metric uses weighted edit
distance divided by the sum of weighted reference-token counts, so it may
exceed 100% when weighted insertions are large. When a `sample_ids` column is
present in the CSV, names are matched only in those samples. Matching ignores
case and punctuation, while scoring uses the same normalized text as the
ordinary WER. The report adds a separate `proper_name_weighted_wer` section;
the regular WER remains unchanged.
When `--characteristics-file` is supplied, the report adds a separate
`stratified_wer` section. It reports corpus WER and edit counts for median F0
ranges 85–155 Hz and 165–255 Hz, and for median active-frame SNR bins from
−20 dB to 20 dB in 5 dB intervals. The final SNR interval includes 20 dB;
missing and out-of-range SNR values are listed separately. Pitch ranges are
acoustic proxies only, not verified speaker gender. Every prediction must have
a matching unique `sample_id` in the characteristics file. Stratified WER
uses the same normalization options as the overall WER.
Scoring with `--characteristics-file` also saves two English PNG figures next
to the score report: `*-snr-wer.png` and `*-pitch-wer.png`. The
pitch figure labels the selected F0 bands as proxies and includes clips outside
those bands or without a usable pitch measurement.
Every score report also includes `error_analysis`, with up to the 10 most
frequent substitution pairs, deleted reference words, and inserted hypothesis
words and their counts. It reflects whichever normalization settings were
selected for that score.
Normalization uses the
[EnglishTextNormalizer and mappings](https://github.com/huggingface/open_asr_leaderboard/tree/7e8d1a1003c201324e9003dd5bc34244340a63b0/normalizer)
from the Hugging Face Open ASR Leaderboard, pinned at commit
`7e8d1a1003c201324e9003dd5bc34244340a63b0`.

The JSONL file contains one record per utterance with `sample_id`, `reference`,
`hypothesis`, dataset/config/split, and model metadata. It can be retained and
scored again independently.

## Extending the pipeline

- `asr_eval.data` loads a streaming dataset and maps its audio/text/id columns
  into a common example shape. Change the dataset id, config, split, or column
  names on the `transcribe` command for other compatible datasets.
- `asr_eval.inference` contains the model adapter. A new backend can implement
  the same `transcribe(audio, sampling_rate)` interface.
- `asr_eval.scoring` scores saved references and hypotheses. The initial
  scoring path preserves case and punctuation; it only splits on whitespace
  to identify words. Put future label normalization in this module so it can
  be reused when scoring existing predictions.

The inference command streams datasets. It defaults to at most 100 samples;
use `--all-samples` to process every row or `--max-samples N` to choose a
different positive limit. During inference, the terminal logs when each sample
finishes and confirms when the prediction file is complete. Each prediction is
flushed to disk as it is written. On macOS, if the completion message appears
but Python does not return to the shell prompt, the predictions have been
saved; press Ctrl+C to interrupt the stuck process cleanup.
