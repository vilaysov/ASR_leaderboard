"""Whisper model adapter and prediction-file writer."""

import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from asr_eval.audio_filter import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    bandpass_filter_audio,
)
from asr_eval import __version__
from asr_eval.data import (
    DEFAULT_SAMPLE_RATE,
    URGENT24_CONFIG,
    URGENT24_DATASET_ID,
    URGENT24_SPLIT,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
QWEN3_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
GRANITE_TURBOCTC_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc-nc"
GRANITE_TURBOCTC_REVISION = "05b33f57fe08aae7a0365c7e096a42658f8a28ac"
DEFAULT_MODEL_REVISION: Optional[str] = None
QWEN3_ASR_MAX_NEW_TOKENS = 512
GRANITE_FRAME_BUCKET = 32


class WhisperTranscriber:
    """Run Hugging Face Whisper ASR inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ) -> None:
        import torch
        from transformers import pipeline

        logger.info("Loading ASR model %s at revision %s", model_id, revision)
        model_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available():
            device: Any = 0
            model_kwargs["attn_implementation"] = "sdpa"
            model_kwargs["torch_dtype"] = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            device = -1
            model_kwargs["torch_dtype"] = torch.float32

        if revision is not None:
            model_kwargs["revision"] = revision

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            model_kwargs=model_kwargs,
        )
        generation_config = self._pipeline.model.generation_config
        generation_config.forced_decoder_ids = None
        generation_config.suppress_tokens = []
        generation_config.begin_suppress_tokens = []
        self.model_id = model_id
        model_config = getattr(self._pipeline.model, "config", None)
        self.revision = revision or getattr(model_config, "_commit_hash", None)
        self.inference_settings = {
            "task": "transcribe",
            "language": "en",
            "num_beams": 1,
            "do_sample": False,
            "device": str(device),
            "dtype": str(model_kwargs["torch_dtype"]),
            "attention": model_kwargs.get("attn_implementation", "default"),
        }
        logger.info(
            "Loaded ASR model %s at revision %s on %s",
            model_id,
            self.revision or "default Hub revision",
            device,
        )

    def transcribe(self, audio: Any, sampling_rate: int) -> str:
        return self.transcribe_batch([audio], sampling_rate)[0]

    def transcribe_batch(self, audios: Iterable[Any], sampling_rate: int) -> list[str]:
        audio_batch = list(audios)
        results = self._pipeline(
            [
                {"raw": audio, "sampling_rate": sampling_rate}
                for audio in audio_batch
            ],
            batch_size=len(audio_batch),
            generate_kwargs={
                "task": "transcribe",
                "language": "en",
                "num_beams": 1,
                "do_sample": False,
            },
        )
        return [str(result["text"]) for result in results]


class Qwen3ASRTranscriber:
    """Run Qwen3-ASR with the Hugging Face Transformers native API."""

    def __init__(
        self,
        model_id: str = QWEN3_ASR_MODEL_ID,
        revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ) -> None:
        import torch

        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as error:
            raise ImportError(
                "Qwen3-ASR requires transformers>=5.13.0; install it with "
                "`python -m pip install -e '.[qwen]'`."
            ) from error

        if torch.cuda.is_available():
            self._device: Any = torch.device("cuda:0")
            self._dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._dtype = torch.float16
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        logger.info(
            "Loading Qwen3-ASR model %s at revision %s",
            model_id,
            revision or "current Hub revision",
        )
        model_kwargs: Dict[str, Any] = {
            "dtype": self._dtype,
            "attn_implementation": "sdpa",
        }
        if revision is not None:
            model_kwargs["revision"] = revision

        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_id, **model_kwargs
        )
        self._model.to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            model_id, **({"revision": revision} if revision is not None else {})
        )
        self.model_id = model_id
        model_config = getattr(self._model, "config", None)
        self.revision = revision or getattr(model_config, "_commit_hash", None)
        self.inference_settings = {
            "task": "transcribe",
            "language": "en",
            "max_new_tokens": QWEN3_ASR_MAX_NEW_TOKENS,
            "do_sample": False,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "attention": "sdpa",
        }
        logger.info(
            "Loaded Qwen3-ASR model %s at revision %s on %s with %s",
            model_id,
            self.revision or "current Hub revision",
            self._device,
            self._dtype,
        )

    def transcribe(self, audio: Any, sampling_rate: int) -> str:
        return self.transcribe_batch([audio], sampling_rate)[0]

    def transcribe_batch(self, audios: Iterable[Any], sampling_rate: int) -> list[str]:
        audio_batch = list(audios)
        inputs = self._processor.apply_transcription_request(
            audio=audio_batch,
            language="en",
        ).to(self._device, self._dtype)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=QWEN3_ASR_MAX_NEW_TOKENS,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.decode(
            generated_ids,
            return_format="transcription_only",
        )
        return [str(text) for text in decoded]


class GraniteTurboCTCTranscriber:
    """Run Granite Speech 5 TurboCTC with its leaderboard CTC inference path."""

    def __init__(
        self,
        model_id: str = GRANITE_TURBOCTC_MODEL_ID,
        revision: Optional[str] = GRANITE_TURBOCTC_REVISION,
    ) -> None:
        import torch

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise ImportError(
                "Granite Speech 5 TurboCTC requires transformers>=5.16.0; "
                "install it with `python -m pip install -e '.[granite]'`."
            ) from error

        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
            self._dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._dtype = torch.float32
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        logger.info(
            "Loading Granite TurboCTC model %s at revision %s",
            model_id,
            revision or "current Hub revision",
        )
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if revision is not None:
            model_kwargs["revision"] = revision
        self._model = AutoModel.from_pretrained(model_id, **model_kwargs)
        self._model.eval().to(self._device).to(self._dtype)
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            **({"revision": revision} if revision is not None else {}),
        )

        if self._device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 2048
            layers = getattr(getattr(self._model, "encoder", None), "layers", None)
            if layers is not None:
                for layer in layers:
                    layer.compile(mode="default", dynamic=False)
            elif getattr(self._model, "encoder", None) is not None:
                self._model.encoder = torch.compile(
                    self._model.encoder,
                    mode="reduce-overhead",
                    dynamic=False,
                )

        self.model_id = model_id
        model_config = getattr(self._model, "config", None)
        self.revision = revision or getattr(model_config, "_commit_hash", None)
        self.inference_settings = {
            "decoder": "greedy_ctc",
            "sampling_rate_hz": DEFAULT_SAMPLE_RATE,
            "frame_bucket": GRANITE_FRAME_BUCKET,
            "device": str(self._device),
            "dtype": str(self._dtype),
        }
        logger.info(
            "Loaded Granite TurboCTC model %s at revision %s on %s with %s",
            model_id,
            self.revision or "current Hub revision",
            self._device,
            self._dtype,
        )

    def transcribe(self, audio: Any, sampling_rate: int) -> str:
        return self.transcribe_batch([audio], sampling_rate)[0]

    def transcribe_batch(self, audios: Iterable[Any], sampling_rate: int) -> list[str]:
        import torch

        if sampling_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(
                "Granite TurboCTC expects 16 kHz audio; got {} Hz".format(
                    sampling_rate
                )
            )

        audio_batch = list(audios)
        if not audio_batch:
            return []

        with torch.inference_mode():
            waveforms, lengths = self._processor.prepare_audio(
                audio_batch,
                device=self._device,
                return_lengths=True,
            )
            processor = self._processor
            mel_frames = waveforms.shape[1] // processor.hop_length
            stacked_frames = -(-mel_frames // processor.stack_factor)
            bucketed_frames = GRANITE_FRAME_BUCKET * -(
                -stacked_frames // GRANITE_FRAME_BUCKET
            )
            padded_samples = (
                bucketed_frames * processor.stack_factor * processor.hop_length
            )
            if waveforms.shape[1] < padded_samples:
                waveforms = torch.nn.functional.pad(
                    waveforms, (0, padded_samples - waveforms.shape[1])
                )
            features = processor(waveforms)["input_features"]
            attention_mask = processor._build_mask(
                lengths, features.shape[1], features.device
            )
            if features.dtype != self._dtype:
                features = features.to(self._dtype)
            output = self._model.transcribe(
                input_features=features,
                attention_mask=attention_mask,
            )
            decoded = processor.batch_decode(output.preds)
        return [str(text) for text in decoded]


def create_transcriber(
    model_id: str = DEFAULT_MODEL_ID,
    revision: Optional[str] = DEFAULT_MODEL_REVISION,
) -> Any:
    """Select the adapter for a supported model family."""
    if "granite-speech-5.0-470m-turboctc" in model_id.lower():
        return GraniteTurboCTCTranscriber(
            model_id,
            revision=revision or GRANITE_TURBOCTC_REVISION,
        )
    if "qwen3-asr" in model_id.lower():
        return Qwen3ASRTranscriber(model_id, revision=revision)
    if "whisper" in model_id.lower():
        return WhisperTranscriber(model_id, revision=revision)
    raise ValueError(
        "Unsupported model {!r}; choose Whisper, Qwen3-ASR, or Granite.".format(
            model_id
        )
    )


def _runtime_metadata() -> Dict[str, str]:
    """Capture package versions that affect inference and dataset decoding."""
    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "asr_eval": __version__,
    }


def write_predictions(
    examples: Iterable[Dict[str, Any]],
    transcriber: Any,
    output_path: Path,
    model_revision: Optional[str] = None,
    batch_size: int = 1,
    bandpass_filter: bool = False,
) -> int:
    """Transcribe examples in bounded batches and persist one JSON object per line."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    runtime = _runtime_metadata()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def write_batch(
        batch: list[Dict[str, Any]], output_file: Any, completed: int
    ) -> int:
        audios = [
            bandpass_filter_audio(example["audio"], example["sampling_rate"])
            if bandpass_filter
            else example["audio"]
            for example in batch
        ]
        batch_transcriber = getattr(transcriber, "transcribe_batch", None)
        if len(batch) == 1 or not callable(batch_transcriber):
            hypotheses = [
                transcriber.transcribe(audio, example["sampling_rate"])
                for example, audio in zip(batch, audios)
            ]
        else:
            hypotheses = batch_transcriber(
                audios,
                batch[0]["sampling_rate"],
            )
        if len(hypotheses) != len(batch):
            raise ValueError("Transcriber returned a different number of hypotheses")

        for example, hypothesis in zip(batch, hypotheses):
            record = {
                "sample_id": example["sample_id"],
                "reference": example["reference"],
                "hypothesis": hypothesis,
                "dataset": URGENT24_DATASET_ID,
                "dataset_config": URGENT24_CONFIG,
                "split": URGENT24_SPLIT,
                "model": transcriber.model_id,
                "model_revision": model_revision or transcriber.revision,
                "batch_size": batch_size,
                "inference_settings": transcriber.inference_settings,
                "runtime": runtime,
                "audio_filter": None,
            }
            if bandpass_filter:
                record["audio_filter"] = {
                    "type": "butterworth_bandpass",
                    "low_cutoff_hz": BANDPASS_LOW_HZ,
                    "high_cutoff_hz": BANDPASS_HIGH_HZ,
                    "order": BANDPASS_ORDER,
                    "phase": "zero_phase",
                }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            completed += 1
            logger.info("Finished sample %d", completed)
        return completed

    with output_path.open("w", encoding="utf-8") as output_file:
        batch: list[Dict[str, Any]] = []
        for example in examples:
            if batch and example["sampling_rate"] != batch[0]["sampling_rate"]:
                count = write_batch(batch, output_file, count)
                batch = []
            batch.append(example)
            if len(batch) == batch_size:
                count = write_batch(batch, output_file, count)
                batch = []
        if batch:
            count = write_batch(batch, output_file, count)

    if count == 0:
        raise ValueError("The selected dataset split yielded no examples")
    return count
