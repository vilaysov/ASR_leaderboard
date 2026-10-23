import json
import logging
import sys
from types import ModuleType, SimpleNamespace

from asr_eval.inference import (
    GRANITE_FRAME_BUCKET,
    GRANITE_TURBOCTC_MODEL_ID,
    GRANITE_TURBOCTC_REVISION,
    QWEN3_ASR_MAX_NEW_TOKENS,
    GraniteTurboCTCTranscriber,
    Qwen3ASRTranscriber,
    WhisperTranscriber,
    create_transcriber,
    write_predictions,
)


class FakeTranscriber:
    model_id = "test-model"
    revision = "test-revision"

    def transcribe(self, audio, sampling_rate):
        assert sampling_rate == 16000
        return "recognized words"

    def transcribe_batch(self, audios, sampling_rate):
        assert sampling_rate == 16000
        return ["recognized words {}".format(index) for index, _ in enumerate(audios)]


def test_write_predictions_logs_progress_and_completion(tmp_path, caplog) -> None:
    output_path = tmp_path / "nested" / "predictions.jsonl"
    examples = [
        {
            "sample_id": "sample-1",
            "audio": [0.0],
            "sampling_rate": 16000,
            "reference": "reference words",
        }
    ]

    with caplog.at_level(logging.INFO):
        count = write_predictions(
            examples,
            FakeTranscriber(),
            "test-dataset",
            "test",
            output_path,
        )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert count == 1
    assert records[0]["hypothesis"] == "recognized words"
    assert records[0]["model_revision"] == "test-revision"
    assert "Transcribing sample" not in caplog.text
    assert "Finished sample 1" in caplog.text


def test_write_predictions_batches_examples(tmp_path) -> None:
    output_path = tmp_path / "predictions.jsonl"
    examples = [
        {
            "sample_id": "sample-{}".format(index),
            "audio": [float(index)],
            "sampling_rate": 16000,
            "reference": "reference",
        }
        for index in range(3)
    ]

    count = write_predictions(
        examples,
        FakeTranscriber(),
        "dataset",
        "test",
        output_path,
        batch_size=2,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert count == 3
    assert [record["hypothesis"] for record in records] == [
        "recognized words 0",
        "recognized words 1",
        "recognized words",
    ]


def test_write_predictions_applies_optional_bandpass_filter(
    tmp_path, monkeypatch
) -> None:
    received = {}
    output_path = tmp_path / "filtered.jsonl"

    def fake_filter(audio, sample_rate):
        received["input"] = audio
        received["sample_rate"] = sample_rate
        return [0.25, -0.25]

    class CapturingTranscriber(FakeTranscriber):
        def transcribe(self, audio, sampling_rate):
            received["filtered_audio"] = audio
            return "recognized words"

    monkeypatch.setattr(
        "asr_eval.inference.bandpass_filter_audio",
        fake_filter,
    )
    example_audio = [0.1, -0.1]
    write_predictions(
        [
            {
                "sample_id": "filtered-sample",
                "audio": example_audio,
                "sampling_rate": 16000,
                "reference": "reference words",
            }
        ],
        CapturingTranscriber(),
        "test-dataset",
        "test",
        output_path,
        bandpass_filter=True,
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert received == {
        "input": example_audio,
        "sample_rate": 16000,
        "filtered_audio": [0.25, -0.25],
    }
    assert record["audio_filter"] == {
        "type": "butterworth_bandpass",
        "low_cutoff_hz": 70.0,
        "high_cutoff_hz": 7000.0,
        "order": 5,
        "phase": "zero_phase",
    }


def test_whisper_transcriber_uses_eager_attention_on_mps(monkeypatch) -> None:
    pipeline_calls = {}
    generation_config = SimpleNamespace()
    pipeline_model = SimpleNamespace(
        generation_config=generation_config,
        config=SimpleNamespace(_commit_hash="resolved-commit"),
    )
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True)
    )
    fake_torch.device = lambda name: name
    fake_transformers = ModuleType("transformers")

    def fake_pipeline(*args, **kwargs):
        pipeline_calls["args"] = args
        pipeline_calls["kwargs"] = kwargs
        return SimpleNamespace(model=pipeline_model)

    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from asr_eval.inference import WhisperTranscriber

    transcriber = WhisperTranscriber("test-model", revision="test-revision")

    assert pipeline_calls["kwargs"]["device"] == "mps"
    assert pipeline_calls["kwargs"]["model_kwargs"] == {
        "attn_implementation": "eager",
        "torch_dtype": "float16",
        "revision": "test-revision",
    }
    assert "chunk_length_s" not in pipeline_calls["kwargs"]
    assert generation_config.forced_decoder_ids is None
    assert generation_config.suppress_tokens == []
    assert generation_config.begin_suppress_tokens == []
    assert transcriber.revision == "test-revision"


def test_whisper_transcriber_uses_leaderboard_cuda_precision(monkeypatch) -> None:
    pipeline_calls = {}
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_transformers = ModuleType("transformers")

    def fake_pipeline(*args, **kwargs):
        pipeline_calls["kwargs"] = kwargs
        return SimpleNamespace(
            model=SimpleNamespace(
                generation_config=SimpleNamespace(),
                config=SimpleNamespace(_commit_hash="resolved-commit"),
            )
        )

    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from asr_eval.inference import WhisperTranscriber

    WhisperTranscriber("test-model", revision=None)

    assert pipeline_calls["kwargs"]["device"] == 0
    assert pipeline_calls["kwargs"]["model_kwargs"] == {
        "attn_implementation": "sdpa",
        "torch_dtype": "bfloat16",
    }


def test_transcribe_uses_leaderboard_greedy_english_settings(monkeypatch) -> None:
    pipeline_calls = {}
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_transformers = ModuleType("transformers")

    def fake_pipeline(*args, **kwargs):
        def transcribe(audios, batch_size, generate_kwargs):
            pipeline_calls["audio"] = audios
            pipeline_calls["batch_size"] = batch_size
            pipeline_calls["generate_kwargs"] = generate_kwargs
            return [{"text": "transcript"} for _ in audios]

        transcribe.model = SimpleNamespace(
            generation_config=SimpleNamespace()
        )
        return transcribe

    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    transcriber = WhisperTranscriber("test-model", revision=None)
    text = transcriber.transcribe([0.0], 16000)

    assert text == "transcript"
    assert pipeline_calls["generate_kwargs"] == {
        "task": "transcribe",
        "language": "en",
        "num_beams": 1,
        "do_sample": False,
    }
    assert pipeline_calls["batch_size"] == 1


def test_qwen3_asr_uses_leaderboard_inference_settings(monkeypatch) -> None:
    calls = {}
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: name

    class FakeInputs(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self["input_ids"] = FakeTensor(shape=(2, 2))

        def to(self, device, dtype):
            calls["inputs_to"] = (device, dtype)
            return self

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            calls["generated_slice"] = item
            return "generated"

    class FakeModel:
        config = SimpleNamespace(_commit_hash="resolved-qwen-commit")

        def to(self, device):
            calls["model_to"] = device
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return FakeTensor(shape=(2, 4))

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_load"] = (model_id, kwargs)
            return FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["processor_load"] = (model_id, kwargs)
            return FakeProcessor()

        def apply_transcription_request(self, **kwargs):
            calls["request"] = kwargs
            return FakeInputs(input_ids=[[1, 2]])

        def decode(self, generated, return_format):
            calls["decode"] = (generated, return_format)
            return ["transcript", "transcript"]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForMultimodalLM = FakeAutoModel
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    transcriber = Qwen3ASRTranscriber(revision=None)
    hypotheses = transcriber.transcribe_batch([[0.0], [0.1]], 16000)

    assert hypotheses == ["transcript", "transcript"]
    assert calls["model_load"] == (
        "Qwen/Qwen3-ASR-1.7B-hf",
        {"dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    assert calls["request"] == {
        "audio": [[0.0], [0.1]],
        "language": "en",
    }
    assert calls["generate"]["max_new_tokens"] == QWEN3_ASR_MAX_NEW_TOKENS
    assert calls["decode"][1] == "transcription_only"
    assert transcriber.revision == "resolved-qwen-commit"


def test_create_transcriber_selects_qwen_by_model_id(monkeypatch) -> None:
    from asr_eval import inference

    class FakeQwenTranscriber:
        def __init__(self, model_id, revision):
            self.model_id = model_id
            self.revision = revision

    monkeypatch.setattr(inference, "Qwen3ASRTranscriber", FakeQwenTranscriber)
    transcriber = create_transcriber("Qwen/Qwen3-ASR-1.7B-hf", revision="rev")

    assert transcriber.model_id == "Qwen/Qwen3-ASR-1.7B-hf"
    assert transcriber.revision == "rev"


def test_granite_turboctc_uses_leaderboard_inference_settings(monkeypatch) -> None:
    from contextlib import nullcontext

    calls = {}

    class FakeTensor:
        def __init__(self, shape, dtype="float32", device="cpu"):
            self.shape = shape
            self.dtype = dtype
            self.device = device

        def to(self, dtype):
            calls["features_dtype"] = dtype
            self.dtype = dtype
            return self

    class FakeModel:
        config = SimpleNamespace(_commit_hash="resolved-granite-commit")

        def eval(self):
            calls["eval"] = True
            return self

        def to(self, device):
            calls.setdefault("model_to", []).append(device)
            return self

        def transcribe(self, **kwargs):
            calls["transcribe"] = kwargs
            return SimpleNamespace(preds=["decoded-ids"])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model_load"] = (model_id, kwargs)
            return FakeModel()

    class FakeProcessor:
        hop_length = 2
        stack_factor = 2

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["processor_load"] = (model_id, kwargs)
            return FakeProcessor()

        def prepare_audio(self, audios, device, return_lengths):
            calls["prepare_audio"] = (audios, device, return_lengths)
            return FakeTensor((len(audios), 130)), [130] * len(audios)

        def __call__(self, waveforms):
            calls["waveform_shape"] = waveforms.shape
            return {"input_features": FakeTensor((2, 64, 8))}

        def _build_mask(self, lengths, frame_count, device):
            calls["mask"] = (lengths, frame_count, device)
            return "mask"

        def batch_decode(self, predictions):
            calls["decode"] = predictions
            return ["first transcript", "second transcript"]

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: SimpleNamespace(type=name, name=name)
    fake_torch.inference_mode = nullcontext

    def fake_pad(tensor, padding):
        calls["padding"] = padding
        return FakeTensor((tensor.shape[0], tensor.shape[1] + padding[1]))

    fake_torch.nn = SimpleNamespace(
        functional=SimpleNamespace(pad=fake_pad)
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    transcriber = GraniteTurboCTCTranscriber()
    hypotheses = transcriber.transcribe_batch([[0.0], [0.1]], 16000)

    assert hypotheses == ["first transcript", "second transcript"]
    assert calls["model_load"] == (
        GRANITE_TURBOCTC_MODEL_ID,
        {
            "trust_remote_code": True,
            "revision": GRANITE_TURBOCTC_REVISION,
        },
    )
    assert calls["processor_load"] == (
        GRANITE_TURBOCTC_MODEL_ID,
        {
            "trust_remote_code": True,
            "revision": GRANITE_TURBOCTC_REVISION,
        },
    )
    assert calls["padding"] == (0, 126)
    assert calls["mask"] == ([130, 130], 64, "cpu")
    assert calls["transcribe"]["attention_mask"] == "mask"
    assert calls["decode"] == ["decoded-ids"]
    assert transcriber.revision == GRANITE_TURBOCTC_REVISION


def test_granite_turboctc_uses_cuda_bfloat16_and_compiles_layers(monkeypatch) -> None:
    from asr_eval import inference

    calls = {}

    class FakeLayer:
        def compile(self, **kwargs):
            calls["layer_compile"] = kwargs

    class FakeModel:
        config = SimpleNamespace()
        encoder = SimpleNamespace(layers=[FakeLayer(), FakeLayer()])

        def eval(self):
            return self

        def to(self, value):
            calls.setdefault("model_to", []).append(value)
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeProcessor()

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.backends = SimpleNamespace()
    fake_torch.device = lambda name: SimpleNamespace(type="cuda", name=name)
    fake_torch.set_float32_matmul_precision = lambda precision: calls.setdefault(
        "matmul_precision", precision
    )
    fake_torch._dynamo = SimpleNamespace(
        config=SimpleNamespace(
            cache_size_limit=None,
            accumulated_cache_size_limit=None,
        )
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    GraniteTurboCTCTranscriber()

    assert calls["model_to"] == [
        SimpleNamespace(type="cuda", name="cuda:0"),
        "bfloat16",
    ]
    assert calls["matmul_precision"] == "high"
    assert calls["layer_compile"] == {"mode": "default", "dynamic": False}
    assert fake_torch._dynamo.config.cache_size_limit == 512
    assert fake_torch._dynamo.config.accumulated_cache_size_limit == 2048


def test_granite_turboctc_rejects_non_16khz_audio(monkeypatch) -> None:
    from contextlib import nullcontext

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: SimpleNamespace(type=name, name=name)
    fake_torch.inference_mode = nullcontext

    class FakeModel:
        config = SimpleNamespace()

        def eval(self):
            return self

        def to(self, _value):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeProcessor()

    fake_torch.nn = SimpleNamespace()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeAutoModel
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    transcriber = GraniteTurboCTCTranscriber(revision=None)

    import pytest

    with pytest.raises(ValueError, match="expects 16 kHz audio"):
        transcriber.transcribe_batch([[0.0]], 8000)


def test_create_transcriber_selects_granite_by_model_id(monkeypatch) -> None:
    from asr_eval import inference

    class FakeGraniteTranscriber:
        def __init__(self, model_id, revision):
            self.model_id = model_id
            self.revision = revision

    monkeypatch.setattr(inference, "GraniteTurboCTCTranscriber", FakeGraniteTranscriber)
    transcriber = create_transcriber(GRANITE_TURBOCTC_MODEL_ID)

    assert transcriber.model_id == GRANITE_TURBOCTC_MODEL_ID
    assert transcriber.revision == GRANITE_TURBOCTC_REVISION
