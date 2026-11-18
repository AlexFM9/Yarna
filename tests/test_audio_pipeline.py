"""Tests unitarios para el pipeline acústico del Bloque 1 (VAD, STT, TTS y WER)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from src.audio.stt import FasterWhisperSTT, resolve_whisper_model_name
from src.audio.tts import PiperTTSEngine
from src.audio.vad import CircularPCMBuffer, VoiceActivityDetector, normalize_pcm
from src.config import SAMPLE_RATE
from tests.eval_wer import (
    compute_wer_score,
    discover_l2_arctic_samples,
    evaluate_wer,
    format_markdown_table,
    normalize_transcript,
)


class TestPCMNormalizationAndBuffer(unittest.TestCase):
    """Pruebas de normalización PCM 16 kHz y buffer circular."""

    def test_normalize_int16_with_dc_offset_and_stereo(self) -> None:
        t = np.linspace(0.0, 0.2, int(0.2 * SAMPLE_RATE), endpoint=False)
        mono = 4000.0 * np.sin(2.0 * np.pi * 440.0 * t) + 1500.0
        stereo_int16 = np.column_stack([mono, mono]).astype(np.int16)

        norm = normalize_pcm(stereo_int16)
        self.assertEqual(norm.dtype, np.float32)
        self.assertEqual(norm.ndim, 1)
        self.assertEqual(norm.size, mono.size)
        self.assertAlmostEqual(float(np.mean(norm)), 0.0, places=3)
        self.assertLessEqual(float(np.max(np.abs(norm))), 1.0)
        self.assertGreater(float(np.max(np.abs(norm))), 0.95)

    def test_normalize_does_not_amplify_silence(self) -> None:
        silence = np.full(512, 1e-6, dtype=np.float32)
        norm = normalize_pcm(silence)
        self.assertLess(float(np.max(np.abs(norm))), 1e-4)

    def test_circular_buffer_fifo_overflow(self) -> None:
        buf = CircularPCMBuffer(max_seconds=0.1, sample_rate=100)  # máx 10 muestras
        buf.append(np.arange(6, dtype=np.float32) / 20.0)
        buf.append(np.arange(6, 14, dtype=np.float32) / 20.0)
        stored = buf.read_all()
        expected = np.arange(4, 14, dtype=np.float32) / 20.0
        np.testing.assert_allclose(stored, expected, atol=1e-6)

        buf.clear()
        self.assertEqual(buf.read_all().size, 0)


class TestVoiceActivityDetector(unittest.TestCase):
    """Pruebas del detector de actividad vocal en modo streaming y bloque."""

    @staticmethod
    def _tone(seconds: float, amp: float = 0.4) -> np.ndarray:
        t = np.linspace(0.0, seconds, int(round(seconds * SAMPLE_RATE)), endpoint=False)
        return (amp * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

    def test_streaming_vad_respects_level_silence_windows(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, use_silero=False)
        pre_silence = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
        speech = self._tone(0.5)
        short_pause = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
        long_pause = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)

        # A2 requiere 2.0 s de silencio -> 1.0 s aún no cierra turno
        self.assertIsNone(vad.process_chunk(pre_silence, level="A2"))
        self.assertIsNone(vad.process_chunk(speech, level="A2"))
        self.assertIsNone(vad.process_chunk(short_pause, level="A2"))

        utterance_a2 = vad.process_chunk(long_pause, level="A2")
        self.assertIsNotNone(utterance_a2)
        assert utterance_a2 is not None
        self.assertEqual(utterance_a2.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(utterance_a2))), 1.0)

        # B2/C1 requiere 0.9 s -> 1.0 s sí cierra turno directamente
        vad.reset()
        self.assertIsNone(vad.process_chunk(speech, level="B2/C1"))
        utterance_b2 = vad.process_chunk(short_pause, level="B2/C1")
        self.assertIsNotNone(utterance_b2)

    def test_extract_speech_segment_trims_silence(self) -> None:
        vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, use_silero=False)
        leading = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)
        speech = self._tone(0.6)
        trailing = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)
        full = np.concatenate([leading, speech, trailing])

        trimmed = vad.extract_speech_segment(full, level="A2")
        self.assertGreater(trimmed.size, int(0.5 * SAMPLE_RATE))
        self.assertLess(trimmed.size, int(1.5 * SAMPLE_RATE))
        self.assertEqual(vad.extract_speech_segment(leading, level="A2").size, 0)


class TestFasterWhisperSTT(unittest.TestCase):
    """Pruebas del wrapper faster-whisper y autodetección GPU/CPU."""

    def test_resolve_english_model_names(self) -> None:
        self.assertEqual(resolve_whisper_model_name("base", True), "base.en")
        self.assertEqual(resolve_whisper_model_name("small", True), "small.en")
        self.assertEqual(resolve_whisper_model_name("base.en", True), "base.en")
        self.assertEqual(resolve_whisper_model_name("base", False), "base")

    def test_cuda_autodetect_and_fallback_to_cpu(self) -> None:
        with patch("src.audio.stt.detect_cuda_available", return_value=True):
            stt = FasterWhisperSTT(model_size="base")
            self.assertEqual(stt.device, "cuda")
            self.assertEqual(stt.compute_type, "float16")
            self.assertEqual(stt.model_name, "base.en")

        with patch("src.audio.stt.detect_cuda_available", return_value=False):
            stt_cpu = FasterWhisperSTT(model_size="base")
            self.assertEqual(stt_cpu.device, "cpu")
            self.assertEqual(stt_cpu.compute_type, "int8")

    def test_transcribe_contract_and_resampling(self) -> None:
        class FakeWhisperModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def transcribe(self, audio: np.ndarray, **kwargs: object):
                self.calls.append({"len": audio.size, **kwargs})
                segs = [
                    SimpleNamespace(text=" Hello there ", start=0.1, end=0.9, avg_logprob=-0.2)
                ]
                return iter(segs), SimpleNamespace(language="en")

        fake_model = FakeWhisperModel()
        stt = FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8")
        stt._model = fake_model

        audio_44k = 0.3 * np.sin(
            2.0 * np.pi * 220.0 * np.linspace(0.0, 0.5, 22050, endpoint=False)
        ).astype(np.float32)

        res = stt.transcribe(audio_44k, sample_rate=44100, language="en")
        self.assertEqual(res["text"], "Hello there")
        self.assertEqual(res["inicio_ms"], 100)
        self.assertEqual(res["fin_ms"], 900)
        self.assertGreater(res["confidence"], 0.8)
        self.assertEqual(fake_model.calls[0]["len"], int(0.5 * SAMPLE_RATE))
        self.assertEqual(fake_model.calls[0]["beam_size"], 1)
        self.assertFalse(fake_model.calls[0]["vad_filter"])


class TestPiperTTS(unittest.TestCase):
    """Pruebas del motor Piper TTS, length_scale y re-muestreo a 16 kHz."""

    def test_compute_length_scale_for_cefr_levels(self) -> None:
        self.assertEqual(PiperTTSEngine.compute_length_scale(0.80), 1.25)
        self.assertEqual(PiperTTSEngine.compute_length_scale(1.05), 0.952)
        with self.assertRaises(ValueError):
            PiperTTSEngine.compute_length_scale(0.0)

    def test_resample_from_22050_to_16000(self) -> None:
        t = np.linspace(0.0, 1.0, 22050, endpoint=False)
        audio_22k = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        resampled = PiperTTSEngine._resample_to_target(audio_22k, orig_sr=22050, target_sr=16000)
        self.assertEqual(resampled.dtype, np.float32)
        self.assertAlmostEqual(resampled.size, 16000, delta=10)
        self.assertLessEqual(float(np.max(np.abs(resampled))), 1.0)

    def test_synthesize_returns_16khz_float32_and_scales_duration(self) -> None:
        tts = PiperTTSEngine(use_cuda=False, auto_download=False)
        tts.model_path = Path("non_existent_model.onnx")
        tts.config_path = Path("non_existent_model.onnx.json")

        sr_a2, audio_a2 = tts.synthesize("Could you repeat that please", rate=0.80)
        sr_b2, audio_b2 = tts.synthesize("Could you repeat that please", rate=1.05)

        self.assertEqual(sr_a2, SAMPLE_RATE)
        self.assertEqual(sr_b2, SAMPLE_RATE)
        self.assertEqual(audio_a2.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(audio_a2))), 1.0)
        self.assertGreater(audio_a2.size, audio_b2.size)


class TestEvalWER(unittest.TestCase):
    """Pruebas de normalización de texto, cálculo WER y lectura L2-ARCTIC."""

    def test_normalize_and_wer_score(self) -> None:
        self.assertEqual(normalize_transcript("I'm happy, aren't you?"), "i am happy are not you")
        perfect = compute_wer_score(["I'm ready."], ["I am ready"])
        self.assertAlmostEqual(perfect["wer"], 0.0, places=4)

        one_error = compute_wer_score(["one two three four"], ["one two five four"])
        self.assertAlmostEqual(one_error["wer"], 0.25, places=4)

    def test_evaluate_wer_l2_arctic_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wav_dir = root / "EBVS" / "wav"
            txt_dir = root / "EBVS" / "transcript"
            wav_dir.mkdir(parents=True)
            txt_dir.mkdir(parents=True)

            wav_path = wav_dir / "arctic_a0001.wav"
            txt_path = txt_dir / "arctic_a0001.txt"
            txt_path.write_text("Hello world", encoding="utf-8")

            pcm_int16 = (np.sin(2.0 * np.pi * 200.0 * np.linspace(0, 0.5, 8000)) * 10000).astype(
                np.int16
            )
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_int16.tobytes())

            samples = discover_l2_arctic_samples(root, speakers=["EBVS"], max_samples=5)
            self.assertEqual(len(samples), 1)

            mock_stt = SimpleNamespace(
                model_name="base.en",
                device="cpu",
                transcribe=lambda audio, sample_rate, language="en": {"text": "Hello world"},
            )
            metrics = evaluate_wer(root, speakers=["EBVS"], stt_engine=mock_stt)
            self.assertEqual(metrics["num_samples"], 1)
            self.assertAlmostEqual(metrics["wer"], 0.0, places=4)
            table = format_markdown_table([metrics])
            self.assertIn("`base.en`", table)


if __name__ == "__main__":
    unittest.main()
