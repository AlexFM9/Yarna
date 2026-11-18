"""Wrapper de faster-whisper (CTranslate2) para transcripción STT con timestamps."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from src.audio.vad import normalize_pcm
from src.config import SAMPLE_RATE, Settings

logger = logging.getLogger(__name__)

_ENGLISH_ONLY_SIZES: frozenset[str] = frozenset({"tiny", "base", "small", "medium"})


def resolve_whisper_model_name(model_size: str, prefer_english_model: bool = True) -> str:
    """Añade el sufijo `.en` a modelos base en inglés cuando corresponde."""
    cleaned = model_size.strip()
    if prefer_english_model and cleaned in _ENGLISH_ONLY_SIZES:
        return f"{cleaned}.en"
    return cleaned


def detect_cuda_available() -> bool:
    """Comprueba si hay GPU NVIDIA utilizable por CTranslate2 o PyTorch."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FasterWhisperSTT:
    """Motor STT local cuantizado (FP16 en GPU NVIDIA con fallback a INT8 en CPU)."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        prefer_english_model: bool = True,
        download_root: str | None = None,
    ) -> None:
        self._explicit_settings = settings is not None
        self.settings = settings or Settings()

        raw_size = model_size or self.settings.whisper_model_size
        self.model_name = resolve_whisper_model_name(
            raw_size, prefer_english_model=prefer_english_model
        )
        self.download_root = download_root or os.getenv("WHISPER_MODEL_DIR", "models/whisper")
        self.device, self.compute_type = self._resolve_device_and_compute_type(
            device=device,
            compute_type=compute_type,
        )
        self._model: Any = None

    def _resolve_device_and_compute_type(
        self,
        device: str | None,
        compute_type: str | None,
    ) -> tuple[str, str]:
        env_device = os.getenv("WHISPER_DEVICE")
        env_compute = os.getenv("WHISPER_COMPUTE_TYPE")

        if device is not None:
            chosen_device = device
        elif self._explicit_settings:
            chosen_device = self.settings.whisper_device
        elif env_device and env_device.lower() != "auto":
            chosen_device = env_device
        else:
            chosen_device = "auto"

        if chosen_device.lower() == "auto":
            chosen_device = "cuda" if detect_cuda_available() else "cpu"

        if compute_type is not None:
            chosen_compute = compute_type
        elif self._explicit_settings:
            chosen_compute = self.settings.whisper_compute_type
        elif env_compute and chosen_device == env_device:
            chosen_compute = env_compute
        elif chosen_device == "cuda":
            chosen_compute = "float16"
        else:
            chosen_compute = "int8"

        if chosen_device == "cuda" and chosen_compute == "int8":
            chosen_compute = "float16"

        return chosen_device, chosen_compute

    def _ensure_model_loaded(self) -> None:
        """Carga perezosa de WhisperModel con degradación automática a CPU si falla CUDA."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        except Exception as exc:
            if self.device != "cuda":
                raise
            logger.warning(
                "Error cargando faster-whisper en CUDA (%s). Cambiando a CPU INT8.",
                exc,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == SAMPLE_RATE or audio.size == 0:
            return audio
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        up = SAMPLE_RATE // divisor
        down = sample_rate // divisor
        resampled = resample_poly(audio.astype(np.float64), up, down)
        return np.clip(resampled, -1.0, 1.0).astype(np.float32)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str = "en",
    ) -> dict[str, Any]:
        """Transcribe audio PCM 16 kHz devolviendo texto, marcas temporales y confianza."""
        pcm = normalize_pcm(audio, peak_normalize=True)
        if pcm.size == 0:
            return {
                "text": "",
                "inicio_ms": 0,
                "fin_ms": 0,
                "confidence": 0.0,
                "segments": [],
                "device": self.device,
                "model_name": self.model_name,
            }

        pcm = self._resample_if_needed(pcm, sample_rate)
        self._ensure_model_loaded()

        effective_lang = "en" if self.model_name.endswith(".en") else language
        segments_iter, _ = self._model.transcribe(
            pcm,
            language=effective_lang,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments: list[dict[str, Any]] = []
        weighted_conf_sum = 0.0
        total_weight = 0.0
        texts: list[str] = []

        for seg in segments_iter:
            text_piece = str(getattr(seg, "text", "")).strip()
            start_ms = int(round(float(getattr(seg, "start", 0.0)) * 1000))
            end_ms = int(round(float(getattr(seg, "end", 0.0)) * 1000))
            avg_logprob = float(getattr(seg, "avg_logprob", -1.0))
            seg_conf = float(min(1.0, max(0.0, math.exp(max(-10.0, min(0.0, avg_logprob))))))

            if text_piece:
                texts.append(text_piece)
            weight = float(max(1, len(text_piece)))
            weighted_conf_sum += seg_conf * weight
            total_weight += weight

            segments.append(
                {
                    "text": text_piece,
                    "inicio_ms": start_ms,
                    "fin_ms": end_ms,
                    "confidence": round(seg_conf, 4),
                }
            )

        full_text = " ".join(texts).strip()
        inicio_ms = segments[0]["inicio_ms"] if segments else 0
        fin_ms = segments[-1]["fin_ms"] if segments else int(round((pcm.size / SAMPLE_RATE) * 1000))
        confidence = (weighted_conf_sum / total_weight) if total_weight > 0 else 0.0

        return {
            "text": full_text,
            "inicio_ms": inicio_ms,
            "fin_ms": fin_ms,
            "confidence": round(confidence, 4),
            "segments": segments,
            "device": self.device,
            "model_name": self.model_name,
        }
