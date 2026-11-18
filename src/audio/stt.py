"""[PLANTILLA GUIADA — RETO 4] Módulo STT: wrapper de `faster-whisper` con GPU CUDA.

En este reto vas a implementar las 3 partes principales de `stt.py`:
  1. `resolve_whisper_model_name(model_size, prefer_english_model)`
  2. `FasterWhisperSTT._resample_if_needed(audio, sample_rate)`
  3. `FasterWhisperSTT.transcribe(audio, sample_rate, language)`

La autodetección de GPU NVIDIA (`detect_cuda_available`), resolución de `(device, compute_type)`
y carga perezosa con degradación a CPU (`_ensure_model_loaded`) ya están resueltas abajo.
"""

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
    """Añade `.en` a tamaños estándar (`tiny`, `base`, `small`, `medium`) si procede.

    TODO (Reto 4.1):
    1. Limpia espacios con `cleaned = model_size.strip()`.
    2. Si `prefer_english_model` es `True` y `cleaned` está en `_ENGLISH_ONLY_SIZES`,
       devuelve `f"{cleaned}.en"`.
    3. En otro caso devuelve `cleaned`.
    """
    raise NotImplementedError("Reto 4.1 pendiente: resolve_whisper_model_name")


def detect_cuda_available() -> bool:
    """Comprueba de forma segura si existe alguna GPU NVIDIA lista para CUDA."""
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
    """Motor STT local cuantizado (`FP16` en GPU NVIDIA con fallback a `INT8` en CPU)."""

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
        """Selecciona `(cuda, float16)` si hay GPU disponible o `(cpu, int8)`."""
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
        """Carga perezosa de `WhisperModel` con fallback automático a CPU INT8."""
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
            logger.warning("Error CUDA (%s). Cambiando a CPU INT8.", exc)
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
        """Re-muestrea el audio a `SAMPLE_RATE` (16 000 Hz) si viene a otra frecuencia.

        TODO (Reto 4.2):
        1. Si `sample_rate == SAMPLE_RATE` o `audio.size == 0`, devuelve `audio`.
        2. Calcula `divisor = math.gcd(sample_rate, SAMPLE_RATE)`, `up = SAMPLE_RATE // divisor`
           y `down = sample_rate // divisor`.
        3. Re-muestrea con `resample_poly(audio.astype(np.float64), up, down)` y devuelve
           el resultado acotado en `[-1.0, 1.0]` como `np.float32`.
        """
        raise NotImplementedError("Reto 4.2 pendiente: _resample_if_needed")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str = "en",
    ) -> dict[str, Any]:
        """Transcribe audio PCM devolviendo texto, marcas temporales y confianza.

        TODO (Reto 4.3):
        1. Normaliza el audio con `pcm = normalize_pcm(audio, peak_normalize=True)`.
           Si `pcm.size == 0`, devuelve diccionario vacío con `inicio_ms=0`, `fin_ms=0`, `confidence=0.0`.
        2. Re-muestrea con `pcm = self._resample_if_needed(pcm, sample_rate)` y llama a `self._ensure_model_loaded()`.
        3. Ejecuta `segments_iter, _ = self._model.transcribe(pcm, language=..., beam_size=1, best_of=1, vad_filter=False, condition_on_previous_text=False)`.
           (Si `self.model_name.endswith(".en")`, usa `language="en"`).
        4. Recorre cada segmento de `segments_iter`:
           - Extrae `text_piece = str(getattr(seg, "text", "")).strip()`, `start_ms = int(round(seg.start * 1000))`,
             `end_ms = int(round(seg.end * 1000))`.
           - Calcula la confianza del segmento: `math.exp(max(-10.0, min(0.0, seg.avg_logprob)))` acotada en `[0.0, 1.0]`.
           - Acumula confianza ponderada por longitud de texto y guarda cada segmento en `segments`.
        5. Devuelve el diccionario con claves:
           `"text"`, `"inicio_ms"`, `"fin_ms"`, `"confidence"`, `"segments"`, `"device"`, `"model_name"`.
        """
        raise NotImplementedError("Reto 4.3 pendiente: transcribe")
