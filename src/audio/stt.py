"""Wrapper de faster-whisper (CTranslate2) para transcripción STT con timestamps."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import SAMPLE_RATE, Settings


class FasterWhisperSTT:
    """Motor STT local cuantizado (INT8 en CPU o FP16 en GPU)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        # TODO: Cargar WhisperModel de faster_whisper con device y compute_type.

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        language: str = "en",
    ) -> dict[str, Any]:
        """Transcribe audio PCM 16 kHz devolviendo texto, marcas temporales y confianza."""
        # TODO: Ejecutar inferencia con faster-whisper y retornar texto puntuado,
        # inicio_ms, fin_ms y puntuación de confianza.
        raise NotImplementedError("Pendiente de implementar FasterWhisperSTT.transcribe.")
