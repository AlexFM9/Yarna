"""Wrapper de Piper TTS (ONNX) con modulación dinámica de velocidad."""

from __future__ import annotations

import numpy as np

from src.config import Settings


class PiperTTSEngine:
    """Motor local de síntesis vocal VITS en formato ONNX."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        # TODO: Cargar la voz ONNX de Piper desde self.settings.piper_model_path.

    @staticmethod
    def compute_length_scale(rate: float) -> float:
        """Calcula length_scale como inverso de prosody_flags.rate."""
        # TODO: Retornar round(1.0 / rate, 3) (~1.25 para A2, ~0.95 para B2/C1).
        raise NotImplementedError("Pendiente de implementar compute_length_scale.")

    def synthesize(self, text: str, rate: float = 1.0) -> tuple[int, np.ndarray]:
        """Sintetiza texto oral devolviendo (sample_rate, audio_pcm)."""
        # TODO: Inyectar length_scale en Piper TTS y devolver la onda PCM generada.
        raise NotImplementedError("Pendiente de implementar PiperTTSEngine.synthesize.")
