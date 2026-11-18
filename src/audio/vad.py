"""Normalización PCM 16 kHz, buffer circular y Silero VAD adaptativo."""

from __future__ import annotations

import numpy as np

from src.config import SAMPLE_RATE


def normalize_pcm(audio: np.ndarray) -> np.ndarray:
    """Normaliza una señal PCM al rango [-1.0, 1.0] en float32."""
    # TODO: Convertir audio int16/float a float32 normalizado en [-1.0, 1.0].
    raise NotImplementedError("Pendiente de implementar normalize_pcm.")


class CircularPCMBuffer:
    """Buffer circular para conservar los primeros fonemas tras dispararse el VAD."""

    def __init__(self, max_seconds: float = 0.5, sample_rate: int = SAMPLE_RATE) -> None:
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        # TODO: Inicializar estructura circular (p. ej. collections.deque).

    def append(self, samples: np.ndarray) -> None:
        """Añade un bloque de muestras al buffer deslizante."""
        # TODO: Insertar muestras normalizadas en el buffer circular.
        raise NotImplementedError("Pendiente de implementar CircularPCMBuffer.append.")

    def read_all(self) -> np.ndarray:
        """Recupera las muestras almacenadas en la ventana circular."""
        # TODO: Concatenar y devolver muestras como np.ndarray float32.
        raise NotImplementedError("Pendiente de implementar CircularPCMBuffer.read_all.")


class VoiceActivityDetector:
    """Detector de actividad vocal basado en Silero VAD y umbral adaptativo."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        # TODO: Cargar modelo Silero VAD e inicializar umbral adaptativo (E_base + alpha*sigma).

    def process_chunk(self, audio_chunk: np.ndarray, level: str = "A2") -> np.ndarray | None:
        """Procesa un fragmento PCM y devuelve el segmento completo al finalizar la voz."""
        # TODO: Detectar inicio/fin de voz aplicando la ventana de silencio de LEVEL_PARAMS[level].
        raise NotImplementedError("Pendiente de implementar VoiceActivityDetector.process_chunk.")
