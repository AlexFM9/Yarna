"""[PLANTILLA GUIADA — RETOS 2 Y 3] Normalización PCM, Buffer Circular y Silero VAD.

Este archivo está dividido en dos retos:
- **Reto 2 (`normalize_pcm` y `CircularPCMBuffer`)**: estandarización de señal a
  16 kHz `float32` en `[-1.0, 1.0]` y buffer circular de pre-roll.
- **Reto 3 (`VoiceActivityDetector`)**: lógica de detección por ventanas de 512 muestras
  (umbral adaptativo $E_{base} + \\alpha \\cdot \\sigma$, streaming y recorte por bloque).

La carga de Silero VAD (`_load_silero_model`), el cálculo del estadístico EMA de ruido
(`_update_noise_stats`) y la conversión base a vector mono (`_to_mono_float32`) ya están
resueltos para facilitarte el desarrollo.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

import numpy as np

from src.config import LEVEL_PARAMS, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Silero VAD v5 opera en ventanas exactas de 512 muestras a 16 kHz (32 ms).
SILERO_WINDOW_SAMPLES: int = 512


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convierte cualquier array PCM entero o flotante a 1D `float32` en `[-1.0, 1.0]`."""
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)

    if arr.ndim > 1:
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            channel_axis = 1 if arr.shape[1] <= arr.shape[0] else 0
            arr = np.mean(arr.astype(np.float64), axis=channel_axis)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min == 0:
            midpoint = (float(info.max) + 1.0) / 2.0
            float_arr = (arr.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), info.max))
            float_arr = arr.astype(np.float32) / scale
    else:
        float_arr = np.nan_to_num(
            arr.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    return np.clip(float_arr, -1.0, 1.0).astype(np.float32, copy=False)


def normalize_pcm(
    audio: np.ndarray,
    noise_floor: float = 1e-4,
    peak_normalize: bool = True,
) -> np.ndarray:
    """Normaliza una señal PCM a `float32` en `[-1.0, 1.0]` con centrado DC seguro.

    TODO (Reto 2.1):
    1. Llama a `float_arr = _to_mono_float32(audio)`. Si `float_arr.size == 0`, devuélvelo.
    2. Centra el nivel DC restando la media (`np.mean(float_arr, dtype=np.float64)`).
    3. Si `peak_normalize` es `True`:
       - Calcula `peak = float(np.max(np.abs(centered)))`.
       - Solo divide `centered` por `peak` si `peak >= noise_floor` (para no amplificar silencio).
    4. Devuelve `np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)`.
    """
    raise NotImplementedError("Reto 2.1 pendiente: normalize_pcm")


class CircularPCMBuffer:
    """Buffer circular deslizante para conservar el audio inmediatamente previo al habla."""

    def __init__(self, max_seconds: float = 0.5, sample_rate: int = SAMPLE_RATE) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds debe ser positivo.")
        if sample_rate <= 0:
            raise ValueError("sample_rate debe ser positivo.")
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(round(max_seconds * sample_rate)))
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0

    def append(self, samples: np.ndarray) -> None:
        """Añade un bloque al buffer descartando por la izquierda lo que exceda `max_samples`.

        TODO (Reto 2.2):
        1. Convierte `samples` con `chunk = _to_mono_float32(samples)`. Si está vacío, retorna.
        2. Si `chunk.size >= self.max_samples`: limpia `self._chunks`, añade solo
           `chunk[-self.max_samples:].copy()` y fija `self._total_samples = self.max_samples`.
        3. Si no: añade `chunk.copy()` a `self._chunks` y suma `chunk.size` a `_total_samples`.
        4. Mientras `self._total_samples > self.max_samples`:
           recorta o extrae por la izquierda (`self._chunks.popleft()` o `oldest[excess:]`)
           actualizando `self._total_samples`.
        """
        raise NotImplementedError("Reto 2.2 pendiente: CircularPCMBuffer.append")

    def read_all(self) -> np.ndarray:
        """Concatena y devuelve las muestras del buffer en un vector 1D `float32`.

        TODO (Reto 2.3):
        - Si `self._chunks` está vacío, devuelve `np.empty(0, dtype=np.float32)`.
        - Si no, devuelve `np.concatenate(list(self._chunks)).astype(np.float32, copy=False)`.
        """
        raise NotImplementedError("Reto 2.3 pendiente: CircularPCMBuffer.read_all")

    def clear(self) -> None:
        """Vacía `self._chunks` y reinicia `self._total_samples` a 0.

        TODO (Reto 2.4): implementa el borrado del buffer.
        """
        raise NotImplementedError("Reto 2.4 pendiente: CircularPCMBuffer.clear")


class VoiceActivityDetector:
    """Detector de actividad vocal (VAD) dual basado en Silero VAD y umbral adaptativo."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        speech_threshold: float = 0.5,
        alpha: float = 2.0,
        pre_roll_sec: float = 0.5,
        use_silero: bool = True,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD requiere sample_rate de 8000 o 16000 Hz.")
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.alpha = alpha
        self.window_samples = SILERO_WINDOW_SAMPLES if sample_rate == 16000 else 256

        self._pre_roll = CircularPCMBuffer(max_seconds=pre_roll_sec, sample_rate=sample_rate)
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_frames: list[np.ndarray] = []
        self._is_speaking: bool = False
        self._silence_samples: int = 0
        self._speech_samples: int = 0

        self._e_base: float = 0.005
        self._sigma: float = 0.002
        self._min_energy_floor: float = 0.01

        self._model: Any = None
        self._torch: Any = None
        if use_silero:
            self._load_silero_model()

    def _load_silero_model(self) -> None:
        """Carga Silero VAD (ONNX o TorchScript) si está instalado."""
        try:
            import torch
            from silero_vad import load_silero_vad

            self._torch = torch
            try:
                self._model = load_silero_vad(onnx=True)
            except Exception:
                self._model = load_silero_vad(onnx=False)
        except Exception as exc:
            logger.warning("Silero VAD no disponible (%s). Usando detector RMS.", exc)
            self._model = None

    def _update_noise_stats(self, rms: float) -> None:
        """Actualiza media (`_e_base`) y desviación típica (`_sigma`) del ruido de fondo."""
        decay = 0.95
        diff = rms - self._e_base
        self._e_base = decay * self._e_base + (1.0 - decay) * rms
        var = decay * (self._sigma**2) + (1.0 - decay) * (diff**2)
        self._sigma = float(np.sqrt(max(var, 1e-10)))

    def _adaptive_energy_threshold(self) -> float:
        """Calcula el umbral adaptativo $E_{base} + \\alpha \\cdot \\sigma$.

        TODO (Reto 3.1):
        - Devuelve `max(self._min_energy_floor, self._e_base + self.alpha * self._sigma)`.
        """
        raise NotImplementedError("Reto 3.1 pendiente: _adaptive_energy_threshold")

    def _is_window_speech(self, window: np.ndarray) -> bool:
        """Decide si una ventana de 512 muestras contiene voz.

        TODO (Reto 3.2):
        1. Calcula el RMS de `window`: `rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))`.
        2. Si `self._model is not None and self._torch is not None`:
           - Calcula `prob = float(self._model(self._torch.from_numpy(window), self.sample_rate).item())`
             dentro de `with self._torch.no_grad():`.
           - Devuelve `True` si `prob >= max(0.75, self.speech_threshold + 0.15)` o si
             `(prob >= self.speech_threshold and rms >= max(1e-4, self._e_base * 0.5))`.
        3. Si no hay modelo Silero cargado, devuelve `rms >= self._adaptive_energy_threshold()`.
        """
        raise NotImplementedError("Reto 3.2 pendiente: _is_window_speech")

    def _reset_silero_states(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def reset(self) -> None:
        """Reinicia el estado interno de captura entre turnos o sesiones."""
        self._pre_roll.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_frames.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._reset_silero_states()

    def _finalize_utterance(self, tail_keep_sec: float = 0.25) -> np.ndarray:
        """Concatena frames capturados, recorta exceso de silencio final y normaliza."""
        full = np.concatenate(self._speech_frames)
        tail_keep_samples = int(round(tail_keep_sec * self.sample_rate))
        excess_silence = max(0, self._silence_samples - tail_keep_samples)
        if excess_silence > 0 and full.size > excess_silence:
            full = full[:-excess_silence]

        self._speech_frames.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._pre_roll.clear()
        self._reset_silero_states()
        return normalize_pcm(full, peak_normalize=True)

    def process_chunk(self, audio_chunk: np.ndarray, level: str = "A2") -> np.ndarray | None:
        """Procesa un fragmento en modo streaming y devuelve el turno al expirar el silencio.

        TODO (Reto 3.3):
        1. Obtén `silence_limit_samples = int(round(float(LEVEL_PARAMS.get(level, LEVEL_PARAMS["A2"])["vad_silence_sec"]) * self.sample_rate))`.
        2. Convierte `audio_chunk` con `_to_mono_float32` (sin normalizar pico por trozo) y concatena `self._pending` si no está vacío.
        3. Calcula cuántas ventanas enteras de `self.window_samples` caben; guarda el sobrante en `self._pending`.
        4. Para cada ventana:
           - Si `not self._is_speaking`:
             - Si hay voz (`_is_window_speech`): recupera `self._pre_roll.read_all()`, añade ambos a `self._speech_frames`,
               limpia `_pre_roll`, marca `_is_speaking = True` y `_silence_samples = 0`.
             - Si no hay voz: añade la ventana a `_pre_roll` y actualiza `_update_noise_stats(rms)`.
           - Si `self._is_speaking`:
             - Añade la ventana a `self._speech_frames`.
             - Si hay voz: `_silence_samples = 0`.
             - Si es silencio: suma `window.size` a `_silence_samples`. Si `_silence_samples >= silence_limit_samples`,
               devuelve `self._finalize_utterance()`.
        5. Si no se cerró turno, devuelve `None`.
        """
        raise NotImplementedError("Reto 3.3 pendiente: process_chunk")

    def flush(self) -> np.ndarray | None:
        """Entrega la locución en curso si se detiene manualmente."""
        if not self._is_speaking or not self._speech_frames:
            return None
        return self._finalize_utterance()

    def extract_speech_segment(
        self,
        audio: np.ndarray,
        level: str = "A2",
        padding_sec: float = 0.25,
    ) -> np.ndarray:
        """Modo bloque completo: recorta silencios inicial/final de una grabación entera.

        TODO (Reto 3.4):
        1. Convierte `audio` con `_to_mono_float32(audio)`. Si está vacío, devuélvelo.
        2. Recorre las ventanas de `self.window_samples` guardando los índices donde `_is_window_speech(window)` sea `True`.
        3. Si no hay ventanas con voz, devuelve `np.empty(0, dtype=np.float32)`.
        4. Calcula `pad_samples = int(round(padding_sec * self.sample_rate))`, recorta entre
           `first_idx * window_samples - pad_samples` y `(last_idx + 1) * window_samples + pad_samples`
           (acotado a `[0, signal.size]`), y devuelve `normalize_pcm(..., peak_normalize=True)`.
        """
        raise NotImplementedError("Reto 3.4 pendiente: extract_speech_segment")
