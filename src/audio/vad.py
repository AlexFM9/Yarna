"""Normalización PCM 16 kHz, buffer circular y Silero VAD adaptativo."""

from __future__ import annotations

from collections import deque
import logging
from typing import Any

import numpy as np

from src.config import LEVEL_PARAMS, SAMPLE_RATE

logger = logging.getLogger(__name__)

SILERO_WINDOW_SAMPLES: int = 512


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convierte cualquier array PCM entero o flotante a 1D float32 en [-1.0, 1.0]."""
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
    """Normaliza una señal PCM a float32 en [-1.0, 1.0] con centrado DC seguro."""
    float_arr = _to_mono_float32(audio)
    if float_arr.size == 0:
        return float_arr

    centered = float_arr - np.float32(np.mean(float_arr, dtype=np.float64))

    if peak_normalize:
        peak = float(np.max(np.abs(centered)))
        if peak >= noise_floor:
            centered = centered / np.float32(peak)

    return np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)


class CircularPCMBuffer:
    """Buffer circular para conservar los primeros fonemas tras dispararse el VAD."""

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
        """Añade un bloque de muestras al buffer deslizante."""
        chunk = _to_mono_float32(samples)
        if chunk.size == 0:
            return

        if chunk.size >= self.max_samples:
            self._chunks.clear()
            self._chunks.append(chunk[-self.max_samples :].copy())
            self._total_samples = self.max_samples
            return

        self._chunks.append(chunk.copy())
        self._total_samples += chunk.size

        while self._total_samples > self.max_samples and self._chunks:
            excess = self._total_samples - self.max_samples
            oldest = self._chunks[0]
            if oldest.size <= excess:
                self._chunks.popleft()
                self._total_samples -= oldest.size
            else:
                self._chunks[0] = oldest[excess:]
                self._total_samples -= excess

    def read_all(self) -> np.ndarray:
        """Recupera las muestras almacenadas en la ventana circular."""
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(self._chunks)).astype(np.float32, copy=False)

    def clear(self) -> None:
        """Vacía el contenido del buffer circular."""
        self._chunks.clear()
        self._total_samples = 0


class VoiceActivityDetector:
    """Detector de actividad vocal basado en Silero VAD y umbral adaptativo."""

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
        """Carga Silero VAD en ONNX (o TorchScript como respaldo) si está disponible."""
        try:
            import torch
            from silero_vad import load_silero_vad

            self._torch = torch
            try:
                self._model = load_silero_vad(onnx=True)
            except Exception:
                self._model = load_silero_vad(onnx=False)
        except Exception as exc:
            logger.warning(
                "Silero VAD no disponible (%s). Usando detector adaptativo por energía RMS.",
                exc,
            )
            self._model = None

    def _update_noise_stats(self, rms: float) -> None:
        """Actualiza media y desviación del ruido de fondo mediante media móvil exponencial."""
        decay = 0.95
        diff = rms - self._e_base
        self._e_base = decay * self._e_base + (1.0 - decay) * rms
        var = decay * (self._sigma**2) + (1.0 - decay) * (diff**2)
        self._sigma = float(np.sqrt(max(var, 1e-10)))

    def _adaptive_energy_threshold(self) -> float:
        return max(self._min_energy_floor, self._e_base + self.alpha * self._sigma)

    def _is_window_speech(self, window: np.ndarray) -> bool:
        """Evalúa una ventana de 512 muestras combinando Silero VAD y energía adaptativa."""
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        energy_thresh = self._adaptive_energy_threshold()

        if self._model is not None and self._torch is not None:
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(window)
                prob = float(self._model(tensor, self.sample_rate).item())
            if prob >= max(0.75, self.speech_threshold + 0.15):
                return True
            return prob >= self.speech_threshold and rms >= max(1e-4, self._e_base * 0.5)

        return rms >= energy_thresh

    def _reset_silero_states(self) -> None:
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def reset(self) -> None:
        """Reinicia el estado interno de detección entre turnos o sesiones."""
        self._pre_roll.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_frames.clear()
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_samples = 0
        self._reset_silero_states()

    def _finalize_utterance(self, tail_keep_sec: float = 0.25) -> np.ndarray:
        """Concatena los frames capturados recortando el exceso de silencio final."""
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
        """Procesa un fragmento PCM y devuelve el segmento completo al finalizar la voz."""
        params = LEVEL_PARAMS.get(level, LEVEL_PARAMS["A2"])
        silence_limit_samples = int(round(float(params["vad_silence_sec"]) * self.sample_rate))

        chunk = _to_mono_float32(audio_chunk)
        if chunk.size == 0:
            return None

        if self._pending.size > 0:
            chunk = np.concatenate([self._pending, chunk])

        total_windows = chunk.size // self.window_samples
        consumed = total_windows * self.window_samples
        self._pending = chunk[consumed:].copy()

        completed_utterance: np.ndarray | None = None

        for idx in range(total_windows):
            start = idx * self.window_samples
            window = chunk[start : start + self.window_samples]
            rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
            is_speech = self._is_window_speech(window)

            if not self._is_speaking:
                if is_speech:
                    pre_roll_audio = self._pre_roll.read_all()
                    if pre_roll_audio.size > 0:
                        self._speech_frames.append(pre_roll_audio)
                    self._speech_frames.append(window.copy())
                    self._pre_roll.clear()
                    self._is_speaking = True
                    self._silence_samples = 0
                    self._speech_samples = window.size
                else:
                    self._pre_roll.append(window)
                    self._update_noise_stats(rms)
            else:
                self._speech_frames.append(window.copy())
                if is_speech:
                    self._silence_samples = 0
                    self._speech_samples += window.size
                else:
                    self._silence_samples += window.size
                    if self._silence_samples >= silence_limit_samples:
                        completed_utterance = self._finalize_utterance()
                        break

        return completed_utterance

    def flush(self) -> np.ndarray | None:
        """Devuelve la locución en curso si se detiene la captura manualmente."""
        if not self._is_speaking or not self._speech_frames:
            return None
        return self._finalize_utterance()

    def extract_speech_segment(
        self,
        audio: np.ndarray,
        level: str = "A2",
        padding_sec: float = 0.25,
    ) -> np.ndarray:
        """Modo bloque completo: recorta silencios inicial/final de una grabación entera."""
        signal = _to_mono_float32(audio)
        if signal.size == 0:
            return signal

        self._reset_silero_states()
        speech_indices: list[int] = []
        total_windows = signal.size // self.window_samples

        for idx in range(total_windows):
            start = idx * self.window_samples
            window = signal[start : start + self.window_samples]
            if self._is_window_speech(window):
                speech_indices.append(idx)

        self._reset_silero_states()

        if not speech_indices:
            return np.empty(0, dtype=np.float32)

        pad_samples = int(round(padding_sec * self.sample_rate))
        start_sample = max(0, speech_indices[0] * self.window_samples - pad_samples)
        end_sample = min(
            signal.size,
            (speech_indices[-1] + 1) * self.window_samples + pad_samples,
        )
        _ = LEVEL_PARAMS.get(level, LEVEL_PARAMS["A2"])
        return normalize_pcm(signal[start_sample:end_sample], peak_normalize=True)
