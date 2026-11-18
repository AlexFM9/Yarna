"""[PLANTILLA GUIADA — RETO 1] Módulo TTS: Piper TTS (ONNX) y control prosódico.

En este primer reto vas a implementar las 3 funciones clave de síntesis y
adaptación de velocidad/frecuencia de `PiperTTSEngine`:
  1. `compute_length_scale(rate)`
  2. `_resample_to_target(audio, orig_sr, target_sr)`
  3. `synthesize(text, rate)`

La parte mecánica (carga de `.onnx`, descarga desde HuggingFace, compatibilidad
entre versiones de `piper-tts` y fallback armónico) ya está implementada abajo
para que te centres en la lógica de señal y prosodia.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any
import urllib.request

import numpy as np
from scipy.signal import resample_poly

from src.config import SAMPLE_RATE, Settings

logger = logging.getLogger(__name__)

_DEFAULT_PIPER_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
_DEFAULT_PIPER_JSON_URL = f"{_DEFAULT_PIPER_ONNX_URL}.json"


def _detect_onnx_cuda() -> bool:
    """Comprueba si ONNX Runtime dispone del proveedor de ejecución CUDA."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class PiperTTSEngine:
    """Motor local de síntesis vocal VITS en formato ONNX."""

    def __init__(
        self,
        settings: Settings | None = None,
        use_cuda: bool | None = None,
        auto_download: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.model_path = Path(self.settings.piper_model_path)
        self.config_path = Path(
            os.getenv("PIPER_CONFIG_PATH", f"{self.settings.piper_model_path}.json")
        )
        self.use_cuda = _detect_onnx_cuda() if use_cuda is None else use_cuda
        self.auto_download = auto_download
        self._voice: Any = None

    @staticmethod
    def compute_length_scale(rate: float) -> float:
        """Calcula `length_scale` como inverso de `prosody_flags.rate`.

        TODO (Reto 1.1):
        1. Si `rate <= 0`, lanza `ValueError("El parámetro rate debe ser mayor que 0.")`.
        2. Acota `rate` entre `0.5` y `2.0` (`min(2.0, max(0.5, float(rate)))`).
        3. Devuelve `round(1.0 / clamped, 3)` (ej.: `0.80 -> 1.25`, `1.05 -> 0.952`).
        """
        raise NotImplementedError("Reto 1.1 pendiente: compute_length_scale")

    def ensure_model_files(self, download_if_missing: bool = False) -> bool:
        """Comprueba si existen `.onnx` y `.onnx.json`; los descarga si se solicita."""
        if self.model_path.exists() and self.config_path.exists():
            return True
        if not download_if_missing:
            return False

        downloads = [
            (self.model_path, _DEFAULT_PIPER_ONNX_URL),
            (self.config_path, _DEFAULT_PIPER_JSON_URL),
        ]
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            for target_path, url in downloads:
                if target_path.exists():
                    continue
                tmp_path = target_path.with_suffix(f"{target_path.suffix}.part")
                logger.info("Descargando modelo Piper desde %s...", url)
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(target_path)
            return self.model_path.exists() and self.config_path.exists()
        except Exception as exc:
            logger.warning("No se pudo descargar la voz de Piper (%s).", exc)
            return False

    def _ensure_voice_loaded(self) -> bool:
        """Carga en memoria `PiperVoice` desde ONNX con degradación a CPU si falla CUDA."""
        if self._voice is not None:
            return True
        if not self.ensure_model_files(download_if_missing=self.auto_download):
            return False

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            logger.warning("piper-tts no instalado (%s). Usando síntesis fallback.", exc)
            return False

        try:
            self._voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_cuda,
            )
            return True
        except Exception as exc:
            if self.use_cuda:
                logger.warning("Error CUDA en Piper (%s). Reintentando en CPU.", exc)
                try:
                    self.use_cuda = False
                    self._voice = PiperVoice.load(
                        str(self.model_path),
                        config_path=str(self.config_path),
                        use_cuda=False,
                    )
                    return True
                except Exception:
                    return False
            return False

    @staticmethod
    def _resample_to_target(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Re-muestrea la señal PCM (p. ej. 22 050 Hz de Piper) a 16 000 Hz `float32`.

        TODO (Reto 1.2):
        1. Convierte `audio` a array `float32`. Si es entero (`int16`), escálalo a [-1, 1]
           dividiendo por `32768.0` (o `float(np.iinfo(arr.dtype).max)`).
        2. Si `arr.size == 0`, devuélvelo vacío en `float32`.
        3. Si `orig_sr != target_sr`:
           - Calcula `divisor = math.gcd(orig_sr, target_sr)`.
           - `up = target_sr // divisor` y `down = orig_sr // divisor`.
           - Llama a `resample_poly(arr.astype(np.float64), up, down)`.
        4. Acota con `np.clip(..., -1.0, 1.0)` y devuelve el vector como `np.float32`.
        """
        raise NotImplementedError("Reto 1.2 pendiente: _resample_to_target")

    def _synthesize_with_piper(self, text: str, length_scale: float) -> tuple[int, np.ndarray]:
        """Ejecuta `PiperVoice` soportando piper-tts >=1.3 y 1.2.x."""
        native_sr = int(getattr(getattr(self._voice, "config", None), "sample_rate", 22050))
        try:
            from piper.config import SynthesisConfig

            syn_cfg = SynthesisConfig(length_scale=length_scale)
            chunks = list(self._voice.synthesize(text, syn_config=syn_cfg))
            arrays: list[np.ndarray] = []
            for chunk in chunks:
                if hasattr(chunk, "audio_float_array"):
                    arrays.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
                elif hasattr(chunk, "audio_int16_bytes"):
                    arrays.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
                if hasattr(chunk, "sample_rate"):
                    native_sr = int(chunk.sample_rate)
            raw_audio = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
        except ImportError:
            raw_bytes = b"".join(
                self._voice.synthesize_stream_raw(text, length_scale=length_scale)
            )
            raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        return SAMPLE_RATE, self._resample_to_target(raw_audio, native_sr, SAMPLE_RATE)

    @staticmethod
    def _synthesize_fallback(text: str, length_scale: float) -> tuple[int, np.ndarray]:
        """Síntesis armónica determinista de respaldo cuando faltan pesos ONNX."""
        word_count = max(1, len(text.split()))
        duration_sec = min(4.0, max(0.25, word_count * 0.12 * length_scale))
        num_samples = int(round(duration_sec * SAMPLE_RATE))
        t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float64)
        envelope = np.sin(np.pi * np.linspace(0.0, 1.0, num_samples, dtype=np.float64))
        tone = 0.25 * envelope * (
            0.7 * np.sin(2.0 * np.pi * 220.0 * t) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        )
        return SAMPLE_RATE, np.clip(tone, -1.0, 1.0).astype(np.float32)

    def synthesize(self, text: str, rate: float = 1.0) -> tuple[int, np.ndarray]:
        """Sintetiza texto oral devolviendo `(16000, audio_float32)` en `[-1.0, 1.0]`.

        TODO (Reto 1.3):
        1. Calcula `length_scale = self.compute_length_scale(rate)`.
        2. Limpia espacios con `cleaned = text.strip()`. Si queda vacío, devuelve
           `(SAMPLE_RATE, np.empty(0, dtype=np.float32))`.
        3. Si `self._ensure_voice_loaded()` es `True`, llama y devuelve
           `self._synthesize_with_piper(cleaned, length_scale)`.
        4. En caso contrario, devuelve `self._synthesize_fallback(cleaned, length_scale)`.
        """
        raise NotImplementedError("Reto 1.3 pendiente: synthesize")
