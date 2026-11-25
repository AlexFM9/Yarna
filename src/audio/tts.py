"""Módulo TTS (Text-to-Speech): wrapper de Piper TTS (ONNX) con control prosódico.

Este módulo implementa la etapa final del pipeline de audio (Capa 3) de YARNA:
convierte en voz hablada la respuesta textual (`oral_response`) generada por el LLM,
adaptando la velocidad del habla al nivel del alumno sin distorsionar el tono natural.

Conceptos clave para entender este módulo:
1. **Modelos VITS en ONNX (Piper)**:
   Piper ejecuta en local modelos neuronales VITS exportados a ONNX con muy baja
   latencia (`Texto` -> `Fonemas` -> `Espectrograma` -> `Onda PCM`).
2. **Modulación de velocidad mediante `length_scale`**:
   En los modelos VITS, el parámetro `length_scale` controla la duración de cada fonema
   (es decir, es el inverso de la velocidad `rate` del contrato JSON):
   - Nivel Básico (`A2`): `rate = 0.80` -> `length_scale = 1.25` (habla más pausada).
   - Nivel Avanzado (`B2/C1`): `rate = 1.05` -> `length_scale = 0.952` (ritmo nativo).
   Al actuar sobre las duraciones fonémicas internas, cambia la velocidad sin provocar
   efecto "ardilla" ni voz grave artificial.
3. **Estandarización a 16 000 Hz `float32`**:
   Las voces `medium` de Piper generan audio nativo a `22 050 Hz`. Para respetar la
   invariante del proyecto (`SAMPLE_RATE = 16000 Hz`, `float32` en `[-1.0, 1.0]`), la onda
   resultante se re-muestrea y normaliza antes de entregarse a la interfaz Gradio.
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

from src.audio.vad import normalize_pcm
from src.config import SAMPLE_RATE, Settings

logger = logging.getLogger(__name__)

# URLs oficiales de la voz inglesa femenina por defecto (`en_US-lessac-medium`) en HuggingFace.
_DEFAULT_PIPER_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
_DEFAULT_PIPER_JSON_URL = f"{_DEFAULT_PIPER_ONNX_URL}.json"


def _detect_onnx_cuda() -> bool:
    """Comprueba si ONNX Runtime dispone del proveedor de ejecución CUDA (GPU NVIDIA)."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class PiperTTSEngine:
    """Motor local de síntesis vocal VITS en formato ONNX con fallback seguro."""

    def __init__(
        self,
        settings: Settings | None = None,
        use_cuda: bool | None = None,
        auto_download: bool = False,
    ) -> None:
        """Inicializa la configuración de rutas y dispositivo para Piper TTS.

        Args:
            settings: Configuración global (`Settings`) con `piper_model_path`.
            use_cuda: Fuerza uso de GPU (`True`) o CPU (`False`). Si es `None`, se
                autodetecta si `onnxruntime` tiene `CUDAExecutionProvider`.
            auto_download: Si es `True`, descarga automáticamente el modelo `.onnx` y su
                `.onnx.json` desde HuggingFace si faltan en `models/`.
        """
        self.settings = settings or Settings()
        self.model_path = Path(self.settings.piper_model_path)
        self.config_path = Path(
            os.getenv("PIPER_CONFIG_PATH", f"{self.settings.piper_model_path}.json")
        )
        self.use_cuda = _detect_onnx_cuda() if use_cuda is None else use_cuda
        self.auto_download = auto_download
        # La voz se carga bajo demanda en memoria la primera vez que se llama a `synthesize()`.
        self._voice: Any = None

    @staticmethod
    def compute_length_scale(rate: float) -> float:
        """Convierte la velocidad relativa (`prosody_flags.rate`) en `length_scale` de Piper.

        Al ser magnitudes inversas (`length_scale = 1 / rate`), un valor mayor que 1.0
        ralentiza la locución y un valor menor que 1.0 la acelera.

        Args:
            rate: Factor de velocidad deseado (por ejemplo `0.80` en A2 o `1.05` en B2/C1).

        Returns:
            Valor `length_scale` acotado y redondeado a 3 decimales.

        Raises:
            ValueError: Si `rate` es menor o igual que cero.
        """
        if rate <= 0:
            raise ValueError("El parámetro rate debe ser mayor que 0.")
        # Acotamos en [0.5, 2.0] para evitar velocidades extremas ininteligibles.
        clamped = min(2.0, max(0.5, float(rate)))
        return round(1.0 / clamped, 3)

    def ensure_model_files(self, download_if_missing: bool = False) -> bool:
        """Comprueba que existan el modelo `.onnx` y su configuración `.onnx.json`.

        Si `download_if_missing` es `True`, descarga de forma atómica (usando extensión
        temporal `.part`) los ficheros de la voz por defecto `en_US-lessac-medium`.

        Args:
            download_if_missing: Indica si se debe intentar la descarga automática.

        Returns:
            `True` si ambos ficheros existen en disco; `False` si falta alguno.
        """
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
                # Descargamos primero en `.part` para no dejar archivos corruptos si se corta la red.
                tmp_path = target_path.with_suffix(f"{target_path.suffix}.part")
                logger.info("Descargando modelo Piper desde %s...", url)
                urllib.request.urlretrieve(url, tmp_path)
                tmp_path.replace(target_path)
            return self.model_path.exists() and self.config_path.exists()
        except Exception as exc:
            logger.warning("No se pudo descargar la voz de Piper (%s).", exc)
            return False

    def _ensure_voice_loaded(self) -> bool:
        """Carga en memoria `PiperVoice` desde ONNX, degradando a CPU si falla CUDA."""
        if self._voice is not None:
            return True

        if not self.ensure_model_files(download_if_missing=self.auto_download):
            return False

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            logger.warning("Librería piper-tts no instalada (%s). Usando síntesis fallback.", exc)
            return False

        try:
            self._voice = PiperVoice.load(
                str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=self.use_cuda,
            )
            return True
        except Exception as exc:
            # Si ONNX Runtime CUDA falla (p. ej. falta onnxruntime-gpu), reintentamos en CPU.
            if self.use_cuda:
                logger.warning("Error cargando Piper en CUDA (%s). Reintentando en CPU.", exc)
                try:
                    self.use_cuda = False
                    self._voice = PiperVoice.load(
                        str(self.model_path),
                        config_path=str(self.config_path),
                        use_cuda=False,
                    )
                    return True
                except Exception as cpu_exc:
                    logger.warning("Error cargando Piper en CPU (%s).", cpu_exc)
                    return False
            logger.warning("Error cargando Piper ONNX (%s).", exc)
            return False

    @staticmethod
    def _resample_to_target(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Re-muestrea la salida nativa de Piper (ej. 22 050 Hz) a `16 000 Hz` `float32`.

        Args:
            audio: Onda PCM devuelta por Piper.
            orig_sr: Frecuencia de muestreo original de la voz ONNX en Hz.
            target_sr: Frecuencia objetivo del pipeline (16 000 Hz).

        Returns:
            Onda `1D` `np.float32` normalizada en `[-1.0, 1.0]` a `target_sr`.
        """
        if audio.size == 0 or orig_sr == target_sr:
            return normalize_pcm(audio, peak_normalize=True)

        # Reducimos la razón (target_sr / orig_sr) por su MCD para un re-muestreo polifásico rápido.
        divisor = math.gcd(orig_sr, target_sr)
        up = target_sr // divisor
        down = orig_sr // divisor
        pcm_float = normalize_pcm(audio, peak_normalize=False)
        resampled = resample_poly(pcm_float.astype(np.float64), up, down)
        return normalize_pcm(resampled.astype(np.float32), peak_normalize=True)

    def _synthesize_with_piper(self, text: str, length_scale: float) -> tuple[int, np.ndarray]:
        """Ejecuta la inferencia real con `PiperVoice` soportando `piper-tts` >=1.3 y 1.2.x."""
        native_sr = int(getattr(getattr(self._voice, "config", None), "sample_rate", 22050))

        try:
            # API moderna (piper-tts >= 1.3): usa `SynthesisConfig` y emite objetos `AudioChunk`.
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
            # Compatibilidad con piper-tts 1.2.x: emite bytes PCM int16 en streaming.
            raw_bytes = b"".join(
                self._voice.synthesize_stream_raw(text, length_scale=length_scale)
            )
            raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        return SAMPLE_RATE, self._resample_to_target(raw_audio, native_sr, SAMPLE_RATE)

    @staticmethod
    def _synthesize_fallback(text: str, length_scale: float) -> tuple[int, np.ndarray]:
        """Genera una señal acústica determinista de respaldo si faltan los pesos `.onnx`.

        Su duración es proporcional al número de palabras y al `length_scale`, permitiendo
        probar la máquina de estados, el *Input Lock* y la interfaz UI sin pesos locales.
        """
        word_count = max(1, len(text.split()))
        duration_sec = min(4.0, max(0.25, word_count * 0.12 * length_scale))
        num_samples = int(round(duration_sec * SAMPLE_RATE))
        t = np.linspace(0.0, duration_sec, num_samples, endpoint=False, dtype=np.float64)

        # Envolvente sinusoidal suave para evitar chasquidos al inicio y final del tono.
        envelope = np.sin(np.pi * np.linspace(0.0, 1.0, num_samples, dtype=np.float64))
        tone = 0.25 * envelope * (
            0.7 * np.sin(2.0 * np.pi * 220.0 * t) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        )
        return SAMPLE_RATE, normalize_pcm(tone.astype(np.float32), peak_normalize=True)

    def synthesize(self, text: str, rate: float = 1.0) -> tuple[int, np.ndarray]:
        """Sintetiza texto oral en inglés devolviendo `(sample_rate, audio_pcm)` a 16 kHz.

        Args:
            text: Respuesta oral (`oral_response`) que debe pronunciar el tutor.
            rate: Velocidad prosódica deseada (`prosody_flags.rate`).

        Returns:
            Tupla `(16000, audio_float32)` con la forma de onda normalizada en `[-1.0, 1.0]`.
        """
        length_scale = self.compute_length_scale(rate)
        cleaned_text = text.strip()
        if not cleaned_text:
            return SAMPLE_RATE, np.empty(0, dtype=np.float32)

        if self._ensure_voice_loaded():
            return self._synthesize_with_piper(cleaned_text, length_scale)

        logger.debug("Piper ONNX no disponible; usando señal acústica de respaldo.")
        return self._synthesize_fallback(cleaned_text, length_scale)
