"""Módulo STT (Speech-to-Text): wrapper de `faster-whisper` sobre CTranslate2.

Este módulo constituye la segunda etapa del pipeline acústico (Capa 3) de YARNA.
Convierte el segmento de voz normalizado que entrega el VAD en texto transcrito en inglés
con marcas temporales y puntuación de confianza para el Orquestador Cognitivo (Capa 2).

Decisiones de diseño incluidas:
1. **Modelos monolingües `.en` (`base.en` / `small.en`)**: al tratarse de un tutor de
   inglés, los modelos específicos `.en` evitan la fase de detección automática de idioma,
   reducen la latencia y no "autocorrigen" en exceso los errores reales del alumno L2
   (algo necesario para que el LLM pueda detectarlos y aplicar *recast*).
2. **Aceleración GPU NVIDIA con degradación automática a CPU**: si el sistema dispone de
   GPU CUDA compatible (como una RTX 3050), ejecuta en `cuda` con precisión `float16`.
   Si faltan librerías CUDA/cuDNN o no hay GPU, cae automáticamente a `cpu` con
   cuantización `int8` sin interrumpir el servicio.
3. **Baja latencia (`beam_size=1`, `vad_filter=False`)**: el filtrado de silencios ya lo
   realizó `VoiceActivityDetector` aguas arriba, por lo que desactivar el VAD interno de
   Whisper y usar búsqueda voraz (`beam_size=1`) ayuda a cumplir el objetivo $p95 < 1.8\\text{ s}$.
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

# Tamaños de Whisper oficiales que disponen de variante específica solo en inglés (`.en`).
_ENGLISH_ONLY_SIZES: frozenset[str] = frozenset({"tiny", "base", "small", "medium"})


def resolve_whisper_model_name(model_size: str, prefer_english_model: bool = True) -> str:
    """Resuelve el identificador exacto del modelo Whisper a cargar.

    Si `prefer_english_model` es `True` y se pide un tamaño estándar (`tiny`, `base`,
    `small`, `medium`), añade el sufijo `.en`. Rutas locales o modelos como `large-v3`
    se mantienen intactos.

    Args:
        model_size: Nombre o ruta del modelo configurado (p. ej. `"base"`, `"small.en"`).
        prefer_english_model: Si es `True`, prioriza variantes monolingües en inglés.

    Returns:
        Nombre normalizado del modelo listo para pasar a `WhisperModel`.
    """
    cleaned = model_size.strip()
    if prefer_english_model and cleaned in _ENGLISH_ONLY_SIZES:
        return f"{cleaned}.en"
    return cleaned


def detect_cuda_available() -> bool:
    """Comprueba de forma segura si existe alguna GPU NVIDIA lista para CUDA.

    Consulta primero `ctranslate2` (motor real de `faster-whisper`) y, si no está
    instalado aún, consulta `torch.cuda.is_available()`.

    Returns:
        `True` si hay al menos una GPU CUDA utilizable; `False` en caso contrario.
    """
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
    """Motor STT local cuantizado (`FP16` en GPU NVIDIA con fallback a `INT8` en CPU).

    La carga de pesos en memoria se realiza de forma perezosa (*lazy loading*) la primera
    vez que se invoca `transcribe()`, evitando ralentizar el arranque de la interfaz UI.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        prefer_english_model: bool = True,
        download_root: str | None = None,
    ) -> None:
        """Configura los parámetros del motor `faster-whisper`.

        Args:
            settings: Configuración global opcional (`Settings`). Si se omite, se leen las
                variables de entorno con autodetección automática de GPU.
            model_size: Sobrescribe el tamaño de modelo (`tiny`, `base`, `small`, etc.).
            device: Fuerza el dispositivo (`"cuda"`, `"cpu"` o `"auto"`).
            compute_type: Tipo numérico en CTranslate2 (`"float16"`, `"int8"`, etc.).
            prefer_english_model: Añade `.en` automáticamente a los tamaños estándar.
            download_root: Carpeta donde almacenar en caché los pesos descargados.
        """
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
        # El modelo real `WhisperModel` se instancia en la primera llamada a `transcribe()`.
        self._model: Any = None

    def _resolve_device_and_compute_type(
        self,
        device: str | None,
        compute_type: str | None,
    ) -> tuple[str, str]:
        """Determina el par `(device, compute_type)` óptimo respetando prioridades.

        Orden de prioridad:
        1. Argumento explícito recibido en el constructor (`device` / `compute_type`).
        2. Objeto `Settings` pasado explícitamente por código.
        3. Variables de entorno (`WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE`).
        4. Autodetección de GPU (`cuda` + `float16` si hay GPU, `cpu` + `int8` si no).
        """
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

        # Evitamos la combinación cuda + int8 por defecto si proviene de un valor pensado para CPU,
        # ya que float16 ofrece mejor soporte y estabilidad en GPUs RTX para modelos pequeños.
        if chosen_device == "cuda" and chosen_compute == "int8":
            chosen_compute = "float16"

        return chosen_device, chosen_compute

    def _ensure_model_loaded(self) -> None:
        """Carga en memoria `WhisperModel` bajo demanda y aplica fallback a CPU si falla CUDA."""
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
            # Si CUDA falla (p. ej. falta cuDNN/cuBLAS o memoria VRAM), no bloqueamos el tutor:
            # degradamos automáticamente a CPU INT8 y actualizamos el estado interno.
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
        """Re-muestrea el audio a 16 000 Hz si procede de un micrófono o WAV a otra frecuencia.

        Usa filtrado polifásico racional (`resample_poly`) reduciendo la fracción por el
        máximo común divisor (`math.gcd`) para máxima calidad y rapidez.
        """
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
        """Transcribe un fragmento de audio PCM devolviendo texto, timestamps y confianza.

        Args:
            audio: Señal de voz (entera o flotante, mono o estéreo). Se normalizará a
                16 000 Hz `float32` en `[-1.0, 1.0]` antes de entrar a Whisper.
            sample_rate: Frecuencia de muestreo del audio de entrada en Hz.
            language: Código ISO del idioma esperado (por defecto `"en"`).

        Returns:
            Diccionario con:
            - `text` (`str`): Transcripción completa concatenada.
            - `inicio_ms` (`int`): Instante inicial de habla en milisegundos.
            - `fin_ms` (`int`): Instante final de habla en milisegundos.
            - `confidence` (`float`): Confianza media ponderada en `[0.0, 1.0]`.
            - `segments` (`list[dict]`): Detalle por segmento con marcas temporales.
            - `device` (`str`): Dispositivo usado finalmente (`"cuda"` o `"cpu"`).
            - `model_name` (`str`): Modelo ejecutado (p. ej. `"base.en"`).
        """
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

        # En modelos `.en` fijamos siempre "en" para ahorrar cualquier comprobación de idioma.
        effective_lang = "en" if self.model_name.endswith(".en") else language
        raw_segments = self._decode_with_cuda_fallback(pcm, effective_lang)

        segments: list[dict[str, Any]] = []
        weighted_conf_sum = 0.0
        total_weight = 0.0
        texts: list[str] = []

        for seg in raw_segments:
            text_piece = str(getattr(seg, "text", "")).strip()
            start_ms = int(round(float(getattr(seg, "start", 0.0)) * 1000))
            end_ms = int(round(float(getattr(seg, "end", 0.0)) * 1000))
            avg_logprob = float(getattr(seg, "avg_logprob", -1.0))

            # Convertimos el logaritmo de probabilidad media (<= 0) en probabilidad [0, 1].
            seg_conf = float(min(1.0, max(0.0, math.exp(max(-10.0, min(0.0, avg_logprob))))))

            if text_piece:
                texts.append(text_piece)

            # Ponderamos la confianza global según la longitud de cada segmento.
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

    def _decode_with_cuda_fallback(
        self, pcm: np.ndarray, effective_lang: str
    ) -> list[Any]:
        """Ejecuta `transcribe()` y materializa el generador con fallback a CPU si falta una DLL CUDA."""
        try:
            segments_iter, _ = self._model.transcribe(
                pcm,
                language=effective_lang,
                beam_size=1,
                best_of=1,
                vad_filter=False,  # El filtrado VAD ya lo realizó `VoiceActivityDetector`.
                condition_on_previous_text=False,  # Evita bucles repetitivos en frases cortas L2.
            )
            return list(segments_iter)
        except RuntimeError as exc:
            if self.device != "cuda":
                raise
            logger.warning(
                "Fallo en inferencia CUDA de faster-whisper (%s). Reintentando en CPU INT8.",
                exc,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = None
            self._ensure_model_loaded()
            segments_iter, _ = self._model.transcribe(
                pcm,
                language=effective_lang,
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return list(segments_iter)
