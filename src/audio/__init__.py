"""Capa 3: Módulo de entrada acústica (VAD, STT) y síntesis vocal (Piper TTS).

Expone las interfaces públicas que consumen la Capa 2 (Orquestador) y la Capa 1 (UI)
manteniendo el desacoplamiento estricto: este paquete no importa nada de `src/ui/` ni
de Gradio.
"""

from src.audio.stt import FasterWhisperSTT, detect_cuda_available, resolve_whisper_model_name
from src.audio.tts import PiperTTSEngine
from src.audio.vad import CircularPCMBuffer, VoiceActivityDetector, normalize_pcm

__all__ = [
    "CircularPCMBuffer",
    "FasterWhisperSTT",
    "PiperTTSEngine",
    "VoiceActivityDetector",
    "detect_cuda_available",
    "normalize_pcm",
    "resolve_whisper_model_name",
]
