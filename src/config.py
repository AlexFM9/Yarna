"""Configuración base del sistema: constantes acústicas, niveles MCER y entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


SAMPLE_RATE: int = 16000
CHANNELS: int = 1
LATENCY_P95_TARGET_SEC: float = 1.8

LEVEL_PARAMS: dict[str, dict[str, float]] = {
    "A2": {
        "speech_rate": 0.80,
        "piper_length_scale": 1.25,
        "vad_silence_sec": 2.0,
        "silence_timeout_sec": 10.0,
    },
    "B2/C1": {
        "speech_rate": 1.05,
        "piper_length_scale": 0.95,
        "vad_silence_sec": 0.9,
        "silence_timeout_sec": 7.0,
    },
}

DEFAULT_SCENARIOS: list[str] = [
    "Entrevista de trabajo",
    "Pedir en una cafetería",
    "Charla cotidiana",
]


@dataclass(frozen=True)
class Settings:
    """Variables de entorno para modelos locales y clientes LLM."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    piper_model_path: str = os.getenv(
        "PIPER_MODEL_PATH", "models/en_US-lessac-medium.onnx"
    )
    sample_rate: int = int(os.getenv("SAMPLE_RATE", str(SAMPLE_RATE)))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))
