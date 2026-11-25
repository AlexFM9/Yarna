"""Orquestador cognitivo: memoria deslizante, llamada LLM y degradación elegante."""

from __future__ import annotations

from collections import deque
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from src.config import LEVEL_PARAMS, Settings
from src.core.prompts import build_system_prompt
from src.core.schemas import (
    ProsodyFlags,
    SessionProfile,
    TutorResponse,
    VisualCard,
)

logger = logging.getLogger(__name__)

CONTINGENCY_ORAL_RESPONSE = "Sorry, I didn't quite catch that, could you say it again?"
SUPPORTED_PROVIDERS = {"ollama", "openai", "gemini", "mock"}


class CognitiveOrchestrator:
    """Coordina historial conversacional, validación Pydantic y fallbacks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._provider = self.settings.llm_provider.lower().strip()
        if self._provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Proveedor LLM no soportado: {self._provider}")

        self._client: Any = None
        # Cada turno consta de 2 mensajes (user + assistant)
        self._history: deque[dict[str, str]] = deque(
            maxlen=self.settings.max_history_turns * 2
        )

    def process_turn(self, transcript: str, profile: SessionProfile) -> TutorResponse:
        """Procesa un turno del usuario y devuelve una respuesta validada."""
        clean_transcript = transcript.strip()
        if not clean_transcript:
            return self._build_contingency_response(transcript, profile)

        messages = self._build_messages(clean_transcript, profile)

        # Hasta 2 intentos (llamada inicial + 1 reintento en caso de fallo completo)
        for attempt in range(1, 3):
            try:
                raw_output = self._call_llm(messages, profile)
                response = self._parse_with_degradation(
                    raw_output, clean_transcript, profile
                )
                self._history.append({"role": "user", "content": clean_transcript})
                self._history.append(
                    {"role": "assistant", "content": response.oral_response}
                )
                return response
            except Exception as exc:
                logger.warning("Intento %d fallido en CognitiveOrchestrator: %s", attempt, exc)

        return self._build_contingency_response(clean_transcript, profile)

    def reset_history(self) -> None:
        """Limpia el buffer de memoria conversacional."""
        self._history.clear()

    def _get_client(self) -> Any:
        """Inicializa y reutiliza el cliente del proveedor LLM configurado."""
        if self._client is not None:
            return self._client

        if self._provider == "ollama":
            import ollama

            self._client = ollama.Client(host=self.settings.ollama_base_url)
        elif self._provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key)
        elif self._provider == "gemini":
            from google import genai

            self._client = genai.Client(api_key=self.settings.gemini_api_key)

        return self._client

    def _build_messages(
        self, transcript: str, profile: SessionProfile
    ) -> list[dict[str, str]]:
        """Construye la lista de mensajes con el System Prompt y la ventana deslizante."""
        return [
            {"role": "system", "content": build_system_prompt(profile)},
            *list(self._history),
            {"role": "user", "content": transcript},
        ]

    def _call_llm(
        self, messages: list[dict[str, str]], profile: SessionProfile
    ) -> str:
        """Invoca al proveedor LLM configurado exigiendo salida JSON."""
        if self._provider == "ollama":
            res = self._get_client().chat(
                model=self.settings.ollama_model,
                messages=messages,
                format=TutorResponse.model_json_schema(),
            )
            return res["message"]["content"]

        if self._provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            res = self._get_client().chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
            )
            return res.choices[0].message.content or ""

        if self._provider == "gemini":
            from google.genai import types

            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            system_instruction = messages[0]["content"]
            contents = [
                f"{m['role'].upper()}: {m['content']}" for m in messages[1:]
            ]
            res = self._get_client().models.generate_content(
                model=model,
                contents="\n".join(contents),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=TutorResponse,
                ),
            )
            return res.text or ""

        default_rate = self._default_rate(profile)
        return json.dumps(
            {
                "transcript": messages[-1]["content"],
                "oral_response": "That sounds great! Could you tell me a bit more?",
                "visual_card": {
                    "correction": "",
                    "category": "ok",
                    "explanation": "",
                },
                "prosody_flags": {"rate": default_rate, "pitch": 0.0},
            }
        )

    @staticmethod
    def _clean_json(raw_output: str) -> str:
        """Elimina posibles bloques Markdown (```json ... ```) alrededor del JSON."""
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _default_rate(profile: SessionProfile) -> float:
        """Obtiene la velocidad de habla por defecto según el nivel MCER."""
        return LEVEL_PARAMS.get(profile.level.value, {}).get("speech_rate", 1.0)

    def _parse_with_degradation(
        self, raw_output: str, transcript: str, profile: SessionProfile
    ) -> TutorResponse:
        """Valida la salida JSON preservando oral_response si falla visual_card."""
        data: dict[str, Any] = json.loads(self._clean_json(raw_output))
        if not isinstance(data, dict):
            raise ValueError("El JSON devuelto no es un objeto.")

        oral = data.get("oral_response")
        if not isinstance(oral, str) or not oral.strip():
            raise ValueError("oral_response ausente o vacío.")

        try:
            visual_card = VisualCard.model_validate(data.get("visual_card") or {})
        except ValidationError:
            visual_card = VisualCard()

        default_rate = self._default_rate(profile)
        try:
            prosody_flags = ProsodyFlags.model_validate(
                data.get("prosody_flags") or {"rate": default_rate, "pitch": 0.0}
            )
        except ValidationError:
            prosody_flags = ProsodyFlags(rate=default_rate, pitch=0.0)

        return TutorResponse(
            transcript=transcript,
            oral_response=oral.strip(),
            visual_card=visual_card,
            prosody_flags=prosody_flags,
        )

    def _build_contingency_response(
        self, transcript: str, profile: SessionProfile
    ) -> TutorResponse:
        """Construye una respuesta acústica de contingencia cuando falla todo tras reintento."""
        return TutorResponse(
            transcript=transcript,
            oral_response=CONTINGENCY_ORAL_RESPONSE,
            visual_card=VisualCard(),
            prosody_flags=ProsodyFlags(rate=self._default_rate(profile), pitch=0.0),
        )