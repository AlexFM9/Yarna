"""Orquestador cognitivo: memoria deslizante, llamada LLM y degradación elegante."""

from __future__ import annotations

from src.config import Settings
from src.core.schemas import SessionProfile, TutorResponse


class CognitiveOrchestrator:
    """Coordina historial conversacional, validación Pydantic y fallbacks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        # TODO: Inicializar el buffer deslizante de memoria conversacional (últimos N turnos)
        # y configurar el cliente LLM (ollama / openai / gemini).

    def process_turn(self, transcript: str, profile: SessionProfile) -> TutorResponse:
        """Procesa un turno del usuario y devuelve una respuesta validada.

        TODO:
        1. Construir prompt con build_system_prompt(profile) + historial deslizante.
        2. Invocar al LLM con Structured Outputs / formato JSON.
        3. Validar el contrato TutorResponse con Pydantic.
        4. Aplicar degradación elegante: si falla visual_card, preservar oral_response;
           si falla todo tras un reintento, emitir respuesta acústica de contingencia.
        """
        raise NotImplementedError("Pendiente de implementar process_turn.")

    def reset_history(self) -> None:
        """Limpia el buffer de memoria conversacional."""
        # TODO: Vaciar la ventana deslizante de turnos.
        raise NotImplementedError("Pendiente de implementar reset_history.")
