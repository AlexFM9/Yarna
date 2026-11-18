"""Máquina de estados finitos (IDLE, LISTENING, PROCESSING, SPEAKING)."""

from __future__ import annotations

from enum import Enum


class SystemState(str, Enum):
    """Estados operativos del ciclo de interacción por voz."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class StateMachine:
    """Controla las transiciones de estado y el bloqueo de micrófono (Input Lock)."""

    def __init__(self, initial_state: SystemState = SystemState.IDLE) -> None:
        self._state = initial_state

    @property
    def state(self) -> SystemState:
        """Estado actual del sistema."""
        return self._state

    @property
    def is_input_locked(self) -> bool:
        """Indica si la entrada de micrófono está bloqueada (Input Lock en SPEAKING)."""
        # TODO: Devolver True durante SPEAKING (y/o PROCESSING) para evitar eco acústico.
        raise NotImplementedError("Pendiente de implementar is_input_locked.")

    def start_session(self) -> SystemState:
        """Transición IDLE -> LISTENING al iniciar la sesión."""
        # TODO: Validar y ejecutar transición a LISTENING.
        raise NotImplementedError("Pendiente de implementar start_session.")

    def on_speech_ended(self) -> SystemState:
        """Transición LISTENING -> PROCESSING al detectar fin de voz en el VAD."""
        # TODO: Validar y ejecutar transición a PROCESSING.
        raise NotImplementedError("Pendiente de implementar on_speech_ended.")

    def on_response_ready(self) -> SystemState:
        """Transición PROCESSING -> SPEAKING activando Input Lock."""
        # TODO: Validar y ejecutar transición a SPEAKING.
        raise NotImplementedError("Pendiente de implementar on_response_ready.")

    def on_playback_finished(self) -> SystemState:
        """Transición SPEAKING -> LISTENING al terminar el audio TTS."""
        # TODO: Validar y ejecutar transición a LISTENING.
        raise NotImplementedError("Pendiente de implementar on_playback_finished.")

    def stop_session(self) -> SystemState:
        """Detiene la sesión volviendo a IDLE."""
        # TODO: Ejecutar transición segura a IDLE.
        raise NotImplementedError("Pendiente de implementar stop_session.")
