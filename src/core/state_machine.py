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
        """Indica si la entrada de micrófono está bloqueada (PROCESSING y SPEAKING)."""
        return self._state in (SystemState.PROCESSING, SystemState.SPEAKING)

    def _transition(self, expected: SystemState, target: SystemState) -> SystemState:
        if self._state != expected:
            raise ValueError(
                f"Transición inválida: no se puede pasar a {target.value} desde {self._state.value} "
                f"(se esperaba {expected.value})."
            )
        self._state = target
        return self._state

    def start_session(self) -> SystemState:
        """Transición IDLE -> LISTENING al iniciar la sesión."""
        return self._transition(SystemState.IDLE, SystemState.LISTENING)

    def on_speech_ended(self) -> SystemState:
        """Transición LISTENING -> PROCESSING al detectar fin de voz en el VAD."""
        return self._transition(SystemState.LISTENING, SystemState.PROCESSING)

    def on_response_ready(self) -> SystemState:
        """Transición PROCESSING -> SPEAKING activando Input Lock."""
        return self._transition(SystemState.PROCESSING, SystemState.SPEAKING)

    def on_playback_finished(self) -> SystemState:
        """Transición SPEAKING -> LISTENING al terminar el audio TTS."""
        return self._transition(SystemState.SPEAKING, SystemState.LISTENING)

    def stop_session(self) -> SystemState:
        """Detiene la sesión volviendo a IDLE."""
        self._state = SystemState.IDLE
        return self._state