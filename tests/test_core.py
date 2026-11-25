"""Tests unitarios del Bloque 2 (Capa 2: schemas, state_machine, prompts, orchestrator)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.core import (
    CEFRLevel,
    CognitiveOrchestrator,
    CorrectionCategory,
    ProsodyFlags,
    SessionProfile,
    SocialRegister,
    StateMachine,
    SystemState,
    TutorResponse,
    VisualCard,
    build_system_prompt,
    get_reengagement_prompt,
)
from src.core.orchestrator import CONTINGENCY_ORAL_RESPONSE


# --- 1. Schemas ---


def test_tutor_response_defaults_and_validation() -> None:
    resp = TutorResponse(oral_response="Hello there!")
    assert resp.transcript == ""
    assert resp.visual_card == VisualCard()
    assert resp.prosody_flags == ProsodyFlags()

    with pytest.raises(ValidationError):
        TutorResponse(oral_response="")

    with pytest.raises(ValidationError):
        ProsodyFlags(rate=2.0)


# --- 2. StateMachine ---


def test_state_machine_full_cycle_and_input_lock() -> None:
    sm = StateMachine()
    assert sm.state == SystemState.IDLE
    assert sm.is_input_locked is False

    assert sm.start_session() == SystemState.LISTENING
    assert sm.is_input_locked is False

    assert sm.on_speech_ended() == SystemState.PROCESSING
    assert sm.is_input_locked is True

    assert sm.on_response_ready() == SystemState.SPEAKING
    assert sm.is_input_locked is True

    assert sm.on_playback_finished() == SystemState.LISTENING
    assert sm.is_input_locked is False

    assert sm.stop_session() == SystemState.IDLE
    assert sm.is_input_locked is False


def test_state_machine_invalid_transitions_raise_value_error() -> None:
    sm = StateMachine()
    with pytest.raises(ValueError):
        sm.on_speech_ended()
    with pytest.raises(ValueError):
        sm.on_response_ready()
    with pytest.raises(ValueError):
        sm.on_playback_finished()

    sm.start_session()
    with pytest.raises(ValueError):
        sm.start_session()


# --- 3. Prompts ---


def test_build_system_prompt_includes_profile_params() -> None:
    profile = SessionProfile(
        level=CEFRLevel.B2_C1,
        register_mode=SocialRegister.FORMAL,
        scenario="Entrevista de trabajo",
    )
    prompt = build_system_prompt(profile)
    assert "Entrevista de trabajo" in prompt
    assert "B2/C1" in prompt
    assert "Formal" in prompt
    assert "1.05" in prompt


def test_reengagement_prompt_varies_by_level() -> None:
    p_a2 = get_reengagement_prompt(SessionProfile(level=CEFRLevel.A2))
    p_b2 = get_reengagement_prompt(SessionProfile(level=CEFRLevel.B2_C1))
    assert p_a2 != p_b2
    assert len(p_a2) > 0 and len(p_b2) > 0


# --- 4. CognitiveOrchestrator ---


def test_orchestrator_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError):
        CognitiveOrchestrator(Settings(llm_provider="desconocido"))


def test_orchestrator_mock_turn_and_history_window() -> None:
    orch = CognitiveOrchestrator(Settings(llm_provider="mock", max_history_turns=2))
    profile = SessionProfile(level=CEFRLevel.A2)

    for i in range(3):
        resp = orch.process_turn(f"Hello turn {i}", profile)
        assert resp.transcript == f"Hello turn {i}"
        assert resp.prosody_flags.rate == 0.8

    # max_history_turns=2 -> 4 mensajes (2 user + 2 assistant)
    assert len(orch._history) == 4
    assert orch._history[0]["content"] == "Hello turn 1"

    orch.reset_history()
    assert len(orch._history) == 0


def test_orchestrator_parses_markdown_json_and_preserves_exact_transcript() -> None:
    orch = CognitiveOrchestrator(Settings(llm_provider="mock"))
    profile = SessionProfile(level=CEFRLevel.B2_C1)
    wrapped = (
        "```json\n"
        + json.dumps(
            {
                "transcript": "texto alterado por el llm",
                "oral_response": "Sure, I'd be happy to help you with that.",
                "visual_card": {
                    "correction": "Could you help me?",
                    "category": "pragmatics",
                    "explanation": "Usa una forma más cortés.",
                },
                "prosody_flags": {"rate": 1.05, "pitch": 0.0},
            }
        )
        + "\n```"
    )

    with patch.object(orch, "_call_llm", return_value=wrapped):
        resp = orch.process_turn("Help me now", profile)

    assert resp.transcript == "Help me now"
    assert resp.oral_response == "Sure, I'd be happy to help you with that."
    assert resp.visual_card.category == CorrectionCategory.PRAGMATICS


def test_orchestrator_graceful_degradation_on_broken_visual_card() -> None:
    orch = CognitiveOrchestrator(Settings(llm_provider="mock"))
    profile = SessionProfile(level=CEFRLevel.A2)
    broken_card_json = json.dumps(
        {
            "oral_response": "I went to the park yesterday too!",
            "visual_card": {"category": "categoria_invalida"},
            "prosody_flags": {"rate": 99.0},
        }
    )

    with patch.object(orch, "_call_llm", return_value=broken_card_json):
        resp = orch.process_turn("I goed to the park", profile)

    assert resp.oral_response == "I went to the park yesterday too!"
    assert resp.visual_card == VisualCard()
    assert resp.prosody_flags.rate == 0.8


def test_orchestrator_retries_once_then_returns_contingency() -> None:
    orch = CognitiveOrchestrator(Settings(llm_provider="mock"))
    profile = SessionProfile(level=CEFRLevel.A2)

    with patch.object(orch, "_call_llm", return_value="no es json") as mock_call:
        resp = orch.process_turn("Hello", profile)

    assert mock_call.call_count == 2
    assert resp.oral_response == CONTINGENCY_ORAL_RESPONSE
    assert resp.prosody_flags.rate == 0.8
    assert len(orch._history) == 0


def test_orchestrator_empty_transcript_skips_llm() -> None:
    orch = CognitiveOrchestrator(Settings(llm_provider="mock"))
    profile = SessionProfile()

    with patch.object(orch, "_call_llm") as mock_call:
        resp = orch.process_turn("   ", profile)

    mock_call.assert_not_called()
    assert resp.oral_response == CONTINGENCY_ORAL_RESPONSE