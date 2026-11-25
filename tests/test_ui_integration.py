"""Tests de integración entre Capa 1 (UI), Capa 2 (FSM estricta) y Capa 3 (Audio)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np


def _ensure_gradio_stub() -> None:
    """Inyecta un stub mínimo de `gradio` si el paquete no está instalado en el entorno."""
    if importlib.util.find_spec("gradio") is not None:
        return
    if "gradio" in sys.modules:
        return

    fake_gr = ModuleType("gradio")
    fake_gr.update = lambda **kwargs: {"__update__": True, **kwargs}
    fake_gr.themes = SimpleNamespace(Soft=lambda **kwargs: SimpleNamespace(**kwargs))
    fake_gr.Blocks = object
    sys.modules["gradio"] = fake_gr


_ensure_gradio_stub()

from src.audio.tts import PiperTTSEngine  # noqa: E402
from src.audio.vad import VoiceActivityDetector  # noqa: E402
from src.config import SAMPLE_RATE  # noqa: E402
from src.core.schemas import (  # noqa: E402
    CEFRLevel,
    CorrectionCategory,
    ProsodyFlags,
    SessionProfile,
    SocialRegister,
    TutorResponse,
    VisualCard,
)
from src.core.state_machine import StateMachine, SystemState  # noqa: E402
from src.ui import app as ui_app  # noqa: E402
from tests.eval_latency import run_latency_profile  # noqa: E402


class _StrictStateMachine(StateMachine):
    """Réplica exacta de la StateMachine estricta de la PR #2 para validar compatibilidad."""

    @property
    def is_input_locked(self) -> bool:
        return self._state in (SystemState.PROCESSING, SystemState.SPEAKING)

    def _transition(self, expected: SystemState, target: SystemState) -> SystemState:
        if self._state != expected:
            raise ValueError(
                f"Transición inválida: {self._state.value} -> {target.value}"
            )
        self._state = target
        return self._state

    def start_session(self) -> SystemState:
        return self._transition(SystemState.IDLE, SystemState.LISTENING)

    def on_speech_ended(self) -> SystemState:
        return self._transition(SystemState.LISTENING, SystemState.PROCESSING)

    def on_response_ready(self) -> SystemState:
        return self._transition(SystemState.PROCESSING, SystemState.SPEAKING)

    def on_playback_finished(self) -> SystemState:
        return self._transition(SystemState.SPEAKING, SystemState.LISTENING)

    def stop_session(self) -> SystemState:
        self._state = SystemState.IDLE
        return self._state


class TestUIAudioAndFSMIntegration(unittest.TestCase):
    """Valida los puntos de integración corregidos entre UI, Audio y Core."""

    @staticmethod
    def _build_session_with_strict_fsm() -> ui_app.UISessionState:
        session = ui_app.UISessionState()
        session.fsm = _StrictStateMachine()
        thread = session._create_thread(
            title="Pedir en una cafetería",
            folder="General",
            level_label="A2",
            register_mode=SocialRegister.EVERYDAY,
        )
        session.chats[thread.chat_id] = thread
        session.active_chat_id = thread.chat_id
        return session

    def test_run_pipeline_turn_uses_vad_trimming_and_unpacked_tts_tuple(self) -> None:
        """Comprueba que _run_pipeline_turn devuelve (sr, 1D ndarray) y recorta silencios."""
        session = self._build_session_with_strict_fsm()
        chat = session.active_chat
        assert chat is not None

        recorded_lengths: list[int] = []

        def fake_transcribe(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
            recorded_lengths.append(len(audio))
            return {"text": "I want a coffee"}

        session.vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, use_silero=False)
        session.stt = SimpleNamespace(transcribe=fake_transcribe)
        session.orchestrator = SimpleNamespace(
            process_turn=lambda text, profile: TutorResponse(
                transcript=text,
                oral_response="Sure, would you like milk with that?",
                visual_card=VisualCard(
                    correction="I would like a coffee.",
                    category=CorrectionCategory.PRAGMATICS,
                    explanation="Usa 'I would like' para sonar más cortés.",
                ),
                prosody_flags=ProsodyFlags(rate=0.80, pitch=0.0),
            )
        )
        tts = PiperTTSEngine(use_cuda=False, auto_download=False)
        tts.model_path = Path("missing.onnx")
        tts.config_path = Path("missing.onnx.json")
        session.tts = tts

        leading = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)
        t = np.linspace(0.0, 0.5, int(0.5 * SAMPLE_RATE), endpoint=False)
        speech = (0.4 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        trailing = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)
        full_audio = np.concatenate([leading, speech, trailing])

        resp, tts_audio = ui_app._run_pipeline_turn(session, chat, full_audio)
        self.assertEqual(resp.transcript, "I want a coffee")
        self.assertEqual(len(recorded_lengths), 1)
        self.assertLess(recorded_lengths[0], full_audio.size)

        self.assertIsNotNone(tts_audio)
        assert tts_audio is not None
        self.assertIsInstance(tts_audio, tuple)
        self.assertEqual(tts_audio[0], SAMPLE_RATE)
        self.assertIsInstance(tts_audio[1], np.ndarray)
        self.assertEqual(tts_audio[1].dtype, np.float32)

    def test_strict_state_machine_transitions_do_not_raise_in_ui_handlers(self) -> None:
        """Verifica que cambiar de chat o reanudar con sesión ya activa no lanza ValueError."""
        session = self._build_session_with_strict_fsm()
        ui_app._safe_fsm_transition(session.fsm, "start_session")
        self.assertEqual(session.fsm.state, SystemState.LISTENING)

        # Llamadas repetidas a switch_chat_handler / resume_session_handler estando en LISTENING
        chat = session.active_chat
        assert chat is not None
        ui_app.switch_chat_handler(chat.chat_id, session)
        self.assertEqual(session.fsm.state, SystemState.LISTENING)

        ui_app.resume_session_handler(session)
        self.assertEqual(session.fsm.state, SystemState.LISTENING)

        # Transición desde SPEAKING hacia nuevo chat mediante _safe_fsm_transition
        session.fsm.on_speech_ended()
        session.fsm.on_response_ready()
        self.assertEqual(session.fsm.state, SystemState.SPEAKING)
        ui_app.switch_chat_handler(chat.chat_id, session)
        self.assertEqual(session.fsm.state, SystemState.LISTENING)

    def test_process_turn_handler_releases_mic_when_tts_is_none_and_locks_with_audio(self) -> None:
        """Sin audio TTS el micrófono queda interactivo; con audio TTS queda bloqueado."""
        session = self._build_session_with_strict_fsm()
        ui_app._safe_fsm_transition(session.fsm, "start_session")

        sample_pcm = (
            SAMPLE_RATE,
            (0.2 * np.sin(2.0 * np.pi * 220.0 * np.linspace(0, 0.3, int(0.3 * SAMPLE_RATE)))).astype(
                np.float32
            ),
        )

        # Caso 1: sin audio TTS (simulación) -> debe volver a LISTENING e interactive=True
        orig_runner = ui_app._run_pipeline_turn
        try:
            ui_app._run_pipeline_turn = lambda sess, ch, aud: (
                TutorResponse(
                    transcript="Hello",
                    oral_response="Hi there!",
                    visual_card=VisualCard(category=CorrectionCategory.OK),
                    prosody_flags=ProsodyFlags(rate=0.8),
                ),
                None,
            )
            res_no_audio = ui_app.process_turn_handler(sample_pcm, session)
            self.assertEqual(session.fsm.state, SystemState.LISTENING)
            self.assertIsNone(res_no_audio[2])
            self.assertTrue(res_no_audio[6]["interactive"])

            # Caso 2: con audio TTS -> debe quedar en SPEAKING e interactive=False (Input Lock)
            ui_app._run_pipeline_turn = lambda sess, ch, aud: (
                TutorResponse(
                    transcript="Hello",
                    oral_response="Hi there!",
                    visual_card=VisualCard(category=CorrectionCategory.OK),
                    prosody_flags=ProsodyFlags(rate=0.8),
                ),
                (SAMPLE_RATE, np.zeros(1600, dtype=np.float32)),
            )
            res_with_audio = ui_app.process_turn_handler(sample_pcm, session)
            self.assertEqual(session.fsm.state, SystemState.SPEAKING)
            self.assertIsNotNone(res_with_audio[2])
            self.assertFalse(res_with_audio[6]["interactive"])
        finally:
            ui_app._run_pipeline_turn = orig_runner

    def test_eval_latency_profile_runs_cleanly(self) -> None:
        """Comprueba que eval_latency.py ejecuta sin errores con las firmas reales de Capa 3."""
        metrics = run_latency_profile(num_turns=2, audio_duration_sec=0.5)
        self.assertEqual(metrics["num_turns"], 2.0)
        self.assertIn("p95_latency_sec", metrics)
        self.assertIn("rtf", metrics)


if __name__ == "__main__":
    unittest.main()
