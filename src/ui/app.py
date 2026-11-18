"""Capa 1: Interfaz web Gradio de doble canal e integración End-to-End."""

from __future__ import annotations

import time
from typing import Any

import gradio as gr
import numpy as np

from src.config import DEFAULT_SCENARIOS, LEVEL_PARAMS, SAMPLE_RATE, Settings
from src.core.schemas import (
    CEFRLevel,
    CorrectionCategory,
    ProsodyFlags,
    SessionProfile,
    SocialRegister,
    TutorResponse,
    VisualCard,
)
from src.core.state_machine import StateMachine, SystemState


class UISessionState:
    """Estado aislado por usuario en Gradio."""

    def __init__(self) -> None:
        self.fsm: StateMachine = StateMachine()
        self.profile: SessionProfile = SessionProfile()
        self.history: list[tuple[str, str]] = []
        self.last_activity_ts: float = time.monotonic()
        # Instancias lazy de los módulos de Capa 2 y Capa 3 (cuando estén implementados)
        self.orchestrator: Any = None
        self.vad: Any = None
        self.stt: Any = None
        self.tts: Any = None


def _ensure_audio_16k_mono(audio_tuple: tuple[int, np.ndarray] | None) -> np.ndarray | None:
    """Convierte el audio de Gradio a PCM mono 16 kHz float32 en [-1.0, 1.0]."""
    if audio_tuple is None:
        return None
    sr, data = audio_tuple
    if data is None or len(data) == 0:
        return None

    audio = np.asarray(data)
    # Convertir a mono si es estéreo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Normalizar enteros PCM (int16/int32) a float32 [-1.0, 1.0]
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

    # Remuestreo simple si el navegador envía 44.1kHz o 48kHz en lugar de 16kHz
    if sr != SAMPLE_RATE and len(audio) > 0:
        target_len = int(round(len(audio) * SAMPLE_RATE / sr))
        old_idx = np.linspace(0, len(audio) - 1, num=len(audio))
        new_idx = np.linspace(0, len(audio) - 1, num=target_len)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def _format_visual_card(card: VisualCard) -> str:
    """Formatea la tarjeta visual asíncrona para el panel lateral."""
    badges = {
        CorrectionCategory.OK: "✅ **¡Buen trabajo! (Adecuado)**",
        CorrectionCategory.GRAMMAR: "📝 **Sugerencia Gramatical**",
        CorrectionCategory.PRAGMATICS: "🎭 **Matiz Pragmático / Registro**",
    }
    header = badges.get(card.category, "💡 **Feedback**")
    correction_block = f"> **{card.correction}**\n\n" if card.correction else ""
    explanation_block = card.explanation or "_Sin observaciones adicionales en este turno._"
    return f"### {header}\n\n{correction_block}{explanation_block}"


def _format_transcript(history: list[tuple[str, str]]) -> str:
    """Renderiza el historial reciente de la conversación."""
    if not history:
        return "_La transcripción aparecerá aquí cuando inicies la conversación._"
    lines = []
    for speaker, text in history[-6:]:
        prefix = "🧑 **Tú:**" if speaker == "user" else "🤖 **YARNA:**"
        lines.append(f"{prefix} {text}")
    return "\n\n".join(lines)


def _safe_fsm_transition(fsm: StateMachine, action: str) -> SystemState:
    """Ejecuta transiciones de StateMachine tolerando que aún tenga NotImplementedError."""
    try:
        return getattr(fsm, action)()
    except NotImplementedError:
        fallback_map = {
            "start_session": SystemState.LISTENING,
            "on_speech_ended": SystemState.PROCESSING,
            "on_response_ready": SystemState.SPEAKING,
            "on_playback_finished": SystemState.LISTENING,
            "stop_session": SystemState.IDLE,
        }
        fsm._state = fallback_map.get(action, SystemState.IDLE)
        return fsm.state


def _is_locked(fsm: StateMachine) -> bool:
    """Consulta is_input_locked con respaldo mientras Bloque 2 termina su implementación."""
    try:
        return fsm.is_input_locked
    except NotImplementedError:
        return fsm.state in (SystemState.PROCESSING, SystemState.SPEAKING)


def _run_pipeline_turn(
    session: UISessionState,
    audio_16k: np.ndarray,
) -> tuple[TutorResponse, tuple[int, np.ndarray] | None]:
    """Conecta [VAD -> STT -> Orchestrator -> TTS] con fallback si aún no están implementados."""
    settings = Settings()
    level_cfg = LEVEL_PARAMS[session.profile.level.value]

    try:
        from src.audio.stt import FasterWhisperSTT
        from src.audio.tts import PiperTTSEngine
        from src.audio.vad import VoiceActivityDetector
        from src.core.orchestrator import CognitiveOrchestrator

        if session.vad is None:
            try:
                session.vad = VoiceActivityDetector(level=session.profile.level.value)
            except TypeError:
                session.vad = VoiceActivityDetector()
        if session.stt is None:
            try:
                session.stt = FasterWhisperSTT(
                    model_size=settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
            except TypeError:
                session.stt = FasterWhisperSTT()
        if session.orchestrator is None:
            try:
                session.orchestrator = CognitiveOrchestrator(settings=settings)
            except TypeError:
                session.orchestrator = CognitiveOrchestrator()
        if session.tts is None:
            try:
                session.tts = PiperTTSEngine(model_path=settings.piper_model_path)
            except TypeError:
                session.tts = PiperTTSEngine()

        # 1. VAD + STT
        _ = session.vad.process_chunk(audio_16k)
        stt_result = session.stt.transcribe(audio_16k)
        user_text = stt_result["text"] if isinstance(stt_result, dict) else str(stt_result)

        # 2. Orquestador Cognitivo (LLM + Pydantic)
        tutor_resp: TutorResponse = session.orchestrator.process_turn(
            user_text,
            session.profile,
        )

        # 3. Síntesis TTS con modulación prosódica
        tts_audio = session.tts.synthesize(
            tutor_resp.oral_response,
            tutor_resp.prosody_flags.rate,
        )
        return tutor_resp, (SAMPLE_RATE, tts_audio)

    except (ImportError, NotImplementedError, AttributeError, TypeError):
        # Respuesta simulada mientras los Bloques 1 y 2 completan sus clases
        duration_sec = round(len(audio_16k) / SAMPLE_RATE, 2)
        simulated = TutorResponse(
            transcript=f"[Audio capturado: {duration_sec}s a 16 kHz]",
            oral_response=(
                f"Great! We are practicing '{session.profile.scenario}' "
                f"at level {session.profile.level.value}."
            ),
            visual_card=VisualCard(
                correction="Everything sounds clear so far!",
                category=CorrectionCategory.OK,
                explanation=(
                    "Modo de integración UI activo. En cuanto los módulos de audio "
                    "y el orquestador estén fusionados, aquí verás tus correcciones reales."
                ),
            ),
            prosody_flags=ProsodyFlags(rate=level_cfg["speech_rate"], pitch=0.0),
        )
        return simulated, None


def start_session_handler(
    level: str,
    register_mode: str,
    scenario: str,
    session: UISessionState,
):
    """Inicia la sesión de práctica oral y configura el perfil."""
    session.profile = SessionProfile(
        level=CEFRLevel(level),
        register_mode=SocialRegister(register_mode),
        scenario=scenario or DEFAULT_SCENARIOS[0],
    )
    session.history.clear()
    session.last_activity_ts = time.monotonic()
    state = _safe_fsm_transition(session.fsm, "start_session")
    timeout = LEVEL_PARAMS[level]["silence_timeout_sec"]

    return (
        session,
        f"**Estado:** `{state.value}` | **Timeout silencio:** `{timeout}s`",
        gr.update(interactive=True),
        gr.update(interactive=True),
        _format_transcript(session.history),
        "_Escuchando... Graba tu turno y pulsa **Enviar Turno**._",
    )


def stop_session_handler(session: UISessionState):
    """Detiene la sesión y vuelve a IDLE."""
    state = _safe_fsm_transition(session.fsm, "stop_session")
    return (
        session,
        f"**Estado:** `{state.value}`",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def process_turn_handler(
    audio_input: tuple[int, np.ndarray] | None,
    level: str,
    register_mode: str,
    scenario: str,
    session: UISessionState,
):
    """Ejecuta un turno completo respetando Input Lock y actualización de doble canal."""
    if _is_locked(session.fsm):
        return (
            session,
            f"**Estado:** `{session.fsm.state.value}` (Micrófono bloqueado por Input Lock)",
            None,
            _format_transcript(session.history),
            gr.update(),
            gr.update(interactive=False),
        )

    session.profile = SessionProfile(
        level=CEFRLevel(level),
        register_mode=SocialRegister(register_mode),
        scenario=scenario or DEFAULT_SCENARIOS[0],
    )

    audio_16k = _ensure_audio_16k_mono(audio_input)
    if audio_16k is None:
        return (
            session,
            f"**Estado:** `{session.fsm.state.value}` — No se detectó audio en el micrófono.",
            None,
            _format_transcript(session.history),
            gr.update(),
            gr.update(interactive=True),
        )

    # LISTENING -> PROCESSING
    _safe_fsm_transition(session.fsm, "on_speech_ended")
    tutor_response, tts_audio = _run_pipeline_turn(session, audio_16k)

    # PROCESSING -> SPEAKING (Input Lock activo)
    speaking_state = _safe_fsm_transition(session.fsm, "on_response_ready")
    session.history.append(("user", tutor_response.transcript))
    session.history.append(("tutor", tutor_response.oral_response))
    session.last_activity_ts = time.monotonic()

    return (
        session,
        f"**Estado:** `{speaking_state.value}` (Reproduciendo respuesta...)",
        tts_audio,
        _format_transcript(session.history),
        _format_visual_card(tutor_response.visual_card),
        gr.update(interactive=False),
    )


def on_audio_playback_end(session: UISessionState):
    """Desbloquea el micrófono (SPEAKING -> LISTENING) al finalizar el audio TTS."""
    state = _safe_fsm_transition(session.fsm, "on_playback_finished")
    session.last_activity_ts = time.monotonic()
    return (
        session,
        f"**Estado:** `{state.value}` (Tu turno)",
        gr.update(value=None, interactive=True),
    )


def check_silence_timeout(session: UISessionState):
    """Comprueba el timeout de silencio en LISTENING y lanza mensaje de reenganche."""
    if session.fsm.state != SystemState.LISTENING:
        return session, gr.update(), gr.update()

    timeout = LEVEL_PARAMS[session.profile.level.value]["silence_timeout_sec"]
    if time.monotonic() - session.last_activity_ts < timeout:
        return session, gr.update(), gr.update()

    session.last_activity_ts = time.monotonic()
    try:
        from src.core.prompts import get_reengagement_prompt

        prompt_text = get_reengagement_prompt(session.profile)
    except (ImportError, NotImplementedError):
        prompt_text = "Are you still there? Take your time and tell me what you think."

    session.history.append(("tutor", prompt_text))
    return (
        session,
        f"**Estado:** `{session.fsm.state.value}` (Reenganche por silencio > {timeout}s)",
        _format_transcript(session.history),
    )


def build_app() -> gr.Blocks:
    """Construye la interfaz web Gradio de doble canal de YARNA."""
    with gr.Blocks(title="YARNA — Tutor Conversacional Adaptativo") as demo:
        session_state = gr.State(UISessionState)

        gr.Markdown(
            "# YARNA — Tutor Conversacional de Doble Canal\n"
            "Práctica oral adaptativa en inglés con feedback en doble canal."
        )

        # Selectores de perfil: [Nivel MCER, Registro Social, Escenario]
        with gr.Row():
            level_selector = gr.Dropdown(
                choices=[CEFRLevel.A2.value, CEFRLevel.B2_C1.value],
                value=CEFRLevel.A2.value,
                label="Nivel MCER",
            )
            register_selector = gr.Dropdown(
                choices=[
                    SocialRegister.FORMAL.value,
                    SocialRegister.INFORMAL.value,
                    SocialRegister.EVERYDAY.value,
                ],
                value=SocialRegister.EVERYDAY.value,
                label="Registro Pragmático",
            )
            scenario_selector = gr.Dropdown(
                choices=DEFAULT_SCENARIOS,
                value=DEFAULT_SCENARIOS[0],
                allow_custom_value=True,
                label="Escenario (Roleplay)",
            )

        with gr.Row():
            start_btn = gr.Button("Iniciar Sesión", variant="primary")
            stop_btn = gr.Button("Detener Sesión", variant="stop")
            status_banner = gr.Markdown("**Estado:** `IDLE`")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Canal Auditivo (Síncrono)")
                mic_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Micrófono / Audio de prueba (PCM 16 kHz)",
                    interactive=False,
                )
                send_btn = gr.Button("Enviar Turno", variant="primary", interactive=False)
                tts_output = gr.Audio(
                    label="Respuesta Oral (Piper TTS)",
                    autoplay=True,
                    interactive=False,
                )
                transcript_box = gr.Markdown(
                    "_Pulsa **Iniciar Sesión** para comenzar el diálogo._"
                )

            with gr.Column(scale=2):
                gr.Markdown("## Canal Visual (Asíncrono / Persistente)")
                visual_card_box = gr.Markdown(
                    "_La tarjeta `visual_card` (gramática / pragmática) aparecerá aquí._"
                )

        # Eventos de control de sesión
        start_btn.click(
            fn=start_session_handler,
            inputs=[level_selector, register_selector, scenario_selector, session_state],
            outputs=[
                session_state,
                status_banner,
                mic_input,
                send_btn,
                transcript_box,
                visual_card_box,
            ],
        )

        stop_btn.click(
            fn=stop_session_handler,
            inputs=[session_state],
            outputs=[session_state, status_banner, mic_input, send_btn],
        )

        # Procesamiento de turno por voz
        send_btn.click(
            fn=process_turn_handler,
            inputs=[
                mic_input,
                level_selector,
                register_selector,
                scenario_selector,
                session_state,
            ],
            outputs=[
                session_state,
                status_banner,
                tts_output,
                transcript_box,
                visual_card_box,
                mic_input,
            ],
        )

        # Liberación de Input Lock al terminar la reproducción del tutor
        tts_output.stop(
            fn=on_audio_playback_end,
            inputs=[session_state],
            outputs=[session_state, status_banner, mic_input],
        )

        # Temporizador periódico para detectar silencio prolongado en LISTENING
        silence_timer = gr.Timer(value=1.0)
        silence_timer.tick(
            fn=check_silence_timeout,
            inputs=[session_state],
            outputs=[session_state, status_banner, transcript_box],
        )

    return demo


def main() -> None:
    """Punto de entrada para lanzar la interfaz Gradio."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()