"""Capa 1: Interfaz web Gradio estilo WhatsApp con vista de configuración, carpetas y doble canal."""

from __future__ import annotations

import difflib
import html
import os
import time
import uuid
from dataclasses import dataclass, field
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


DEFAULT_FOLDERS: list[str] = ["General", "Trabajo y Estudios", "Viajes y Ocio"]

CUSTOM_CSS = """
*, *::before, *::after {
    box-sizing: border-box;
}

:root {
    --wa-bg-app: #efeae2;
    --wa-sidebar-bg: #ffffff;
    --wa-header-grad: linear-gradient(135deg, #075e54 0%, #128c7e 100%);
    --wa-chat-bg: #efeae2;
    --wa-bubble-user: #d9fdd3;
    --wa-bubble-tutor: #ffffff;
    --wa-text-main: #111b21;
    --wa-text-muted: #54656f;
    --wa-border: #d1d7db;
    --wa-panel-bg: #ffffff;
    --wa-err-grammar-bg: rgba(245, 158, 11, 0.28);
    --wa-err-grammar-line: #d97706;
    --wa-err-prag-bg: rgba(139, 92, 246, 0.25);
    --wa-err-prag-line: #7c3aed;
}

.dark {
    --wa-bg-app: #0b141a;
    --wa-sidebar-bg: #111b21;
    --wa-header-grad: linear-gradient(135deg, #111b21 0%, #202c33 100%);
    --wa-chat-bg: #0b141a;
    --wa-bubble-user: #005c4b;
    --wa-bubble-tutor: #202c33;
    --wa-text-main: #e9edef;
    --wa-text-muted: #8696a0;
    --wa-border: #2a3942;
    --wa-panel-bg: #111b21;
    --wa-err-grammar-bg: rgba(251, 191, 36, 0.25);
    --wa-err-grammar-line: #fbbf24;
    --wa-err-prag-bg: rgba(167, 139, 250, 0.25);
    --wa-err-prag-line: #a78bfa;
}

.gradio-container {
    max-width: 1400px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
    overflow-x: hidden !important;
    font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Asegurar que filas y columnas no desborden nunca */
.gradio-container .gr-row {
    flex-wrap: wrap !important;
    gap: 12px !important;
}
.gradio-container .gr-column {
    min-width: 0 !important;
}
.gradio-container button {
    width: 100% !important;
    max-width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.wa-top-banner {
    background: var(--wa-header-grad);
    color: #ffffff;
    padding: 16px 22px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.14);
    width: 100%;
}
.wa-top-banner h1 {
    margin: 0 0 4px 0 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.wa-top-banner p {
    margin: 0 !important;
    font-size: 0.9rem !important;
    color: #d1fae5 !important;
}

.setup-card {
    max-width: 780px !important;
    width: 100% !important;
    margin: 10px auto !important;
    background: var(--wa-panel-bg) !important;
    border: 1px solid var(--wa-border) !important;
    border-radius: 18px !important;
    padding: 24px 26px !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08) !important;
}

.wa-column {
    background: var(--wa-panel-bg) !important;
    border: 1px solid var(--wa-border) !important;
    border-radius: 14px !important;
    padding: 14px !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

.wa-chat-window {
    background-color: var(--wa-chat-bg);
    background-image: radial-gradient(var(--wa-border) 0.75px, transparent 0.75px);
    background-size: 18px 18px;
    border: 1px solid var(--wa-border);
    border-radius: 14px;
    padding: 14px;
    height: 360px;
    width: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.wa-msg-row {
    display: flex;
    width: 100%;
}
.wa-msg-row.user {
    justify-content: flex-end;
}
.wa-msg-row.tutor {
    justify-content: flex-start;
}

.wa-bubble {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.94rem;
    line-height: 1.45;
    color: var(--wa-text-main);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.wa-bubble.user {
    background: var(--wa-bubble-user);
    border-top-right-radius: 2px;
}
.wa-bubble.tutor {
    background: var(--wa-bubble-tutor);
    border-top-left-radius: 2px;
}

.wa-sender {
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 4px;
    color: var(--wa-text-muted);
    display: flex;
    justify-content: space-between;
    gap: 10px;
}

.wa-underline-grammar {
    background: var(--wa-err-grammar-bg);
    border-bottom: 2.5px wavy var(--wa-err-grammar-line);
    padding: 1px 4px;
    border-radius: 4px;
    font-weight: 600;
}
.wa-underline-pragmatics {
    background: var(--wa-err-prag-bg);
    border-bottom: 2.5px wavy var(--wa-err-prag-line);
    padding: 1px 4px;
    border-radius: 4px;
    font-weight: 600;
}

.wa-card-link {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed var(--wa-border);
    font-size: 0.8rem;
    font-weight: 600;
}
.wa-card-link.grammar {
    color: var(--wa-err-grammar-line);
}
.wa-card-link.pragmatics {
    color: var(--wa-err-prag-line);
}
.wa-card-link.ok {
    color: #10b981;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.84rem;
    border: 1px solid var(--wa-border);
    background: var(--wa-panel-bg);
    color: var(--wa-text-main);
    width: 100%;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.status-pill span {
    overflow: hidden;
    text-overflow: ellipsis;
}
.status-listening {
    background: #ecfdf5;
    border-color: #10b981;
    color: #065f46;
}
.dark .status-listening {
    background: rgba(6, 95, 70, 0.35);
    color: #a7f3d0;
}
.status-processing {
    background: #fffbeb;
    border-color: #f59e0b;
    color: #92400e;
}
.dark .status-processing {
    background: rgba(146, 64, 14, 0.35);
    color: #fde68a;
}
.status-speaking {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1e40af;
}
.dark .status-speaking {
    background: rgba(30, 64, 175, 0.35);
    color: #bfdbfe;
}

.error-cards-feed {
    height: 510px;
    width: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding-right: 2px;
}
.err-card {
    background: var(--wa-panel-bg);
    border: 1px solid var(--wa-border);
    border-radius: 12px;
    padding: 13px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.err-card.grammar {
    border-left: 5px solid var(--wa-err-grammar-line);
}
.err-card.pragmatics {
    border-left: 5px solid var(--wa-err-prag-line);
}
.err-card.ok {
    border-left: 5px solid #10b981;
}
.err-card-header {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}
.err-card-quote {
    font-size: 0.86rem;
    color: var(--wa-text-muted);
    margin-bottom: 6px;
}
.err-card-fix {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--wa-text-main);
    background: rgba(16, 185, 129, 0.12);
    padding: 6px 10px;
    border-radius: 8px;
    margin-bottom: 6px;
}
.err-card-exp {
    font-size: 0.86rem;
    color: var(--wa-text-main);
    line-height: 1.4;
}
"""

THEME_TOGGLE_JS = """
(theme_choice) => {
    const root = document.documentElement;
    const body = document.body;
    const app = document.querySelector('gradio-app');
    const isDark = theme_choice === '🌙 Oscuro';

    [root, body, app].forEach(el => {
        if (!el) return;
        if (isDark) {
            el.classList.add('dark');
        } else {
            el.classList.remove('dark');
        }
    });

    const url = new URL(window.location.href);
    url.searchParams.set('__theme', isDark ? 'dark' : 'light');
    window.history.replaceState({}, '', url);
    return theme_choice;
}
"""


@dataclass
class ChatMessage:
    """Burbuja individual dentro de un chat."""

    sender: str  # "user" | "tutor"
    text: str
    timestamp: str
    card: VisualCard | None = None
    card_index: int | None = None


@dataclass
class ChatThread:
    """Conversación independiente organizada dentro de una carpeta."""

    chat_id: str
    title: str
    folder: str
    level_label: str
    profile: SessionProfile
    messages: list[ChatMessage] = field(default_factory=list)
    error_cards: list[tuple[int, str, VisualCard]] = field(default_factory=list)


class UISessionState:
    """Estado aislado por usuario con carpetas y múltiples chats."""

    def __init__(self) -> None:
        self.fsm: StateMachine = StateMachine()
        self.last_activity_ts: float = time.monotonic()
        self.orchestrator: Any = None
        self.vad: Any = None
        self.stt: Any = None
        self.tts: Any = None

        self.folders: list[str] = list(DEFAULT_FOLDERS)
        self.selected_folder_filter: str = "Todas"
        self.chats: dict[str, ChatThread] = {}
        self.active_chat_id: str | None = None

    @staticmethod
    def _create_thread(
        title: str,
        folder: str,
        level_label: str,
        register_mode: SocialRegister,
    ) -> ChatThread:
        cefr_enum = _parse_cefr_level(level_label)
        return ChatThread(
            chat_id=str(uuid.uuid4())[:8],
            title=title,
            folder=folder,
            level_label=level_label,
            profile=SessionProfile(
                level=cefr_enum,
                register_mode=register_mode,
                scenario=title,
            ),
        )

    @property
    def active_chat(self) -> ChatThread | None:
        if self.active_chat_id and self.active_chat_id in self.chats:
            return self.chats[self.active_chat_id]
        return None

    def folder_filter_choices(self) -> list[str]:
        return ["Todas"] + self.folders

    def chat_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for cid, chat in self.chats.items():
            if (
                self.selected_folder_filter != "Todas"
                and chat.folder != self.selected_folder_filter
            ):
                continue
            err_count = sum(
                1 for _, _, c in chat.error_cards if c.category != CorrectionCategory.OK
            )
            badge = f" ({err_count} corr.)" if err_count > 0 else ""
            label = f"📁 {chat.folder} · {chat.title} [{chat.level_label}]{badge}"
            choices.append((label, cid))
        return choices


def _parse_cefr_level(level_str: str) -> CEFRLevel:
    """Convierte la selección de la UI ('A2', 'B2', 'C1') al enum interno CEFRLevel."""
    if level_str in ("B2", "C1", CEFRLevel.B2_C1.value):
        return CEFRLevel.B2_C1
    return CEFRLevel.A2


def _highlight_user_mistakes(user_text: str, card: VisualCard | None) -> str:
    """Subraya en color las palabras donde difiere la frase del usuario respecto a la corrección."""
    safe_text = html.escape(user_text)
    if card is None or card.category == CorrectionCategory.OK or not card.correction:
        return safe_text

    css_class = (
        "wa-underline-grammar"
        if card.category == CorrectionCategory.GRAMMAR
        else "wa-underline-pragmatics"
    )

    user_words = user_text.split()
    corr_words = card.correction.split()
    matcher = difflib.SequenceMatcher(
        None,
        [w.lower().strip(".,!?") for w in user_words],
        [w.lower().strip(".,!?") for w in corr_words],
    )

    highlighted_tokens: list[str] = []
    has_diff = False
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        segment = html.escape(" ".join(user_words[i1:i2]))
        if not segment:
            continue
        if tag == "equal":
            highlighted_tokens.append(segment)
        else:
            has_diff = True
            highlighted_tokens.append(f"<span class='{css_class}'>{segment}</span>")

    if not has_diff:
        return f"<span class='{css_class}'>{safe_text}</span>"
    return " ".join(highlighted_tokens)


def _render_chat_window(chat: ChatThread | None) -> str:
    """Renderiza las burbujas de conversación estilo WhatsApp con errores subrayados."""
    if chat is None:
        return "<div class='wa-chat-window'></div>"

    if not chat.messages:
        return (
            "<div class='wa-chat-window'>"
            "<div style='margin:auto; text-align:center; color:var(--wa-text-muted); max-width:360px;'>"
            f"<b>📁 {html.escape(chat.folder)} · {html.escape(chat.title)}</b><br/>"
            f"<small>Nivel {html.escape(chat.level_label)} · Registro {html.escape(chat.profile.register_mode.value)}</small><br/><br/>"
            "El micrófono ya está listo. Graba tu primera intervención y pulsa <b>Enviar ➤</b>."
            "</div></div>"
        )

    bubbles_html: list[str] = ["<div class='wa-chat-window'>"]
    for msg in chat.messages:
        if msg.sender == "user":
            body_html = _highlight_user_mistakes(msg.text, msg.card)
            footer_html = ""
            if msg.card is not None:
                if msg.card.category == CorrectionCategory.GRAMMAR:
                    footer_html = (
                        f"<div class='wa-card-link grammar'>"
                        f"🟠 Error gramatical subrayado → Tarjeta #{msg.card_index}"
                        f"</div>"
                    )
                elif msg.card.category == CorrectionCategory.PRAGMATICS:
                    footer_html = (
                        f"<div class='wa-card-link pragmatics'>"
                        f"🟣 Matiz de registro subrayado → Tarjeta #{msg.card_index}"
                        f"</div>"
                    )
                else:
                    footer_html = "<div class='wa-card-link ok'>✅ Expresión natural</div>"

            bubbles_html.append(
                f"<div class='wa-msg-row user'>"
                f"<div class='wa-bubble user'>"
                f"<div class='wa-sender'><span>Tú</span><span>{msg.timestamp}</span></div>"
                f"<div>{body_html}</div>"
                f"{footer_html}"
                f"</div></div>"
            )
        else:
            bubbles_html.append(
                f"<div class='wa-msg-row tutor'>"
                f"<div class='wa-bubble tutor'>"
                f"<div class='wa-sender'><span>🤖 YARNA Tutor</span><span>{msg.timestamp}</span></div>"
                f"<div>{html.escape(msg.text)}</div>"
                f"</div></div>"
            )

    bubbles_html.append("</div>")
    return "".join(bubbles_html)


def _render_error_cards(chat: ChatThread | None) -> str:
    """Genera las tarjetas visuales acumuladas a partir de los errores subrayados del chat."""
    if chat is None or not chat.error_cards:
        return (
            "<div class='error-cards-feed'>"
            "<div style='color:var(--wa-text-muted); padding:16px; text-align:center;'>"
            "Las partes subrayadas en tus mensajes generarán aquí sus tarjetas de corrección."
            "</div></div>"
        )

    cards_html: list[str] = ["<div class='error-cards-feed'>"]
    for idx, original_text, card in reversed(chat.error_cards):
        if card.category == CorrectionCategory.GRAMMAR:
            kind_cls = "grammar"
            badge = f"🟠 Tarjeta #{idx} · Gramática"
        elif card.category == CorrectionCategory.PRAGMATICS:
            kind_cls = "pragmatics"
            badge = f"🟣 Tarjeta #{idx} · Pragmática / Registro"
        else:
            kind_cls = "ok"
            badge = f"✅ Tarjeta #{idx} · Adecuado"

        fix_html = (
            f"<div class='err-card-fix'>✨ {html.escape(card.correction)}</div>"
            if card.correction
            else ""
        )
        exp_text = card.explanation or "Expresión correcta para el contexto."

        cards_html.append(
            f"<div class='err-card {kind_cls}'>"
            f"<div class='err-card-header'>{badge}</div>"
            f"<div class='err-card-quote'>Dijiste: <i>\"{html.escape(original_text)}\"</i></div>"
            f"{fix_html}"
            f"<div class='err-card-exp'>{html.escape(exp_text)}</div>"
            f"</div>"
        )

    cards_html.append("</div>")
    return "".join(cards_html)


def _render_status_badge(state: SystemState, detail: str = "") -> str:
    """Genera un indicador HTML estilizado según el estado de la StateMachine."""
    styles = {
        SystemState.IDLE: ("status-pill", "⚪", "PAUSADO"),
        SystemState.LISTENING: ("status-pill status-listening", "🟢", "EN LÍNEA"),
        SystemState.PROCESSING: ("status-pill status-processing", "🟡", "PROCESANDO..."),
        SystemState.SPEAKING: ("status-pill status-speaking", "🔵", "HABLANDO (BLOQUEADO)"),
    }
    css_class, icon, label = styles.get(state, ("status-pill", "⚪", state.value))
    safe_detail = html.escape(detail)
    extra = f" · <span style='font-weight:400'>{safe_detail}</span>" if safe_detail else ""
    return f"<div class='{css_class}' title='{safe_detail}'>{icon} <span>{label}{extra}</span></div>"


def _ensure_audio_16k_mono(audio_tuple: tuple[int, np.ndarray] | None) -> np.ndarray | None:
    """Convierte el audio de Gradio a PCM mono 16 kHz float32 en [-1.0, 1.0]."""
    if audio_tuple is None:
        return None
    sr, data = audio_tuple
    if data is None or len(data) == 0:
        return None

    audio = np.asarray(data)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak

    if sr != SAMPLE_RATE and len(audio) > 0:
        target_len = int(round(len(audio) * SAMPLE_RATE / sr))
        old_idx = np.linspace(0, len(audio) - 1, num=len(audio))
        new_idx = np.linspace(0, len(audio) - 1, num=target_len)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def _safe_fsm_transition(fsm: StateMachine, action: str) -> SystemState:
    """Ejecuta transiciones de StateMachine tolerando NotImplementedError y estados ya activos."""
    fallback_map = {
        "start_session": SystemState.LISTENING,
        "on_speech_ended": SystemState.PROCESSING,
        "on_response_ready": SystemState.SPEAKING,
        "on_playback_finished": SystemState.LISTENING,
        "stop_session": SystemState.IDLE,
    }
    try:
        current = fsm.state
        if action == "start_session":
            if current == SystemState.LISTENING:
                return current
            if current in (SystemState.PROCESSING, SystemState.SPEAKING):
                fsm.stop_session()
        elif action == "on_speech_ended":
            if current == SystemState.PROCESSING:
                return current
            if current == SystemState.IDLE:
                fsm.start_session()
        elif action == "on_response_ready":
            if current == SystemState.SPEAKING:
                return current
        elif action == "on_playback_finished":
            if current == SystemState.LISTENING:
                return current
            if current != SystemState.SPEAKING:
                fsm.stop_session()
                return fsm.start_session()

        return getattr(fsm, action)()
    except (NotImplementedError, ValueError):
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
    chat: ChatThread,
    audio_16k: np.ndarray,
) -> tuple[TutorResponse, tuple[int, np.ndarray] | None]:
    """Conecta [VAD -> STT -> Orchestrator -> TTS] con fallback realista si aún no están fusionados."""
    settings = Settings()
    level_cfg = LEVEL_PARAMS[chat.profile.level.value]

    try:
        from src.audio.stt import FasterWhisperSTT
        from src.audio.tts import PiperTTSEngine
        from src.audio.vad import VoiceActivityDetector
        from src.core.orchestrator import CognitiveOrchestrator

        if session.vad is None:
            try:
                session.vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
            except TypeError:
                session.vad = VoiceActivityDetector()
        if session.stt is None:
            try:
                session.stt = FasterWhisperSTT(
                    model_size=settings.whisper_model_size,
                    device=os.getenv("WHISPER_DEVICE") or "auto",
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
                session.tts = PiperTTSEngine(settings=settings)
            except TypeError:
                session.tts = PiperTTSEngine()

        level_key = chat.profile.level.value
        if hasattr(session.vad, "extract_speech_segment"):
            trimmed_audio = session.vad.extract_speech_segment(audio_16k, level=level_key)
        else:
            trimmed_audio = session.vad.process_chunk(audio_16k, level=level_key)
        audio_for_stt = (
            trimmed_audio
            if trimmed_audio is not None and len(trimmed_audio) > 0
            else audio_16k
        )
        stt_result = session.stt.transcribe(audio_for_stt, sample_rate=SAMPLE_RATE)
        user_text = stt_result["text"] if isinstance(stt_result, dict) else str(stt_result)

        tutor_resp: TutorResponse = session.orchestrator.process_turn(
            user_text,
            chat.profile,
        )

        tts_result = session.tts.synthesize(
            tutor_resp.oral_response,
            tutor_resp.prosody_flags.rate,
        )
        if isinstance(tts_result, tuple) and len(tts_result) == 2:
            tts_sr, tts_pcm = int(tts_result[0]), tts_result[1]
        else:
            tts_sr, tts_pcm = SAMPLE_RATE, tts_result
        tts_output = (tts_sr, tts_pcm) if tts_pcm is not None and len(tts_pcm) > 0 else None
        return tutor_resp, tts_output

    except (ImportError, NotImplementedError, AttributeError, TypeError):
        turn_num = len(chat.messages) // 2 + 1
        if turn_num % 2 == 1:
            simulated = TutorResponse(
                transcript="Hello, I want a coffee and I go to work yesterday.",
                oral_response=(
                    f"Sure! Let's practice '{chat.title}'. "
                    "What else would you like to say?"
                ),
                visual_card=VisualCard(
                    correction="Hello, I would like a coffee and I went to work yesterday.",
                    category=CorrectionCategory.GRAMMAR,
                    explanation=(
                        "Para acciones pasadas ('yesterday') usamos 'went' en lugar de 'go', "
                        "y 'I would like' suena más natural que 'I want'."
                    ),
                ),
                prosody_flags=ProsodyFlags(rate=level_cfg["speech_rate"], pitch=0.0),
            )
        else:
            simulated = TutorResponse(
                transcript="Give me the menu right now.",
                oral_response="Of course, here is the menu. Take your time.",
                visual_card=VisualCard(
                    correction="Could you please bring me the menu?",
                    category=CorrectionCategory.PRAGMATICS,
                    explanation=(
                        "En este registro social, 'Give me... right now' resulta demasiado directo. "
                        "Usa 'Could you please...' para sonar cortés."
                    ),
                ),
                prosody_flags=ProsodyFlags(rate=level_cfg["speech_rate"], pitch=0.0),
            )
        return simulated, None


def _apply_scenario_suggestion(suggestion: str | None) -> str:
    return suggestion or ""


def start_new_chat_from_setup(
    level: str,
    register_mode: str,
    folder_choice: str,
    new_folder_name: str,
    scenario: str,
    session: UISessionState,
):
    """Crea el chat desde la pantalla inicial, oculta la configuración y abre la vista de chat."""
    folder = new_folder_name.strip() if new_folder_name and new_folder_name.strip() else folder_choice
    if not folder:
        folder = DEFAULT_FOLDERS[0]
    if folder not in session.folders:
        session.folders.append(folder)

    title = scenario.strip() if scenario and scenario.strip() else DEFAULT_SCENARIOS[0]
    new_thread = session._create_thread(
        title=title,
        folder=folder,
        level_label=level,
        register_mode=SocialRegister(register_mode),
    )
    session.chats[new_thread.chat_id] = new_thread
    session.active_chat_id = new_thread.chat_id
    session.selected_folder_filter = "Todas"

    state = _safe_fsm_transition(session.fsm, "start_session")
    session.last_activity_ts = time.monotonic()
    timeout = LEVEL_PARAMS[new_thread.profile.level.value]["silence_timeout_sec"]

    return (
        session,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(choices=session.folder_filter_choices(), value="Todas"),
        gr.update(choices=session.chat_choices(), value=new_thread.chat_id),
        gr.update(choices=session.folders, value=new_thread.folder),
        _render_status_badge(state, f"{title} · Nivel {level} · {timeout}s"),
        _render_chat_window(new_thread),
        _render_error_cards(new_thread),
        gr.update(value=None, interactive=True),
        gr.update(interactive=True),
    )


def open_setup_for_another_chat(session: UISessionState):
    """Vuelve a mostrar las opciones de configuración al pulsar 'Otro Chat'."""
    _safe_fsm_transition(session.fsm, "stop_session")
    return (
        session,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(choices=session.folders, value=session.folders[0]),
        "",
        "",
        None,
    )


def filter_chats_by_folder(folder_filter: str, session: UISessionState):
    """Filtra la lista lateral de conversaciones según la carpeta elegida."""
    session.selected_folder_filter = folder_filter or "Todas"
    choices = session.chat_choices()
    current_val = (
        session.active_chat_id
        if any(cid == session.active_chat_id for _, cid in choices)
        else (choices[0][1] if choices else None)
    )
    return session, gr.update(choices=choices, value=current_val)


def move_active_chat_to_folder(target_folder: str, session: UISessionState):
    """Permite mover el chat actual a otra carpeta."""
    chat = session.active_chat
    if chat is None or not target_folder:
        return session, gr.update(), gr.update()
    chat.folder = target_folder
    return (
        session,
        gr.update(choices=session.chat_choices(), value=chat.chat_id),
        _render_status_badge(session.fsm.state, f"📁 {chat.folder} · {chat.title}"),
    )


def switch_chat_handler(selected_chat_id: str, session: UISessionState):
    """Cambia entre los chats guardados en la barra lateral."""
    if selected_chat_id and selected_chat_id in session.chats:
        session.active_chat_id = selected_chat_id
    chat = session.active_chat
    if chat is None:
        return session, gr.update(), gr.update(), gr.update(), gr.update()

    state = _safe_fsm_transition(session.fsm, "start_session")
    session.last_activity_ts = time.monotonic()
    return (
        session,
        gr.update(choices=session.folders, value=chat.folder),
        _render_status_badge(state, f"{chat.title} ({chat.level_label})"),
        _render_chat_window(chat),
        _render_error_cards(chat),
    )


def stop_session_handler(session: UISessionState):
    """Pausa la escucha activa en el chat actual."""
    state = _safe_fsm_transition(session.fsm, "stop_session")
    return (
        session,
        _render_status_badge(state, "Sesión pausada"),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def resume_session_handler(session: UISessionState):
    """Reanuda la escucha activa en el chat actual."""
    chat = session.active_chat
    state = _safe_fsm_transition(session.fsm, "start_session")
    session.last_activity_ts = time.monotonic()
    detail = f"{chat.title}" if chat else ""
    return (
        session,
        _render_status_badge(state, detail),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def process_turn_handler(
    audio_input: tuple[int, np.ndarray] | None,
    session: UISessionState,
):
    """Procesa el turno de voz, subraya errores en la burbuja y genera la tarjeta visual."""
    chat = session.active_chat
    if chat is None:
        return (
            session,
            _render_status_badge(SystemState.IDLE, "Crea un chat primero"),
            None,
            _render_chat_window(None),
            _render_error_cards(None),
            gr.update(),
            gr.update(interactive=False),
        )

    if _is_locked(session.fsm):
        return (
            session,
            _render_status_badge(session.fsm.state, "Input Lock activo"),
            None,
            _render_chat_window(chat),
            _render_error_cards(chat),
            gr.update(),
            gr.update(interactive=False),
        )

    audio_16k = _ensure_audio_16k_mono(audio_input)
    if audio_16k is None:
        return (
            session,
            _render_status_badge(session.fsm.state, "No se detectó audio"),
            None,
            _render_chat_window(chat),
            _render_error_cards(chat),
            gr.update(),
            gr.update(interactive=True),
        )

    _safe_fsm_transition(session.fsm, "on_speech_ended")
    tutor_response, tts_audio = _run_pipeline_turn(session, chat, audio_16k)

    speaking_state = _safe_fsm_transition(session.fsm, "on_response_ready")
    now_str = time.strftime("%H:%M")

    card_idx = len(chat.error_cards) + 1
    chat.error_cards.append((card_idx, tutor_response.transcript, tutor_response.visual_card))

    chat.messages.append(
        ChatMessage(
            sender="user",
            text=tutor_response.transcript,
            timestamp=now_str,
            card=tutor_response.visual_card,
            card_index=card_idx,
        )
    )
    chat.messages.append(
        ChatMessage(
            sender="tutor",
            text=tutor_response.oral_response,
            timestamp=now_str,
        )
    )
    session.last_activity_ts = time.monotonic()

    if tts_audio is None:
        final_state = _safe_fsm_transition(session.fsm, "on_playback_finished")
        status_html = _render_status_badge(final_state, "Modo simulado · Tu turno")
        mic_update = gr.update(value=None, interactive=True)
    else:
        status_html = _render_status_badge(speaking_state, "Reproduciendo respuesta")
        mic_update = gr.update(interactive=False)

    return (
        session,
        status_html,
        tts_audio,
        _render_chat_window(chat),
        _render_error_cards(chat),
        gr.update(choices=session.chat_choices(), value=chat.chat_id),
        mic_update,
    )


def on_audio_playback_end(session: UISessionState):
    """Desbloquea el micrófono (SPEAKING -> LISTENING) al finalizar el audio TTS."""
    state = _safe_fsm_transition(session.fsm, "on_playback_finished")
    session.last_activity_ts = time.monotonic()
    return (
        session,
        _render_status_badge(state, "Tu turno para hablar"),
        gr.update(value=None, interactive=True),
    )


def check_silence_timeout(session: UISessionState):
    """Comprueba el timeout de silencio en LISTENING y envía mensaje de reenganche al chat."""
    chat = session.active_chat
    if chat is None or session.fsm.state != SystemState.LISTENING:
        return session, gr.update(), gr.update()

    timeout = LEVEL_PARAMS[chat.profile.level.value]["silence_timeout_sec"]
    if time.monotonic() - session.last_activity_ts < timeout:
        return session, gr.update(), gr.update()

    session.last_activity_ts = time.monotonic()
    try:
        from src.core.prompts import get_reengagement_prompt

        prompt_text = get_reengagement_prompt(chat.profile)
    except (ImportError, NotImplementedError):
        prompt_text = "Are you still there? Take your time and tell me what you think."

    chat.messages.append(
        ChatMessage(
            sender="tutor",
            text=prompt_text,
            timestamp=time.strftime("%H:%M"),
        )
    )
    return (
        session,
        _render_status_badge(session.fsm.state, f"Reenganche tras {timeout}s"),
        _render_chat_window(chat),
    )


def build_app() -> gr.Blocks:
    """Construye la interfaz estilo WhatsApp con pantalla de inicio, carpetas y doble canal."""
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
    )

    with gr.Blocks(
        title="YARNA — Tutor Conversacional estilo Chat",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:
        session_state = gr.State(UISessionState)

        # Cabecera global + Selector de Tema
        with gr.Row():
            with gr.Column(scale=5, min_width=280):
                gr.HTML(
                    """
                    <div class="wa-top-banner">
                        <h1>💬 YARNA — Tutor Conversacional Adaptativo</h1>
                        <p>Configura tu escenario, practica por voz estilo chat y aprende con subrayado inteligente y tarjetas de error.</p>
                    </div>
                    """
                )
            with gr.Column(scale=1, min_width=170):
                theme_toggle = gr.Radio(
                    choices=["☀️ Claro", "🌙 Oscuro"],
                    value="☀️ Claro",
                    label="Apariencia",
                    interactive=True,
                )

        theme_toggle.change(
            fn=None,
            inputs=[theme_toggle],
            outputs=[theme_toggle],
            js=THEME_TOGGLE_JS,
        )

        # =====================================================================
        # VISTA 1: CONFIGURACIÓN PREVIA AL CHAT (Visible al inicio o al pulsar "Otro Chat")
        # =====================================================================
        with gr.Column(visible=True, elem_classes=["setup-card"]) as setup_view:
            gr.Markdown("## ✨ Configura tu nueva conversación")
            gr.Markdown(
                "Elige tu nivel, el tono social, la carpeta donde guardarla y el tema sobre el que quieres hablar."
            )

            with gr.Row():
                level_selector = gr.Dropdown(
                    choices=["A2", "B2", "C1"],
                    value="A2",
                    label="🎯 Nivel MCER",
                    info="Adapta vocabulario y velocidad.",
                )
                register_selector = gr.Dropdown(
                    choices=[
                        SocialRegister.FORMAL.value,
                        SocialRegister.INFORMAL.value,
                        SocialRegister.EVERYDAY.value,
                    ],
                    value=SocialRegister.EVERYDAY.value,
                    label="🎭 Registro Pragmático",
                    info="Formal, informal o cotidiano.",
                )

            with gr.Row():
                folder_selector = gr.Dropdown(
                    choices=DEFAULT_FOLDERS,
                    value=DEFAULT_FOLDERS[0],
                    label="📁 Guardar en carpeta",
                    info="Organiza tus chats por categorías.",
                )
                new_folder_input = gr.Textbox(
                    value="",
                    label="➕ O crea una carpeta nueva",
                    placeholder="Ej. Entrevistas, Exámenes, Médico...",
                    info="Si escribes aquí, se creará esa carpeta.",
                )

            with gr.Row():
                scenario_input = gr.Textbox(
                    value="",
                    label="✍️ ¿De qué quieres hablar hoy?",
                    placeholder="Escribe libremente el tema o situación...",
                    info="Describe el escenario de roleplay.",
                    scale=3,
                )
                scenario_suggestions = gr.Dropdown(
                    choices=DEFAULT_SCENARIOS,
                    value=None,
                    label="💡 ¿No tienes ideas? Elige una",
                    info="Rellena el cuadro de la izquierda.",
                    scale=2,
                )

            start_chat_btn = gr.Button(
                "🚀 Empezar Chat",
                variant="primary",
                size="lg",
            )

        # =====================================================================
        # VISTA 2: ENTORNO DE CHAT ESTILO WHATSAPP (Oculta opciones de setup)
        # =====================================================================
        with gr.Row(visible=False) as chat_view:
            # COLUMNA 1: Botón "Otro Chat" + Carpetas + Lista de Chats
            with gr.Column(scale=3, min_width=240, elem_classes=["wa-column"]):
                another_chat_btn = gr.Button(
                    "➕ Otro Chat",
                    variant="primary",
                    size="md",
                )
                gr.Markdown("### 📁 Carpetas y Chats")
                folder_filter_dd = gr.Dropdown(
                    choices=["Todas"] + DEFAULT_FOLDERS,
                    value="Todas",
                    label="Filtrar por carpeta",
                    interactive=True,
                )
                chat_list = gr.Radio(
                    choices=[],
                    value=None,
                    label="Tus conversaciones",
                    interactive=True,
                )
                move_folder_dd = gr.Dropdown(
                    choices=DEFAULT_FOLDERS,
                    value=DEFAULT_FOLDERS[0],
                    label="Mover este chat a carpeta",
                    interactive=True,
                )

            # COLUMNA 2: Ventana de Chat (Burbujas + Micrófono)
            with gr.Column(scale=5, min_width=320, elem_classes=["wa-column"]):
                status_banner = gr.HTML(
                    value=_render_status_badge(SystemState.IDLE),
                )
                with gr.Row():
                    resume_btn = gr.Button("▶ Escuchar", variant="primary", size="sm", scale=1)
                    stop_btn = gr.Button("⏹ Pausar", variant="stop", size="sm", scale=1)

                chat_window = gr.HTML(value=_render_chat_window(None))

                mic_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎙️ Mensaje de voz (16 kHz)",
                    interactive=True,
                )
                send_btn = gr.Button(
                    "Enviar ➤",
                    variant="primary",
                    size="md",
                    interactive=True,
                )

                tts_output = gr.Audio(
                    label="🔊 Nota de voz de YARNA",
                    autoplay=True,
                    interactive=False,
                )

            # COLUMNA 3: Tarjetas de Errores (Canal Visual)
            with gr.Column(scale=4, min_width=280, elem_classes=["wa-column"]):
                gr.Markdown("### 🃏 Tarjetas de Corrección")
                gr.Markdown(
                    "<small>🟠 <b>Ámbar:</b> Gramática &nbsp;|&nbsp; 🟣 <b>Violeta:</b> Registro/Pragmática</small>"
                )
                error_cards_panel = gr.HTML(value=_render_error_cards(None))

        # Eventos de pantalla de configuración
        scenario_suggestions.change(
            fn=_apply_scenario_suggestion,
            inputs=[scenario_suggestions],
            outputs=[scenario_input],
        )

        start_chat_btn.click(
            fn=start_new_chat_from_setup,
            inputs=[
                level_selector,
                register_selector,
                folder_selector,
                new_folder_input,
                scenario_input,
                session_state,
            ],
            outputs=[
                session_state,
                setup_view,
                chat_view,
                folder_filter_dd,
                chat_list,
                move_folder_dd,
                status_banner,
                chat_window,
                error_cards_panel,
                mic_input,
                send_btn,
            ],
        )

        # Botón "Otro Chat": vuelve a mostrar la pantalla de configuración
        another_chat_btn.click(
            fn=open_setup_for_another_chat,
            inputs=[session_state],
            outputs=[
                session_state,
                setup_view,
                chat_view,
                folder_selector,
                new_folder_input,
                scenario_input,
                scenario_suggestions,
            ],
        )

        # Carpetas y cambio entre chats
        folder_filter_dd.change(
            fn=filter_chats_by_folder,
            inputs=[folder_filter_dd, session_state],
            outputs=[session_state, chat_list],
        )

        move_folder_dd.change(
            fn=move_active_chat_to_folder,
            inputs=[move_folder_dd, session_state],
            outputs=[session_state, chat_list, status_banner],
        )

        chat_list.change(
            fn=switch_chat_handler,
            inputs=[chat_list, session_state],
            outputs=[
                session_state,
                move_folder_dd,
                status_banner,
                chat_window,
                error_cards_panel,
            ],
        )

        # Controles de voz y turnos
        resume_btn.click(
            fn=resume_session_handler,
            inputs=[session_state],
            outputs=[session_state, status_banner, mic_input, send_btn],
        )

        stop_btn.click(
            fn=stop_session_handler,
            inputs=[session_state],
            outputs=[session_state, status_banner, mic_input, send_btn],
        )

        send_btn.click(
            fn=process_turn_handler,
            inputs=[mic_input, session_state],
            outputs=[
                session_state,
                status_banner,
                tts_output,
                chat_window,
                error_cards_panel,
                chat_list,
                mic_input,
            ],
        )

        tts_output.stop(
            fn=on_audio_playback_end,
            inputs=[session_state],
            outputs=[session_state, status_banner, mic_input],
        )

        silence_timer = gr.Timer(value=1.0)
        silence_timer.tick(
            fn=check_silence_timeout,
            inputs=[session_state],
            outputs=[session_state, status_banner, chat_window],
        )

    return demo


def main() -> None:
    """Punto de entrada para lanzar la interfaz Gradio."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()