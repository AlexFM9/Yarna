"""Capa 1: Maqueta base de la interfaz web Gradio de doble canal."""

from __future__ import annotations

import gradio as gr

from src.config import DEFAULT_SCENARIOS


def build_app() -> gr.Blocks:
    """Construye la maqueta base de la interfaz de voz bidireccional en Gradio."""
    with gr.Blocks(title="YARNA — Tutor Conversacional Adaptativo") as demo:
        gr.Markdown(
            "# YARNA — Tutor Conversacional de Doble Canal\n"
            "Estructura base lista para conectar el pipeline `[VAD -> STT -> LLM -> TTS]`."
        )

        # Selectores de perfil: [Nivel MCER, Registro Social, Escenario]
        with gr.Row():
            level_selector = gr.Dropdown(
                choices=["A2", "B2/C1"],
                value="A2",
                label="Nivel MCER",
            )
            register_selector = gr.Dropdown(
                choices=["formal", "informal", "everyday"],
                value="everyday",
                label="Registro Pragmático",
            )
            scenario_selector = gr.Dropdown(
                choices=DEFAULT_SCENARIOS,
                value=DEFAULT_SCENARIOS[0],
                allow_custom_value=True,
                label="Escenario (Roleplay)",
            )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## Canal Auditivo (Síncrono)")
                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Micrófono (PCM 16 kHz)",
                )
                send_btn = gr.Button("Enviar Turno", variant="primary")
                tts_output = gr.Audio(label="Respuesta Oral (Piper TTS)", autoplay=True)
                transcript_box = gr.Markdown("_La transcripción aparecerá aquí._")

            with gr.Column(scale=2):
                gr.Markdown("## Canal Visual (Asíncrono / Persistente)")
                visual_card_box = gr.Markdown(
                    "_La tarjeta `visual_card` (gramática / pragmática) aparecerá aquí._"
                )

        # TODO: Conectar send_btn.click (y eventos de sesión/streaming) con el
        # orquestador de Capa 2 y el pipeline de audio de Capa 3.
        _ = (
            level_selector,
            register_selector,
            scenario_selector,
            mic_input,
            send_btn,
            tts_output,
            transcript_box,
            visual_card_box,
        )

    return demo


def main() -> None:
    """Punto de entrada para lanzar la interfaz Gradio."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
