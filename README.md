# YARNA — Tutor Conversacional Inteligente de Doble Canal

Estructura base del proyecto **YARNA** para la asignatura **Interfaces Inteligentes**.

* **Equipo de Desarrollo:** Alexander Fontán Moreira, Antonio Formoso Rey, Carlota Marra Rodríguez
* **Especificación Maestra:** [CONTEXTO_YARNA.md](./CONTEXTO_YARNA.md)

## Estructura del Repositorio

```text
yarna/
├── CONTEXTO_YARNA.md          # Documento de especificación técnica
├── README.md                  # Guía de inicio y estructura
├── requirements.txt           # Dependencias del proyecto
├── .env.example               # Plantilla de variables de entorno
├── models/                    # Pesos locales (Whisper / Piper ONNX)
├── src/
│   ├── __init__.py
│   ├── config.py              # Hiperparámetros por Nivel (A2/B2-C1) y Registro
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── vad.py             # Normalización PCM 16kHz + Silero VAD adaptativo
│   │   ├── stt.py             # Wrapper de faster-whisper con timestamps
│   │   └── tts.py             # Wrapper de Piper TTS con modulación de rate
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Modelos Pydantic (Contrato JSON y tuplas de control)
│   │   ├── prompts.py         # Plantillas de System Prompt por Nivel y Registro
│   │   ├── orchestrator.py    # Memoria deslizante, llamada LLM y fallbacks
│   │   └── state_machine.py   # Control de estados (IDLE, LISTENING, PROCESSING, SPEAKING)
│   └── ui/
│       ├── __init__.py
│       └── app.py             # Aplicación base en Gradio (Doble canal + Selectores)
└── tests/
    ├── __init__.py
    ├── eval_wer.py            # Plantilla benchmark WER sobre L2-ARCTIC
    └── eval_latency.py        # Plantilla perfilado de latencia (RTF y p95 < 1.8s)
```

## Preparación del Entorno

```bash
python -m venv .venv
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

## Reparto de Tareas y Trabajo en Paralelo (3 Bloques Desacoplados)

La arquitectura en 3 capas permite dividir el desarrollo en paralelo entre los 3 integrantes del equipo trabajando sobre ramas independientes:

### Bloque 1 — Pipeline Acústico Local: Capa 3 (`feat/audio-pipeline`)
* **Archivos asignados:** [`src/audio/vad.py`](src/audio/vad.py), [`src/audio/stt.py`](src/audio/stt.py), [`src/audio/tts.py`](src/audio/tts.py), [`tests/eval_wer.py`](tests/eval_wer.py)
* **Tareas:**
  * Implementar `normalize_pcm` (entrada PCM 16 kHz escalada a `float32` en `[-1.0, 1.0]`) y `CircularPCMBuffer` para evitar recorte en los primeros fonemas.
  * Implementar `VoiceActivityDetector` con Silero VAD, umbral adaptativo ($E_{\text{base}} + \alpha \cdot \sigma$) y ventana de silencio parametrizada según nivel (`A2` / `B2/C1`).
  * Implementar `FasterWhisperSTT` cuantizado (`INT8` en CPU / `FP16` en GPU) devolviendo texto puntuado, marcas temporales (`inicio_ms`, `fin_ms`) y confianza.
  * Implementar `PiperTTSEngine` con cálculo de `length_scale` (inverso de `prosody_flags.rate`) y síntesis ONNX local.
  * Completar el benchmark offline de WER sobre el subconjunto L2-ARCTIC en `tests/eval_wer.py` usando `jiwer`.

### Bloque 2 — Orquestador Cognitivo, Prompts y FSM: Capa 2 (`feat/core-orchestrator`)
* **Archivos asignados:** [`src/core/schemas.py`](src/core/schemas.py), [`src/core/prompts.py`](src/core/prompts.py), [`src/core/orchestrator.py`](src/core/orchestrator.py), [`src/core/state_machine.py`](src/core/state_machine.py)
* **Tareas:**
  * Implementar las transiciones de `StateMachine` (`IDLE -> LISTENING -> PROCESSING -> SPEAKING -> LISTENING`) y el bloqueo `is_input_locked` (*Input Lock*) durante `SPEAKING`.
  * Implementar `build_system_prompt` parametrizado por `[Nivel MCER, Registro Social, Escenario]` (exigiendo *recast* implícito y contrato JSON `TutorResponse`) y `get_reengagement_prompt` para `silencio_timeout`.
  * Implementar `CognitiveOrchestrator` con buffer deslizante de los últimos $N$ turnos (`MAX_HISTORY_TURNS`), cliente LLM configurable (`ollama` / `openai` / `gemini` con *Structured Outputs*) y validación Pydantic.
  * Implementar degradación elegante: priorizar siempre `oral_response` si falla `visual_card` y devolver respuesta acústica de contingencia tras reintento fallido.

### Bloque 3 — Interfaz Web de Doble Canal e Integración End-to-End: Capa 1 (`feat/ui-gradio`)
* **Archivos asignados:** [`src/ui/app.py`](src/ui/app.py), [`src/config.py`](src/config.py), [`tests/eval_latency.py`](tests/eval_latency.py)
* **Tareas:**
  * Conectar la UI en Gradio (`python -m src.ui.app`): captura de micrófono web (PCM 16 kHz), selectores de `[Nivel MCER, Registro, Escenario]`, canal auditivo síncrono con `autoplay` e *Input Lock* y panel lateral persistente para `visual_card`.
  * Integrar de extremo a extremo el ciclo `[VAD -> STT -> Orchestrator/LLM -> TTS]` gestionado por `StateMachine`.
  * Completar el perfilado de latencia por etapa ($T_{\text{VAD}} + T_{\text{STT}} + T_{\text{LLM}} + T_{\text{TTS}}$) y RTF en `tests/eval_latency.py` verificando el objetivo $p95 < 1.8\text{ s}$.
  * Coordinar las pruebas de usabilidad con cuestionario SUS (*System Usability Scale*).

### Contratos Comunes y Flujo de Trabajo en Git
* **Puntos de sincronización:** Respetar los modelos Pydantic de [`src/core/schemas.py`](src/core/schemas.py) (`SessionProfile`, `TutorResponse`, `VisualCard`, `ProsodyFlags`) y las constantes e hiperparámetros de [`src/config.py`](src/config.py) (`SAMPLE_RATE`, `LEVEL_PARAMS`, `Settings`) para desarrollar cada capa de forma independiente sin bloqueos.
* **Flujo de ramas:**
  ```bash
  git checkout -b feat/audio-pipeline     # Bloque 1 (Capa 3)
  git checkout -b feat/core-orchestrator  # Bloque 2 (Capa 2)
  git checkout -b feat/ui-gradio          # Bloque 3 (Capa 1 + Integración)
  ```
