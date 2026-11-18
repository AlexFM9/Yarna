# Reparto de Tareas — Proyecto YARNA

Documento de coordinación interna para el desarrollo en paralelo del proyecto **YARNA** (*Interfaces Inteligentes*).  
La especificación técnica completa y las reglas de arquitectura figuran en [AGENTS.md](./AGENTS.md).

---

## Organización en 3 Bloques Desacoplados

La arquitectura en tres capas permite trabajar simultáneamente en tres ramas independientes sin bloqueos mutuos, sincronizándose únicamente a través de los contratos comunes.

### Bloque 1 — Pipeline Acústico Local: Capa 3 (`feat/audio-pipeline`)
* **Archivos asignados:**
  * [`src/audio/vad.py`](src/audio/vad.py)
  * [`src/audio/stt.py`](src/audio/stt.py)
  * [`src/audio/tts.py`](src/audio/tts.py)
  * [`tests/eval_wer.py`](tests/eval_wer.py)
* **Checklist de tareas:**
  * [ ] Implementar `normalize_pcm` en `src/audio/vad.py` (flujo PCM mono a `16000 Hz` normalizado a `float32` en `[-1.0, 1.0]`).
  * [ ] Implementar `CircularPCMBuffer` (`append`, `read_all`) para evitar recortes en los primeros fonemas tras el disparo del VAD.
  * [ ] Implementar `VoiceActivityDetector.process_chunk` con Silero VAD, umbral adaptativo ($E_{\text{base}} + \alpha \cdot \sigma$) y ventana de silencio según `LEVEL_PARAMS[level]["vad_silence_sec"]` (`2.0 s` en `A2`, `0.9 s` en `B2/C1`).
  * [ ] Implementar `FasterWhisperSTT.transcribe` en `src/audio/stt.py` (`INT8` en CPU / `FP16` en GPU) devolviendo texto puntuado, marcas temporales (`inicio_ms`, `fin_ms`) y puntuación de confianza.
  * [ ] Implementar `PiperTTSEngine` (`compute_length_scale` y `synthesize`) en `src/audio/tts.py` modulando la velocidad mediante `length_scale` (inverso de `prosody_flags.rate`).
  * [ ] Completar `evaluate_wer` en `tests/eval_wer.py` calculando el WER con `jiwer` sobre muestras del dataset L2-ARCTIC.

---

### Bloque 2 — Orquestador Cognitivo, Prompts y FSM: Capa 2 (`feat/core-orchestrator`)
* **Archivos asignados:**
  * [`src/core/schemas.py`](src/core/schemas.py)
  * [`src/core/prompts.py`](src/core/prompts.py)
  * [`src/core/orchestrator.py`](src/core/orchestrator.py)
  * [`src/core/state_machine.py`](src/core/state_machine.py)
* **Checklist de tareas:**
  * [ ] Completar `StateMachine` en `src/core/state_machine.py`: transiciones `start_session`, `on_speech_ended`, `on_response_ready`, `on_playback_finished`, `stop_session` (`IDLE` $\rightarrow$ `LISTENING` $\rightarrow$ `PROCESSING` $\rightarrow$ `SPEAKING` $\rightarrow$ `LISTENING`) e `is_input_locked` (*Input Lock* durante `SPEAKING`).
  * [ ] Implementar `build_system_prompt` y `get_reengagement_prompt` en `src/core/prompts.py` parametrizados por `[Nivel MCER, Registro Social, Escenario]` (`SessionProfile`), exigiendo *recast* implícito y salida JSON bajo `TutorResponse`.
  * [ ] Implementar `CognitiveOrchestrator` (`process_turn`, `reset_history`) en `src/core/orchestrator.py` con buffer deslizante de los últimos $N$ turnos (`MAX_HISTORY_TURNS`), soporte para `ollama` / `openai` / `gemini` con *Structured Outputs* y validación Pydantic.
  * [ ] Implementar degradación elegante: si falla `visual_card`, preservar siempre `oral_response`; si falla el JSON completo tras un reintento, emitir una respuesta acústica de contingencia válida bajo `TutorResponse`.
  * [ ] Garantizar la estabilidad de los modelos Pydantic en `src/core/schemas.py` (`SessionProfile`, `TutorResponse`, `VisualCard`, `ProsodyFlags`, `CEFRLevel`, `SocialRegister`, `CorrectionCategory`).

---

### Bloque 3 — Interfaz Web de Doble Canal e Integración End-to-End: Capa 1 (`feat/ui-gradio`)
* **Archivos asignados:**
  * [`src/ui/app.py`](src/ui/app.py)
  * [`src/config.py`](src/config.py)
  * [`tests/eval_latency.py`](tests/eval_latency.py)
* **Checklist de tareas:**
  * [ ] Conectar en `src/ui/app.py` los selectores `[Nivel MCER, Registro Pragmático, Escenario]`, captura de micrófono PCM 16 kHz, salida de audio con `autoplay`, renderizado de transcripción y panel lateral persistente para `visual_card`.
  * [ ] Integrar el ciclo completo `[VAD -> STT -> Orchestrator -> TTS]` gobernado por `StateMachine`, respetando el *Input Lock* durante `SPEAKING` y el disparo de reenganche tras `silence_timeout_sec`.
  * [ ] Mantener los hiperparámetros y configuración global en `src/config.py` (`SAMPLE_RATE`, `LEVEL_PARAMS`, `DEFAULT_SCENARIOS`, `Settings`).
  * [ ] Completar `run_latency_profile` en `tests/eval_latency.py` instrumentando $T_{\text{VAD}} + T_{\text{STT}} + T_{\text{LLM}} + T_{\text{TTS}}$, calculando el RTF y verificando $p95 < 1.8\text{ s}$.
  * [ ] Coordinar la evaluación de usabilidad con usuarios reales mediante el cuestionario SUS (*System Usability Scale*).

---

## Puntos de Sincronización (Contratos Comunes)

Antes de modificar cualquier interfaz compartida, todo el equipo debe respetar los contratos definidos en:
* [`src/core/schemas.py`](src/core/schemas.py): `SessionProfile`, `TutorResponse`, `VisualCard`, `ProsodyFlags`
* [`src/config.py`](src/config.py): `SAMPLE_RATE`, `CHANNELS`, `LATENCY_P95_TARGET_SEC`, `LEVEL_PARAMS`, `Settings`

Además, `src/audio/` (Capa 3) debe mantenerse completamente desacoplado de `src/ui/` (Capa 1) y de Gradio.

---

## Flujo de Trabajo con Ramas Git

Cada bloque se desarrolla en su propia rama creada a partir de `main`:

```bash
# Bloque 1 — Capa 3 (Audio)
git checkout -b feat/audio-pipeline

# Bloque 2 — Capa 2 (Core)
git checkout -b feat/core-orchestrator

# Bloque 3 — Capa 1 (UI + Integración)
git checkout -b feat/ui-gradio
```

Para mantener la rama actualizada antes de abrir un Pull Request hacia `main`:

```bash
git fetch origin
git rebase origin/main
```
