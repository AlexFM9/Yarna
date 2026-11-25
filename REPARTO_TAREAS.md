# Reparto de Tareas — Proyecto YARNA

Documento de coordinación interna para el desarrollo en paralelo del proyecto **YARNA** (*Interfaces Inteligentes*).  
La especificación técnica completa y las reglas de arquitectura figuran en [AGENTS.md](./AGENTS.md).

---

## Sprint 1 — Arquitectura Base en 3 Bloques (Completado e Integrado en `main`)

### Bloque 1 — Pipeline Acústico Local: Capa 3 (`feat/audio-pipeline`)
* **Archivos asignados:** [`src/audio/vad.py`](src/audio/vad.py), [`src/audio/stt.py`](src/audio/stt.py), [`src/audio/tts.py`](src/audio/tts.py), [`tests/eval_wer.py`](tests/eval_wer.py)
* **Checklist de tareas:**
  * [x] Implementar `normalize_pcm` en `src/audio/vad.py` (flujo PCM mono a `16000 Hz` normalizado a `float32` en `[-1.0, 1.0]`).
  * [x] Implementar `CircularPCMBuffer` (`append`, `read_all`) para evitar recortes en los primeros fonemas tras el disparo del VAD.
  * [x] Implementar `VoiceActivityDetector.process_chunk` con Silero VAD, umbral adaptativo ($E_{\text{base}} + \alpha \cdot \sigma$) y ventana de silencio según `LEVEL_PARAMS[level]["vad_silence_sec"]` (`2.0 s` en `A2`, `0.9 s` en `B2/C1`).
  * [x] Implementar `FasterWhisperSTT.transcribe` en `src/audio/stt.py` devolviendo texto puntuado, marcas temporales (`inicio_ms`, `fin_ms`) y puntuación de confianza.
  * [x] Implementar `PiperTTSEngine` (`compute_length_scale` y `synthesize`) en `src/audio/tts.py` modulando la velocidad mediante `length_scale` (inverso de `prosody_flags.rate`).
  * [ ] Ejecutar el benchmark real de `evaluate_wer` en `tests/eval_wer.py` sobre muestras del dataset L2-ARCTIC para la memoria.

### Bloque 2 — Orquestador Cognitivo, Prompts y FSM: Capa 2 (`feat/core-orchestrator`)
* **Archivos asignados:** [`src/core/schemas.py`](src/core/schemas.py), [`src/core/prompts.py`](src/core/prompts.py), [`src/core/orchestrator.py`](src/core/orchestrator.py), [`src/core/state_machine.py`](src/core/state_machine.py)
* **Checklist de tareas:**
  * [x] Completar `StateMachine` en `src/core/state_machine.py`: transiciones `start_session`, `on_speech_ended`, `on_response_ready`, `on_playback_finished`, `stop_session` e `is_input_locked`.
  * [x] Implementar `build_system_prompt` y `get_reengagement_prompt` en `src/core/prompts.py` parametrizados por `[Nivel MCER, Registro Social, Escenario]`.
  * [x] Implementar `CognitiveOrchestrator` (`process_turn`, `reset_history`) en `src/core/orchestrator.py` con buffer deslizante (`MAX_HISTORY_TURNS`), soporte para `ollama` / `openai` / `gemini` / `mock` y validación Pydantic.
  * [x] Implementar degradación elegante: preservar `oral_response` si falla `visual_card` y respuesta acústica de contingencia tras reintento fallido.
  * [x] Garantizar la estabilidad de los modelos Pydantic en `src/core/schemas.py`.

### Bloque 3 — Interfaz Web de Doble Canal e Integración End-to-End: Capa 1 (`feat/ui-gradio`)
* **Archivos asignados:** [`src/ui/app.py`](src/ui/app.py), [`src/config.py`](src/config.py), [`tests/eval_latency.py`](tests/eval_latency.py)
* **Checklist de tareas:**
  * [x] Conectar en `src/ui/app.py` los selectores `[Nivel MCER, Registro Pragmático, Escenario]`, captura de micrófono PCM 16 kHz, salida de audio con `autoplay`, renderizado de transcripción y panel lateral persistente para `visual_card`.
  * [x] Integrar el ciclo completo `[VAD -> STT -> Orchestrator -> TTS]` gobernado por `StateMachine` e *Input Lock*.
  * [x] Mantener los hiperparámetros y configuración global en `src/config.py`.
  * [x] Implementar `run_latency_profile` en `tests/eval_latency.py`.
  * [ ] Coordinar la evaluación de usabilidad con usuarios reales mediante el cuestionario SUS (*System Usability Scale*).

---

## Sprint 2 — Corrección de Bugs, Calidad de Reconocimiento y Latencia

Tras las primeras pruebas reales end-to-end con micrófono, Ollama y Piper TTS, se detectaron cuatro problemas a resolver en paralelo antes de la entrega:
1. Parpadeo continuo del chat en la UI por el temporizador cada 1 s.
2. Spam de mensajes de reenganche del tutor cuando nadie está hablando.
3. Pérdida de precisión en el reconocimiento de voz (re-muestreo lineal `np.interp`, recortes VAD y falta de contexto).
4. Latencia elevada (~14 s en frío / ~3.9 s en caliente) frente al objetivo $p95 < 1.8\text{ s}$.

### Bloque 1 — Alex: Calidad STT, VAD para voces suaves y aceleración (`fix/audio-quality-speed`)
* **Archivos asignados:** [`src/audio/vad.py`](src/audio/vad.py), [`src/audio/stt.py`](src/audio/stt.py), [`src/audio/tts.py`](src/audio/tts.py), [`tests/eval_wer.py`](tests/eval_wer.py)
* **Checklist de tareas:**
  * [ ] Blindar el fallback automático CUDA $\rightarrow$ CPU `INT8` en tiempo de inferencia cuando falte `cublas64_12.dll` en Windows, y añadir soporte opcional para cargar DLLs de `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` con `os.add_dll_directory`.
  * [ ] Mejorar la precisión de reconocimiento en inglés L2 (acento español): evaluar `small.en` frente a `base.en`, probar `beam_size=2/3` y añadir un `initial_prompt` conversacional breve en `FasterWhisperSTT.transcribe()`.
  * [ ] Ajustar `VoiceActivityDetector.extract_speech_segment()` para no cortar inicios/finales en voces suaves: normalizar ganancia previa antes de Silero VAD, ajustar `speech_threshold` (`0.35–0.40`) y ampliar `padding_sec` (`0.35–0.40 s`).
  * [ ] Exponer un método `warmup()` en `FasterWhisperSTT` y `PiperTTSEngine` para precargar modelos al arrancar la aplicación y eliminar la demora del primer turno.
  * [ ] Comparar `tiny.en`, `base.en` y `small.en` en `tests/eval_wer.py` documentando el compromiso WER vs. latencia.

### Bloque 2 — Antonio: Optimización de Latencia LLM y Prompt Compacto (`fix/llm-latency-optimization`)
* **Archivos asignados:** [`src/core/orchestrator.py`](src/core/orchestrator.py), [`src/core/prompts.py`](src/core/prompts.py)
* **Checklist de tareas:**
  * [ ] Optimizar la llamada a Ollama en `CognitiveOrchestrator._call_llm()`: pasar `options={"num_ctx": 1024, "num_predict": 140, "temperature": 0.3}` y `keep_alive="15m"` para que todas las capas (29/29) de `llama3.2` quepan en GPU (4 GB VRAM) sin descargar capas a CPU.
  * [ ] Compactar `build_system_prompt()` en `src/core/prompts.py` de ~536 tokens a **~180–220 tokens** (en inglés directo para mejorar el *recast* implícito y evitar explicaciones erróneas, manteniendo `explanation` en español y evitando duplicar el esquema JSON cuando ya se pasa `format=TutorResponse.model_json_schema()`).
  * [ ] Añadir un método `warmup()` en `CognitiveOrchestrator` para precargar el modelo y compilar la gramática JSON antes del primer turno del usuario.
  * [ ] Evaluar modelos locales aún más rápidos (`llama3.2:1b` / `qwen2.5:1.5b`) y verificar su calidad de *recast* buscando cumplir $p95 < 1.8\text{ s}$.

### Bloque 3 — Carlota: Anti-parpadeo, Anti-spam del Tutor, Audio 16 kHz y Warmup UI (`fix/ui-flicker-and-ux`)
* **Archivos asignados:** [`src/ui/app.py`](src/ui/app.py), [`tests/eval_latency.py`](tests/eval_latency.py)
* **Checklist de tareas:**
  * [ ] **Eliminar el parpadeo del chat:** añadir `show_progress="hidden"` a `silence_timer.tick(...)`, espaciar el `gr.Timer` a `2.0–3.0 s` y devolver `gr.skip()` (en lugar de `gr.update()`) en `check_silence_timeout()` cuando aún no toca actuar.
  * [ ] **Evitar spam del tutor si no habla nadie:** añadir una bandera `reengagement_sent: bool` por chat/sesión para que `check_silence_timeout()` envíe **un único mensaje de reenganche** al superarse `silence_timeout_sec` y **pare** (sin volver a enviar más mensajes ni llenar el chat), reiniciando la bandera únicamente cuando el usuario envíe un nuevo turno en `process_turn_handler()` o reanude/inicie chat.
  * [ ] **Corregir degradación de audio al re-muestrear:** sustituir la interpolación lineal `np.interp` en `_ensure_audio_16k_mono()` por `scipy.signal.resample_poly` (con `math.gcd`) para eliminar aliasing al pasar de `48 kHz` del navegador a `16 kHz`.
  * [ ] **Eliminar los ~14 s del primer turno y limpiar historial entre chats:** compartir una única instancia pre-calentada (`warmup`) de `VAD / STT / Orchestrator / TTS` al arrancar `app.py` (en lugar de instanciarlos en `None` por sesión), llamar a `orchestrator.reset_history()` en `start_new_chat_from_setup()` y `switch_chat_handler()`, devolver audio TTS en `int16` a `gr.Audio` y mover `theme`/`css` a `demo.launch(...)` (eliminando avisos de deprecación).
  * [ ] Actualizar `tests/eval_latency.py` con calentamiento previo + 20 turnos en caliente para validar el cumplimiento de $p95 < 1.8\text{ s}$.

---

## Puntos de Sincronización (Contratos Comunes)

Antes de modificar cualquier interfaz compartida, todo el equipo debe respetar los contratos definidos en:
* [`src/core/schemas.py`](src/core/schemas.py): `SessionProfile`, `TutorResponse`, `VisualCard`, `ProsodyFlags`
* [`src/config.py`](src/config.py): `SAMPLE_RATE`, `CHANNELS`, `LATENCY_P95_TARGET_SEC`, `LEVEL_PARAMS`, `Settings`

Además, `src/audio/` (Capa 3) debe mantenerse completamente desacoplado de `src/ui/` (Capa 1) y de Gradio.

---

## Flujo de Trabajo con Ramas Git

Cada bloque del Sprint 2 se desarrolla en una rama nueva creada a partir de `main` actualizada:

```bash
git checkout main && git pull origin main

# Bloque 1 — Alex (Capa 3: Audio)
git checkout -b fix/audio-quality-speed

# Bloque 2 — Antonio (Capa 2: Core/LLM)
git checkout -b fix/llm-latency-optimization

# Bloque 3 — Carlota (Capa 1: UI + Integración)
git checkout -b fix/ui-flicker-and-ux
```

Para mantener la rama actualizada antes de abrir un Pull Request hacia `main`:

```bash
git fetch origin
git rebase origin/main
```
