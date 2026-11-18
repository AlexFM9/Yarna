# YARNA — Especificación Técnica y Contexto General del Proyecto

* **Asignatura:** Interfaces Inteligentes
* **Equipo de Desarrollo:** Alexander Fontán Moreira, Antonio Formoso Rey, Carlota Marra Rodríguez
* **Versión del Documento:** 1.0 (Arquitectura Base de Implementación)

---

## 1. Visión General y Propósito del Sistema

**YARNA** es un tutor conversacional inteligente orientado al aprendizaje y práctica oral de idiomas (inglés) mediante una **interfaz de voz bidireccional y adaptativa de doble canal**. 

El sistema elimina la dependencia del teclado y permite al usuario mantener diálogos fluidos en escenarios simulados (roleplay), adaptando dinámicamente tanto su comportamiento acústico como su exigencia lingüística y pragmática en función del perfil seleccionado por el alumno.

### Ejes de Adaptabilidad (Perfil de Usuario)
Antes o durante la sesión, el usuario configura una tupla de control `[Nivel MCER, Registro Social, Escenario]`:
1. **Nivel de Competencia (MCER):** Acotado en dos perfiles contrastados principales (extensible de `A2` a `C1`):
   * **Básico (`A2`):** Velocidad de síntesis pausada ($0.80\times$), articulación marcada, ventana de silencio VAD amplia ($\sim 2.0\text{ s}$) para dar tiempo a pensar, y alta tolerancia gramatical.
   * **Avanzado (`B2/C1`):** Velocidad de síntesis nativa ($1.05\times$), habla conectada, ventana de silencio VAD ágil ($\sim 0.8 - 1.0\text{ s}$), y exigencia en colocaciones, *phrasal verbs* y matices léxicos.
2. **Registro Pragmático / Ámbito Social:**
   * **Formal / Laboral:** Evalúa fórmulas de cortesía, estructuras indirectas (*"Could you please..."*) y tono profesional. Penaliza la brusquedad.
   * **Urbano / Informal:** Evalúa naturalidad, uso de contracciones, modismos cotidianos y fluidez. Penaliza el exceso de rigidez o formalidad artificial.
   * **Familiar / Cotidiano:** Enfoque cercano, rutinas diarias y elipsis conversacionales naturales.

---

## 2. Requisitos Operativos y Restricciones Medibles

* **Formato de Audio de Entrada:** Flujo de audio **PCM mono** a **$16\text{ kHz}$** (16 bits), normalizado en amplitud antes de inferencia.
* **Restricción Crítica de Latencia:** Percentil 95 ($p95$) de latencia *end-to-end* inferior a **$1.8\text{ s}$** por turno (medido desde el instante en que el VAD detecta el fin de locución hasta que comienza a reproducirse el primer bloque de audio TTS).
* **Enfoque Local-First:** Inferencia optimizada para CPU/GPU local mediante modelos cuantizados y runtimes ligeros (CTranslate2 y ONNX), sin requerir infraestructura pesada.

---

## 3. Macroarquitectura del Sistema (3 Capas Desacopladas)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 1: CLIENTE UI (Gradio Web Client)                                  │
│ • Captura de micrófono web (PCM 16 kHz) y cola de reproducción de audio │
│ • Selectores de estado [Nivel MCER, Registro, Escenario]                │
│ • Renderizado asíncrono de transcripción y tarjeta visual persistente   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Eventos / Audio Stream
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 2: CONTROL Y BACKEND (Orquestador en Python)                       │
│ • Máquina de Estados Finitos (IDLE, LISTENING, PROCESSING, SPEAKING)    │
│ • Buffer deslizante de memoria conversacional (últimos N turnos)        │
│ • Validación de contratos JSON (Pydantic) y gestión de contingencias    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ Llamadas Modulares
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 3: INFERENCIA IA (Pipeline Desacoplado)                            │
│ • [Silero VAD] ──► [faster-whisper (STT)] ──► [LLM] ──► [Piper TTS]     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Detallado por Módulos

### 4.1. Módulo de Entrada Acústica (VAD + STT)
1. **Normalización PCM:** Escalado de señal en punto flotante $[-1.0, 1.0]$ para neutralizar diferencias de ganancia entre micrófonos.
2. **Buffer Circular:** Almacenamiento continuo en ventana deslizante para evitar recortes en los primeros fonemas tras dispararse la detección.
3. **Silero VAD:** Red neuronal ligera de detección de actividad de voz con umbral adaptativo ($E_{\text{base}} + \alpha \cdot \sigma$) y temporizador de fin de turno parametrizado según el nivel MCER.
4. **faster-whisper (STT):** Decodificación acústica mediante CTranslate2 (modelo `base` o `small` cuantizado en `INT8` para CPU o `FP16` para GPU). Emite texto puntuado, marcas temporales (`inicio_ms`, `fin_ms`) y puntuación de confianza.

### 4.2. Orquestador Cognitivo y Contrato JSON
El orquestador recibe la transcripción junto con el historial deslizante de $N$ turnos y la tupla `[Nivel, Registro, Rol]`. Evalúa en paralelo la **corrección sintáctica** y la **adecuación pragmática**.

El LLM **debe** responder bajo un esquema JSON estricto (validado mediante **Pydantic** o decodificación guiada por gramática):

```json
{
  "transcript": "texto exacto reconocido del usuario",
  "oral_response": "respuesta conversacional aplicando recast implícito (sin explicaciones teóricas)",
  "visual_card": {
    "correction": "versión corregida o más adecuada al registro",
    "category": "grammar | pragmatics | ok",
    "explanation": "explicación breve y clara en español para consulta visual"
  },
  "prosody_flags": {
    "rate": 1.05,
    "pitch": 0.0
  }
}
```

#### Principio de Tolerancia a Fallos (Degradación Elegante)
* El canal oral tiene prioridad crítica: un fallo parcial de parseo o validación en `visual_card` **nunca** debe bloquear la síntesis y emisión de `oral_response`.
* Si falla por completo la estructura JSON tras un reintento, se emite una respuesta acústica de contingencia (ej. *"Sorry, I didn't quite catch that, could you say it again?"*).

### 4.3. Motor de Síntesis Vocal (Piper TTS)
* Ejecución local mediante modelos neuronales VITS exportados a **ONNX**.
* Pipeline: `Texto Oral` $\rightarrow$ `G2P (Grafema a Fonema)` $\rightarrow$ `Espectrograma Neuronal` $\rightarrow$ `Vocoder` $\rightarrow$ `PCM Audio Out`.
* **Modulación Prosódica:** Inyección dinámica del parámetro `length_scale` (inverso de `prosody_flags.rate`: $\approx 1.25$ para velocidad $0.80\times$ en A2, y $\approx 0.95$ para velocidad $1.05\times$ en B2/C1), alterando la duración fonémica sin distorsionar el tono natural de la voz.

---

## 5. Dinámica de Interacción: Doble Canal y Máquina de Estados

### 5.1. Estrategia de Doble Canal (HCI)
Para evitar la saturación de la memoria de trabajo del alumno durante la conversación:
* **Canal Auditivo (Síncrono / Inmersivo):** Reproduce `oral_response`. Utiliza la técnica lingüística de **recast** (reformular implícitamente el error dentro de la respuesta natural del tutor) para mantener el ritmo conversacional sin interrupciones teóricas.
* **Canal Visual (Asíncrono / Persistente):** Renderiza `visual_card` en un panel lateral de la interfaz. Permite al usuario revisar la regla gramatical o el matiz pragmático de forma reflexiva sin frenar el diálogo.

### 5.2. Máquina de Estados Finitos
```text
[IDLE] ──(Inicio sesión)──► [LISTENING] ──(Fin voz VAD)──► [PROCESSING]
                                 ▲                               │
                                 │                               ▼
                           (fin_audio) ◄─────────────────── [SPEAKING]
                                                           (INPUT LOCK)
```
* **Input Lock:** Durante el estado `SPEAKING`, la captura del micrófono permanece desactivada para evitar realimentación acústica (*echo feedback*) en la v1 sin requerir cancelación de eco (AEC) compleja en el navegador.
* **Timeout de Silencio (`silencio_timeout`):** Si el sistema permanece en `LISTENING` sin actividad vocal más allá del umbral configurado, emite un *prompt* corto de reenganche para dinamizar la charla.

---

## 6. Stack Tecnológico y Estructura del Repositorio

### Dependencias Principales (`requirements.txt`)
* `gradio>=4.0.0` (Interfaz web reactiva, componentes de audio I/O y gestión de sesiones)
* `faster-whisper>=1.0.0` (Motor STT optimizado en CTranslate2)
* `torch` / `torchaudio` + `silero-vad` (Detección de actividad de voz neuronal)
* `piper-tts` / `onnxruntime` (Síntesis acústica local ultrarrápida)
* `pydantic>=2.0.0` (Validación estricta de esquemas JSON)
* `openai` / `google-genai` / `ollama` (Cliente LLM configurable con *Structured Outputs*)
* `numpy` / `scipy` / `soundfile` (Procesamiento y normalización de arrays PCM)
* `jiwer` (Cálculo automático de *Word Error Rate* para evaluación)

### Arquitectura de Directorios
```text
yarna/
├── AGENTS.md                  # Este documento de especificación maestra y reglas IA
├── requirements.txt           # Dependencias del proyecto
├── .env.example               # Variables de entorno (API keys / rutas de modelos)
├── models/                    # Pesos locales (modelos ONNX de Piper y Whisper)
├── src/
│   ├── __init__.py
│   ├── config.py              # Hiperparámetros por Nivel (A2/B2) y Registro
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── vad.py             # Normalización PCM 16kHz + Silero VAD adaptativo
│   │   ├── stt.py             # Wrapper de faster-whisper con timestamps
│   │   └── tts.py             # Wrapper de Piper TTS con modulación de rate
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Modelos Pydantic (Contrato JSON y tuplas de control)
│   │   ├── prompts.py         # Plantillas de System Prompt por Nivel y Registro
│   │   ├── orchestrator.py    # Gestión de memoria deslizante, llamada LLM y fallbacks
│   │   └── state_machine.py   # Control de estados (IDLE, LISTENING, PROCESSING, SPEAKING)
│   └── ui/
│       ├── __init__.py
│       └── app.py             # Aplicación principal en Gradio (Doble canal + Selectores)
└── tests/
    ├── eval_wer.py            # Benchmark offline de WER sobre subconjunto L2-ARCTIC
    └── eval_latency.py        # Perfilado de latencia por etapa (RTF y p95 < 1.8s)
```

---

## 7. Protocolo de Validación Experimental

Para la memoria y defensa de la asignatura, el sistema se validará en tres dimensiones:
1. **Evaluación Acústica (WER):** Medición de la tasa de error de palabra (*Word Error Rate*) mediante `jiwer` sobre muestras de hablantes no nativos del dataset público **L2-ARCTIC**.
2. **Perfilado de Rendimiento (RTF y Latencia):** Registro de marcas temporales en cada fase ($T_{\text{VAD}} + T_{\text{STT}} + T_{\text{LLM}} + T_{\text{TTS}}$) verificando el cumplimiento de $p95 < 1.8\text{ s}$.
3. **Evaluación de Usabilidad (SUS):** Pruebas interactivas con usuarios reales de distintos niveles aplicando el cuestionario estandarizado *System Usability Scale* para medir carga cognitiva y fluidez percibida.

---

## 8. Reglas Operativas para Agentes IA (Coding Guidelines)

Cualquier agente IA que colabore en este repositorio debe respetar obligatoriamente las siguientes invariantes:

1. **Desacoplamiento estricto de las 3 capas:**
   * `src/ui/` (Capa 1) solo se comunica con `src/core/` (Capa 2) y `src/audio/` (Capa 3) a través de sus interfaces públicas.
   * `src/audio/` (Capa 3) no debe importar nada de `src/ui/` ni acoplarse a Gradio.
2. **Inmutabilidad del contrato de datos:**
   * Respetar los esquemas Pydantic definidos en `src/core/schemas.py` (`SessionProfile`, `TutorResponse`, `VisualCard`, `ProsodyFlags`) y las constantes de `src/config.py` (`SAMPLE_RATE`, `LEVEL_PARAMS`, `Settings`) para no romper el desarrollo en paralelo de las ramas del equipo.
3. **Prioridad del canal oral (Degradación Elegante):**
   * En `src/core/orchestrator.py`, un error parcial de parseo o validación en `visual_card` **nunca** debe impedir devolver `oral_response` ni bloquear la síntesis con Piper TTS.
   * Si el JSON del LLM falla por completo tras un reintento, debe emitirse una respuesta acústica de contingencia válida bajo `TutorResponse`.
4. **Invariantes acústicas, de latencia y de estados:**
   * Todo flujo de audio interno opera en PCM mono a `16000 Hz` (`SAMPLE_RATE`) normalizado en `float32` dentro del rango `[-1.0, 1.0]`.
   * Mantener el objetivo de latencia $p95 < 1.8\text{ s}$ por turno evitando bloqueos síncronos innecesarios y respetando el *Input Lock* de `StateMachine` durante el estado `SPEAKING`.
