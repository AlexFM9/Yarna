<div align="center">

# YARNA

### Tutor Conversacional Inteligente de Doble Canal para Práctica Oral de Inglés

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Local-First](https://img.shields.io/badge/Inference-Local--First-2EA44F?style=for-the-badge)
![Latency](https://img.shields.io/badge/Latency%20p95-%3C%201.8s-0969DA?style=for-the-badge)

**Asignatura:** Interfaces Inteligentes  
**Equipo:** Alexander Fontán Moreira · Antonio Formoso Rey · Carlota Marra Rodríguez

[Especificación Técnica (`AGENTS.md`)](./AGENTS.md) · [Modelos Locales (`models/`)](./models/README.md)

</div>

---

## Visión General

**YARNA** es un tutor conversacional inteligente orientado al aprendizaje y práctica oral del inglés mediante una **interfaz de voz bidireccional y adaptativa de doble canal**.

Elimina la dependencia del teclado y permite practicar diálogos fluidos en escenarios simulados (*roleplay*), ajustando en tiempo real su comportamiento acústico, ritmo de habla y exigencia lingüística y pragmática al perfil seleccionado.

---

## Características Principales

* **Estrategia de Doble Canal (HCI):**
  * **Canal Auditivo (Síncrono / Inmersivo):** Reproduce la respuesta oral aplicando *recast* implícito para corregir de forma natural sin cortar el ritmo de la conversación.
  * **Canal Visual (Asíncrono / Persistente):** Muestra tarjetas de feedback (`visual_card`) con la corrección, categoría (`grammar`, `pragmatics`, `ok`) y explicación en español para consulta reflexiva.
* **Adaptabilidad por Perfil `[Nivel MCER, Registro Social, Escenario]`:**
  * **Nivel Básico (`A2`):** Síntesis pausada (`0.80x`), ventana de silencio VAD amplia (`2.0 s`) y alta tolerancia gramatical.
  * **Nivel Avanzado (`B2/C1`):** Síntesis a ritmo nativo (`1.05x`), ventana de silencio ágil (`0.9 s`) y exigencia en colocaciones, *phrasal verbs* y matices léxicos.
  * **Registros Pragmáticos:** *Formal / Laboral*, *Urbano / Informal* y *Familiar / Cotidiano*.
* **Arquitectura Local-First de Baja Latencia:** Pipeline optimizado sobre CPU/GPU local (CTranslate2 + ONNX) con audio **PCM mono a 16 kHz** y objetivo de latencia $p95 < 1.8\text{ s}$ por turno.
* **Máquina de Estados con *Input Lock*:** Ciclo controlado (`IDLE` $\rightarrow$ `LISTENING` $\rightarrow$ `PROCESSING` $\rightarrow$ `SPEAKING`) que bloquea el micrófono durante la reproducción TTS para evitar realimentación acústica.

---

## Arquitectura y Stack Tecnológico

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 1: CLIENTE UI (Gradio Web Client)                                  │
│ • Captura de micrófono (PCM 16 kHz) y reproducción automática           │
│ • Selectores [Nivel MCER, Registro Social, Escenario]                   │
│ • Transcripción en vivo y panel visual persistente                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 2: CONTROL Y ORQUESTACIÓN (Python + Pydantic)                      │
│ • Máquina de Estados Finitos (IDLE, LISTENING, PROCESSING, SPEAKING)    │
│ • Memoria deslizante conversacional y degradación elegante              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ CAPA 3: PIPELINE DE INFERENCIA IA                                       │
│ • [Silero VAD] ──► [faster-whisper STT] ──► [LLM] ──► [Piper TTS]       │
└─────────────────────────────────────────────────────────────────────────┘
```

| Componente | Tecnología | Rol en el sistema |
| :--- | :--- | :--- |
| **Interfaz Web** | `Gradio >= 4.0` | Captura/reproducción de audio y vista dual síncrona/asíncrona |
| **Detección de Voz** | `Silero VAD` | Umbral adaptativo y fin de turno dependiente del nivel MCER |
| **Transcripción (STT)** | `faster-whisper` | Inferencia CTranslate2 (`INT8` / `FP16`) con marcas temporales |
| **Orquestación LLM** | `Ollama` / `OpenAI` / `Gemini` + `Pydantic` | Evaluación sintáctica/pragmática con contrato JSON estricto |
| **Síntesis Vocal (TTS)** | `Piper TTS` (ONNX) | Modelos neuronales VITS locales con modulación por `length_scale` |
| **Evaluación** | `jiwer` + perfilado `perf_counter` | Cálculo de WER (L2-ARCTIC), RTF y percentil de latencia $p95$ |

---

## Estructura del Repositorio

```text
yarna/
├── AGENTS.md                  # Especificación técnica maestra y guía de arquitectura
├── README.md                  # Documentación general del proyecto
├── requirements.txt           # Dependencias Python
├── .env.example               # Plantilla de variables de entorno y rutas de modelos
├── models/                    # Pesos locales (Piper ONNX / Whisper)
├── src/
│   ├── __init__.py
│   ├── config.py              # Constantes acústicas, parámetros MCER y configuración
│   ├── audio/
│   │   ├── vad.py             # Normalización PCM 16 kHz, buffer circular y Silero VAD
│   │   ├── stt.py             # Wrapper de faster-whisper
│   │   └── tts.py             # Wrapper de Piper TTS con modulación de velocidad
│   ├── core/
│   │   ├── schemas.py         # Contratos Pydantic (TutorResponse, SessionProfile)
│   │   ├── prompts.py         # System Prompts parametrizados y reenganche
│   │   ├── orchestrator.py    # Historial deslizante, llamadas LLM y fallbacks
│   │   └── state_machine.py   # Máquina de estados finitos e Input Lock
│   └── ui/
│       └── app.py             # Interfaz web Gradio de doble canal
└── tests/
    ├── eval_wer.py            # Evaluación acústica (WER sobre L2-ARCTIC)
    └── eval_latency.py        # Perfilado de latencia por etapa (RTF y p95 < 1.8 s)
```

---

## Instalación y Puesta en Marcha

### 1. Crear entorno e instalar dependencias

```bash
python -m venv .venv
```

```powershell
# En Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# En Linux / macOS
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno y modelos locales

```bash
cp .env.example .env
```

Descarga los pesos locales de voz para **Piper TTS** siguiendo las indicaciones de [`models/README.md`](./models/README.md) y ajusta `.env` si deseas elegir proveedor LLM (`ollama`, `openai` o `gemini`).

### 3. Ejecutar la aplicación

```bash
python -m src.ui.app
```

---

## Evaluación Experimental

El repositorio incluye scripts de evaluación para validar el sistema frente a los objetivos cuantitativos del proyecto:

```bash
# Benchmark de Word Error Rate (WER) sobre muestras L2-ARCTIC
python -m tests.eval_wer

# Perfilado de latencia por etapa (T_VAD + T_STT + T_LLM + T_TTS), RTF y p95
python -m tests.eval_latency
```
