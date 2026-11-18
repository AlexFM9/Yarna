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

## Próximos Pasos de Desarrollo

1. **Capa 3 (`src/audio/`):** Implementar normalización PCM, Silero VAD adaptativo, `faster-whisper` y síntesis `Piper TTS` con `length_scale`.
2. **Capa 2 (`src/core/`):** Implementar máquina de estados, prompts parametrizados y orquestador con buffer deslizante y degradación elegante.
3. **Capa 1 (`src/ui/app.py`):** Conectar los componentes Gradio con el backend y ejecutar `python -m src.ui.app`.
4. **Validación (`tests/`):** Completar `tests/eval_wer.py` (L2-ARCTIC) y `tests/eval_latency.py` (objetivo $p95 < 1.8\text{ s}$).
