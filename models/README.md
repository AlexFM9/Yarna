# Modelos Locales (`models/`)

Este directorio almacena los pesos y artefactos locales para inferencia **Local-First** (no versionados en Git debido a su tamaño):

## 1. Piper TTS (VITS en ONNX)
Coloca aquí la voz en inglés seleccionada junto con su archivo de metadatos JSON:
- `en_US-lessac-medium.onnx`
- `en_US-lessac-medium.onnx.json`

Puedes descargar voces oficiales desde los lanzamientos de [Rhasspy Piper](https://huggingface.co/rhasspy/piper-voices).

## 2. faster-whisper (CTranslate2)
Los modelos cuantizados (`base` o `small` en `INT8`/`FP16`) pueden descargarse automáticamente por `faster-whisper` o guardarse en `models/whisper/` configurando `WHISPER_MODEL_DIR` en `.env`.

## 3. Silero VAD
`silero-vad` gestiona su modelo ligero ONNX/TorchScript de forma automática tras instalar las dependencias de `requirements.txt`.
