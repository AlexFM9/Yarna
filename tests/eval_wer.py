"""[PLANTILLA GUIADA — RETO 5] Benchmark de WER, CER y RTF sobre L2-ARCTIC.

En este último reto implementarás las 3 piezas clave de evaluación acústica:
  1. `normalize_transcript(text)`
  2. `compute_wer_score(references, hypotheses)`
  3. `evaluate_wer(dataset_dir, ...)`

La lectura de WAV (`read_wav_pcm`), descubrimiento de pares en L2-ARCTIC
(`discover_l2_arctic_samples`), tabla Markdown (`format_markdown_table`) y el CLI
(`main`) ya están implementados para que te centres en la métrica experimental.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any
import wave

import numpy as np

from src.audio.stt import FasterWhisperSTT
from src.audio.vad import normalize_pcm

SPANISH_L2_SPEAKERS: tuple[str, ...] = ("EBVS", "ERMS", "MBMPS", "NJS")

_ENGLISH_CONTRACTIONS: dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def normalize_transcript(text: str) -> str:
    """Normaliza una frase inglesa (minúsculas, sin puntuación, contracciones expandidas).

    TODO (Reto 5.1):
    1. Si `jiwer` está instalado (`try: import jiwer`), compón un pipeline con:
       `ToLowerCase()`, `ExpandCommonEnglishContractions()`, `RemovePunctuation()`,
       `RemoveMultipleSpaces()`, `Strip()`.
    2. Si `ImportError` (como respaldo cuando aún no está instalado `jiwer`):
       - Convierte a minúsculas (`text.lower().replace("’", "'")`).
       - Reemplaza cada contracción de `_ENGLISH_CONTRACTIONS` usando `re.sub`.
       - Elimina puntuación (`re.sub(r"[^\\w\\s]", " ", cleaned)`), colapsa espacios y haz `.strip()`.
    """
    raise NotImplementedError("Reto 5.1 pendiente: normalize_transcript")


def _levenshtein_distance(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Distancia mínima de edición usada como respaldo si no está `jiwer`."""
    if len(seq_a) < len(seq_b):
        return _levenshtein_distance(seq_b, seq_a)
    if len(seq_b) == 0:
        return len(seq_a)

    prev_row = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a):
        curr_row = [i + 1]
        for j, item_b in enumerate(seq_b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (item_a != item_b)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_wer_score(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Calcula `wer` y `cer` sobre listas de referencias e hipótesis normalizadas.

    TODO (Reto 5.2):
    1. Valida que `len(references) == len(hypotheses)` (si no, lanza `ValueError`).
    2. Si `references` está vacío, devuelve `{"wer": 0.0, "cer": 0.0}`.
    3. Normaliza ambas listas llamando a `normalize_transcript(...)`.
    4. Usa `jiwer.wer(...)` y `jiwer.cer(...)` si `jiwer` está disponible; si da
       `ImportError`, acumula `_levenshtein_distance` sobre palabras y caracteres
       y divide por el total de palabras/caracteres de referencia.
    """
    raise NotImplementedError("Reto 5.2 pendiente: compute_wer_score")


def read_wav_pcm(wav_path: Path) -> tuple[int, np.ndarray]:
    """Lee un fichero `.wav` devolviendo `(sample_rate, audio_float32)`."""
    try:
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        return int(sr), normalize_pcm(audio, peak_normalize=False)
    except ImportError:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32)
        else:
            arr = np.frombuffer(raw, dtype=np.uint8)

        if channels > 1:
            arr = arr.reshape(-1, channels)
        return int(sr), normalize_pcm(arr, peak_normalize=False)


def discover_l2_arctic_samples(
    dataset_dir: Path | str,
    speakers: list[str] | None = None,
    max_samples: int = 50,
) -> list[tuple[Path, str]]:
    """Localiza pares `(audio.wav, transcripción)` en estructura L2-ARCTIC o plana."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    samples: list[tuple[Path, str]] = []
    target_speakers = {s.upper() for s in speakers} if speakers else None

    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if target_speakers and speaker_dir.name.upper() not in target_speakers:
            continue
        wav_dir = speaker_dir / "wav"
        txt_dir = speaker_dir / "transcript"
        if not wav_dir.is_dir() or not txt_dir.is_dir():
            continue

        for wav_path in sorted(wav_dir.glob("*.wav")):
            txt_path = txt_dir / f"{wav_path.stem}.txt"
            if not txt_path.exists():
                continue
            ref_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if ref_text:
                samples.append((wav_path, ref_text))
                if len(samples) >= max_samples:
                    return samples

    if not samples:
        for wav_path in sorted(root.rglob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            ref_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if ref_text:
                samples.append((wav_path, ref_text))
                if len(samples) >= max_samples:
                    break

    return samples


def evaluate_wer(
    dataset_dir: Path | str,
    max_samples: int = 50,
    model_size: str = "base",
    speakers: list[str] | None = None,
    device: str | None = None,
    stt_engine: FasterWhisperSTT | None = None,
) -> dict[str, Any]:
    """Ejecuta el benchmark completo de WER, CER, RTF y latencia sobre L2-ARCTIC.

    TODO (Reto 5.3):
    1. Obtén `samples = discover_l2_arctic_samples(dataset_dir, speakers=speakers, max_samples=max_samples)`.
       Si está vacío, lanza `ValueError(...)`.
    2. Usa `stt = stt_engine or FasterWhisperSTT(model_size=model_size, device=device)`.
    3. Para cada `(wav_path, ref_text)`:
       - Lee `sr, audio = read_wav_pcm(wav_path)` y acumula su duración (`audio.size / max(1, sr)`).
       - Mide con `time.perf_counter()` cuánto tarda `result = stt.transcribe(audio, sample_rate=sr, language="en")`.
       - Guarda `ref_text`, `result["text"]` y la latencia en ms.
    4. Calcula `scores = compute_wer_score(references, hypotheses)` y `rtf = (sum(latencies_ms)/1000.0) / max(1e-6, total_audio_sec)`.
    5. Devuelve diccionario con claves:
       `"model"`, `"device"`, `"num_samples"`, `"wer"`, `"cer"`, `"mean_latency_ms"`, `"p95_latency_ms"`, `"rtf"`.
    """
    raise NotImplementedError("Reto 5.3 pendiente: evaluate_wer")


def compare_models(
    dataset_dir: Path | str,
    models: tuple[str, ...] = ("tiny", "base", "small"),
    max_samples: int = 50,
    speakers: list[str] | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Compara varios modelos Whisper sobre el mismo subconjunto."""
    results: list[dict[str, Any]] = []
    for model_size in models:
        metrics = evaluate_wer(
            dataset_dir=dataset_dir,
            max_samples=max_samples,
            model_size=model_size,
            speakers=speakers,
            device=device,
        )
        results.append(metrics)
    return results


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    """Genera una tabla Markdown con resultados de WER/CER/latencia/RTF."""
    lines = [
        "| Modelo | Device | Muestras | WER (%) | CER (%) | Latencia Media (ms) | p95 (ms) | RTF |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| `{row['model']}` | `{row['device']}` | {row['num_samples']} | "
            f"{row['wer'] * 100:.2f}% | {row['cer'] * 100:.2f}% | "
            f"{row['mean_latency_ms']:.1f} | {row['p95_latency_ms']:.1f} | {row['rtf']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo WER/CER/RTF de YARNA sobre L2-ARCTIC."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    parser.add_argument("--speakers", nargs="*", default=list(SPANISH_L2_SPEAKERS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = compare_models(
        dataset_dir=args.dataset_dir,
        models=tuple(args.models),
        max_samples=args.max_samples,
        speakers=args.speakers,
        device=args.device,
    )
    print(format_markdown_table(results))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
