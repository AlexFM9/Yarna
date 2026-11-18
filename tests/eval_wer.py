"""Benchmark offline de Word Error Rate (WER) y RTF sobre subconjunto L2-ARCTIC."""

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
    """Normaliza transcripciones inglesas evitando penalizaciones por puntuación."""
    try:
        import jiwer

        transform = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.ExpandCommonEnglishContractions(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        return str(transform(text))
    except ImportError:
        cleaned = text.lower().replace("’", "'")
        for contraction, expanded in _ENGLISH_CONTRACTIONS.items():
            cleaned = re.sub(rf"\b{re.escape(contraction)}\b", expanded, cleaned)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


def _levenshtein_distance(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Distancia de edición de Levenshtein para cálculo fallback de WER/CER."""
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
    """Calcula WER y CER sobre listas de referencias e hipótesis normalizadas."""
    if len(references) != len(hypotheses):
        raise ValueError("references e hypotheses deben tener la misma longitud.")
    if not references:
        return {"wer": 0.0, "cer": 0.0}

    norm_refs = [normalize_transcript(r) for r in references]
    norm_hyps = [normalize_transcript(h) for h in hypotheses]

    try:
        import jiwer

        wer_val = float(jiwer.wer(norm_refs, norm_hyps))
        cer_val = float(jiwer.cer(norm_refs, norm_hyps))
        return {"wer": wer_val, "cer": cer_val}
    except ImportError:
        word_edits = 0
        total_ref_words = 0
        char_edits = 0
        total_ref_chars = 0

        for ref, hyp in zip(norm_refs, norm_hyps):
            ref_words = ref.split()
            hyp_words = hyp.split()
            word_edits += _levenshtein_distance(ref_words, hyp_words)
            total_ref_words += len(ref_words)

            char_edits += _levenshtein_distance(ref, hyp)
            total_ref_chars += len(ref)

        wer_val = word_edits / max(1, total_ref_words)
        cer_val = char_edits / max(1, total_ref_chars)
        return {"wer": wer_val, "cer": cer_val}


def read_wav_pcm(wav_path: Path) -> tuple[int, np.ndarray]:
    """Lee un fichero WAV devolviendo (sample_rate, audio_float32)."""
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
    """Localiza pares (audio.wav, transcripción) en estructura L2-ARCTIC o plana."""
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
    """Evalúa WER, CER, RTF y latencia del STT sobre muestras L2-ARCTIC."""
    samples = discover_l2_arctic_samples(
        dataset_dir=dataset_dir,
        speakers=speakers,
        max_samples=max_samples,
    )
    if not samples:
        raise ValueError(f"No se encontraron pares (.wav, .txt) válidos en {dataset_dir}")

    stt = stt_engine or FasterWhisperSTT(model_size=model_size, device=device)

    references: list[str] = []
    hypotheses: list[str] = []
    latencies_ms: list[float] = []
    total_audio_sec = 0.0

    for wav_path, ref_text in samples:
        sr, audio = read_wav_pcm(wav_path)
        audio_duration_sec = audio.size / max(1, sr)
        total_audio_sec += audio_duration_sec

        t0 = time.perf_counter()
        result = stt.transcribe(audio, sample_rate=sr, language="en")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        references.append(ref_text)
        hypotheses.append(str(result.get("text", "")))
        latencies_ms.append(elapsed_ms)

    scores = compute_wer_score(references, hypotheses)
    total_infer_sec = sum(latencies_ms) / 1000.0
    rtf = total_infer_sec / max(1e-6, total_audio_sec)

    return {
        "model": stt.model_name,
        "device": stt.device,
        "num_samples": len(samples),
        "wer": round(scores["wer"], 4),
        "cer": round(scores["cer"], 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "rtf": round(rtf, 4),
    }


def compare_models(
    dataset_dir: Path | str,
    models: tuple[str, ...] = ("tiny", "base", "small"),
    max_samples: int = 50,
    speakers: list[str] | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Ejecuta evaluación comparativa de WER y latencia entre varios modelos Whisper."""
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
    """Formatea los resultados comparativos en tabla Markdown lista para memoria."""
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
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Ruta raíz al dataset L2-ARCTIC (o carpeta con pares .wav/.txt).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "base", "small"],
        help="Tamaños de modelo a comparar (por defecto: tiny base small).",
    )
    parser.add_argument(
        "--speakers",
        nargs="*",
        default=list(SPANISH_L2_SPEAKERS),
        help="Hablantes a evaluar (por defecto L1 español: EBVS ERMS MBMPS NJS).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Número máximo de muestras a evaluar.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de inferencia (auto, cuda, cpu).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Ruta opcional para guardar métricas en JSON.",
    )
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
