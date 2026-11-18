"""Benchmark offline de Word Error Rate (WER) sobre subconjunto L2-ARCTIC."""

from __future__ import annotations

from pathlib import Path


def evaluate_wer(dataset_dir: Path | str, max_samples: int = 50) -> dict[str, float]:
    """Evalúa el WER del módulo STT usando jiwer sobre muestras de L2-ARCTIC.

    TODO:
    1. Cargar pares (audio.wav, transcripción_referencia) desde dataset_dir.
    2. Ejecutar FasterWhisperSTT.transcribe sobre cada audio normalizado a 16 kHz.
    3. Calcular el WER medio mediante jiwer.wer y devolver el resumen de métricas.
    """
    raise NotImplementedError("Pendiente de implementar evaluate_wer.")


if __name__ == "__main__":
    # TODO: Configurar ruta al subconjunto L2-ARCTIC y ejecutar evaluate_wer().
    print("Benchmark WER listo para implementar sobre el dataset L2-ARCTIC.")
