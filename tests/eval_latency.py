"""Perfilado de latencia por etapa (RTF y p95 < 1.8s)."""

from __future__ import annotations


def run_latency_profile(num_turns: int = 20) -> dict[str, float]:
    """Mide T_VAD + T_STT + T_LLM + T_TTS, RTF y percentil p95 (< 1.8 s).

    TODO:
    1. Instrumentar marcas temporales con time.perf_counter() en cada etapa.
    2. Ejecutar num_turns turnos representativos del pipeline completo.
    3. Calcular RTF y percentil 95 verificando el cumplimiento de p95 < 1.8s.
    """
    raise NotImplementedError("Pendiente de implementar run_latency_profile.")


if __name__ == "__main__":
    # TODO: Invocar run_latency_profile() y reportar tiempos por etapa y p95.
    print("Perfilado de latencia (RTF y p95 < 1.8s) listo para implementar.")
