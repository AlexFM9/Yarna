"""Plantillas de System Prompt parametrizadas por nivel MCER y registro social."""

from __future__ import annotations

from src.core.schemas import SessionProfile


def build_system_prompt(profile: SessionProfile) -> str:
    """Construye el System Prompt exigiendo recast implícito y contrato JSON estricto."""
    # TODO: Construir el prompt combinando profile.level, profile.register_mode y
    # profile.scenario, instruyendo el formato JSON de TutorResponse.
    raise NotImplementedError("Pendiente de implementar build_system_prompt.")


def get_reengagement_prompt(profile: SessionProfile) -> str:
    """Genera un prompt corto de reenganche cuando expira silencio_timeout."""
    # TODO: Devolver una frase breve de reenganche adaptada al nivel MCER del alumno.
    raise NotImplementedError("Pendiente de implementar get_reengagement_prompt.")
