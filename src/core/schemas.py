"""Contratos de datos Pydantic del núcleo de YARNA (Sección 4.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CEFRLevel(str, Enum):
    """Niveles de competencia MCER."""

    A2 = "A2"
    B2_C1 = "B2/C1"


class SocialRegister(str, Enum):
    """Registros pragmáticos / ámbitos sociales."""

    FORMAL = "formal"
    INFORMAL = "informal"
    EVERYDAY = "everyday"


class CorrectionCategory(str, Enum):
    """Categoría del feedback en la tarjeta visual."""

    GRAMMAR = "grammar"
    PRAGMATICS = "pragmatics"
    OK = "ok"


class SessionProfile(BaseModel):
    """Tupla de control [Nivel MCER, Registro Social, Escenario]."""

    level: CEFRLevel = CEFRLevel.A2
    register_mode: SocialRegister = SocialRegister.EVERYDAY
    scenario: str = "Pedir en una cafetería"


class VisualCard(BaseModel):
    """Tarjeta visual asíncrona para corrección reflexiva."""

    correction: str = ""
    category: CorrectionCategory = CorrectionCategory.OK
    explanation: str = ""


class ProsodyFlags(BaseModel):
    """Parámetros de modulación prosódica para Piper TTS."""

    rate: float = Field(default=1.0, ge=0.5, le=1.5)
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)


class TutorResponse(BaseModel):
    """Contrato JSON estricto devuelto por el LLM."""

    transcript: str
    oral_response: str
    visual_card: VisualCard
    prosody_flags: ProsodyFlags
