"""Pydantic schemas for deterministic clinical LLM outputs.

The schemas are intentionally explicit. They force evidence, uncertainty, and
limitations to be represented as data instead of free text hidden in prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


UrgencyLevel = Literal["routine", "urgent_24h", "urgent_4h", "emergency"]
SeverityLevel = Literal["low", "moderate", "high", "critical"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceItem(StrictModel):
    finding: str = Field(..., min_length=1)
    supports: str = Field(..., min_length=1)


class DiagnosisCandidate(StrictModel):
    name: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(..., min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    unsafe_to_exclude: bool = False

    @field_validator("evidence")
    @classmethod
    def require_non_empty_evidence(cls, value: list[EvidenceItem]) -> list[EvidenceItem]:
        if not value:
            raise ValueError("each diagnosis must include supporting evidence")
        return value


class Recommendation(StrictModel):
    name: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    priority: Literal["immediate", "same_day", "routine"]


class TreatmentRecommendation(StrictModel):
    action: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    cautions: list[str] = Field(default_factory=list)
    requires_prescriber: bool = True


class RedFlag(StrictModel):
    sign: str = Field(..., min_length=1)
    why_it_matters: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)


class UrgencyAssessment(StrictModel):
    level: UrgencyLevel
    rationale: str = Field(..., min_length=1)
    escalation_required: bool
    emergency_actions: list[str] = Field(default_factory=list)


class ExtractedPatientData(StrictModel):
    age: str = "not_provided"
    sex: str = "not_provided"
    chief_complaint: str = "not_provided"
    duration: str = "not_provided"
    observed_signs: str = "not_provided"
    history: str = "not_provided"
    medications: str = "not_provided"
    test_results: str = "not_provided"


class ClinicalDiagnosisResponse(StrictModel):
    extracted_data: ExtractedPatientData
    possible_diagnosis: list[DiagnosisCandidate] = Field(..., min_length=1)
    differential_diagnosis: list[DiagnosisCandidate] = Field(..., min_length=1)
    urgency: UrgencyAssessment
    recommended_laboratory_tests: list[Recommendation] = Field(default_factory=list)
    recommended_imaging: list[Recommendation] = Field(default_factory=list)
    treatments: list[TreatmentRecommendation] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    red_flags: list[RedFlag] = Field(default_factory=list)
    questions_for_nurse: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    safety_note: str = Field(
        "Clinical decision support only. A licensed clinician must validate diagnosis and treatment."
    )


class VoicePatientFields(StrictModel):
    age: int | None = Field(default=None, ge=0, le=120)
    sexe: Literal["Homme", "Femme", "Non precise"] | None = None
    taille: int | None = Field(default=None, ge=30, le=250)
    poids: float | None = Field(default=None, ge=1, le=400)
    ethnie: str | None = None
    motif: str | None = None
    signes: str | None = None
    duree: str | None = None
    antecedents: str | None = None
    traitements: str | None = None
    resultats: str | None = None
