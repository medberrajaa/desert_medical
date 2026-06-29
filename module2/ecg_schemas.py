"""Schemas for ECG measurement and LLM interpretation outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ECGIntervals(StrictModel):
    rr_ms: float | None = None
    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    qtc_ms: float | None = None


class ECGFinding(StrictModel):
    finding: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)
    severity: Literal["info", "warning", "critical"]


class ECGDiagnosis(StrictModel):
    diagnosis: str = Field(..., min_length=1)
    evidence: list[str] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ECGInterpretation(StrictModel):
    rhythm: str = Field(..., min_length=1)
    heart_rate: str = Field(..., min_length=1)
    intervals: ECGIntervals
    axis: str = Field(..., min_length=1)
    findings: list[ECGFinding] = Field(default_factory=list)
    possible_diagnosis: list[ECGDiagnosis] = Field(default_factory=list)
    urgency: Literal["routine", "urgent_24h", "urgent_4h", "emergency"]
    recommended_action: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
