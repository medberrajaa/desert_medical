"""LLM interpretation layer for hybrid ECG reports."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from module1.groq_client import DEFAULT_REASONING_MODEL, GroqConfigurationError, GroqJSONError, GroqJsonClient
from module2.ecg_measurements import compact_measurements_for_llm
from module2.ecg_schemas import ECGDiagnosis, ECGFinding, ECGInterpretation, ECGIntervals

logger = logging.getLogger(__name__)


URGENCY_RANK = {"routine": 0, "urgent_24h": 1, "urgent_4h": 2, "emergency": 3}


def _critical_floor(abnormalities: list[dict[str, Any]]) -> str:
    if any(alert.get("severity") == "critical" for alert in abnormalities):
        return "emergency"
    if any(alert.get("severity") == "warning" for alert in abnormalities):
        return "urgent_4h"
    return "routine"


def _rhythm_from_measurements(measurements: dict[str, Any], abnormalities: list[dict[str, Any]]) -> str:
    rules = {alert.get("rule") for alert in abnormalities}
    if "VF" in rules:
        return "chaotic rhythm; possible ventricular fibrillation"
    if "VT" in rules:
        return "regular wide-complex tachycardia"
    if "AF" in rules:
        return "irregularly irregular rhythm; possible atrial fibrillation"
    hr = measurements.get("rhythm", {}).get("heart_rate_bpm")
    if hr is None:
        return "rhythm indeterminate"
    if hr < 60:
        return "sinus bradycardia pattern if P waves are present"
    if hr > 100:
        return "sinus tachycardia pattern if P waves are present"
    return "regular rhythm; sinus rhythm possible if P waves are present"


def deterministic_ecg_interpretation(
    *,
    measurements: dict[str, Any],
    classifier_context: dict[str, Any] | None = None,
    extra_limitation: str | None = None,
) -> ECGInterpretation:
    """Safe fallback when the LLM is unavailable or disabled."""
    abnormalities = measurements.get("detected_abnormalities", [])
    intervals = measurements.get("intervals", {})
    rhythm = measurements.get("rhythm", {})
    urgency = _critical_floor(abnormalities)
    findings: list[ECGFinding] = []
    diagnoses: list[ECGDiagnosis] = []

    for alert in abnormalities:
        evidence = alert.get("evidence") or [alert.get("detail", "Detected by deterministic ECG rule.")]
        findings.append(
            ECGFinding(
                finding=alert.get("label", alert.get("rule", "ECG abnormality")),
                evidence=alert.get("detail", "; ".join(evidence)),
                severity=alert.get("severity", "info"),
            )
        )
        confidence = 0.88 if alert.get("severity") == "critical" else 0.70 if alert.get("severity") == "warning" else 0.55
        diagnoses.append(
            ECGDiagnosis(
                diagnosis=alert.get("label", alert.get("rule", "ECG abnormality")),
                evidence=[str(item) for item in evidence],
                confidence=confidence,
            )
        )

    if not diagnoses:
        findings.append(
            ECGFinding(
                finding="No deterministic high-risk ECG pattern detected",
                evidence="No STEMI, VT/VF, wide QRS, or marked irregular rhythm rule was triggered.",
                severity="info",
            )
        )
        diagnoses.append(
            ECGDiagnosis(
                diagnosis="No acute high-risk ECG pattern detected by deterministic analysis",
                evidence=["No deterministic critical alert was triggered."],
                confidence=0.60,
            )
        )

    limitations = list(measurements.get("limitations", []))
    if classifier_context and classifier_context.get("available") is False:
        limitations.append("The trained ECG classifier file is not available; interpretation uses signal rules and LLM/fallback reasoning.")
    if extra_limitation:
        limitations.append(extra_limitation)

    action = "Routine clinician review."
    if urgency == "emergency":
        action = "Activate emergency pathway now; obtain clinician/cardiology review, repeat ECG, vitals, and troponin as locally available."
    elif urgency == "urgent_4h":
        action = "Arrange urgent clinician review and correlate with symptoms, vitals, and labs."

    return ECGInterpretation(
        rhythm=_rhythm_from_measurements(measurements, abnormalities),
        heart_rate=f"{rhythm.get('heart_rate_bpm')} bpm" if rhythm.get("heart_rate_bpm") is not None else "not measurable",
        intervals=ECGIntervals(
            rr_ms=intervals.get("rr_ms"),
            pr_ms=intervals.get("pr_ms"),
            qrs_ms=intervals.get("qrs_ms"),
            qt_ms=intervals.get("qt_ms"),
            qtc_ms=intervals.get("qtc_ms"),
        ),
        axis=measurements.get("axis", "indeterminate"),
        findings=findings,
        possible_diagnosis=diagnoses,
        urgency=urgency,  # type: ignore[arg-type]
        recommended_action=action,
        confidence=max((diag.confidence for diag in diagnoses), default=0.55),
        limitations=limitations,
    )


def _messages(
    *,
    patient_info: dict[str, Any] | None,
    measurements: dict[str, Any],
    classifier_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    compact = compact_measurements_for_llm(measurements)
    system = (
        "You are a board-certified cardiologist with expertise in electrocardiography. "
        "Interpret the ECG from measured signal features, not from imagination. "
        "Never invent measurements. Never diagnose without evidence. If uncertain, say so in limitations. "
        "Prioritize sensitivity for acute myocardial infarction: STEMI, NSTEMI/ischemia, anterior, inferior, "
        "lateral, posterior, and old infarction patterns must be considered. "
        "ST elevation in contiguous leads must be escalated as emergency until proven otherwise. "
        "Return one JSON object only that conforms to the schema."
    )
    user = {
        "patient_info": patient_info or {},
        "measurements": compact,
        "classifier_context": classifier_context or {},
        "required_safety_rules": [
            "Explain which ECG findings support each possible diagnosis.",
            "Do not call NSTEMI confirmed from ECG alone; state that troponin and clinical context are required.",
            "Escalate STEMI, posterior STEMI equivalent, VT, VF, and severe hyperkalemia patterns as emergency.",
            "If amplitude is relative or image-derived, state measurement limitations.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
    ]


def _apply_urgency_floor(report: ECGInterpretation, abnormalities: list[dict[str, Any]]) -> ECGInterpretation:
    floor = _critical_floor(abnormalities)
    if URGENCY_RANK[report.urgency] >= URGENCY_RANK[floor]:
        return report
    limitations = list(report.limitations)
    limitations.append("Urgency was raised by deterministic ECG safety rules.")
    action = report.recommended_action
    if floor == "emergency":
        action = "Activate emergency pathway now; deterministic ECG rules detected a critical pattern. " + action
    return report.model_copy(update={"urgency": floor, "recommended_action": action, "limitations": limitations})


def interpret_ecg_with_llm(
    *,
    measurements: dict[str, Any],
    patient_info: dict[str, Any] | None = None,
    classifier_context: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Return a validated ECG clinical report, using LLM then fallback rules."""
    abnormalities = measurements.get("detected_abnormalities", [])

    if not use_llm:
        fallback = deterministic_ecg_interpretation(
            measurements=measurements,
            classifier_context=classifier_context,
            extra_limitation="LLM interpretation disabled; deterministic fallback report used.",
        )
        return {
            "success": True,
            "source": "deterministic_fallback",
            "model": None,
            "tokens_used": 0,
            "data": fallback.model_dump(mode="json"),
            "validated_response": fallback,
        }

    client = GroqJsonClient()
    try:
        result = client.call_json(
            messages=_messages(
                patient_info=patient_info,
                measurements=measurements,
                classifier_context=classifier_context,
            ),
            schema_model=ECGInterpretation,
            schema_name="ecg_interpretation",
            model=DEFAULT_REASONING_MODEL,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="low",
        )
        report = ECGInterpretation.model_validate(result.data.model_dump())
        report = _apply_urgency_floor(report, abnormalities)
        return {
            "success": True,
            "source": "llm",
            "model": result.model,
            "tokens_used": result.tokens_used or 0,
            "attempts": result.attempts,
            "cached": result.cached,
            "data": report.model_dump(mode="json"),
            "validated_response": report,
        }
    except (GroqConfigurationError, GroqJSONError, ValidationError, Exception) as exc:
        logger.warning("ecg_llm_interpretation_failed", extra={"error": str(exc)})
        fallback = deterministic_ecg_interpretation(
            measurements=measurements,
            classifier_context=classifier_context,
            extra_limitation=f"LLM interpretation unavailable: {exc}",
        )
        fallback = _apply_urgency_floor(fallback, abnormalities)
        return {
            "success": True,
            "source": "deterministic_fallback",
            "model": DEFAULT_REASONING_MODEL,
            "tokens_used": 0,
            "data": fallback.model_dump(mode="json"),
            "validated_response": fallback,
            "llm_error": str(exc),
        }
