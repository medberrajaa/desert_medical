"""Prompt engineering for the general clinical decision support module.

The public API is intentionally compatible with the original project:
``build_prompt()``, ``build_raw_prompt()``, ``generate_diagnosis()``, and
``generate_raw_diagnosis()`` still exist. The implementation now requires
schema-valid JSON and keeps deterministic DDSS rules outside the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from module1.groq_client import DEFAULT_REASONING_MODEL, GroqConfigurationError, GroqJSONError, GroqJsonClient
from module1.schemas import ClinicalDiagnosisResponse, UrgencyAssessment

logger = logging.getLogger(__name__)


COUNTRY_CONTEXTS = {
    "France": {
        "healthcare": "French primary-care and emergency-care context",
        "resources": "blood pressure cuff, thermometer, stethoscope, 6/12-lead ECG, point-of-care glucose, limited portable blood tests",
        "constraints": "follow local escalation pathways and do not delay emergency transfer for life-threatening presentations",
    },
    "S\u00e9n\u00e9gal": {
        "healthcare": "Senegalese care context with variable access to laboratory and imaging resources",
        "resources": "WHO essential medicines, bedside vitals, point-of-care tests where available",
        "constraints": "prioritize syndromic triage, malaria/sepsis screening, and early referral when resources are limited",
    },
    "Maroc": {
        "healthcare": "Moroccan care context with possible rural access constraints",
        "resources": "standard primary-care medicines, vitals, ECG, and basic laboratory access where available",
        "constraints": "consider limited access to specialists in remote areas and escalate emergencies early",
    },
    "Belgique": {
        "healthcare": "Belgian care context",
        "resources": "standard primary-care and emergency-care diagnostics",
        "constraints": "follow local referral and prescribing rules",
    },
    "C\u00f4te d'Ivoire": {
        "healthcare": "Ivorian care context with variable resources",
        "resources": "WHO essential medicines, vitals, point-of-care tests where available",
        "constraints": "consider endemic tropical diseases and escalate unstable patients early",
    },
    "Cameroun": {
        "healthcare": "Cameroonian care context with regional resource variation",
        "resources": "WHO essential medicines, vitals, point-of-care tests where available",
        "constraints": "consider malaria and other endemic infections; avoid delayed referral for unstable patients",
    },
    "Benin": {
        "healthcare": "Beninese care context with variable access to advanced diagnostics",
        "resources": "WHO essential medicines, vitals, point-of-care tests where available",
        "constraints": "consider tropical infections and local referral constraints",
    },
    "Autre": {
        "healthcare": "local clinical context",
        "resources": "available local resources",
        "constraints": "adapt recommendations to available staff, diagnostics, and referral pathways",
    },
}

LANGUAGE_INSTRUCTIONS = {
    "Fran\u00e7ais": "Write all string values in French. Keep JSON keys unchanged.",
    "English": "Write all string values in English. Keep JSON keys unchanged.",
    "Arabic": "Write all string values in Arabic. Keep JSON keys unchanged.",
}

URGENCY_RANK = {"routine": 0, "urgent_24h": 1, "urgent_4h": 2, "emergency": 3}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text_blob(patient: dict[str, Any]) -> str:
    return " ".join(
        str(patient.get(k, ""))
        for k in ("motif_consultation", "signes_cliniques", "symptomes", "resultats")
    ).lower()


DDSS_RULES = [
    {
        "name": "acute_coronary_syndrome_risk",
        "label": "Syndrome coronarien aigu (SCA)",
        "description": "Douleur thoracique ou termes ECG ischémiques chez l'adulte (≥ 35 ans) → priorité STEMI/NSTEMI.",
        "condition": lambda p: (
            any(kw in _text_blob(p) for kw in ["douleur thoracique", "chest pain", "oppression", "irradiation bras", "st elevation", "sus-decalage", "sus d\u00e9calage"])
            and _safe_int(p.get("age")) >= 35
        ),
        "hint": "Chest pain or ischemic ECG wording in an adult: prioritize ACS/STEMI/NSTEMI and emergency escalation until excluded.",
        "urgency_floor": "emergency",
    },
    {
        "name": "possible_stroke",
        "label": "Accident vasculaire cérébral (AVC)",
        "description": "Déficit neurologique focal (hémiplégie, aphasie, paralysie faciale) → suspicion d'AVC/AIT.",
        "condition": lambda p: any(
            kw in _text_blob(p)
            for kw in ["hemiplegie", "h\u00e9mipl\u00e9gie", "aphasie", "facial droop", "paralysie", "trouble parole", "deviation bouche"]
        ),
        "hint": "Focal neurologic deficit: treat as possible stroke/TIA and escalate immediately.",
        "urgency_floor": "emergency",
    },
    {
        "name": "possible_sepsis",
        "label": "Sepsis",
        "description": "Fièvre associée à une instabilité ou des marqueurs biologiques (CRP, GB, tachycardie).",
        "condition": lambda p: (
            any(kw in _text_blob(p) for kw in ["fievre", "fi\u00e8vre", "frissons", "hypotension", "confusion", "sepsis"])
            and any(kw in _text_blob(p) for kw in ["38", "39", "40", "crp", "gb", "leucocytes", "tachycardie"])
        ),
        "hint": "Fever plus instability/labs may indicate sepsis. Use qSOFA/NEWS-style escalation and urgent clinician review.",
        "urgency_floor": "emergency",
    },
    {
        "name": "respiratory_distress",
        "label": "Détresse respiratoire",
        "description": "Dyspnée, désaturation (SpO₂), cyanose → compromission respiratoire possible.",
        "condition": lambda p: any(
            kw in _text_blob(p)
            for kw in ["dyspnee", "dyspn\u00e9e", "essoufflement", "spo2", "saturation", "cyanose", "respiratory distress"]
        ),
        "hint": "Possible respiratory compromise. Document SpO2, respiratory rate, work of breathing, and escalate if abnormal.",
        "urgency_floor": "urgent_4h",
    },
    {
        "name": "tropical_fever_context",
        "label": "Fièvre en contexte tropical",
        "description": "Fièvre en zone d'endémie (Sénégal, Côte d'Ivoire, Cameroun, Bénin) → inclure paludisme, typhoïde, dengue.",
        "condition": lambda p: (
            p.get("country") in {"S\u00e9n\u00e9gal", "C\u00f4te d'Ivoire", "Cameroun", "Benin"}
            and any(kw in _text_blob(p) for kw in ["fievre", "fi\u00e8vre", "frissons", "cephalee", "c\u00e9phal\u00e9e", "sueurs"])
        ),
        "hint": "Fever in a tropical/endemic context: include malaria, typhoid, dengue, and sepsis where clinically compatible.",
        "urgency_floor": "urgent_4h",
    },
]


URGENCY_FLOOR_LABELS = {
    "emergency": "Urgence absolue",
    "urgent_4h": "Urgent (< 4 h)",
    "urgent_24h": "Urgent (< 24 h)",
    "routine": "Non urgent",
}


def ddss_catalog() -> list[dict[str, str]]:
    """UI-friendly, serializable view of the DDSS safety rules (no lambdas)."""
    return [
        {
            "name": rule["name"],
            "label": rule.get("label", rule["name"]),
            "description": rule.get("description", ""),
            "urgency_floor": rule["urgency_floor"],
            "urgency_label": URGENCY_FLOOR_LABELS.get(rule["urgency_floor"], rule["urgency_floor"]),
        }
        for rule in DDSS_RULES
    ]


def apply_ddss_rules(patient: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply deterministic safety rules before LLM reasoning."""
    triggered: list[dict[str, Any]] = []
    for rule in DDSS_RULES:
        try:
            if rule["condition"](patient):
                triggered.append(rule)
        except Exception as exc:
            logger.debug("ddss_rule_failed", extra={"rule": rule["name"], "error": str(exc)})
    return triggered


def _guess_age(text: str) -> int:
    patterns = [
        r"\b(\d{1,3})\s*ans?\b",
        r"\bage[\s:]+(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:years old|yo)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 0 < age < 120:
                return age
    return 0


def _compact_patient(patient: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "age",
        "sexe",
        "sex",
        "ethnie",
        "taille",
        "poids",
        "duree_symptomes",
        "motif_consultation",
        "signes_cliniques",
        "antecedents",
        "traitements",
        "examens",
        "resultats",
        "symptomes",
    }
    return {k: v for k, v in patient.items() if k in allowed and v not in (None, "", [])}


def _base_system_prompt(country: str, language: str, triggered_rules: list[dict[str, Any]]) -> str:
    ctx = COUNTRY_CONTEXTS.get(country, COUNTRY_CONTEXTS["Autre"])
    language_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["Fran\u00e7ais"])
    ddss_hints = [rule["hint"] for rule in triggered_rules]
    return (
        "You are a clinical decision support assistant for nurses. You are not the treating physician. "
        "Use conservative medical reasoning, prioritize patient safety, and never invent unavailable facts. "
        "Never state a diagnosis unless at least one evidence item supports it. If evidence is weak, say so in "
        "missing_information and limitations. Escalate emergencies. Return one JSON object only.\n\n"
        f"Context: {ctx['healthcare']}.\n"
        f"Available resources: {ctx['resources']}.\n"
        f"Constraints: {ctx['constraints']}.\n"
        f"Language: {language_instruction}\n"
        f"Deterministic safety alerts: {json.dumps(ddss_hints, ensure_ascii=False)}"
    )


def _user_prompt(patient_payload: dict[str, Any], country: str, raw_text: bool = False) -> str:
    mode = "free_text" if raw_text else "structured_form"
    return (
        "Analyze the following patient information for nurse-facing clinical decision support. "
        "Use only the supplied evidence. Keep the output concise but complete. "
        "Include possible diagnosis, differential diagnosis, urgency, labs, imaging, treatments, "
        "confidence, explanation, red flags, questions, uncertainty, and limitations.\n\n"
        f"mode={mode}\n"
        f"country={country}\n"
        f"patient={json.dumps(patient_payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_prompt(patient: dict, country: str, language: str) -> tuple[str, list[dict[str, Any]]]:
    patient_with_country = {**patient, "country": country}
    triggered_rules = apply_ddss_rules(patient_with_country)
    prompt = _base_system_prompt(country, language, triggered_rules) + "\n\n" + _user_prompt(
        _compact_patient(patient),
        country,
        raw_text=False,
    )
    return prompt, triggered_rules


def build_raw_prompt(texte: str, country: str, language: str) -> tuple[str, list[dict[str, Any]]]:
    pseudo_patient = {
        "raw_text": texte.strip(),
        "motif_consultation": texte,
        "signes_cliniques": texte,
        "resultats": texte,
        "age": _guess_age(texte),
        "country": country,
    }
    triggered_rules = apply_ddss_rules(pseudo_patient)
    prompt = _base_system_prompt(country, language, triggered_rules) + "\n\n" + _user_prompt(
        {"raw_text": texte.strip()},
        country,
        raw_text=True,
    )
    return prompt, triggered_rules


def _messages_from_prompt(prompt: str) -> list[dict[str, str]]:
    system, _, user = prompt.partition("\n\nAnalyze the following patient information")
    if not user:
        return [{"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Analyze the following patient information" + user},
    ]


def _urgency_floor(triggered_rules: list[dict[str, Any]]) -> str | None:
    floors = [rule.get("urgency_floor") for rule in triggered_rules if rule.get("urgency_floor")]
    if not floors:
        return None
    return max(floors, key=lambda level: URGENCY_RANK.get(level, 0))


def _apply_urgency_floor(
    response: ClinicalDiagnosisResponse,
    triggered_rules: list[dict[str, Any]],
) -> ClinicalDiagnosisResponse:
    floor = _urgency_floor(triggered_rules)
    if floor is None:
        return response
    if URGENCY_RANK[response.urgency.level] >= URGENCY_RANK[floor]:
        return response

    rule_names = ", ".join(rule["name"] for rule in triggered_rules if rule.get("urgency_floor") == floor)
    new_urgency = UrgencyAssessment(
        level=floor,  # type: ignore[arg-type]
        rationale=(
            f"Urgency raised by deterministic safety rule(s): {rule_names}. "
            f"Original model rationale: {response.urgency.rationale}"
        ),
        escalation_required=floor in {"urgent_4h", "emergency"},
        emergency_actions=response.urgency.emergency_actions
        or ["Seek urgent clinician review and follow local emergency escalation protocol."],
    )
    limitations = list(response.limitations)
    limitations.append("Urgency was raised by deterministic safety rules to minimize false-negative emergencies.")
    return response.model_copy(update={"urgency": new_urgency, "limitations": limitations})


def _result_dict(
    *,
    prompt: str,
    response: ClinicalDiagnosisResponse,
    model: str,
    tokens_used: int | None,
    country: str,
    language: str,
    triggered_rules: list[dict[str, Any]],
    attempts: int,
    cached: bool,
) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    return {
        "success": True,
        "prompt": prompt,
        "response": json.dumps(payload, ensure_ascii=False, indent=2),
        "data": payload,
        "validated_response": response,
        "model": model,
        "tokens_used": tokens_used or 0,
        "country": country,
        "language": language,
        "ddss_alerts": [rule["name"] for rule in triggered_rules],
        "ddss_urgency_floor": _urgency_floor(triggered_rules),
        "attempts": attempts,
        "cached": cached,
    }


def _generate(prompt: str, triggered_rules: list[dict[str, Any]], country: str, language: str) -> dict[str, Any]:
    client = GroqJsonClient()
    try:
        result = client.call_json(
            messages=_messages_from_prompt(prompt),
            schema_model=ClinicalDiagnosisResponse,
            schema_name="clinical_diagnosis_response",
            model=DEFAULT_REASONING_MODEL,
            temperature=0.0,
            max_tokens=4096,
            reasoning_effort="low",
        )
        validated = ClinicalDiagnosisResponse.model_validate(result.data.model_dump())
        validated = _apply_urgency_floor(validated, triggered_rules)
        return _result_dict(
            prompt=prompt,
            response=validated,
            model=result.model,
            tokens_used=result.tokens_used,
            country=country,
            language=language,
            triggered_rules=triggered_rules,
            attempts=result.attempts,
            cached=result.cached,
        )
    except (GroqConfigurationError, GroqJSONError, ValidationError, Exception) as exc:
        logger.exception("clinical_generation_failed")
        return {
            "success": False,
            "error": str(exc),
            "prompt": prompt,
            "model": DEFAULT_REASONING_MODEL,
            "country": country,
            "language": language,
            "ddss_alerts": [rule["name"] for rule in triggered_rules],
            "ddss_urgency_floor": _urgency_floor(triggered_rules),
        }


def generate_diagnosis(patient: dict, country: str = "France", language: str = "Fran\u00e7ais") -> dict:
    """Generate a validated JSON clinical decision support response."""
    prompt, triggered_rules = build_prompt(patient, country, language)
    return _generate(prompt, triggered_rules, country, language)


def generate_raw_diagnosis(texte: str, country: str = "France", language: str = "Fran\u00e7ais") -> dict:
    """Generate a validated JSON response from free-text patient information."""
    prompt, triggered_rules = build_raw_prompt(texte, country, language)
    return _generate(prompt, triggered_rules, country, language)
