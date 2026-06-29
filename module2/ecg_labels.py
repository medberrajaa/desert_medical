"""Shared French display helpers for ECG diagnoses (used by both the Gradio and Streamlit UIs)."""

from __future__ import annotations

# French translation of the diagnosis labels produced by the measurement engine.
FR_ECG_LABELS = {
    "Possible acute anterior STEMI": "Infarctus du myocarde (STEMI antérieur)",
    "Possible acute inferior STEMI": "Infarctus du myocarde (STEMI inférieur)",
    "Possible acute lateral STEMI": "Infarctus du myocarde (STEMI latéral)",
    "Possible acute septal STEMI": "Infarctus du myocarde (STEMI septal)",
    "Possible posterior STEMI equivalent": "Infarctus du myocarde (STEMI postérieur)",
    "Possible ischemia / NSTEMI pattern": "Ischémie myocardique / NSTEMI possible",
    "Possible ventricular tachycardia": "Tachycardie ventriculaire possible",
    "Possible ventricular fibrillation / chaotic rhythm": "Fibrillation ventriculaire possible",
    "Possible atrial fibrillation": "Fibrillation auriculaire possible",
    "Possible right bundle branch block": "Bloc de branche droit possible",
    "Possible left bundle branch block": "Bloc de branche gauche possible",
    "Possible hyperkalemia pattern": "Hyperkaliémie possible",
    "Possible acute pericarditis pattern": "Péricardite aiguë possible",
    "Prolonged QTc": "QTc allongé",
    "Premature beats detected": "Extrasystoles détectées",
    "Possible old anterior infarction": "Séquelle d'infarctus antérieur",
    "Possible old inferior infarction": "Séquelle d'infarctus inférieur",
    "Possible old lateral infarction": "Séquelle d'infarctus latéral",
    "Possible old septal infarction": "Séquelle d'infarctus septal",
    "Possible old posterior infarction": "Séquelle d'infarctus postérieur",
    "Normal sinus rhythm": "Rythme sinusal normal",
    "Atrial fibrillation": "Fibrillation auriculaire",
    "Atrial flutter": "Flutter auriculaire",
    "Sinus tachycardia": "Tachycardie sinusale",
    "Sinus bradycardia": "Bradycardie sinusale",
    "First-degree AV block": "Bloc auriculo-ventriculaire du 1er degré",
    "Premature beats": "Extrasystoles",
    "Ventricular paced rhythm": "Rythme électro-entraîné ventriculaire",
    "Atrial paced rhythm": "Rythme électro-entraîné auriculaire",
    "Sinus arrhythmia": "Arythmie sinusale",
    "Pas d'anomalie aiguë détectée": "Pas d'anomalie aiguë détectée",
}


def fr_ecg_label(label: str) -> str:
    return FR_ECG_LABELS.get(label, label)


def ecg_urgency_badge(urgency) -> tuple[str, str]:
    """Normalise an urgency (French string or English level) to (icon, French label)."""
    u = str(urgency or "").lower()
    if u in ("emergency", "critique", "critical") or "critiqu" in u or "absolu" in u:
        return "🔴", "Urgence absolue"
    if u in ("urgent_4h", "élevé", "eleve", "high") or "élev" in u or "elev" in u:
        return "🔴", "Urgence élevée"
    if u in ("urgent_24h", "modéré", "modere", "moderate") or "modér" in u or "moder" in u:
        return "🟡", "Urgence modérée"
    return "🟢", "Non urgent"
