"""Render the structured clinical diagnosis (ClinicalDiagnosisResponse) as Markdown.

The LLM now returns a validated JSON object instead of free-text Markdown. The UI
must turn that structured data into a readable clinical note for the nurse, instead
of dumping raw JSON.
"""

from __future__ import annotations

from typing import Any

_URGENCY = {
    "emergency": ("🔴", "Urgence absolue"),
    "urgent_4h": ("🟠", "Urgent (< 4 h)"),
    "urgent_24h": ("🟡", "Urgent (< 24 h)"),
    "routine": ("🟢", "Non urgent"),
}

_PRIORITY_FR = {"immediate": "immédiat", "same_day": "le jour même", "routine": "programmé"}


def _pct(value: Any) -> str:
    try:
        return f"{round(float(value) * 100)} %"
    except (TypeError, ValueError):
        return "—"


def _diag_block(title: str, items: list[dict[str, Any]], limit: int = 4) -> list[str]:
    if not items:
        return []
    lines = [f"### {title}"]
    for d in items[:limit]:
        name = d.get("name", "—")
        flag = " · ⚠️ à ne pas exclure" if d.get("unsafe_to_exclude") else ""
        lines.append(f"- **{name}** — probabilité estimée {_pct(d.get('probability'))}{flag}")
        for ev in (d.get("evidence") or [])[:3]:
            finding = ev.get("finding", "")
            supports = ev.get("supports", "")
            if finding:
                lines.append(f"    - {finding} → *{supports}*")
        for miss in (d.get("missing_information") or [])[:2]:
            lines.append(f"    - ❓ Information manquante : {miss}")
    return lines


def format_clinical_markdown(data: dict[str, Any]) -> str:
    """Turn a ClinicalDiagnosisResponse dict into a readable French clinical note."""
    if not data:
        return "_Aucune donnée de diagnostic._"

    lines: list[str] = []

    urgency = data.get("urgency", {})
    icon, label = _URGENCY.get(urgency.get("level", "routine"), ("⚪", urgency.get("level", "—")))
    lines.append(f"## {icon} Niveau d'urgence : {label}")
    if urgency.get("rationale"):
        lines.append(f"> {urgency['rationale']}")
    for action in (urgency.get("emergency_actions") or [])[:4]:
        lines.append(f"- 🚑 {action}")
    lines.append("")

    lines += _diag_block("🩺 Diagnostics possibles", data.get("possible_diagnosis", []))
    lines.append("")
    lines += _diag_block("🔄 Diagnostics différentiels", data.get("differential_diagnosis", []))
    lines.append("")

    red_flags = data.get("red_flags", [])
    if red_flags:
        lines.append("### 🚩 Signes d'alerte (red flags)")
        for rf in red_flags[:5]:
            lines.append(f"- **{rf.get('sign', '')}** — {rf.get('why_it_matters', '')}")
            if rf.get("action"):
                lines.append(f"    - Conduite : {rf['action']}")
        lines.append("")

    labs = data.get("recommended_laboratory_tests", [])
    imaging = data.get("recommended_imaging", [])
    if labs or imaging:
        lines.append("### 🔬 Examens recommandés")
        for t in labs[:6]:
            lines.append(f"- **{t.get('name', '')}** ({_PRIORITY_FR.get(t.get('priority'), t.get('priority', ''))}) — {t.get('rationale', '')}")
        for im in imaging[:4]:
            lines.append(f"- 🖼️ **{im.get('name', '')}** ({_PRIORITY_FR.get(im.get('priority'), im.get('priority', ''))}) — {im.get('rationale', '')}")
        lines.append("")

    treatments = data.get("treatments", [])
    if treatments:
        lines.append("### 💊 Pistes thérapeutiques")
        for tr in treatments[:5]:
            presc = " · ⚕️ prescription médicale requise" if tr.get("requires_prescriber") else ""
            lines.append(f"- **{tr.get('action', '')}** — {tr.get('rationale', '')}{presc}")
            for caution in (tr.get("cautions") or [])[:2]:
                lines.append(f"    - ⚠️ {caution}")
        lines.append("")

    if data.get("explanation"):
        lines.append("### 🧠 Raisonnement clinique")
        lines.append(data["explanation"])
        lines.append("")

    questions = data.get("questions_for_nurse", [])
    if questions:
        lines.append("### ❓ Questions à poser / vérifier")
        for q in questions[:6]:
            lines.append(f"- {q}")
        lines.append("")

    limitations = data.get("limitations", [])
    if limitations:
        lines.append("### ⚠️ Limites")
        for lim in limitations[:5]:
            lines.append(f"- {lim}")
        lines.append("")

    conf = _pct(data.get("confidence_score"))
    lines.append(f"*Confiance globale du modèle : {conf}*")
    if data.get("safety_note"):
        lines.append(f"\n> ℹ️ {data['safety_note']}")

    return "\n".join(lines)


def extract_primary_diagnosis(data: dict[str, Any]) -> str:
    """Short headline diagnosis for history / print sheet."""
    if not data:
        return "—"
    diags = data.get("possible_diagnosis") or []
    if diags:
        name = diags[0].get("name", "—")
        return f"{name} ({_pct(diags[0].get('probability'))})"
    return "—"
