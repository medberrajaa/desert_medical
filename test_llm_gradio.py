"""
test_llm_gradio.py — Test du Module 1 (LLM / Diagnostic général) via Gradio
Exécuter : python test_llm_gradio.py
"""
import gradio as gr
import os
import csv
import json
import base64
import tempfile
import shutil
from datetime import datetime
from module1.prompt_engine import generate_diagnosis, generate_raw_diagnosis
from module2.ecg_predictor import predict_ecg, diagnose_ecg_image
from module1.voice_engine import transcribe_audio
from views.print_sheet import build_print_sheet
from views.format_diagnosis import format_clinical_markdown, extract_primary_diagnosis

# ── Constantes ───────────────────────────────────────────────────────────────
COUNTRIES = ["France", "Sénégal", "Maroc", "Belgique", "Côte d'Ivoire", "Cameroun", "Benin", "Autre"]
LANGUAGES = ["Français", "English", "Arabic"]
EXAMENS_LIST = [
    "Tension artérielle", "Fréquence cardiaque", "Fréquence respiratoire",
    "Température", "Saturation O₂ (SpO₂)", "Glycémie capillaire", "Poids & Taille (IMC)",
    "ECG 6 dérivations", "ECG 12 dérivations",
    "NFS (Numération Formule Sanguine)", "CRP (Protéine C-Réactive)",
    "Ionogramme sanguin", "Créatinine / Urée", "Glycémie à jeun", "HbA1c",
    "Bilan hépatique (ASAT/ALAT)", "TSH (Thyroïde)", "Troponine", "D-Dimères",
    "Gaz du sang (GDS)", "Bandelette urinaire (BU)", "ECBU",
    "Radiographie thoracique", "Échographie abdominale", "Scanner / TDM", "IRM",
    "Test COVID-19 (antigénique)", "Test paludisme (TDR)", "Peak-flow (DEP)",
]

# ── Historique ───────────────────────────────────────────────────────────────
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "evaluation")
HISTORY_FILE = os.path.join(HISTORY_DIR, "consultations.csv")
HISTORY_FIELDS = [
    "id", "date", "age", "sexe", "ethnie", "symptomes", "country", "language",
    "diagnosis_response", "model", "tokens", "ddss_alerts", "status", "doctor_comment",
]


def _init_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()


def _save_consultation(patient: dict, result: dict, country: str, language: str) -> int:
    _init_history()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            next_id = max((int(r.get("id", 0)) for r in rows), default=0) + 1
    except Exception:
        next_id = 1

    row = {
        "id": next_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "age": patient.get("age", ""),
        "sexe": patient.get("sexe", ""),
        "ethnie": patient.get("ethnie", ""),
        "symptomes": patient.get("motif_consultation", patient.get("symptomes", "")),
        "country": country,
        "language": language,
        "diagnosis_response": format_clinical_markdown(result.get("data", {})) or result.get("response", ""),
        "model": result.get("model", ""),
        "tokens": result.get("tokens_used", ""),
        "ddss_alerts": ", ".join(result.get("ddss_alerts", [])),
        "status": "en_attente",
        "doctor_comment": "",
    }
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(row)
    return next_id


def _load_history() -> list[dict]:
    _init_history()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _update_consultation(consultation_id: str, status: str, comment: str):
    rows = _load_history()
    for row in rows:
        if str(row["id"]) == str(consultation_id):
            row["status"] = status
            row["doctor_comment"] = comment
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  INFIRMIER — Formulaire détaillé
# ══════════════════════════════════════════════════════════════════════════════

def generate_detailed(age, sexe, taille, poids, ethnie,
                      motif, signes, duree,
                      antecedents, traitements,
                      examens, resultats,
                      country, language):
    """Génère le diagnostic à partir du formulaire détaillé."""
    if not motif or not motif.strip():
        return ("⚠️ Veuillez renseigner le motif de consultation.",
                "", "", "")

    taille_val = int(taille) if taille and str(taille).strip().isdigit() else "Non renseignée"
    poids_val = poids if poids else "Non renseigné"

    patient = {
        "age": age,
        "sexe": sexe,
        "ethnie": ethnie or "Non renseignée",
        "taille": taille_val,
        "poids": poids_val,
        "duree_symptomes": duree or "Non renseignée",
        "motif_consultation": motif,
        "signes_cliniques": signes or "Aucun noté",
        "antecedents": antecedents or "Aucun connu",
        "traitements": traitements or "Aucun",
        "examens": ", ".join(examens) if examens else "Aucun",
        "resultats": resultats or "Aucun",
    }

    result = generate_diagnosis(patient, country=country, language=language)

    if result["success"]:
        alerts = result.get("ddss_alerts", [])
        alert_text = ""
        if alerts:
            alert_text = (
                f"🚨 **Alertes DDSS : {', '.join(alerts)}**\n\n"
                f"Niveau d'urgence plancher : **{result.get('ddss_urgency_floor', '?')}**"
            )

        cid = _save_consultation(patient, result, country, language)
        status = f"✅ Diagnostic généré · Modèle : {result['model']} · {result['tokens_used']} tokens · Consultation #{cid}"

        diagnosis_md = format_clinical_markdown(result.get("data", {}))
        return (status, alert_text, diagnosis_md, result.get("prompt", ""))
    else:
        return (f"❌ Erreur : {result.get('error', 'Inconnue')}", "", "", result.get("prompt", ""))


# ── Helpers extraction patient depuis texte brut ─────────────────────────────
import re as _re

def _extract_meta_from_raw(texte: str) -> dict:
    """Tente d'extraire âge, sexe et un motif court depuis un texte brut."""
    meta = {"age": "", "sexe": "", "motif": ""}

    # Âge
    age_m = _re.search(r"\b(\d{1,3})\s*ans?\b", texte, _re.IGNORECASE)
    if age_m:
        a = int(age_m.group(1))
        if 0 < a < 120:
            meta["age"] = str(a)

    # Sexe
    if _re.search(r"\b(femme|patiente|dame|f[ée]minin|girl|woman)\b", texte, _re.IGNORECASE):
        meta["sexe"] = "Femme"
    elif _re.search(r"\b(homme|patient|monsieur|masculin|boy|man)\b", texte, _re.IGNORECASE):
        meta["sexe"] = "Homme"

    # Motif : prendre la première phrase non vide (max 120 chars)
    for line in texte.split("\n"):
        line = line.strip().lstrip("•-–*#").strip()
        if len(line) > 10:
            meta["motif"] = line[:120]
            break
    if not meta["motif"]:
        meta["motif"] = texte[:120]

    return meta


def _extract_diag_summary(response: str) -> str:
    """Extrait le premier diagnostic nommé depuis la réponse Markdown formatée."""
    if not response:
        return "—"
    # Nouveau format : section « Diagnostics possibles » puis « - **Nom** — … »
    m = _re.search(r"Diagnostics possibles.*?\n\s*[-*]\s*\*\*(.+?)\*\*", response, _re.IGNORECASE | _re.DOTALL)
    if m:
        return m.group(1).strip()[:200]
    # Sinon : premier élément en gras de la réponse
    m = _re.search(r"\*\*(.+?)\*\*", response)
    if m:
        return m.group(1).strip()[:200]
    # Fallback : première ligne non-titre
    for line in response.split("\n"):
        stripped = line.strip().lstrip("#*•-–>1234567890. ").strip()
        if len(stripped) > 10 and not stripped.isupper():
            return stripped[:200]
    return response[:200].split("\n")[0]


# ══════════════════════════════════════════════════════════════════════════════
#  INFIRMIER — Saisie texte libre
# ══════════════════════════════════════════════════════════════════════════════

def generate_raw_text(texte_brut, country, language):
    """Génère le diagnostic à partir d'un texte brut non structuré."""
    if not texte_brut or not texte_brut.strip():
        return ("⚠️ Veuillez saisir les informations du patient.", "", "", "")

    result = generate_raw_diagnosis(texte_brut, country=country, language=language)

    if result["success"]:
        alerts = result.get("ddss_alerts", [])
        alert_text = ""
        if alerts:
            alert_text = (
                f"🚨 **Alertes DDSS : {', '.join(alerts)}**\n\n"
                f"Niveau d'urgence plancher : **{result.get('ddss_urgency_floor', '?')}**"
            )

        # Extraire les métadonnées pour l'historique
        meta = _extract_meta_from_raw(texte_brut)
        patient_rec = {
            "motif_consultation": meta["motif"],
            "age": meta["age"],
            "sexe": meta["sexe"],
            "signes_cliniques": "",
        }

        cid = _save_consultation(patient_rec, result, country, language)
        status = f"✅ Diagnostic généré · Modèle : {result['model']} · {result['tokens_used']} tokens · Consultation #{cid}"

        diagnosis_md = format_clinical_markdown(result.get("data", {}))
        return (status, alert_text, diagnosis_md, result.get("prompt", ""))
    else:
        return (f"❌ Erreur : {result.get('error', 'Inconnue')}", "", "", result.get("prompt", ""))


# ══════════════════════════════════════════════════════════════════════════════
#  TÉLÉMÉDECIN — Historique & validation
# ══════════════════════════════════════════════════════════════════════════════

def load_history_table(status_filter, search_text):
    """Charge et filtre l'historique des consultations."""
    history = _load_history()
    if not history:
        return "Aucune consultation enregistrée.", gr.update(choices=[], value=None)

    # Filtrage
    filtered = history
    if status_filter and status_filter != "Tous":
        filtered = [h for h in filtered if h.get("status") == status_filter]
    if search_text:
        filtered = [h for h in filtered if search_text.lower() in str(h).lower()]

    if not filtered:
        return "Aucune consultation ne correspond aux filtres.", gr.update(choices=[], value=None)

    # Métriques
    total = len(history)
    en_attente = sum(1 for h in history if h.get("status") == "en_attente")
    validees = sum(1 for h in history if h.get("status") == "validée")
    rejetees = sum(1 for h in history if h.get("status") == "rejetée")
    metrics = (
        f"**Total :** {total} · **En attente :** {en_attente} "
        f"· **Validées :** {validees} · **Rejetées :** {rejetees}"
    )

    # Une entrée = une consultation (une seule ligne, sans retour à la ligne)
    status_icon = {"en_attente": "🟡", "validée": "🟢", "rejetée": "🔴"}
    choices = []
    for h in filtered:
        icon  = status_icon.get(h.get("status", ""), "⚪")
        date  = str(h.get("date", "?"))[:16]
        age   = h.get("age", "?")
        sexe  = h.get("sexe", "?")
        # Nettoyage du motif : retours à la ligne → espace, puces → virgule
        motif_raw = str(h.get("symptomes", ""))
        motif = " ".join(
            part.strip().lstrip("•–-").strip()
            for part in motif_raw.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").split("  ")
            if part.strip()
        )
        motif = motif[:70] + ("…" if len(motif) > 70 else "")
        label = f"#{h.get('id', '?')}  {icon}  {date}  —  {sexe}, {age} ans  —  {motif}"
        choices.append(label)

    return metrics, gr.update(choices=choices, value=None)


def get_consultation_detail(consultation_id):
    """Retourne le détail complet d'une consultation."""
    if not consultation_id or not str(consultation_id).strip().isdigit():
        return "Saisissez un numéro de consultation valide.", "", ""

    history = _load_history()
    match = [h for h in history if str(h.get("id")) == str(consultation_id).strip()]
    if not match:
        return f"Consultation #{consultation_id} introuvable.", "", ""

    h = match[0]
    info = (
        f"**Consultation #{h.get('id')}** — {h.get('date', '?')}\n\n"
        f"- **Âge :** {h.get('age', '?')} ans\n"
        f"- **Sexe :** {h.get('sexe', '?')}\n"
        f"- **Ethnie :** {h.get('ethnie', 'Non renseignée')}\n"
        f"- **Pays :** {h.get('country', '?')}\n"
        f"- **Motif :** {h.get('symptomes', '')}\n"
    )

    if h.get("ddss_alerts"):
        info += f"\n🚨 **Alertes DDSS :** {h['ddss_alerts']}\n"

    info += f"\n*Modèle : {h.get('model', '?')} · Tokens : {h.get('tokens', '?')}*"

    diagnosis = h.get("diagnosis_response", "Non disponible")
    status = h.get("status", "en_attente")

    status_info = f"**Statut actuel :** {status}"
    if h.get("doctor_comment"):
        status_info += f"\n\n**Commentaire médecin :** {h['doctor_comment']}"

    return info, diagnosis, status_info


def validate_consultation(consultation_id, decision, comment):
    """Valide ou rejette une consultation."""
    if not consultation_id or not str(consultation_id).strip().isdigit():
        return "⚠️ Saisissez un numéro de consultation valide."

    cid = str(consultation_id).strip()
    history = _load_history()
    match = [h for h in history if str(h.get("id")) == cid]
    if not match:
        return f"❌ Consultation #{cid} introuvable."

    if match[0].get("status") != "en_attente":
        return f"ℹ️ Consultation #{cid} déjà traitée ({match[0].get('status')})."

    new_status = "validée" if "Valider" in decision else "rejetée"
    _update_consultation(cid, new_status, comment or "")
    return f"✅ Consultation #{cid} → **{new_status}**"


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DE L'INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  SESSION HISTORY HELPERS (Feature 1)
# ═══════════════════════════════════════════════════════════════════════════

def _add_session_entry(history: list, entry: dict) -> list:
    """Ajoute une entrée et garde les 10 dernières."""
    history = list(history)  # copie
    history.insert(0, entry)
    return history[:10]


def _session_table_md(history: list) -> str:
    """Rend un tableau Markdown des 10 dernières entrées."""
    if not history:
        return "_Aucun diagnostic dans cette session._"
    lines = [
        "| # | Heure | Patient | Module | Diagnostic | Confiance |",
        "|---|-------|---------|--------|------------|-----------|"]
    for i, e in enumerate(history, 1):
        lines.append(
            f"| {i} | {e.get('time','')} | {e.get('patient','')} "
            f"| {e.get('module','')} | {e.get('diagnosis','')} "
            f"| {e.get('confidence','')} % |")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  CLIPBOARD JS SNIPPET (Feature 2)
# ═══════════════════════════════════════════════════════════════════════════

_CLIPBOARD_JS_FUNC = """
function mediaCopy(btnId) {
  var btn = document.getElementById(btnId);
  if (!btn) return;
  var text = btn.getAttribute('data-copy');
  navigator.clipboard.writeText(text).then(function() {
    var orig = btn.textContent;
    btn.textContent = '✓ Copié';
    btn.style.background = '#16a34a';
    btn.style.color = '#fff';
    setTimeout(function(){ btn.textContent = orig; btn.style.background = ''; btn.style.color = ''; }, 2000);
  });
}
"""

def _make_copy_button(text: str, btn_id: str = "copyBtn") -> str:
    """HTML d'un bouton copie avec le texte encodé."""
    safe = text.replace('"', '&quot;').replace("'", "&#39;").replace("\n", "&#10;")
    return (
        f'<button id="{btn_id}" data-copy="{safe}" '
        f'onclick="mediaCopy(\'{btn_id}\')" '
        f'style="padding:6px 16px;border-radius:6px;border:1px solid #cbd5e1;'
        f'cursor:pointer;font-size:0.85rem;margin-top:6px;">'
        f'📋 Copier le résumé</button>'
    )


# ═══════════════════════════════════════════════════════════════════════════
#  RELIABILITY INDICATOR (Feature 3)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_reliability_warnings(result: dict, age=None) -> str:
    """Retourne un bandeau Markdown : vert si OK, orange si warnings."""
    warnings = []
    fs = result.get("features_summary", {})
    confidence = result.get("confidence", 100)

    # Âge hors distribution d'entraînement
    if age is not None:
        try:
            a = int(age)
            if a < 18 or a > 85:
                warnings.append(
                    "Patient hors de la tranche d'âge dominante du dataset "
                    "d'entraînement (18-85 ans)")
        except (ValueError, TypeError):
            pass

    # FC extrême
    hr = fs.get("hr_mean")
    if hr not in (None, "N/A"):
        try:
            hr_val = float(hr)
            if hr_val < 30 or hr_val > 200:
                warnings.append(
                    "Fréquence cardiaque extrême peu représentée "
                    "à l'entraînement")
        except (ValueError, TypeError):
            pass

    # Signal trop court
    n_beats = fs.get("n_beats", 99)
    try:
        if int(n_beats) < 5:
            warnings.append("Signal trop court pour une analyse fiable")
    except (ValueError, TypeError):
        pass

    # Confiance faible
    if confidence < 60:
        warnings.append("Le modèle hésite entre plusieurs classes")

    if not warnings:
        return (
            '<div style="background:#dcfce7;border-left:4px solid #16a34a;'
            'padding:8px 12px;border-radius:4px;margin:6px 0;">'
            '✅ <b>Fiabilité OK</b> — Cas similaire aux données d\'entraînement</div>')

    items = "".join(
        f'<div style="margin:2px 0;">⚠ Fiabilité réduite : {w}</div>'
        for w in warnings)
    return (
        f'<div style="background:#fef3c7;border-left:4px solid #d97706;'
        f'padding:8px 12px;border-radius:4px;margin:6px 0;">'
        f'{items}</div>')


# ═══════════════════════════════════════════════════════════════════════════
#  PRINTABLE SHEET HELPER (Feature 4)
# ═══════════════════════════════════════════════════════════════════════════

REPORT_DIR = os.path.join(os.path.dirname(__file__), "report")
os.makedirs(REPORT_DIR, exist_ok=True)


def _save_print_sheet(html_str: str) -> str:
    """Enregistre la fiche HTML dans report/ et retourne le chemin du fichier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fiche_diagnostic_{ts}.html"
    filepath = os.path.join(REPORT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_str)
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  ECG — libellés français et badge d'urgence (partagés avec l'UI Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

from module2.ecg_labels import fr_ecg_label as _fr_ecg_label
from module2.ecg_labels import ecg_urgency_badge as _ecg_urgency_badge


CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
.gr-button-primary { border-radius: 8px !important; font-weight: 600 !important; }

/* Titre principal */
.app-title {
    text-align: center !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.1rem !important;
    letter-spacing: -0.5px;
}
.app-subtitle {
    text-align: center !important;
    font-size: 1.25rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
}

/* Onglets bien visibles */
.tabs > .tab-nav {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 6px 6px 0 6px !important;
    gap: 4px !important;
}
.tabs > .tab-nav > button {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 12px 22px !important;
    border-radius: 10px 10px 0 0 !important;
    border: none !important;
    background: rgba(255,255,255,0.08) !important;
    transition: all 0.2s ease !important;
}
.tabs > .tab-nav > button:hover {
    background: rgba(255,255,255,0.18) !important;
    color: #fff !important;
}
.tabs > .tab-nav > button.selected {
    background: #fff !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.1) !important;
}
"""

with gr.Blocks(
    title="MedIA — Aide au diagnostic",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
    js=_CLIPBOARD_JS_FUNC,
) as app:

    # ── Feature 1 : Session state ─────────────────────────────────────
    session_history = gr.State([])

    # ── Feature 2 : Clipboard JS (injecté via gr.Blocks(js=...)) ─────

    gr.Markdown("<h1 class='app-title'>🩺 MedIA</h1>", elem_classes="app-title")
    gr.Markdown("<p class='app-subtitle'>Outil d'aide au diagnostic — Déserts médicaux</p>", elem_classes="app-subtitle")

    with gr.Row():
        country = gr.Dropdown(COUNTRIES, value="France", label="🌍 Pays / contexte")
        language = gr.Dropdown(LANGUAGES, value="Français", label="🗣️ Langue du diagnostic")

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════
        #  ONGLET 1 : INFIRMIER — Formulaire détaillé
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("👨‍⚕️ Infirmier — Formulaire détaillé"):

            gr.Markdown("## 📝 Informations patient")

            with gr.Row():
                age = gr.Number(value=45, label="Âge", minimum=0, maximum=120, precision=0)
                sexe = gr.Dropdown(["Homme", "Femme", "Non précisé"], value="Homme", label="Sexe")
                taille = gr.Textbox(label="Taille (cm)", placeholder="ex : 170 (optionnel)")
                poids = gr.Textbox(label="Poids (kg)", placeholder="ex : 70 (optionnel)")

            ethnie = gr.Textbox(label="Ethnie (optionnel)", placeholder="ex : Européen, Africain, Asiatique…")

            gr.Markdown("---\n## 📌 Motif de consultation & signes cliniques")

            with gr.Row():
                motif = gr.Textbox(
                    label="Motif de consultation *",
                    placeholder="Raison principale de la visite…",
                    lines=3,
                    scale=2,
                )
                duree = gr.Textbox(label="Durée des symptômes", placeholder="ex : 2 jours", scale=1)

            signes = gr.Textbox(
                label="Signes cliniques observés",
                placeholder="Signes objectifs relevés par l'infirmier… ex : fièvre à 38.5°C, dyspnée, pâleur…",
                lines=3,
            )

            with gr.Row():
                antecedents = gr.Textbox(
                    label="Antécédents médicaux",
                    placeholder="HTA, diabète, cardiopathie…",
                    lines=3,
                )
                traitements = gr.Textbox(
                    label="Traitements en cours",
                    placeholder="Médicaments, posologie…",
                    lines=3,
                )

            gr.Markdown("---\n## 🔬 Examens & résultats")

            examens = gr.CheckboxGroup(EXAMENS_LIST, label="Examens réalisés")
            resultats = gr.Textbox(
                label="Résultats des examens",
                placeholder="TA : 145/90 mmHg, T° : 38.2°C, SpO₂ : 97%…",
                lines=3,
            )

            gr.Markdown("---")

            btn_detailed = gr.Button("🔍  Générer le diagnostic IA", variant="primary", size="lg")

            status_detailed = gr.Textbox(label="Statut", interactive=False)
            alerts_detailed = gr.Markdown(label="Alertes DDSS")
            diag_detailed = gr.Markdown(label="Diagnostic")

            with gr.Accordion("📄 Voir le prompt envoyé à l'IA", open=False):
                prompt_detailed = gr.Code(label="Prompt", language=None)

            btn_new_detailed = gr.Button("🆕  Nouveau diagnostic", size="lg")

            # Feature 2 : bouton copie + Feature 4 : fiche imprimable
            with gr.Row():
                copy_html_detailed = gr.HTML(value="", visible=True)
                print_html_detailed = gr.File(label="📄 Fiche imprimable", visible=True, interactive=False)
            btn_copy_detailed = gr.Button("📋 Copier le résumé", size="sm")
            btn_print_detailed = gr.Button("🖨️ Générer fiche imprimable", size="sm")

            def generate_detailed_with_session(age_v, sexe_v, taille_v, poids_v, ethnie_v,
                                               motif_v, signes_v, duree_v,
                                               antecedents_v, traitements_v,
                                               examens_v, resultats_v,
                                               country_v, language_v, hist):
                result = generate_detailed(age_v, sexe_v, taille_v, poids_v, ethnie_v,
                                           motif_v, signes_v, duree_v,
                                           antecedents_v, traitements_v,
                                           examens_v, resultats_v,
                                           country_v, language_v)
                status_out, alerts_out, diag_out, prompt_out = result
                # Feature 1 : ajouter à l'historique de session
                if status_out.startswith("✅"):
                    entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "patient": f"{sexe_v}, {age_v} ans",
                        "module": "LLM",
                        "diagnosis": diag_out[:80].split("\n")[0] if diag_out else "—",
                        "confidence": "—",
                        "age": age_v, "sexe": sexe_v,
                        "motif": motif_v,
                        "full_diag": diag_out,
                    }
                    hist = _add_session_entry(hist, entry)
                return status_out, alerts_out, diag_out, prompt_out, hist

            btn_detailed.click(
                fn=generate_detailed_with_session,
                inputs=[age, sexe, taille, poids, ethnie,
                        motif, signes, duree,
                        antecedents, traitements,
                        examens, resultats,
                        country, language, session_history],
                outputs=[status_detailed, alerts_detailed, diag_detailed, prompt_detailed, session_history],
            )

            def _copy_detailed(age_v, sexe_v, motif_v, diag_v):
                now = datetime.now().strftime("%d/%m/%Y %H:%M")
                summary = diag_v[:120].split("\n")[0] if diag_v else "—"
                text = (
                    f"Date : {now}\n"
                    f"Patient : {age_v} ans, {sexe_v}\n"
                    f"Motif : {motif_v}\n"
                    f"Diagnostic : {summary}\n"
                    f"Validé par télémédecin : En attente")
                return _make_copy_button(text, "copyDetailed")

            btn_copy_detailed.click(
                fn=_copy_detailed,
                inputs=[age, sexe, motif, diag_detailed],
                outputs=[copy_html_detailed],
            )

            def _print_detailed(age_v, sexe_v, motif_v, diag_v):
                summary = _extract_diag_summary(diag_v) if diag_v else "—"

                urgency = "modérée"
                if diag_v:
                    if "urgence immédiate" in diag_v.lower() or "🔴" in diag_v:
                        urgency = "élevée"
                    elif "non urgent" in diag_v.lower() or "🟢" in diag_v:
                        urgency = "faible"

                sheet = build_print_sheet({
                    "age": age_v, "sexe": sexe_v, "motif": motif_v,
                    "diagnosis": summary, "confidence": "—",
                    "module": "LLM", "urgency": urgency,
                    "hr": "—", "rr": "—", "pr_ms": "—", "qrs_ms": "—",
                })
                return _save_print_sheet(sheet)

            btn_print_detailed.click(
                fn=_print_detailed,
                inputs=[age, sexe, motif, diag_detailed],
                outputs=[print_html_detailed],
            )

            def clear_detailed():
                return (45, "Homme", "", "", "", "", "", "", "", "", [], "",
                        "", "", "", "")

            btn_new_detailed.click(
                fn=clear_detailed,
                outputs=[age, sexe, taille, poids, ethnie,
                         motif, signes, duree,
                         antecedents, traitements,
                         examens, resultats,
                         status_detailed, alerts_detailed, diag_detailed, prompt_detailed],
            )

        # ═══════════════════════════════════════════════════════════════════
        #  ONGLET 2 : INFIRMIER — Saisie texte libre
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📝 Infirmier — Saisie texte libre"):

            gr.Markdown("## 📝 Saisie texte libre")
            gr.Markdown(
                "> Collez ci-dessous toutes les informations du patient en texte libre : "
                "identité, motif, signes cliniques, antécédents, résultats d'examens… "
                "L'IA analysera l'ensemble."
            )

            # ── Saisie vocale ────────────────────────────────────────────
            with gr.Accordion("🎙️ Dicter les informations patient", open=False):
                gr.Markdown(
                    "> Enregistrez votre voix. La transcription sera ajoutée "
                    "directement dans la zone de saisie ci-dessous."
                )
                with gr.Row():
                    voice_audio_raw = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 Microphone",
                        format="wav",
                        scale=3,
                    )
                    btn_voice_raw = gr.Button(
                        "🎙️ Transcrire",
                        variant="secondary",
                        size="sm",
                        scale=1,
                    )
                voice_status_raw = gr.Textbox(
                    label="Statut transcription",
                    interactive=False,
                    lines=1,
                )

            texte_brut = gr.Textbox(
                label="Informations patient (texte brut) *",
                placeholder=(
                    "Exemple :\n"
                    "Homme, 62 ans, ethnie Peul, 78 kg, 175 cm.\n"
                    "Motif : douleur thoracique rétrosternale depuis 2h, irradiant vers le bras gauche.\n"
                    "Signes : sueurs, pâleur, TA 160/95, FC 110, SpO₂ 94%, T° 37.1°C.\n"
                    "Antécédents : HTA sous amlodipine 5mg, tabagisme actif 20PA, diabète type 2.\n"
                    "Traitements : metformine 1000mg x2/j, amlodipine 5mg/j.\n"
                    "ECG 6 dérivations réalisé : sus-décalage ST en V1-V4.\n"
                    "NFS et troponine demandées."
                ),
                lines=12,
            )

            gr.Markdown("---")

            btn_raw = gr.Button("🔍  Générer le diagnostic IA", variant="primary", size="lg")

            status_raw = gr.Textbox(label="Statut", interactive=False)
            alerts_raw = gr.Markdown(label="Alertes DDSS")
            diag_raw = gr.Markdown(label="Diagnostic")

            with gr.Accordion("📄 Voir le prompt envoyé à l'IA", open=False):
                prompt_raw = gr.Code(label="Prompt", language=None)

            btn_new_raw = gr.Button("🆕  Nouveau diagnostic", size="lg")

            # Feature 2 : bouton copie + Feature 4 : fiche imprimable
            with gr.Row():
                copy_html_raw = gr.HTML(value="", visible=True)
                print_html_raw = gr.File(label="📄 Fiche imprimable", visible=True, interactive=False)
            btn_copy_raw = gr.Button("📋 Copier le résumé", size="sm")
            btn_print_raw = gr.Button("🖨️ Générer fiche imprimable", size="sm")

            def generate_raw_with_session(texte, country_v, language_v, hist):
                result = generate_raw_text(texte, country_v, language_v)
                status_out, alerts_out, diag_out, prompt_out = result
                if status_out.startswith("✅"):
                    entry = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "patient": "Texte libre",
                        "module": "LLM",
                        "diagnosis": diag_out[:80].split("\n")[0] if diag_out else "—",
                        "confidence": "—",
                        "motif": texte[:60] if texte else "—",
                        "full_diag": diag_out,
                    }
                    hist = _add_session_entry(hist, entry)
                return status_out, alerts_out, diag_out, prompt_out, hist

            btn_raw.click(
                fn=generate_raw_with_session,
                inputs=[texte_brut, country, language, session_history],
                outputs=[status_raw, alerts_raw, diag_raw, prompt_raw, session_history],
            )

            def _copy_raw(texte, diag_v):
                now = datetime.now().strftime("%d/%m/%Y %H:%M")
                summary = diag_v[:120].split("\n")[0] if diag_v else "—"
                text = (
                    f"Date : {now}\n"
                    f"Patient : Texte libre\n"
                    f"Diagnostic : {summary}\n"
                    f"Validé par télémédecin : En attente")
                return _make_copy_button(text, "copyRaw")

            btn_copy_raw.click(
                fn=_copy_raw,
                inputs=[texte_brut, diag_raw],
                outputs=[copy_html_raw],
            )

            def _print_raw(texte, diag_v):
                meta = _extract_meta_from_raw(texte or "")
                summary = _extract_diag_summary(diag_v) if diag_v else "—"

                # Extraire le niveau d'urgence depuis la réponse LLM
                urgency = "modérée"
                if diag_v:
                    if "urgence immédiate" in diag_v.lower() or "🔴" in diag_v:
                        urgency = "élevée"
                    elif "non urgent" in diag_v.lower() or "🟢" in diag_v:
                        urgency = "faible"
                    elif "urgent" in diag_v.lower() or "🟡" in diag_v:
                        urgency = "modérée"

                sheet = build_print_sheet({
                    "age": meta["age"],
                    "sexe": meta["sexe"],
                    "motif": meta["motif"],
                    "diagnosis": summary,
                    "confidence": "—",
                    "module": "LLM",
                    "urgency": urgency,
                    "hr": "—", "rr": "—", "pr_ms": "—", "qrs_ms": "—",
                })
                return _save_print_sheet(sheet)

            btn_print_raw.click(
                fn=_print_raw,
                inputs=[texte_brut, diag_raw],
                outputs=[print_html_raw],
            )

            # ── Handler commande vocale texte libre ──────────────────────
            def handle_voice_raw(audio_path, current_text):
                """Transcrit l'audio et l'ajoute dans la zone de saisie."""
                if audio_path is None:
                    return "⚠️ Aucun audio enregistré.", current_text or ""

                result = transcribe_audio(audio_path)
                if not result["success"]:
                    return f"❌ {result['error']}", current_text or ""

                transcript = result["transcript"]
                # Ajouter la transcription au texte existant (saut de ligne si non vide)
                existing = (current_text or "").strip()
                new_text = (existing + "\n" + transcript).strip() if existing else transcript
                return f"✅ Transcrit ({len(transcript)} caractères)", new_text

            btn_voice_raw.click(
                fn=handle_voice_raw,
                inputs=[voice_audio_raw, texte_brut],
                outputs=[voice_status_raw, texte_brut],
            )

            btn_new_raw.click(
                fn=lambda: (None, "", "", "", "", "", ""),
                outputs=[voice_audio_raw, voice_status_raw,
                         texte_brut, status_raw, alerts_raw, diag_raw, prompt_raw],
            )

        # ═══════════════════════════════════════════════════════════════════
        #  ONGLET 3 : TÉLÉMÉDECIN — Validation en cours
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("✅ Télémédecin — Validation"):

            gr.Markdown("## ✅ Validation des consultations")

            with gr.Row():
                filter_status = gr.Dropdown(
                    ["Tous", "en_attente", "validée", "rejetée"],
                    value="en_attente",
                    label="Filtrer par statut",
                )
                search_box = gr.Textbox(label="🔍 Rechercher", placeholder="symptômes, pays…")

            btn_refresh = gr.Button("🔄 Charger / Actualiser", variant="secondary")

            metrics_md = gr.Markdown(label="Métriques")
            consult_list = gr.Dropdown(
                choices=[],
                label="📋 Consultations — sélectionnez une ligne pour afficher le détail",
                interactive=True,
                value=None,
            )

            btn_refresh.click(
                fn=load_history_table,
                inputs=[filter_status, search_box],
                outputs=[metrics_md, consult_list],
            )

            gr.Markdown("---\n## 📄 Détail d'une consultation")

            info_md = gr.Markdown(label="Informations patient")
            diag_md = gr.Markdown(label="Diagnostic IA complet")
            status_md = gr.Markdown(label="Statut")

            def _load_detail_from_choice_tab3(choice):
                if not choice:
                    return "", "", ""
                try:
                    consult_id_str = choice.split("#")[1].split(" ")[0].strip()
                except Exception:
                    return "Impossible de lire le numéro de consultation.", "", ""
                return get_consultation_detail(consult_id_str)

            consult_list.change(
                fn=_load_detail_from_choice_tab3,
                inputs=[consult_list],
                outputs=[info_md, diag_md, status_md],
            )

            gr.Markdown("---\n## ⚖️ Décision médicale")

            decision = gr.Radio(
                ["✅ Valider le diagnostic", "❌ Rejeter le diagnostic"],
                label="Action",
                value="✅ Valider le diagnostic",
            )
            comment = gr.Textbox(
                label="Commentaire médical",
                placeholder="Observations, corrections ou contre-indications…",
                lines=3,
            )
            btn_validate = gr.Button("Confirmer la décision", variant="primary")
            validation_result = gr.Markdown()

            btn_validate.click(
                fn=lambda choice, dec, cmt: validate_consultation(
                    choice.split("#")[1].split(" ")[0].strip() if choice else "",
                    dec, cmt
                ),
                inputs=[consult_list, decision, comment],
                outputs=[validation_result],
            )

        # ═══════════════════════════════════════════════════════════════════
        #  ONGLET 4 : MODULE 2 — Analyse ECG 12 dérivations
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("💓 Analyse ECG"):

            gr.Markdown("## 💓 Analyse ECG 12 dérivations")
            gr.Markdown(
                "> **Module 2** : Chargez un **fichier CSV** (12 dérivations @ 500 Hz → analyse "
                "hybride mesures du signal + LLM) ou une **image JPG/PNG** du tracé "
                "(→ lecture par LLM vision, style PULSE). Dans les deux cas le tracé 12 dérivations "
                "est reconstruit et affiché. Le moteur est choisi automatiquement selon le type de fichier."
            )

            gr.Markdown("---\n### 📂 Charger un fichier ECG")

            ecg_file = gr.File(
                label="📄 Fichier ECG — CSV ou Image (JPG / PNG)",
                file_types=[".csv", ".jpg", ".jpeg", ".png"],
                type="filepath",
            )

            btn_ecg = gr.Button("🔍  Analyser l'ECG", variant="primary", size="lg")

            ecg_status = gr.Textbox(label="Statut", interactive=False)

            ecg_diag = gr.Markdown(label="Diagnostic")

            ecg_analysis = gr.Markdown(label="Analyse détaillée")

            # Feature 3 : indicateur de fiabilité contextuel (CSV only)
            ecg_reliability = gr.HTML(value="", visible=True)

            ecg_plot = gr.Plot(label="Tracé ECG")

            with gr.Row():
                ecg_metrics = gr.Markdown(label="Métriques")
                ecg_alerts = gr.Markdown(label="Alertes cliniques (règles)")

            # Features 2 & 4 : boutons copie + fiche imprimable ECG
            with gr.Row():
                copy_html_ecg = gr.HTML(value="", visible=True)
                print_html_ecg = gr.File(label="📄 Fiche imprimable", visible=True, interactive=False)
            with gr.Row():
                btn_copy_ecg = gr.Button("📋 Copier le résumé", size="sm")
                btn_print_ecg = gr.Button("🖨️ Générer fiche imprimable", size="sm")

            # Stocker le dernier résultat ECG pour copy/print
            ecg_last_result = gr.State({})

            def _ecg_metrics_md(fs: dict, approximate: bool = False) -> str:
                if not fs:
                    return ""
                note = " *(estimations approximatives depuis l'image)*" if approximate else ""
                return (
                    f"### 📊 Métriques cliniques{note}\n\n"
                    f"| Paramètre | Valeur |\n"
                    f"|-----------|--------|\n"
                    f"| **Fréquence cardiaque** | {fs.get('hr_mean', '—')} bpm |\n"
                    f"| **Intervalle RR moyen** | {fs.get('rr_mean_ms', '—')} ms |\n"
                    f"| **Intervalle PR** | {fs.get('pr_ms', '—')} ms |\n"
                    f"| **Durée QRS** | {fs.get('qrs_ms', '—')} ms |\n"
                    f"| **Intervalle QT** | {fs.get('qt_ms', '—')} ms |\n"
                    f"| **QTc** | {fs.get('qtc_ms', '—')} ms |\n"
                )

            def _ecg_alerts_md(alerts: list) -> str:
                if not alerts:
                    return "### ⚡ Alertes cliniques\n\n✅ Aucune alerte déclenchée."
                severity_icon = {"info": "🟢", "warning": "🟡", "critical": "🔴"}
                lines = ["### ⚡ Alertes cliniques (règles déterministes)\n",
                         "| Sévérité | Anomalie | Détail |",
                         "|----------|----------|--------|"]
                for a in alerts:
                    icon = severity_icon.get(a.get("severity"), "⚪")
                    lines.append(
                        f"| {icon} {str(a.get('severity', '')).upper()} | **{_fr_ecg_label(a.get('label', ''))}** "
                        f"| {a.get('detail', '')} |"
                    )
                return "\n".join(lines)

            def analyze_ecg(file_path, language, hist):
                empty = ("⚠️ Veuillez charger un fichier CSV ou une image.", "", "", "", None, "", "", hist, {})
                if file_path is None:
                    return empty

                fpath = str(file_path)
                is_image = fpath.lower().endswith((".jpg", ".jpeg", ".png"))

                # ─── IMAGE → analyse visuelle LLM (style PULSE) + cas de référence ───
                if is_image:
                    result = diagnose_ecg_image(fpath, language=language or "Français")
                    if not result.get("success"):
                        return (f"❌ Erreur : {result.get('error')}", "", "", "", None, "", "", hist, {})

                    diagnosis = result["diagnosis"]
                    subtitle = result.get("subtitle", "")
                    confidence = result.get("confidence")
                    urg_icon, urg_label = _ecg_urgency_badge(result.get("urgency"))
                    engine = "cas de référence validé + LLM vision" if result.get("reference_case") else "LLM vision (style PULSE)"

                    diag_lines = ["### 🩺 Pathologie identifiée\n", f"## {urg_icon} {diagnosis}\n"]
                    if subtitle:
                        diag_lines.append(f"**{subtitle}**\n")
                    if confidence is not None:
                        diag_lines.append(f"**Confiance : {confidence}%**  ·  ")
                    diag_lines.append(f"**Niveau d'urgence : {urg_label}**\n")
                    diag_lines.append(f"*Moteur : {engine}*")
                    diag_md = "\n".join(diag_lines)

                    analysis_md = result.get("analysis", "")
                    fs = result.get("features_summary", {})
                    metrics_md = _ecg_metrics_md(fs, approximate=True)
                    alerts_md = _ecg_alerts_md(result.get("clinical_alerts", []))
                    reliability_html = ""
                    if result.get("missing_leads"):
                        reliability_html = (
                            '<div style="background:#fef3c7;border-left:4px solid #d97706;padding:8px 12px;'
                            'border-radius:4px;margin:6px 0;">⚠ Dérivations non reconstruites depuis l\'image : '
                            f'{", ".join(result["missing_leads"])}</div>'
                        )

                    status = f"✅ Analyse terminée · {engine}"
                    entry = {
                        "time": datetime.now().strftime("%H:%M:%S"), "patient": "ECG",
                        "module": "ECG-Image", "diagnosis": f"{diagnosis} {subtitle}".strip(),
                        "confidence": str(confidence) if confidence is not None else "—",
                        "motif": "Analyse ECG image", "full_diag": result.get("vision_raw", ""),
                    }
                    hist = _add_session_entry(hist, entry)
                    stored = {
                        "diagnosis": f"{diagnosis} ({subtitle})" if subtitle else diagnosis,
                        "confidence": confidence if confidence is not None else "—",
                        "urgency": urg_label, "module": engine,
                        "hr": fs.get("hr_mean", "—"), "rr": fs.get("rr_mean_ms", "—"),
                        "pr_ms": fs.get("pr_ms", "—"), "qrs_ms": fs.get("qrs_ms", "—"),
                    }
                    return (status, diag_md, analysis_md, reliability_html,
                            result.get("ecg_plot"), metrics_md, alerts_md, hist, stored)

                # ─── CSV → analyse hybride (mesures du signal + LLM) ───
                if not fpath.endswith(".csv"):
                    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                    shutil.copy(fpath, tmp.name)
                    fpath = tmp.name
                result = predict_ecg(fpath, use_llm=True)

                if not result["success"]:
                    return (f"❌ Erreur : {result['error']}", "", "", "", None, "", "", hist, {})

                diagnosis = _fr_ecg_label(result.get("diagnosis_display") or result.get("diagnosis", "—"))
                confidence = result["confidence"]
                report = result.get("clinical_report", {})
                urg_icon, urg_label = _ecg_urgency_badge(report.get("urgency"))
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"

                diag_lines = [
                    "### 🩺 Pathologie identifiée\n", f"## {urg_icon} {diagnosis}\n",
                    f"{conf_color} **Confiance : {confidence}%**  ·  **Urgence : {urg_label}**\n",
                    "*Moteur : analyse hybride (mesures du signal + LLM)*",
                ]
                diffs = result.get("differentials", [])
                if diffs:
                    diag_lines += ["", "Diagnostics différentiels : "
                                   + " · ".join(f"**{_fr_ecg_label(d['label'])}**" for d in diffs)]
                diag_md = "\n".join(diag_lines)

                fs = result["features_summary"]
                metrics_md = _ecg_metrics_md(fs, approximate=False)
                alerts_md = _ecg_alerts_md(result.get("clinical_alerts", []))
                reliability_html = _compute_reliability_warnings(result)

                # Analyse détaillée issue du rapport LLM
                analysis_md = ""
                if report.get("recommended_action"):
                    analysis_md = f"### 🩺 Conduite recommandée\n{report['recommended_action']}"

                entry = {
                    "time": datetime.now().strftime("%H:%M:%S"), "patient": "ECG",
                    "module": "ECG-Signal+LLM", "diagnosis": diagnosis, "confidence": str(confidence),
                    "hr": str(fs.get("hr_mean", "—")), "rr": str(fs.get("rr_mean_ms", "—")),
                    "pr_ms": str(fs.get("pr_ms", "—")), "qrs_ms": str(fs.get("qrs_ms", "—")),
                    "motif": "Analyse ECG CSV", "full_diag": diagnosis,
                    "differentials": result.get("differentials", []),
                }
                hist = _add_session_entry(hist, entry)
                stored = {
                    "diagnosis": diagnosis, "confidence": confidence,
                    "hr": fs.get("hr_mean", "—"), "rr": fs.get("rr_mean_ms", "—"),
                    "pr_ms": fs.get("pr_ms", "—"), "qrs_ms": fs.get("qrs_ms", "—"),
                    "differentials": result.get("differentials", []), "module": "Signal+LLM",
                    "urgency": urg_label,
                }
                status = "✅ Analyse terminée · analyse hybride (signal + LLM)"
                return (status, diag_md, analysis_md, reliability_html,
                        result.get("ecg_plot"), metrics_md, alerts_md, hist, stored)

            btn_ecg.click(
                fn=analyze_ecg,
                inputs=[ecg_file, language, session_history],
                outputs=[ecg_status, ecg_diag, ecg_analysis, ecg_reliability,
                         ecg_plot, ecg_metrics, ecg_alerts,
                         session_history, ecg_last_result],
            )

            # Feature 2 : copie ECG
            def _copy_ecg(stored):
                if not stored:
                    return ""
                now = datetime.now().strftime("%d/%m/%Y %H:%M")
                text = (
                    f"Date : {now}\n"
                    f"Diagnostic : {stored.get('diagnosis', '—')}\n"
                    f"Moteur : {stored.get('module', '—')}\n")
                conf = stored.get("confidence", "—")
                if conf not in (None, "—"):
                    text += f"Confiance : {conf} %\n"
                if stored.get("urgency"):
                    text += f"Urgence : {stored.get('urgency')}\n"
                text += f"FC {stored.get('hr', '—')} bpm | RR {stored.get('rr', '—')} ms\n"
                text += "Validé par télémédecin : En attente"
                return _make_copy_button(text, "copyECG")

            btn_copy_ecg.click(
                fn=_copy_ecg,
                inputs=[ecg_last_result],
                outputs=[copy_html_ecg],
            )

            # Feature 4 : fiche imprimable ECG
            def _print_ecg(stored):
                if not stored:
                    return None
                sheet = build_print_sheet({
                    "diagnosis": stored.get("diagnosis", "—"),
                    "confidence": stored.get("confidence", "—"),
                    "differentials": stored.get("differentials", []),
                    "hr": stored.get("hr", "—"),
                    "rr": stored.get("rr", "—"),
                    "pr_ms": stored.get("pr_ms", "—"),
                    "qrs_ms": stored.get("qrs_ms", "—"),
                    "module": stored.get("module", "ECG"),
                    "urgency": stored.get("urgency", "modérée"),
                    "motif": "Analyse ECG",
                })
                return _save_print_sheet(sheet)

            btn_print_ecg.click(
                fn=_print_ecg,
                inputs=[ecg_last_result],
                outputs=[print_html_ecg],
            )

            btn_new_ecg = gr.Button("🆕  Nouvelle analyse", size="lg")
            btn_new_ecg.click(
                fn=lambda: (None, "", "", "", "", None, "", "",
                            "", None, {}),
                outputs=[ecg_file, ecg_status, ecg_diag, ecg_analysis,
                         ecg_reliability, ecg_plot, ecg_metrics, ecg_alerts,
                         copy_html_ecg, print_html_ecg, ecg_last_result],
            )

        # ═══════════════════════════════════════════════════════════════════
        #  ONGLET 5 : TÉLÉMÉDECIN — Historique complet
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📋 Historique des consultations"):

            gr.Markdown("## 📋 Historique complet")

            with gr.Row():
                hist_filter = gr.Dropdown(
                    ["Tous", "en_attente", "validée", "rejetée"],
                    value="Tous",
                    label="Filtrer par statut",
                )
                hist_search = gr.Textbox(label="🔍 Rechercher", placeholder="symptômes, pays…")

            btn_hist_refresh = gr.Button("🔄 Charger / Actualiser", variant="secondary")

            hist_metrics = gr.Markdown()

            hist_list = gr.Dropdown(
                choices=[],
                label="📋 Consultations — sélectionnez une ligne pour afficher le détail",
                interactive=True,
                value=None,
            )

            btn_hist_refresh.click(
                fn=load_history_table,
                inputs=[hist_filter, hist_search],
                outputs=[hist_metrics, hist_list],
            )

            gr.Markdown("---\n## 📄 Détail de la consultation sélectionnée")

            hist_info   = gr.Markdown()
            hist_diag   = gr.Markdown()
            hist_status = gr.Markdown()

            def _load_detail_from_choice(choice):
                """Extrait le N° de la ligne sélectionnée et charge le détail."""
                if not choice:
                    return "", "", ""
                # Le format est : "#ID  …"
                try:
                    consult_id = choice.split("#")[1].split(" ")[0].strip()
                except Exception:
                    return "Impossible de lire le numéro de consultation.", "", ""
                return get_consultation_detail(consult_id)

            hist_list.change(
                fn=_load_detail_from_choice,
                inputs=[hist_list],
                outputs=[hist_info, hist_diag, hist_status],
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  Feature 1 : Historique de session (collapsible, en bas de page)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Accordion("📊 Historique de session (derniers diagnostics)", open=False):
        session_table_md = gr.Markdown("_Aucun diagnostic dans cette session._")
        btn_refresh_session = gr.Button("🔄 Actualiser", size="sm")

        btn_refresh_session.click(
            fn=_session_table_md,
            inputs=[session_history],
            outputs=[session_table_md],
        )

        gr.Markdown("---\n### 🔍 Comparer deux diagnostics")
        with gr.Row():
            compare_a = gr.Number(label="N° entrée A", value=1, minimum=1, maximum=10, precision=0)
            compare_b = gr.Number(label="N° entrée B", value=2, minimum=1, maximum=10, precision=0)
        btn_compare = gr.Button("⚖️ Comparer", size="sm")

        with gr.Row():
            compare_col_a = gr.Markdown()
            compare_col_b = gr.Markdown()

        def _compare_entries(hist, idx_a, idx_b):
            """Affiche deux entrées côte à côte."""
            def _fmt(entry, idx):
                if not entry:
                    return f"### Entrée #{idx}\n\n_Pas de données._"
                return (
                    f"### Entrée #{idx}\n\n"
                    f"**Heure :** {entry.get('time', '—')}\n\n"
                    f"**Patient :** {entry.get('patient', '—')}\n\n"
                    f"**Module :** {entry.get('module', '—')}\n\n"
                    f"**Diagnostic :** {entry.get('diagnosis', '—')}\n\n"
                    f"**Confiance :** {entry.get('confidence', '—')} %"
                )
            ia = int(idx_a) - 1 if idx_a else 0
            ib = int(idx_b) - 1 if idx_b else 1
            a = hist[ia] if ia < len(hist) else None
            b = hist[ib] if ib < len(hist) else None
            return _fmt(a, ia + 1), _fmt(b, ib + 1)

        btn_compare.click(
            fn=_compare_entries,
            inputs=[session_history, compare_a, compare_b],
            outputs=[compare_col_a, compare_col_b],
        )

    gr.Markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem; margin-top:2rem;'>"
        "MedIA v2.0 — Usage professionnel uniquement · Interface Gradio</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  🩺  MedIA — Aide au diagnostic")
    print("  🌐  Interface disponible sur : http://localhost:7860 (ou port suivant si occupé)")
    print("═" * 50 + "\n")
    app.launch(server_name="0.0.0.0", share=False, max_threads=40)
