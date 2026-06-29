"""
views/view_module2.py — Interface Module 2 (Diagnostic cardiaque ECG)

Accepte un CSV 12 dérivations OU une image de tracé (JPG/PNG). Dans les deux cas
le tracé 12 dérivations est reconstruit et affiché de la même façon. Les images
sont lues par un LLM vision (style PULSE) avec ancrage sur les cas de référence
validés ; les CSV passent par l'analyse hybride mesures du signal + LLM.
"""
import os
import tempfile

import pandas as pd
import streamlit as st

from module2.ecg_predictor import PATHOLOGIES, diagnose_ecg_image, predict_ecg
from module2.ecg_labels import ecg_urgency_badge, fr_ecg_label

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def render_module2(role: str, language: str = "Français"):
    st.title("Module 2 — Diagnostic cardiaque (ECG)")
    st.caption("Analyse d'ECG 12 dérivations — fichier CSV ou image du tracé")

    is_nurse = "Infirmier" in role

    if is_nurse:
        _render_ecg_upload(language)
    else:
        _render_ecg_validation()


def _render_ecg_upload(language: str):
    """Upload et analyse ECG pour l'infirmier (CSV ou image)."""
    st.subheader("Uploader un ECG")
    st.info(
        "Formats acceptés : **CSV** (une colonne par dérivation I, II, III, aVR, aVL, "
        "aVF, V1–V6 · 500 Hz) ou **image** du tracé (JPG/PNG). Le moteur d'analyse est "
        "choisi automatiquement selon le type de fichier."
    )

    uploaded = st.file_uploader(
        "Sélectionner le fichier ECG (.csv, .jpg, .png)",
        type=["csv", "jpg", "jpeg", "png"],
    )
    if uploaded is None:
        return

    is_image = uploaded.name.lower().endswith(_IMAGE_EXTS)

    # Persister le fichier sur disque pour les pipelines (qui attendent un chemin)
    suffix = os.path.splitext(uploaded.name)[1] or (".png" if is_image else ".csv")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    uploaded.seek(0)

    if is_image:
        st.image(uploaded, caption=f"Image chargée : {uploaded.name}", width='stretch')
    else:
        with st.expander("Aperçu des données brutes"):
            df_preview = pd.read_csv(uploaded)
            uploaded.seek(0)
            st.dataframe(df_preview.head(20), width='stretch')
            st.caption(f"{len(df_preview)} lignes · {len(df_preview.columns)} colonnes")

    label = "🫀 Analyser l'image ECG" if is_image else "🫀 Analyser l'ECG"
    if st.button(label, type="primary", width='stretch'):
        spinner = "Lecture du tracé par le LLM vision…" if is_image else "Mesures du signal et interprétation…"
        with st.spinner(spinner):
            if is_image:
                result = diagnose_ecg_image(tmp_path, language=language)
            else:
                result = predict_ecg(tmp_path, use_llm=True)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if result.get("success"):
            result["_source"] = "image" if is_image else "csv"
            _display_ecg_result(result)
            st.session_state["last_ecg_result"] = result
        else:
            st.error(f"Erreur : {result.get('error')}")


def _diagnosis_fr(result: dict) -> tuple[str, str]:
    """Retourne (titre, sous-titre) du diagnostic en français selon la source."""
    if result.get("_source") == "image":
        return result.get("diagnosis", "—"), result.get("subtitle", "")
    return fr_ecg_label(result.get("diagnosis_display") or result.get("diagnosis", "—")), ""


def _urgency_of(result: dict):
    if result.get("_source") == "image":
        return result.get("urgency")
    return result.get("clinical_report", {}).get("urgency")


def _display_ecg_result(result: dict):
    """Affiche le résultat de l'analyse ECG (image ou CSV) de façon unifiée."""
    diagnosis, subtitle = _diagnosis_fr(result)
    confidence = result.get("confidence")
    urg_icon, urg_label = ecg_urgency_badge(_urgency_of(result))
    is_image = result.get("_source") == "image"
    engine = result.get("interpretation_source") or ("LLM vision" if is_image else "signal + LLM")
    if is_image and result.get("reference_case"):
        engine = "cas de référence validé + LLM vision"

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diagnostic principal", diagnosis)
    with col2:
        st.metric("Confiance du modèle", f"{confidence} %" if confidence is not None else "—")
    with col3:
        st.metric("Fréquence cardiaque", f"{result.get('features_summary', {}).get('hr_mean', '?')} bpm")

    if subtitle:
        st.markdown(f"**{subtitle}**")

    # Bandeau d'urgence
    banner = f"{urg_icon} **Niveau d'urgence : {urg_label}**  ·  *Moteur : {engine}*"
    if urg_icon == "🔴":
        st.error(banner)
    elif urg_icon == "🟡":
        st.warning(banner)
    else:
        st.success(banner)

    # ── Tracé ECG 12 dérivations (identique pour CSV et image) ──
    if result.get("ecg_plot") is not None:
        st.subheader("Tracé ECG — 12 dérivations")
        st.pyplot(result["ecg_plot"])
        if is_image:
            st.caption("Tracé reconstruit depuis l'image — amplitudes approximatives.")

    # ── Analyse détaillée (LLM vision, style PULSE) ──
    if result.get("analysis"):
        st.subheader("Analyse détaillée (LLM vision)")
        st.markdown(result["analysis"])

    # ── Diagnostics différentiels (CSV) ──
    diffs = result.get("differentials", [])
    if diffs:
        st.subheader("Diagnostics différentiels")
        st.markdown(" · ".join(f"**{fr_ecg_label(d['label'])}** ({d.get('confidence', '—')} %)" for d in diffs))

    # ── Conduite recommandée (CSV/LLM) ──
    report = result.get("clinical_report", {})
    if report.get("recommended_action"):
        st.subheader("Conduite recommandée")
        st.markdown(report["recommended_action"])

    # ── Alertes cliniques déterministes ──
    alerts = result.get("clinical_alerts", [])
    if alerts:
        st.subheader("Alertes cliniques (règles déterministes)")
        icon = {"info": "🟢", "warning": "🟡", "critical": "🔴"}
        st.table(
            pd.DataFrame(
                [
                    {
                        "Sévérité": f"{icon.get(a.get('severity'), '⚪')} {str(a.get('severity', '')).upper()}",
                        "Anomalie": fr_ecg_label(a.get("label", "")),
                        "Détail": a.get("detail", ""),
                    }
                    for a in alerts
                ]
            )
        )

    # ── Métriques cliniques ──
    fs = result.get("features_summary", {})
    if fs:
        with st.expander("📊 Métriques cliniques"):
            note = " (approximatives, depuis l'image)" if is_image else ""
            st.markdown(
                f"| Paramètre | Valeur{note} |\n|---|---|\n"
                f"| Fréquence cardiaque | {fs.get('hr_mean', '—')} bpm |\n"
                f"| Intervalle RR moyen | {fs.get('rr_mean_ms', '—')} ms |\n"
                f"| Intervalle PR | {fs.get('pr_ms', '—')} ms |\n"
                f"| Durée QRS | {fs.get('qrs_ms', '—')} ms |\n"
                f"| Intervalle QT | {fs.get('qt_ms', '—')} ms |\n"
                f"| QTc | {fs.get('qtc_ms', '—')} ms |\n"
            )

    if result.get("missing_leads"):
        st.warning(
            "⚠ Dérivations non reconstruites depuis l'image : "
            + ", ".join(result["missing_leads"])
        )


def _render_ecg_validation():
    """Interface de validation pour le médecin."""
    st.subheader("Validation du diagnostic ECG")

    if "last_ecg_result" not in st.session_state:
        st.info("Aucun ECG en attente de validation.")
        return

    result = st.session_state["last_ecg_result"]
    _display_ecg_result(result)

    st.divider()
    decision = st.radio(
        "Décision médicale",
        ["✅ Confirmer le diagnostic", "⚠️ Diagnostic alternatif", "❌ Rejeter"],
    )

    if decision == "⚠️ Diagnostic alternatif":
        st.selectbox("Diagnostic retenu", PATHOLOGIES)

    st.text_area("Observations cliniques complémentaires", height=80)

    if st.button("Valider et enregistrer", type="primary"):
        st.success("Validation enregistrée.")
