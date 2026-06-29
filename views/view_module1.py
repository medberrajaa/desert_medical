"""
views/view_module1.py — Interface Module 1 (Diagnostic général)
"""
import streamlit as st
import os
import csv
import tempfile
from datetime import datetime
from module1.prompt_engine import generate_diagnosis, generate_raw_diagnosis, ddss_catalog, URGENCY_FLOOR_LABELS
from views.format_diagnosis import format_clinical_markdown

# Icône par niveau d'urgence plancher (affichage des règles DDSS)
_FLOOR_ICON = {"emergency": "🔴", "urgent_4h": "🟠", "urgent_24h": "🟡", "routine": "🟢"}


def _render_ddss_panel():
    """Panneau de référence : les règles de sécurité DDSS vérifiées automatiquement."""
    with st.expander("🛡️ Règles de sécurité DDSS — vérifiées automatiquement avant l'IA", expanded=False):
        st.caption(
            "Filets de sécurité **déterministes** (sans IA) appliqués au texte du patient. "
            "Si une règle se déclenche, elle impose un **niveau d'urgence minimal**, "
            "indépendamment de la réponse du LLM — pour ne jamais sous-estimer une urgence vitale."
        )
        for r in ddss_catalog():
            icon = _FLOOR_ICON.get(r["urgency_floor"], "⚪")
            st.markdown(
                f"- {icon} **{r['label']}** — {r['description']}  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;↳ *Urgence plancher imposée : {r['urgency_label']}*"
            )


def _render_triggered_ddss(result: dict):
    """Affiche les règles DDSS déclenchées en clair (libellés français)."""
    names = result.get("ddss_alerts", [])
    if not names:
        return
    lut = {r["name"]: r for r in ddss_catalog()}
    labels = [lut.get(n, {}).get("label", n) for n in names]
    floor = result.get("ddss_urgency_floor")
    floor_label = URGENCY_FLOOR_LABELS.get(floor, floor or "?")
    icon = _FLOOR_ICON.get(floor, "🚨")
    st.error(
        f"🚨 **Règles de sécurité DDSS déclenchées :** {', '.join(labels)}\n\n"
        f"{icon} Niveau d'urgence plancher imposé : **{floor_label}**",
        icon="🚨",
    )

# ── Fichier d'historique ─────────────────────────────────────────────────────
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation")
HISTORY_FILE = os.path.join(HISTORY_DIR, "consultations.csv")
HISTORY_FIELDS = [
    "id", "date", "age", "sexe", "ethnie", "symptomes", "country", "language",
    "diagnosis_response", "model", "tokens", "ddss_alerts", "status", "doctor_comment",
]


def _init_history():
    """Crée le fichier d'historique s'il n'existe pas."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()


def _save_consultation(patient: dict, result: dict, country: str, language: str):
    """Sauvegarde une consultation dans l'historique CSV."""
    _init_history()
    # Déterminer le prochain ID
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
    """Charge l'historique des consultations."""
    _init_history()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _update_consultation(consultation_id: str, status: str, comment: str):
    """Met à jour le statut d'une consultation."""
    rows = _load_history()
    for row in rows:
        if str(row["id"]) == str(consultation_id):
            row["status"] = status
            row["doctor_comment"] = comment
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ── Rendu principal ──────────────────────────────────────────────────────────

def render_module1(role: str, country: str, language: str):
    st.markdown(
        "<div class='main-header'>"
        "<h2>🩺 Module 1 — Diagnostic général</h2>"
        f"<p>Contexte : {country} · Langue : {language}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    is_nurse = "Infirmier" in role

    if is_nurse:
        _render_nurse_form(country, language)
    else:
        _render_doctor_dashboard()


# ── Formulaire infirmière ────────────────────────────────────────────────────

# Clés des widgets du formulaire infirmier (pour reset complet)
FORM_KEYS_DEFAULTS = {
    "form_age": 45,
    "form_sexe": "Homme",
    "form_taille": "",
    "form_poids": "",
    "form_ethnie": "",
    "form_motif": "",
    "form_signes": "",
    "form_duree": "",
    "form_antecedents": "",
    "form_traitements": "",
    "form_examens": [],
    "form_resultats": "",
    "form_texte_brut": "",
    "form_mode": "Formulaire détaillé",
}


def _clear_form():
    """Réinitialise tous les champs du formulaire à leurs valeurs par défaut."""
    for key, default in FORM_KEYS_DEFAULTS.items():
        st.session_state[key] = default
    for key in ["last_diagnosis", "last_patient"]:
        st.session_state.pop(key, None)


def _render_nurse_form(country: str, language: str):
    """Formulaire de saisie pour l'infirmier."""

    # ── Choix du mode de saisie ──
    mode = st.radio(
        "Mode de saisie",
        ["Formulaire détaillé", "📝 Saisie texte libre"],
        horizontal=True,
        key="form_mode",
    )

    _render_ddss_panel()

    if "Saisie texte libre" in mode:
        _render_raw_text_form(country, language)
        return

    st.subheader("📝 Informations patient")

    # ── Identité ──
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Âge", min_value=0, max_value=120, value=45, key="form_age")
        with col2:
            sexe = st.selectbox("Sexe", ["Homme", "Femme", "Non précisé"], key="form_sexe")
        with col3:
            taille_raw = st.text_input("Taille (cm)", placeholder="ex : 170  (optionnel)", key="form_taille")
            taille = int(taille_raw) if taille_raw.strip().isdigit() else None
        with col4:
            poids_raw = st.text_input("Poids (kg)", placeholder="ex : 70  (optionnel)", key="form_poids")
            poids = float(poids_raw.replace(",", ".")) if poids_raw.strip().replace(",", ".").replace(".", "", 1).isdigit() else None

    ethnie = st.text_input("Ethnie (optionnel)", placeholder="ex : Européen, Africain, Asiatique…", key="form_ethnie")

    st.divider()

    # ── Motif & signes cliniques ──
    st.subheader("📌 Motif de consultation & signes cliniques")

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        motif_consultation = st.text_area(
            "Motif de consultation *",
            placeholder="Raison principale de la visite… ex : douleurs abdominales, fièvre depuis 3 jours, contrôle de routine…",
            height=90,
            key="form_motif",
        )
    with col_m2:
        duree = st.text_input("Durée des symptômes", placeholder="ex : 2 jours", key="form_duree")

    signes_cliniques = st.text_area(
        "Signes cliniques observés",
        placeholder="Signes objectifs relevés par l'infirmière… ex : fièvre à 38.5°C, dyspnée, oедème des membres inférieurs, pâleur…",
        height=90,
        key="form_signes",
    )

    # ── Antécédents & traitements ──
    col6, col7 = st.columns(2)
    with col6:
        antecedents = st.text_area(
            "Antécédents médicaux",
            placeholder="HTA, diabète, cardiopathie…",
            height=80,
            key="form_antecedents",
        )
    with col7:
        traitements = st.text_area(
            "Traitements en cours",
            placeholder="Médicaments, posologie…",
            height=80,
            key="form_traitements",
        )

    st.divider()

    # ── Examens ──
    st.subheader("🔬 Examens & résultats")
    examens = st.multiselect(
        "Examens réalisés",
        [
            # Constantes vitales
            "Tension artérielle",
            "Fréquence cardiaque",
            "Fréquence respiratoire",
            "Température",
            "Saturation O₂ (SpO₂)",
            "Glycémie capillaire",
            "Poids & Taille (IMC)",
            # Cardiologie
            "ECG 6 dérivations",
            "ECG 12 dérivations",
            # Biologie
            "NFS (Numération Formule Sanguine)",
            "CRP (Protéine C-Réactive)",
            "Ionogramme sanguin",
            "Créatinine / Urée",
            "Glycémie à jeun",
            "HbA1c",
            "Bilan hépatique (ASAT/ALAT)",
            "TSH (Thyroïde)",
            "Troponine",
            "D-Dimères",
            "Gaz du sang (GDS)",
            "Bandelette urinaire (BU)",
            "ECBU",
            # Imagerie
            "Radiographie thoracique",
            "Échographie abdominale",
            "Scanner / TDM",
            "IRM",
            # Autres
            "Test COVID-19 (antigénique)",
            "Test paludisme (TDR)",
            "Peak-flow (DEP)",
        ],
        key="form_examens",
    )
    resultats = st.text_area(
        "Résultats des examens",
        placeholder="TA : 145/90 mmHg, T° : 38.2°C, SpO₂ : 97%…",
        height=80,
        key="form_resultats",
    )

    # L'analyse ECG (CSV ou image) est gérée par le Module 2 — Cardiologie.
    st.info("🫀 Pour analyser un **ECG** (fichier CSV ou image du tracé), utilisez le **Module 2 — Cardiologie (ECG)**.")

    st.divider()

    # ── Génération ──
    if st.button("🔍  Générer le diagnostic IA", type="primary", width='stretch'):
        if not motif_consultation.strip():
            st.error("Veuillez renseigner le motif de consultation.")
            return

        patient = {
            "age": age,
            "sexe": sexe,
            "ethnie": ethnie or "Non renseignée",
            "taille": taille if taille is not None else "Non renseignée",
            "poids": poids if poids is not None else "Non renseigné",
            "duree_symptomes": duree,
            "motif_consultation": motif_consultation,
            "signes_cliniques": signes_cliniques or "Aucun noté",
            "antecedents": antecedents or "Aucun connu",
            "traitements": traitements or "Aucun",
            "examens": ", ".join(examens) if examens else "Aucun",
            "resultats": resultats or "Aucun",
        }

        with st.spinner("Analyse en cours..."):
            result = generate_diagnosis(patient, country=country, language=language)

        if result["success"]:
            # ── Alertes DDSS (règles de sécurité déterministes) ──
            _render_triggered_ddss(result)

            st.success(f"✅ Diagnostic généré · Modèle : {result['model']} · {result['tokens_used']} tokens")
            st.divider()
            st.markdown(format_clinical_markdown(result.get("data", {})))
            with st.expander("🧬 Données structurées (JSON)"):
                st.json(result.get("data", {}))

            # Sauvegarder dans l'historique
            cid = _save_consultation(patient, result, country, language)
            st.info(f"📋 Consultation #{cid} enregistrée dans l'historique.")

            # Session pour validation immédiate
            st.session_state["last_diagnosis"] = result
            st.session_state["last_patient"] = patient
        else:
            st.error(f"Erreur API : {result.get('error', 'Erreur inconnue')}")
            st.info("Vérifiez votre GROQ_API_KEY dans le fichier .env")

        with st.expander("📄 Voir le prompt envoyé à l'IA"):
            st.code(result.get("prompt", ""), language="text")

        # Bouton nouveau diagnostic
        st.divider()
        if st.button("🆕  Nouveau diagnostic", width='stretch'):
            _clear_form()
            st.rerun()


# ── Saisie texte libre ───────────────────────────────────────────────────────

def _render_raw_text_form(country: str, language: str):
    """Permet à l'infirmier de coller un bloc de texte brut avec toutes les infos patient."""
    st.subheader("📝 Saisie texte libre")

    st.info(
        "Collez ci-dessous toutes les informations du patient en texte libre : "
        "identité, motif, signes cliniques, antécédents, résultats d'examens… "
        "L'IA analysera l'ensemble."
    )

    texte_brut = st.text_area(
        "Informations patient (texte brut) *",
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
        height=250,
        key="form_texte_brut",
    )

    st.divider()

    if st.button("🔍  Générer le diagnostic IA", type="primary", width='stretch'):
        if not texte_brut.strip():
            st.error("Veuillez saisir les informations du patient.")
            return

        patient = {
            "motif_consultation": texte_brut,
            "signes_cliniques": "",
            "age": 0,
            "sexe": "Non précisé",
        }

        with st.spinner("Analyse en cours..."):
            result = generate_raw_diagnosis(texte_brut, country=country, language=language)

        if result["success"]:
            alerts = result.get("ddss_alerts", [])
            if alerts:
                st.error(
                    f"🚨 **Alertes système DDSS :** {', '.join(alerts)}\n\n"
                    f"Niveau d'urgence plancher : **{result.get('ddss_urgency_floor', '?')}**",
                    icon="🚨",
                )

            st.success(f"✅ Diagnostic généré · Modèle : {result['model']} · {result['tokens_used']} tokens")
            st.divider()
            st.markdown(format_clinical_markdown(result.get("data", {})))
            with st.expander("🧬 Données structurées (JSON)"):
                st.json(result.get("data", {}))

            cid = _save_consultation(patient, result, country, language)
            st.info(f"📋 Consultation #{cid} enregistrée dans l'historique.")

            st.session_state["last_diagnosis"] = result
            st.session_state["last_patient"] = patient
        else:
            st.error(f"Erreur API : {result.get('error', 'Erreur inconnue')}")
            st.info("Vérifiez votre GROQ_API_KEY dans le fichier .env")

        with st.expander("📄 Voir le prompt envoyé à l'IA"):
            st.code(result.get("prompt", ""), language="text")

        st.divider()
        if st.button("🆕  Nouveau diagnostic", width='stretch', key="btn_new_raw"):
            _clear_form()
            st.rerun()


# ── Dashboard Télémédecin ────────────────────────────────────────────────────

def _render_doctor_dashboard():
    """Interface complète pour le télémédecin : validation + historique."""
    tab_validate, tab_history = st.tabs(["✅ Validation en cours", "📋 Historique des consultations"])

    with tab_validate:
        _render_validation_tab()

    with tab_history:
        _render_history_tab()


def _render_history_tab():
    """Affiche l'historique des consultations avec filtres."""
    import pandas as pd

    history = _load_history()

    if not history:
        st.info("Aucune consultation enregistrée pour le moment.")
        return

    df = pd.DataFrame(history)

    # ── Métriques résumé ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total consultations", len(df))
    with col2:
        en_attente = len(df[df["status"] == "en_attente"])
        st.metric("En attente", en_attente)
    with col3:
        validees = len(df[df["status"] == "validée"])
        st.metric("Validées", validees)
    with col4:
        rejetees = len(df[df["status"] == "rejetée"])
        st.metric("Rejetées", rejetees)

    st.divider()

    # ── Filtres ──
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        status_filter = st.selectbox(
            "Filtrer par statut",
            ["Tous", "en_attente", "validée", "rejetée"],
            index=0,
        )
    with col_f2:
        search = st.text_input("🔍 Rechercher (symptômes, pays…)", "")

    df_filtered = df.copy()
    if status_filter != "Tous":
        df_filtered = df_filtered[df_filtered["status"] == status_filter]
    if search:
        mask = df_filtered.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        df_filtered = df_filtered[mask]

    # ── Tableau ──
    if df_filtered.empty:
        st.warning("Aucune consultation ne correspond aux filtres.")
        return

    display_cols = ["id", "date", "age", "sexe", "ethnie", "symptomes", "country", "status"]
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    st.dataframe(
        df_filtered[available_cols],
        width='stretch',
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("N°", width="small"),
            "date": st.column_config.TextColumn("Date"),
            "age": st.column_config.NumberColumn("Âge"),
            "sexe": st.column_config.TextColumn("Sexe"),
            "ethnie": st.column_config.TextColumn("Ethnie"),
            "symptomes": st.column_config.TextColumn("Symptômes", width="large"),
            "country": st.column_config.TextColumn("Pays"),
            "status": st.column_config.TextColumn("Statut"),
        },
    )

    # ── Sélection d'une consultation ──
    st.divider()
    st.subheader("📄 Détail d'une consultation")
    ids = df_filtered["id"].tolist()
    selected_id = st.selectbox("Sélectionner une consultation", ids, format_func=lambda x: f"Consultation #{x}")

    if selected_id:
        row = df_filtered[df_filtered["id"] == selected_id].iloc[0]
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Âge", f"{row.get('age', '?')} ans")
        with col_d2:
            st.metric("Sexe", row.get("sexe", "?"))
        with col_d3:
            ethnie_val = row.get("ethnie", "")
            st.metric("Ethnie", ethnie_val if ethnie_val else "Non renseignée")
        with col_d4:
            st.metric("Pays", row.get("country", "?"))

        st.markdown(f"**Symptômes :** {row.get('symptomes', '')}")
        if row.get("ddss_alerts"):
            st.warning(f"🚨 Alertes DDSS : {row['ddss_alerts']}")

        st.divider()
        st.subheader("📋 Diagnostic IA complet")
        st.markdown(row.get("diagnosis_response", "Non disponible"))

        st.caption(f"Modèle : {row.get('model', '?')} · Tokens : {row.get('tokens', '?')}")

        # ── Actions de validation ──
        current_status = row.get("status", "en_attente")
        if current_status == "en_attente":
            st.divider()
            st.subheader("⚖️ Décision médicale")
            decision = st.radio(
                "Action",
                ["✅ Valider le diagnostic", "❌ Rejeter le diagnostic"],
                key=f"decision_{selected_id}",
            )
            comment = st.text_area(
                "Commentaire médical",
                placeholder="Observations, corrections ou contre-indications…",
                key=f"comment_{selected_id}",
            )
            if st.button("Confirmer la décision", type="primary", key=f"btn_{selected_id}"):
                new_status = "validée" if "Valider" in decision else "rejetée"
                _update_consultation(selected_id, new_status, comment)
                st.success(f"Consultation #{selected_id} → **{new_status}**")
                st.rerun()
        else:
            status_icon = "✅" if current_status == "validée" else "❌"
            st.info(f"{status_icon} Cette consultation a été **{current_status}**.")
            if row.get("doctor_comment"):
                st.markdown(f"**Commentaire :** {row['doctor_comment']}")


def _render_validation_tab():
    """Affiche uniquement les consultations en attente de validation."""
    history = _load_history()
    pending = [h for h in history if h.get("status") == "en_attente"]

    if not pending:
        st.success("🎉 Aucune consultation en attente de validation.")
        return

    st.info(f"**{len(pending)}** consultation(s) en attente de validation.")

    for consultation in pending:
        with st.expander(
            f"#{consultation['id']} — {consultation.get('date', '?')} · "
            f"{consultation.get('symptomes', '')[:60]}…",
            expanded=False,
        ):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Âge", f"{consultation.get('age', '?')} ans")
            with col2:
                st.metric("Sexe", consultation.get("sexe", "?"))
            with col3:
                ethnie_val = consultation.get("ethnie", "")
                st.metric("Ethnie", ethnie_val if ethnie_val else "Non renseignée")
            with col4:
                st.metric("Pays", consultation.get("country", "?"))

            st.markdown(f"**Symptômes :** {consultation.get('symptomes', '')}")
            st.divider()
            st.markdown("**Diagnostic IA :**")
            st.markdown(consultation.get("diagnosis_response", ""))

            decision = st.radio(
                "Décision",
                ["✅ Valider", "❌ Rejeter"],
                key=f"pending_decision_{consultation['id']}",
            )
            comment = st.text_area(
                "Commentaire",
                key=f"pending_comment_{consultation['id']}",
                height=60,
            )
            if st.button("Confirmer", type="primary", key=f"pending_btn_{consultation['id']}"):
                new_status = "validée" if "Valider" in decision else "rejetée"
                _update_consultation(consultation["id"], new_status, comment)
                st.success(f"Consultation #{consultation['id']} → **{new_status}**")
                st.rerun()
