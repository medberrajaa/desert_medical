"""
app.py — Application principale MedIA
Point d'entrée Streamlit. Orchestre les deux modules.
"""
import streamlit as st

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="MedIA — Aide au diagnostic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1b2d 0%, #1a2940 100%);
        color: #e0e6ed;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #ffffff !important;
    }
    /* Section headers (widget labels) — accent cyan, bien lisibles */
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] .stSelectbox > label {
        color: #22d3ee !important;       /* cyan */
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px;
    }
    /* Options des boutons radio — texte clair et lisible */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: #e6eef5 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    /* Pastille du bouton radio en cyan */
    section[data-testid="stSidebar"] div[role="radiogroup"] input {
        accent-color: #22d3ee !important;
    }
    /* Option sélectionnée : surlignage cyan */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: rgba(34, 211, 238, 0.14);
        border-radius: 6px;
        padding: 2px 6px;
        margin-left: -6px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #67e8f9 !important;       /* cyan clair */
        font-weight: 700 !important;
    }
    /* Survol */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
        color: #a5f3fc !important;
    }
    /* Texte des menus déroulants (Pays / Langue) bien contrasté */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        color: #0f1b2d !important;
    }
    /* Cards / containers */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
    .stMetric {
        background: #f8fafc;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    /* Buttons */
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }
    /* Main header */
    .main-header {
        padding: 1rem 0 0.5rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .main-header h2 {
        margin: 0;
        color: #1a2940;
    }
    .main-header p {
        margin: 0.2rem 0 0 0;
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar : navigation et paramètres ──────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MedIA")
    st.caption("Outil d'aide au diagnostic — Déserts médicaux")
    st.divider()

    role = st.radio(
        "👤 Profil utilisateur",
        ["⚕️ Infirmier", "👨‍⚕️ Télémédecin"],
        index=0,
    )
    st.divider()

    module = st.radio(
        "📋 Module actif",
        ["Module 1 — Diagnostic général", "Module 2 — Cardiologie (ECG)"],
        index=0,
    )
    st.divider()

    country = st.selectbox(
        "🌍 Pays / contexte",
        ["France", "Sénégal", "Maroc", "Belgique", "Côte d'Ivoire", "Cameroun", "Benin", "Autre"],
        index=0,
    )
    language = st.selectbox(
        "🗣️ Langue du diagnostic",
        ["Français", "English", "Arabic"],
        index=0,
    )

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        "MedIA v2.0 — Usage professionnel uniquement</div>",
        unsafe_allow_html=True,
    )

# ── Routage principal ────────────────────────────────────────────────────────
if module == "Module 1 — Diagnostic général":
    from views.view_module1 import render_module1
    render_module1(role=role, country=country, language=language)
else:
    from views.view_module2 import render_module2
    render_module2(role=role, language=language)
