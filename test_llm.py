"""
test_llm.py — Test isolé du Module 1 (LLM / Diagnostic général)
Exécuter : streamlit run test_llm.py
"""
import streamlit as st
from views.view_module1 import render_module1

st.set_page_config(
    page_title="Test LLM — Module 1",
    page_icon="🧪",
    layout="wide",
)

with st.sidebar:
    st.title("🧪 Test Module 1")
    st.caption("Test isolé du LLM sans ECG")
    st.divider()

    role = st.radio(
        "Profil utilisateur",
        ["�‍⚕️ Infirmier", "👨‍⚕️ Télémédecin"],
        index=0,
    )
    st.divider()

    country = st.selectbox(
        "Pays / contexte",
        ["France", "Sénégal", "Maroc", "Belgique", "Côte d'Ivoire", "Cameroun", "Benin", "Autre"],
        index=0,
    )
    language = st.selectbox(
        "Langue du diagnostic",
        ["Français", "English", "Arabic"],
        index=0,
    )

render_module1(role=role, country=country, language=language)
