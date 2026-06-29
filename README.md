# MedIA — Outil d'aide au diagnostic en déserts médicaux

Projet MIAS M2 — Centrale Lille · Mars 2026  
Encadrant : M. Broucqsault (EXOFIT)

---

## Prérequis

- Python 3.10+
- Les données brutes ECG (dossier `data/ECG AHP/`) — **non versionnées**, à récupérer séparément
- Une clé API Groq gratuite : https://console.groq.com

---

## Installation

```bash
# 1. Créer et activer un environnement virtuel
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration

Créer un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=votre_cle_groq
```

---

## Cas 1 — Le modèle `.pkl` est déjà disponible

Si vous avez reçu le fichier `models/xgboost_ecg.pkl` (et `models/label_encoder.pkl`), vous pouvez lancer l'application directement :

```bash
streamlit run app.py
```

L'application sera accessible sur http://localhost:8501.

---

## Cas 2 — Entraîner le modèle depuis zéro

Les données brutes et le modèle ne sont pas versionnés. Si vous devez tout reconstruire, suivez ces étapes **dans l'ordre**.

### Étape 1 — Générer le fichier de métadonnées

```bash
python training/generate_full_meta.py
# Produit : data/samples/full_meta.csv
```

> Variante avec sous-échantillonnage stratifié (plus rapide) :
>
> ```bash
> python training/sample_ecg.py
> # Produit : data/samples/sample_meta.csv
> ```

### Étape 2 — Prétraitement des signaux ECG (feature extraction)

```bash
# Sur le dataset complet
python training/preprocess_ecg.py --data_dir "data/ECG AHP/data_ahp_ecg/data" --out_dir data/processed_full --n_jobs 4

# Ou sur l'échantillon (plus rapide, pour tester)
python training/preprocess_ecg.py --sample_file data/samples/sample_meta.csv --out_dir data/processed_sample --n_jobs 4
```

> `--n_jobs 4` parallélise le traitement. Augmenter selon votre nombre de cœurs CPU.  
> Produit : `data/processed_full/features.parquet` (~100 MB)

### Étape 3 — Entraînement XGBoost

```bash
python training/train_ecg.py
# Produit : models/xgboost_ecg.pkl, models/label_encoder.pkl, models/model_meta.json
```

Options disponibles :

```bash
python training/train_ecg.py --n_estimators 500 --seed 42
```

### Étape 4 — Lancer l'application

```bash
streamlit run app.py
```

---

## Structure du projet

```
app.py                  — Point d'entrée Streamlit
module1/                — Module diagnostic général (LLM via Groq)
  prompt_engine.py
  voice_engine.py
module2/                — Module diagnostic cardiaque (ECG + XGBoost)
  ecg_predictor.py
  ecg_image_reader.py
training/               — Pipeline d'entraînement (à exécuter dans l'ordre)
  generate_full_meta.py — Étape 1 : métadonnées
  sample_ecg.py         — Étape 1 (variante) : sous-échantillonnage
  preprocess_ecg.py     — Étape 2 : feature extraction
  train_ecg.py          — Étape 3 : entraînement XGBoost
data/                   — Données brutes et prétraitées (non versionnées)
models/                 — Modèles entraînés .pkl (non versionnés)
evaluation/             — Rapports de performance (classification report)
views/                  — Composants UI Streamlit par module
```

---

## Notes importantes

- `data/` et `models/*.pkl` sont exclus du dépôt git (fichiers trop volumineux).
- Le fichier `data/processed_full/features.parquet` (~100 MB) dépasse la limite GitHub — ne pas le commiter.
- Le modèle entraîné supporte 10 classes : AFIB, ARYTHMIE_SINUSALE, BAV1, BRADYCARDIE, EXTRASYSTOLES, FLUTTER, NORMAL, PACE_AURIC, PACE_VENT, TACHYCARDIE.
