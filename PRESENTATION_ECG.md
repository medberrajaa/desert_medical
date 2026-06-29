# Présentation du Module ECG — MedIA

## 1. Vue d'ensemble

Le **Module 2 (ECG)** est un système de **classification automatique d'ECG 12 dérivations** basé sur du Machine Learning (XGBoost) combiné à un **moteur de règles cliniques déterministes**.

**Objectif** : À partir d'un fichier CSV brut d'ECG 12 dérivations, le système :

1. Nettoie le signal (filtrage, normalisation)
2. Extrait 484 features (temporelles, fréquentielles, inter-dérivations)
3. Prédit la pathologie cardiaque parmi 10 classes via XGBoost
4. Applique 8 règles cliniques pour émettre des alertes complémentaires
5. Affiche le tracé ECG, les métriques cliniques (HR, RR, PR, QRS, RMSSD, LF/HF) et les diagnostics différentiels

---

## 2. Les données

### Source

- **Base de données** : ECG AHP (Assistance Hôpitaux Publics)
- **Volume total** : 32 536 enregistrements ECG
- **Format** : fichiers CSV, chacun = 1 patient
- **Structure par fichier** : 5 000 lignes × 12 colonnes (= 12 dérivations)
- **Fréquence d'échantillonnage** : 500 Hz → chaque fichier = **10 secondes** d'enregistrement
- **Unité** : microvolts (µV), valeurs entières

### Les 12 dérivations

| Dérivations des membres | Dérivations précordiales |
| ----------------------- | ------------------------ |
| I, II, III              | V1, V2, V3               |
| aVR, aVL, aVF           | V4, V5, V6               |

### Métadonnées

- Fichier `df_meta.pkl` : 32 536 lignes × 11 colonnes
- Contient : `patient_id`, `diagnosis` (liste de diagnostics textuels), `age`, `gender`, `ecg_file_path`

---

## 3. Pipeline d'entraînement (3 étapes)

Le pipeline se lance en 3 commandes successives :

```bash
python training/sample_ecg.py          # Étape 1 : Échantillonnage
python training/preprocess_ecg.py      # Étape 2 : Prétraitement
python training/train_ecg.py           # Étape 3 : Entraînement
```

---

### Étape 1 — Échantillonnage stratifié (`sample_ecg.py`)

**Problème** : Les 32 536 ECG sont très déséquilibrés (beaucoup de NORMAL, peu de FLUTTER).

**Solution** :

- Mapping des diagnostics textuels français → 10 classes simplifiées
- Échantillonnage stratifié avec caps par classe
- Exclusion des classes trop rares (BAV2 : 23, BBG : 21) et trop hétérogènes (SINUSAL_AUTRE)

**Configuration finale de l'échantillonnage :**

| Classe            | Disponible | Sélectionné | Stratégie                        |
| ----------------- | ---------- | ----------- | -------------------------------- |
| NORMAL            | ~18 000    | 7 000       | Sous-échantillonné (cap)         |
| BRADYCARDIE       | ~2 800     | 2 000       | Sous-échantillonné               |
| EXTRASYSTOLES     | ~1 830     | 1 200       | Sous-échantillonné               |
| AFIB              | ~1 760     | 1 200       | Sous-échantillonné               |
| BAV1              | ~1 492     | 1 492       | **Tout pris** (classe difficile) |
| TACHYCARDIE       | ~1 155     | 1 000       | Sous-échantillonné               |
| ARYTHMIE_SINUSALE | ~306       | 306         | **Tout pris**                    |
| FLUTTER           | ~271       | 271         | **Tout pris**                    |
| PACE_VENT         | ~239       | 239         | **Tout pris**                    |
| PACE_AURIC        | ~138       | 138         | **Tout pris**                    |
| **TOTAL**         | 32 536     | **14 846**  | —                                |

**Sortie** : `data/samples/sample_meta.csv` (14 846 lignes avec chemins + labels)

---

### Étape 2 — Prétraitement (`preprocess_ecg.py`)

Pour chaque patient (fichier CSV), on applique :

#### 2a. Nettoyage du signal (par dérivation)

1. **Clip des artefacts** : quantiles 0.1 % / 99.9 %
2. **Filtre passe-haut 0.5 Hz** (Butterworth ordre 4) → supprime la dérive de la ligne de base
3. **Filtre coupe-bande 50 Hz** (notch, Q=30) → supprime l'interférence du secteur électrique européen
4. **Filtre passe-bas 150 Hz** (Butterworth ordre 4) → supprime le bruit haute fréquence
5. **Normalisation Z-score** : μ=0, σ=1

#### 2b. Extraction de features (484 au total)

**Features temporelles (26 par dérivation × 12 = 312)** :

- Statistiques de base : mean, std, min, max, median, Q25, Q75, IQR, skewness, kurtosis, RMS, peak-to-peak, energy
- Morphologie : zero_crossings, paramètres de Hjorth (activity, mobility, complexity), dérivées 1ère et 2ème
- HRV (variabilité cardiaque) : heart_rate_mean, rr_mean, rr_std, rr_cv, RMSSD, pNN50, n_beats
  - Détection des pics R par recherche de maxima locaux (hauteur > 0.5 × max, distance > 150 ms, proéminence > 0.3)

**Features fréquentielles / FFT (14 par dérivation × 12 = 168)** :

- Fenêtrage de Hann avant FFT (réduction des fuites spectrales)
- Puissance par bande : VLF (0.003–0.04 Hz), LF (0.04–0.15 Hz), HF (0.15–0.4 Hz), cardiaque (0.4–10 Hz), large (0.5–40 Hz), bruit HF (40–150 Hz)
- Ratios : **LF/HF** (balance sympatho-vagale), cardiaque/total
- Fréquence dominante (Hz et bpm)
- Features spectrales : centroïde, entropie, rolloff 85 %, spread

**Features inter-dérivations (4)** :

- Corrélation croisée II–III (axe électrique)
- Corrélation croisée V1–V6 (axe transversal)
- Ratio énergie précordiales/membres
- Énergie totale du signal

**Total : 312 + 168 + 4 = 484 features par patient**

**Sortie** : `data/processed_sample/features.parquet` (14 846 lignes × 486 colonnes)

> La parallélisation (joblib, 4 cœurs) permet de traiter les 14 846 patients rapidement.

---

### Étape 3 — Entraînement XGBoost (`train_ecg.py`)

#### Préparation

- Encodage des labels : LabelEncoder (10 classes → 0–9)
- Split stratifié : **80 % train / 20 % test** (seed 42)
  - Train : 11 876 patients
  - Test : 2 970 patients
- **Gestion du déséquilibre** : `compute_sample_weight(class_weight="balanced")` → poids inversement proportionnels à la fréquence de chaque classe

#### Hyperparamètres XGBoost

| Paramètre             | Valeur         | Rôle                             |
| --------------------- | -------------- | -------------------------------- |
| n_estimators          | 500 (max)      | Nombre max d'arbres              |
| max_depth             | 6              | Profondeur maximale              |
| learning_rate         | 0.05           | Taux d'apprentissage             |
| subsample             | 0.8            | Fraction de données par arbre    |
| colsample_bytree      | 0.8            | Fraction de features par arbre   |
| min_child_weight      | 3              | Minimum de poids par feuille     |
| gamma                 | 0.1            | Seuil de gain minimum pour split |
| objective             | multi:softprob | Classification multiclasse       |
| eval_metric           | mlogloss       | Log-loss multiclasse             |
| early_stopping_rounds | 30             | Arrêt si pas d'amélioration      |

→ **Early stopping à l'itération 395** (sur 500 max)

---

## 4. Résultats du modèle

### Performance globale

| Métrique              | Valeur     |
| --------------------- | ---------- |
| **Accuracy**          | **81.6 %** |
| **Macro F1-score**    | **0.683**  |
| **Weighted F1-score** | **0.812**  |

### Performance par classe

| Classe            | Precision | Recall | F1-score  | Support (test) |
| ----------------- | --------- | ------ | --------- | -------------- |
| NORMAL            | 0.905     | 0.886  | **0.895** | 1 400          |
| BRADYCARDIE       | 0.905     | 0.973  | **0.937** | 400            |
| TACHYCARDIE       | 0.923     | 0.900  | **0.911** | 200            |
| PACE_VENT         | 0.800     | 0.833  | **0.816** | 48             |
| AFIB              | 0.673     | 0.825  | **0.742** | 240            |
| EXTRASYSTOLES     | 0.651     | 0.792  | **0.714** | 240            |
| BAV1              | 0.548     | 0.458  | **0.499** | 299            |
| PACE_AURIC        | 0.667     | 0.357  | **0.465** | 28             |
| ARYTHMIE_SINUSALE | 0.480     | 0.393  | **0.432** | 61             |
| FLUTTER           | 0.727     | 0.296  | **0.421** | 54             |

### Analyse des résultats

**Points forts** (F1 > 0.8) :

- **Bradycardie (0.937)** : La fréquence cardiaque est un signal très clair
- **Tachycardie (0.911)** : Idem, HR élevé est sans ambiguïté
- **Normal (0.895)** : Classe la plus représentée, bien apprise
- **Pacemaker ventriculaire (0.816)** : Spikes réguliers au signal → pattern distinctif

**Points faibles** (F1 < 0.5) :

- **Flutter (0.421)** : Seulement 54 exemples de test, confusion avec AFIB et NORMAL
- **Arythmie sinusale (0.432)** : Classe subtile, peu d'exemples (61 test)
- **Pacemaker auriculaire (0.465)** : Seulement 28 exemples de test, trop rare

> Les classes les moins performantes sont celles avec le moins d'exemples d'entraînement. C'est un problème classique en ML médical.

---

## 5. Moteur de règles cliniques (8 règles)

En plus de la prédiction XGBoost, un système de **règles déterministes** analyse le signal et émet des alertes. Ces règles sont inspirées des seuils cliniques standards.

| #   | Règle                    | Condition                                                           | Sévérité                                 |
| --- | ------------------------ | ------------------------------------------------------------------- | ---------------------------------------- |
| 1   | **Tachycardie sinusale** | FC > 100 bpm                                                        | ⚠️ Warning (< 150) / 🔴 Critical (≥ 150) |
| 2   | **Bradycardie sinusale** | FC < 60 bpm                                                         | ⚠️ Warning (> 40) / 🔴 Critical (≤ 40)   |
| 3   | **Arythmie sinusale**    | CV des RR > 10 %                                                    | ℹ️ Info                                  |
| 4   | **Suspicion de FA**      | RMSSD > 50 ms ET CV_RR > 15 %                                       | ⚠️ Warning                               |
| 5   | **Suspicion de Flutter** | FC entre 130–170 bpm ET RR réguliers (CV < 5 %)                     | ⚠️ Warning                               |
| 6   | **BAV 1er degré**        | Intervalle PR > 200 ms (exclu si FA suspectée)                      | ⚠️ Warning                               |
| 7   | **Extrasystoles**        | Battements prématurés (RR < 80 % du RR médian)                      | ℹ️ Info (< 3) / ⚠️ Warning (≥ 3)         |
| 8   | **Pacemaker**            | Spikes de stimulation réguliers (slew-rate > Q75+10×IQR, CV < 20 %) | ℹ️ Info                                  |

### Fonctions d'analyse du signal

- **`_measure_pr_interval()`** : Détecte l'onde P (pic positif) dans une fenêtre 120–350 ms avant chaque pic R. Retourne la médiane des intervalles PR mesurés.
- **`_measure_qrs_duration()`** : Mesure la largeur du complexe QRS autour de chaque pic R (seuil 15 % de l'amplitude R). Retourne la médiane des durées.
- **`_detect_premature_beats()`** : Battement prématuré si RR précédent < 80 % du RR médian.
- **`_detect_pacemaker_spikes()`** : Détecte les déflexions à slew-rate extrême (> Q75 + 10×IQR), regroupe les spikes consécutifs, vérifie leur régularité (CV < 20 %).

> **Intérêt du double système ML + Règles** : Le XGBoost peut se tromper sur des cas subtils. Les règles cliniques apportent une couche de vérification indépendante basée sur des seuils médicaux reconnus. Les deux systèmes sont complémentaires.

---

## 6. Interface utilisateur (Gradio)

L'onglet "💓 Analyse ECG" dans l'interface Gradio permet :

1. **Upload** : L'utilisateur charge un fichier CSV ECG 12 dérivations
2. **Diagnostic** : Présenté en phrase naturelle :
   > _"Le modèle identifie une **Fibrillation auriculaire** avec un niveau de confiance de 🟢 **92.6 %**"_
   > _"Diagnostics différentiels : **Arythmie sinusale** (3.2 %) · **Normal** (2.1 %)"_
3. **Tracé ECG graphique** : Visualisation de 6 dérivations (I, II, III, aVR, V1, V5) sur 2.5 secondes
4. **Métriques cliniques** :
   - **HR** (fréquence cardiaque en bpm)
   - **RR** (intervalle RR moyen en ms)
   - **PR** (intervalle PR en ms — conduction auriculo-ventriculaire)
   - **QRS** (durée du complexe QRS en ms — dépolarisation ventriculaire)
   - **RMSSD** (variabilité des intervalles RR en ms — marqueur du système nerveux autonome)
   - **LF/HF ratio** (balance sympatho-vagale, analyser via FFT)
5. **Alertes cliniques** : Tableau des règles déclenchées avec sévérité et détail

---

## 7. Architecture des fichiers

```
training/
├── sample_ecg.py        → Échantillonnage stratifié (32k → 14.8k)
├── preprocess_ecg.py    → Nettoyage + extraction 484 features
└── train_ecg.py         → Entraînement XGBoost + évaluation

module2/
└── ecg_predictor.py     → Prédiction + règles cliniques + tracé ECG

models/
├── ecg_xgboost.pkl      → Modèle XGBoost sérialisé
├── label_encoder.pkl    → Encodeur des labels (10 classes)
└── model_meta.json      → Métadonnées (accuracy, F1, classes, etc.)

evaluation/
├── classification_report.txt   → Rapport precision/recall/F1
├── confusion_matrix.png        → Matrice de confusion
└── feature_importance.png      → Top 30 features importantes

data/
├── ECG AHP/data_ahp_ecg/data/  → 32 536 fichiers CSV bruts
├── samples/sample_meta.csv      → Échantillon stratifié (14 846)
└── processed_sample/
    ├── features.parquet         → Dataset de features
    └── feature_names.txt        → Liste ordonnée des 484 features
```

---

## 8. Technologies utilisées

| Composant         | Technologie                                         |
| ----------------- | --------------------------------------------------- |
| Langage           | Python 3.13                                         |
| ML                | XGBoost (gradient boosting)                         |
| Traitement signal | SciPy (filtres Butterworth, FFT, détection de pics) |
| Data              | Pandas, NumPy                                       |
| Parallélisation   | Joblib (multi-cœurs)                                |
| Visualisation     | Matplotlib, Seaborn                                 |
| Interface         | Gradio                                              |
| Sérialisation     | Joblib (pickle optimisé)                            |

---

## 9. Choix techniques justifiés

### Pourquoi XGBoost et pas un réseau de neurones (CNN) ?

- Les ECG sont transformés en 484 features tabulaires → XGBoost excelle sur les données tabulaires
- Entraînement rapide (< 5 min vs heures pour un CNN)
- Interprétabilité : on peut voir les features les plus importantes (feature_importance)
- Fonctionne bien même avec des classes déséquilibrées grâce aux sample_weights

### Pourquoi 484 features et pas le signal brut ?

- Le signal brut = 5 000 × 12 = 60 000 valeurs par patient → trop lourd pour XGBoost
- Le feature engineering permet de capturer l'information médicale pertinente (HR, HRV, spectre) dans un vecteur compact
- Les features ont une signification clinique interprétable

### Pourquoi un moteur de règles EN PLUS du ML ?

- Le ML peut se tromper (81.6 % d'accuracy ≠ 100 %)
- Les règles cliniques sont des seuils reconnus médicalement (FC > 100 = tachycardie, PR > 200 ms = BAV1)
- Double vérification → augmente la fiabilité globale
- Les règles détectent des choses que le ML ne capture pas bien (ex : spikes de pacemaker)

### Pourquoi early stopping ?

- Évite le sur-apprentissage (overfitting) : le modèle s'arrête à 395 itérations au lieu de 500
- Le critère est la log-loss sur le jeu de test : si elle ne s'améliore plus pendant 30 itérations → stop

---

## 10. Glossaire des termes médicaux ECG

| Terme                               | Explication                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| **Rythme sinusal normal**           | Rythme cardiaque normal, initié par le nœud sinusal                                      |
| **Fibrillation auriculaire (AFIB)** | Activité électrique chaotique des oreillettes, rythme irrégulier                         |
| **Flutter auriculaire**             | Activation rapide et régulière des oreillettes (~300/min), souvent 2:1                   |
| **Tachycardie sinusale**            | Rythme sinusal accéléré > 100 bpm                                                        |
| **Bradycardie sinusale**            | Rythme sinusal ralenti < 60 bpm                                                          |
| **BAV1**                            | Bloc auriculo-ventriculaire du 1er degré : retard de conduction (PR > 200 ms)            |
| **Extrasystoles**                   | Battements prématurés d'origine auriculaire ou ventriculaire                             |
| **Pacemaker**                       | Stimulateur cardiaque artificiel, visible par des spikes électriques                     |
| **Arythmie sinusale**               | Variation physiologique du rythme sinusal (souvent bénigne)                              |
| **HR**                              | Heart Rate — fréquence cardiaque (bpm)                                                   |
| **RR**                              | Intervalle entre deux pics R consécutifs (ms)                                            |
| **PR**                              | Intervalle entre l'onde P et le pic R — conduction AV (ms)                               |
| **QRS**                             | Durée du complexe de dépolarisation ventriculaire (ms), normal : 80–120 ms               |
| **RMSSD**                           | Root Mean Square of Successive Differences — marqueur de variabilité RR                  |
| **LF/HF ratio**                     | Ratio des basses / hautes fréquences du spectre RR → balance sympathique/parasympathique |
| **FFT**                             | Fast Fourier Transform — décompose le signal en fréquences                               |
| **HRV**                             | Heart Rate Variability — variabilité de la fréquence cardiaque                           |

---

## 11. Résumé en une phrase

> Le Module ECG prend un fichier CSV brut de 12 dérivations, nettoie le signal (filtrage Butterworth + notch 50 Hz), extrait 484 features (temporelles + FFT + inter-dérivations), prédit la pathologie parmi 10 classes via XGBoost (accuracy 81.6 %, F1 0.683), puis complète le diagnostic par 8 règles cliniques déterministes et affiche le tracé ECG avec les métriques cliniques (HR, RR, PR, QRS, RMSSD, LF/HF).
