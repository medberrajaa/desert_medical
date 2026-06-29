"""training/train_ecg.py — Entraînement XGBoost sur features ECG
================================================================
Lit data/processed_sample/features.parquet (produit par preprocess_ecg.py),
entraîne un modèle XGBoost multiclasse stratifié et sauvegarde le modèle.

Workflow complet :
    1) python training/sample_ecg.py
    2) python training/preprocess_ecg.py --sample_file data/samples/sample_meta.csv
                                          --n_jobs 4
    3) python training/train_ecg.py

Usage :
    python training/train_ecg.py
    python training/train_ecg.py --n_estimators 500 --seed 42
"""

import argparse
import os
import sys

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Sous-échantillonnage des classes dominantes (appliqué sur le dataset complet)
UNDERSAMPLE_CAPS = {
    "NORMAL":      5000,
    "BRADYCARDIE": 2000,
}

# SMOTE — sur-échantillonnage des classes minoritaires (appliqué sur train uniquement)
# Clés = noms de classe, valeurs = taille cible après SMOTE
SMOTE_TARGETS = {
    "FLUTTER":           500,
    "PACE_AURIC":        500,
    "PACE_VENT":         500,
    "ARYTHMIE_SINUSALE": 500,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_features(parquet_path: str):
    """
    Charge le dataset de features depuis le parquet.
    Retourne (X, y_raw, feature_names).
    """
    if not os.path.exists(parquet_path):
        print(f"❌ Fichier introuvable : {parquet_path}", file=sys.stderr)
        print("   → Lancez d'abord: python training/preprocess_ecg.py \\")
        print("       --sample_file data/samples/sample_meta.csv --n_jobs 4")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"   Parquet chargé : {df.shape[0]:,} patients × {df.shape[1]} colonnes")

    if "label" not in df.columns:
        print("❌ Colonne 'label' absente du parquet.", file=sys.stderr)
        print("   → Utilisez --sample_file lors du prétraitement pour attacher les labels.")
        sys.exit(1)

    # Supprime les patients sans label
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if len(df) < before:
        print(f"   ⚠  {before - len(df)} patients sans label — ignorés")

    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    return X, y, feature_cols


def plot_confusion_matrix(cm, class_names, out_path):
    """Sauvegarde la matrice de confusion normalisée."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Absolue
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=axes[0],
    )
    axes[0].set_title("Matrice de confusion (valeurs absolues)")
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")
    axes[0].tick_params(axis="x", rotation=45)

    # Normalisée
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", ax=axes[1], vmin=0, vmax=1,
    )
    axes[1].set_title("Matrice de confusion (normalisée par classe réelle)")
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Réel")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Classification ECG multi-classes", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Matrice sauvegardée : {out_path}")


def plot_feature_importance(model, feature_names, out_path, top_n=30, title=""):
    """Sauvegarde un barplot des top features importantes."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:][::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
    )
    plt.xlabel("Importance (gain moyen)")
    plt.title(f"Top {top_n} features — {title or 'ECG'}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   → Feature importance sauvegardée : {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline d'entraînement
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  Entraînement d'un modèle unique (retourne modèle + métriques)
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, X_test, y_train, y_test, le, n_estimators, seed):
    n_classes = len(le.classes_)
    print(f"\n── XGBoost ({n_estimators} estimateurs max, early stopping 30) ──")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print("   Poids par classe :")
    for cls_idx in np.unique(y_train):
        mask = y_train == cls_idx
        w = sample_weights[mask][0]
        print(f"     {le.classes_[cls_idx]:<22} weight={w:.3f}  (n={mask.sum()})")

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print(f"   → Meilleure itération : {model.best_iteration}")
    return model


def evaluate_model(model, X_test, y_test, le, model_name):
    """Évalue le modèle et retourne (rapport_str, macro_f1, y_pred)."""
    n_classes = len(le.classes_)
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        labels=np.arange(n_classes),
        digits=3,
        zero_division=0,
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc = np.mean(y_pred == y_test)
    print(f"\n{'='*60}")
    print(f"  {model_name} — Résultats")
    print(f"  Accuracy : {acc*100:.1f}%   Macro F1 : {macro_f1:.3f}")
    print('='*60)
    print(report)
    return report, macro_f1, acc, y_pred


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

def train(features_path: str, output_dir: str, n_estimators: int, seed: int):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    # ── 1. Chargement ────────────────────────────────────────────────────
    print("\n1. Chargement des features...")
    X, y_raw, feature_names = load_features(features_path)

    unique, counts = np.unique(y_raw, return_counts=True)
    print(f"\n   Distribution des classes ({len(unique)} classes) :")
    for cls, cnt in sorted(zip(unique, counts), key=lambda t: -t[1]):
        bar = "█" * (cnt // 20)
        print(f"   {cls:<22} {cnt:>5}  {bar}")

    # ── 1b. Sous-échantillonnage des classes dominantes ──────────────────
    rng = np.random.default_rng(seed)
    caps_applied = False
    for cls_name, cap in UNDERSAMPLE_CAPS.items():
        idx_cls = np.where(y_raw == cls_name)[0]
        if len(idx_cls) > cap:
            if not caps_applied:
                print("\n   Sous-échantillonnage :")
                caps_applied = True
            keep = rng.choice(idx_cls, size=cap, replace=False)
            mask = np.ones(len(y_raw), dtype=bool)
            mask[idx_cls] = False
            mask[keep] = True
            X, y_raw = X[mask], y_raw[mask]
            print(f"     {cls_name:<22} {len(idx_cls):>5} → {cap}")
    if caps_applied:
        print(f"   Total après sous-échantillonnage : {len(X):,}")

    # ── 2. Encodage + Split ───────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=seed, stratify=y_enc,
    )
    print(f"\n2. Split stratifié : {len(X_train):,} train  /  {len(X_test):,} test")

    # ── 2b. SMOTE sur classes minoritaires (train uniquement) ─────────────
    smote_targets_enc = {
        le.transform([cls])[0]: target
        for cls, target in SMOTE_TARGETS.items()
        if cls in le.classes_
    }
    # N'appliquer SMOTE qu'aux classes dont le count actuel < cible
    current_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    smote_strategy = {
        cls_enc: target
        for cls_enc, target in smote_targets_enc.items()
        if current_counts.get(cls_enc, 0) < target
    }
    if smote_strategy:
        print("\n   SMOTE sur classes minoritaires :")
        for cls_enc, target in smote_strategy.items():
            cls_name = le.inverse_transform([cls_enc])[0]
            n_before = current_counts.get(cls_enc, 0)
            print(f"     {cls_name:<22} {n_before:>4} → {target}")
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=seed, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   Train après SMOTE : {len(X_train):,} samples")
    else:
        print("   SMOTE : toutes les cibles déjà atteintes, pas d'oversampling.")

    # ── 3. Entraînement XGBoost ──────────────────────────────────────────
    print("\n3. Entraînement XGBoost...")
    model = train_xgboost(X_train, X_test, y_train, y_test, le, n_estimators, seed)
    report, macro_f1, acc, y_pred = evaluate_model(model, X_test, y_test, le, "XGBoost")

    # ── 4. Sauvegarde du rapport ─────────────────────────────────────────
    print("\n4. Sauvegarde des rapports...")
    for fname in ("evaluation/classification_report_xgboost.txt",
                  "evaluation/classification_report.txt"):
        with open(fname, "w", encoding="utf-8") as f:
            f.write("XGBoost — Rapport de classification ECG\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"   → {fname}")

    # ── 5. Graphiques ────────────────────────────────────────────────────
    print("\n5. Génération des graphiques...")
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    plot_confusion_matrix(cm, le.classes_, "evaluation/confusion_matrix.png")
    plot_feature_importance(
        model, feature_names,
        "evaluation/feature_importance.png",
        title="XGBoost",
    )

    # ── 6. Sauvegarde du modèle ──────────────────────────────────────────
    print(f"\n6. Sauvegarde du modèle dans {output_dir}/...")
    model_path = os.path.join(output_dir, "ecg_xgboost.pkl")
    le_path = os.path.join(output_dir, "label_encoder.pkl")
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)

    meta = {
        "model_type": "XGBoost",
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "classes": list(le.classes_),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features_parquet": features_path,
        "best_iteration": model.best_iteration,
    }
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Entraînement terminé !")
    print(f"   Modèle production : {model_path}")
    print(f"   Encodeur          : {le_path}")
    print(f"   Métadonnées       : {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Entrée
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîne XGBoost sur les features ECG"
    )
    parser.add_argument(
        "--features",
        default="data/processed_sample/features.parquet",
        help="Chemin du parquet produit par preprocess_ecg.py",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Dossier de sortie du modèle (défaut: models/)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="Nombre max d'arbres (défaut: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire (défaut: 42)",
    )
    args = parser.parse_args()
    train(args.features, args.output_dir, args.n_estimators, args.seed)
