"""
training/preprocess_ecg.py — Pipeline de prétraitement ECG 12 dérivations
==========================================================================
Pour chaque fichier patient (5 000 lignes × 12 dérivations @ 500 Hz) :
  1. Nettoyage  — suppression de la dérive, filtre passe-bande, détection d'artefacts
  2. Feature engineering temporel  — statistiques, morphologie, HRV basique
  3. Transformée de Fourier  — puissance par bande, entropie spectrale, fréquence dominante

Sortie : data/processed/features.parquet  (1 ligne = 1 patient, N colonnes de features)
         + data/processed/feature_names.txt

Usage :
    python training/preprocess_ecg.py
    python training/preprocess_ecg.py --data_dir "data/ECG AHP/data_ahp_ecg/data"
                                       --out_dir data/processed
                                       --n_jobs 4  (parallélisation)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed, Memory
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

FS = 500          # Fréquence d'échantillonnage (Hz)
DURATION = 10     # Secondes d'enregistrement
N_SAMPLES = 5000  # FS × DURATION

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Bandes fréquentielles ECG
FREQ_BANDS = {
    "vlf":     (0.003, 0.04),   # Très basses fréquences (dérive baseline résiduelle)
    "lf":      (0.04,  0.15),   # Basses fréquences
    "hf":      (0.15,  0.4),    # Hautes fréquences (respiration)
    "cardiac": (0.4,   10.0),   # Fréquences cardiaques (24–600 bpm)
    "broad":   (0.5,   40.0),   # Spectre large ECG utile
    "hi_noise":(40.0, 150.0),   # Bruit haute fréquence
}

CLIP_QUANTILE = 0.001  # Clip les valeurs extrêmes (0.1 % et 99.9 %)


# ═══════════════════════════════════════════════════════════════════════════
#  1. NETTOYAGE DU SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

def _design_filters(fs: int = FS):
    """Pré-calcule les filtres Butterworth (appelé une seule fois)."""
    nyq = fs / 2.0
    # Passe-haut 0.5 Hz  (suppression dérive baseline)
    hp = signal.butter(4, 0.5 / nyq, btype="high", output="sos")
    # Passe-bas 150 Hz  (suppression bruit haute fréquence)
    lp = signal.butter(4, min(150.0, nyq - 1) / nyq, btype="low", output="sos")
    # Coupe-bande 50 Hz ± 1 Hz  (interférence secteur Europe)
    notch_b, notch_a = signal.iirnotch(50.0 / nyq, Q=30)
    return hp, lp, (notch_b, notch_a)


# Filtres calculés une seule fois au chargement du module
_FILTERS = _design_filters(FS)


def clean_signal(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Nettoie un signal ECG mono-dérivation.

    Étapes :
    --------
    1. Clip des artefacts extrêmes (quantile 0.1 % / 99.9 %)
    2. Filtre passe-haut 0.5 Hz (supprime dérive baseline)
    3. Filtre coupe-bande 50 Hz (supprime interférence réseau)
    4. Filtre passe-bas 150 Hz (supprime bruit HF)
    5. Normalisation Z-score (μ=0, σ=1)

    Paramètres
    ----------
    sig : np.ndarray, shape (N,)
    fs  : int, fréquence d'échantillonnage (défaut 500 Hz)

    Retour
    ------
    sig_clean : np.ndarray, shape (N,) — signal nettoyé, normalisé
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size == 0:
        return np.zeros(N_SAMPLES, dtype=np.float64)

    if not np.isfinite(sig).all():
        series = pd.Series(sig)
        sig = (
            series.interpolate(limit_direction="both")
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    if np.nanstd(sig) < 1e-12:
        return np.zeros_like(sig, dtype=np.float64)

    # 1. Clip artefacts
    lo = np.nanquantile(sig, CLIP_QUANTILE)
    hi = np.nanquantile(sig, 1.0 - CLIP_QUANTILE)
    sig = np.clip(sig, lo, hi)

    # 2. Passe-haut
    hp_sos = _FILTERS[0]
    sig = signal.sosfiltfilt(hp_sos, sig)

    # 3. Coupe-bande 50 Hz
    notch_b, notch_a = _FILTERS[2]
    sig = signal.filtfilt(notch_b, notch_a, sig)

    # 4. Passe-bas
    lp_sos = _FILTERS[1]
    sig = signal.sosfiltfilt(lp_sos, sig)

    # 5. Normalisation Z-score
    std = sig.std()
    if std > 1e-8:
        sig = (sig - sig.mean()) / std
    else:
        sig = sig - sig.mean()

    return sig


def clean_ecg_df(df: pd.DataFrame) -> np.ndarray:
    """
    Nettoie toutes les dérivations d'un DataFrame patient.

    Retour
    ------
    cleaned : np.ndarray, shape (N_SAMPLES, 12)
    """
    # Supprimer les espaces dans les noms de colonnes + colonnes vides
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df[[c for c in df.columns if c in LEADS]].reindex(columns=LEADS)
    for lead in LEADS:
        df[lead] = pd.to_numeric(df[lead], errors="coerce")
    df = df.interpolate(limit_direction="both").fillna(0.0)

    # Tronquer ou compléter à N_SAMPLES lignes
    n = len(df)
    if n < N_SAMPLES:
        pad = pd.DataFrame(np.zeros((N_SAMPLES - n, 12)), columns=LEADS)
        df = pd.concat([df, pad], ignore_index=True)
    elif n > N_SAMPLES:
        df = df.iloc[:N_SAMPLES]

    cleaned = np.zeros((N_SAMPLES, 12))
    for i, lead in enumerate(LEADS):
        cleaned[:, i] = clean_signal(df[lead].values)

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING TEMPOREL
# ═══════════════════════════════════════════════════════════════════════════

def _detect_r_peaks_simple(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Détection simple des pics R par recherche de maxima locaux.
    Utilise la dérivée et un seuil adaptatif.

    Retour : indices des pics R (ou tableau vide)
    """
    # Distance minimale entre deux R-peaks (150 ms → 75 Hz)
    min_dist = int(0.15 * fs)
    threshold = 0.5 * sig.max() if sig.max() > 0 else 0.5

    peaks, _ = signal.find_peaks(
        sig,
        height=threshold,
        distance=min_dist,
        prominence=0.3,
    )
    return peaks


def _hrv_features(r_peaks: np.ndarray, fs: int = FS) -> dict:
    """
    Calcule les features de variabilité cardiaque (HRV) à partir des R-peaks.

    Features calculées :
    - heart_rate_mean  : fréquence cardiaque moyenne (bpm)
    - rr_mean          : intervalle RR moyen (ms)
    - rr_std           : écart-type des RR → SDNN (ms)
    - rr_cv            : coefficient de variation des RR
    - rmssd            : racine carrée de la moyenne des différences successives au carré
    - pnn50            : proportion de différences successives > 50 ms
    - n_beats          : nombre de battements détectés
    """
    out = {
        "heart_rate_mean": np.nan,
        "rr_mean": np.nan,
        "rr_std": np.nan,
        "rr_cv": np.nan,
        "rmssd": np.nan,
        "pnn50": np.nan,
        "n_beats": len(r_peaks),
    }

    if len(r_peaks) < 3:
        return out

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples / fs * 1000.0  # en millisecondes

    out["heart_rate_mean"] = 60_000.0 / rr_ms.mean() if rr_ms.mean() > 0 else np.nan
    out["rr_mean"] = rr_ms.mean()
    out["rr_std"] = rr_ms.std()
    out["rr_cv"] = rr_ms.std() / rr_ms.mean() if rr_ms.mean() > 0 else np.nan

    diff_rr = np.diff(rr_ms)
    out["rmssd"] = np.sqrt(np.mean(diff_rr ** 2))
    out["pnn50"] = np.mean(np.abs(diff_rr) > 50)

    return out


def extract_temporal_features(sig: np.ndarray, lead_name: str, fs: int = FS) -> dict:
    """
    Extrait les features temporelles pour une dérivation.

    Features (préfixées par lead_name) :
    -------------------------------------
    Statistiques de base :
        mean, std, min, max, median, q25, q75, iqr
        skewness, kurtosis, rms, peak_to_peak, energy

    Morphologie :
        zero_crossings      : taux de passage par zéro
        hjorth_activity     : variance du signal (Hjorth)
        hjorth_mobility     : mobilité de Hjorth
        hjorth_complexity   : complexité de Hjorth
        mean_abs_deriv1     : dérivée 1ère (vitesse moyenne)
        mean_abs_deriv2     : dérivée 2ème (accélération moyenne)

    HRV (sur dérivation II uniquement par défaut, calculé pour toutes) :
        heart_rate_mean, rr_mean, rr_std, rr_cv, rmssd, pnn50, n_beats
    """
    prefix = f"{lead_name}_"
    out = {}

    # ── Statistiques de base ──────────────────────────────────────────────
    out[f"{prefix}mean"] = sig.mean()
    out[f"{prefix}std"] = sig.std()
    out[f"{prefix}min"] = sig.min()
    out[f"{prefix}max"] = sig.max()
    out[f"{prefix}median"] = np.median(sig)
    q25, q75 = np.percentile(sig, [25, 75])
    out[f"{prefix}q25"] = q25
    out[f"{prefix}q75"] = q75
    out[f"{prefix}iqr"] = q75 - q25
    out[f"{prefix}skewness"] = stats.skew(sig)
    out[f"{prefix}kurtosis"] = stats.kurtosis(sig)
    out[f"{prefix}rms"] = np.sqrt(np.mean(sig ** 2))
    out[f"{prefix}peak_to_peak"] = sig.max() - sig.min()
    out[f"{prefix}energy"] = np.sum(sig ** 2) / len(sig)

    # ── Morphologie ───────────────────────────────────────────────────────
    # Passage par zéro
    out[f"{prefix}zero_crossings"] = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)

    # Paramètres de Hjorth
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var_sig = np.var(sig) + 1e-10
    var_d1 = np.var(d1) + 1e-10
    var_d2 = np.var(d2) + 1e-10
    out[f"{prefix}hjorth_activity"] = var_sig
    out[f"{prefix}hjorth_mobility"] = np.sqrt(var_d1 / var_sig)
    mob_d1 = np.sqrt(var_d1 / var_sig)
    mob_d2 = np.sqrt(var_d2 / var_d1)
    out[f"{prefix}hjorth_complexity"] = mob_d2 / (mob_d1 + 1e-10)

    # Dérivées
    out[f"{prefix}mean_abs_deriv1"] = np.mean(np.abs(d1))
    out[f"{prefix}mean_abs_deriv2"] = np.mean(np.abs(d2))

    # ── HRV ───────────────────────────────────────────────────────────────
    r_peaks = _detect_r_peaks_simple(sig, fs)
    hrv = _hrv_features(r_peaks, fs)
    for k, v in hrv.items():
        out[f"{prefix}{k}"] = v

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  3. TRANSFORMÉE DE FOURIER
# ═══════════════════════════════════════════════════════════════════════════

def extract_fft_features(sig: np.ndarray, lead_name: str, fs: int = FS) -> dict:
    """
    Extrait les features fréquentielles par FFT pour une dérivation.

    Features (préfixées par lead_name) :
    -------------------------------------
    Puissance par bande :
        pwr_vlf, pwr_lf, pwr_hf, pwr_cardiac, pwr_broad, pwr_hi_noise

    Ratios de puissance :
        ratio_lf_hf        : ratio LF/HF (balance sympatho-vagale)
        ratio_cardiac_total: part de la puissance cardiaque sur le total

    Fréquence dominante :
        dominant_freq      : fréquence du pic d'énergie maximal (Hz)
        dominant_freq_bpm  : fréquence dominante convertie en bpm

    Spectral features :
        spectral_centroid  : centroïde spectral (Hz)
        spectral_entropy   : entropie de la distribution d'énergie spectrale
        spectral_rolloff   : fréquence sous laquelle 85 % de l'énergie est contenue (Hz)
        spectral_spread    : étalement spectral autour du centroïde (Hz)
    """
    prefix = f"{lead_name}_fft_"
    out = {}

    n = len(sig)
    # Fenêtrage de Hann pour réduire les fuites spectrales
    window = np.hanning(n)
    sig_win = sig * window

    # FFT unilatérale
    fft_vals = np.fft.rfft(sig_win)
    fft_mag = np.abs(fft_vals)
    fft_power = fft_mag ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)  # Hz

    total_power = fft_power.sum() + 1e-10

    # ── Puissance par bande ───────────────────────────────────────────────
    for band_name, (f_lo, f_hi) in FREQ_BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        out[f"{prefix}pwr_{band_name}"] = fft_power[mask].sum() / total_power

    # ── Ratios ────────────────────────────────────────────────────────────
    pwr_lf = fft_power[(freqs >= 0.04) & (freqs < 0.15)].sum()
    pwr_hf = fft_power[(freqs >= 0.15) & (freqs < 0.4)].sum()
    out[f"{prefix}ratio_lf_hf"] = pwr_lf / (pwr_hf + 1e-10)

    pwr_cardiac = fft_power[(freqs >= 0.4) & (freqs < 10.0)].sum()
    out[f"{prefix}ratio_cardiac_total"] = pwr_cardiac / total_power

    # ── Fréquence dominante ───────────────────────────────────────────────
    # On cherche le pic dans la bande cardiaque (0.4–10 Hz)
    cardiac_mask = (freqs >= 0.4) & (freqs < 10.0)
    if cardiac_mask.any():
        dom_idx_local = np.argmax(fft_power[cardiac_mask])
        dom_freq = freqs[cardiac_mask][dom_idx_local]
    else:
        dom_freq = 0.0
    out[f"{prefix}dominant_freq"] = dom_freq
    out[f"{prefix}dominant_freq_bpm"] = dom_freq * 60.0

    # ── Features spectrales ───────────────────────────────────────────────
    # Centroïde spectral
    if total_power > 1e-8:
        centroid = np.sum(freqs * fft_power) / total_power
    else:
        centroid = 0.0
    out[f"{prefix}spectral_centroid"] = centroid

    # Entropie spectrale (entropie de la distribution d'énergie normalisée)
    psd_norm = fft_power / total_power
    psd_norm = np.clip(psd_norm, 1e-12, None)
    out[f"{prefix}spectral_entropy"] = -np.sum(psd_norm * np.log2(psd_norm))

    # Spread spectral (std autour du centroïde)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_power) / total_power)
    out[f"{prefix}spectral_spread"] = spread

    # Rolloff 85 %
    cumsum = np.cumsum(fft_power)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    out[f"{prefix}spectral_rolloff"] = freqs[min(rolloff_idx, len(freqs) - 1)]

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  4. PIPELINE COMPLET PAR PATIENT
# ═══════════════════════════════════════════════════════════════════════════

def process_one_file(filepath: str):
    """
    Pipeline complet pour un fichier CSV patient.

    Étapes :
    --------
    1. Lecture du CSV
    2. Nettoyage des 12 dérivations
    3. Features temporelles par dérivation
    4. Features FFT par dérivation
    5. Features inter-dérivations (corrélation, énergie relative)

    Retour
    ------
    dict avec toutes les features + clé 'filename', ou None en cas d'erreur.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
    except Exception as e:
        return None

    try:
        cleaned = clean_ecg_df(df)    # (5000, 12)
    except Exception:
        return None

    features = {"filename": os.path.basename(filepath)}

    # ── Features par dérivation ───────────────────────────────────────────
    for i, lead in enumerate(LEADS):
        sig = cleaned[:, i]

        try:
            temp_feats = extract_temporal_features(sig, lead, FS)
            features.update(temp_feats)
        except Exception:
            pass

        try:
            fft_feats = extract_fft_features(sig, lead, FS)
            features.update(fft_feats)
        except Exception:
            pass

    # ── Features inter-dérivations ────────────────────────────────────────
    try:
        # Matrice de corrélation entre dérivations (limbs vs. précordiales)
        lead_idx = {l: i for i, l in enumerate(LEADS)}

        # Corrélation II–III (axe électrique)
        features["xcorr_II_III"] = float(np.corrcoef(
            cleaned[:, lead_idx["II"]], cleaned[:, lead_idx["III"]]
        )[0, 1])

        # Corrélation V1–V6 (axe transversal)
        features["xcorr_V1_V6"] = float(np.corrcoef(
            cleaned[:, lead_idx["V1"]], cleaned[:, lead_idx["V6"]]
        )[0, 1])

        # Énergie relative précordiales vs. dérivations des membres
        limb_energy = sum(np.sum(cleaned[:, j] ** 2)
                          for j, l in enumerate(LEADS) if l in ("I", "II", "III", "aVR", "aVL", "aVF"))
        precord_energy = sum(np.sum(cleaned[:, j] ** 2)
                             for j, l in enumerate(LEADS) if l.startswith("V"))
        total_energy = limb_energy + precord_energy + 1e-10
        features["ratio_precord_limb_energy"] = precord_energy / (limb_energy + 1e-10)
        features["total_signal_energy"] = total_energy / N_SAMPLES

    except Exception:
        pass

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  5. SCRIPT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline prétraitement ECG 12 dérivations")
    parser.add_argument(
        "--data_dir",
        default=r"data/ECG AHP/data_ahp_ecg/data",
        help="Dossier contenant les CSV patients (défaut : data/ECG AHP/data_ahp_ecg/data)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed",
        help="Dossier de sortie du dataset de features (défaut : data/processed)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Nombre de cœurs CPU pour la parallélisation (défaut : -1 = tous)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Si > 0, traite seulement N fichiers (pour test rapide)",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="CSV optionnel avec colonnes [filename, label] à joindre au résultat",
    )
    parser.add_argument(
        "--sample_file",
        default="",
        help=(
            "CSV généré par sample_ecg.py (colonnes full_ecg_path, label). "
            "Si fourni, remplace --data_dir et attache automatiquement les labels."
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collecte des fichiers ─────────────────────────────────────────────
    sample_label_map = {}  # full_path → label (rempli si --sample_file est fourni)

    if args.sample_file and os.path.exists(args.sample_file):
        # Mode sample_file : utilise les chemins complets du CSV
        print(f"Mode sample_file : lecture de {args.sample_file}")
        sample_meta = pd.read_csv(args.sample_file)
        if "full_ecg_path" not in sample_meta.columns:
            print("❌ Le CSV doit contenir une colonne 'full_ecg_path'", file=sys.stderr)
            sys.exit(1)
        all_files = sample_meta["full_ecg_path"].tolist()
        if "label" in sample_meta.columns:
            sample_label_map = dict(
                zip(sample_meta["full_ecg_path"], sample_meta["label"])
            )
        # Filtre les fichiers manquants
        missing = [f for f in all_files if not os.path.exists(f)]
        if missing:
            print(f"⚠  {len(missing)} fichiers introuvables — ignorés")
            all_files = [f for f in all_files if os.path.exists(f)]
    else:
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        )
        if not all_files:
            print(f"❌ Aucun CSV trouvé dans {data_dir}")
            sys.exit(1)

    if args.sample > 0 and not args.sample_file:
        all_files = all_files[: args.sample]
        print(f"⚠  Mode test : {len(all_files)} fichiers uniquement")
    elif args.sample > 0 and args.sample_file:
        all_files = all_files[: args.sample]
        print(f"⚠  Mode test (sample_file) : {len(all_files)} fichiers uniquement")

    print(f"\n{'═' * 60}")
    print(f"  Pipeline ECG — {len(all_files):,} patients")
    print(f"  FS={FS} Hz  |  {N_SAMPLES} pts  |  {len(LEADS)} dérivations")
    print(f"  Parallélisation : {args.n_jobs} cœur(s)")
    print(f"{'═' * 60}\n")

    # ── Traitement parallèle ──────────────────────────────────────────────
    print("1. Nettoyage + Feature engineering + FFT...")
    results = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(process_one_file)(fp) for fp in tqdm(all_files, unit="patient")
    )

    # Filtrage des erreurs
    results = [r for r in results if r is not None]
    n_errors = len(all_files) - len(results)
    if n_errors:
        print(f"   ⚠  {n_errors} fichiers ignorés (erreurs de lecture/traitement)")

    print(f"\n2. Construction du DataFrame...")
    df_features = pd.DataFrame(results)
    print(f"   → {len(df_features):,} patients × {df_features.shape[1]} colonnes")

    # ── Vérification NaN ─────────────────────────────────────────────────
    nan_counts = df_features.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols):
        print(f"   ⚠  {len(nan_cols)} colonnes avec NaN (ex : HRV quand < 3 R-peaks)")
        df_features.fillna(df_features.median(numeric_only=True), inplace=True)
        print(f"   → NaN remplacés par la médiane de la colonne")

    # ── Jointure labels — depuis sample_file ou --labels ────────────────
    if sample_label_map:
        # Construit un mapping basename → label pour un accès rapide O(1)
        basename_label_map = {
            os.path.basename(k): v for k, v in sample_label_map.items()
        }
        df_features["label"] = df_features["filename"].map(basename_label_map)
        n_labeled = df_features["label"].notna().sum()
        print(f"\n3. Labels attachés depuis sample_file : {n_labeled:,} / {len(df_features):,}")
    elif args.labels and os.path.exists(args.labels):
        print(f"\n3. Jointure avec les labels ({args.labels})...")
        labels_df = pd.read_csv(args.labels)
        df_features = df_features.merge(labels_df, on="filename", how="left")
        n_labeled = df_features["label"].notna().sum() if "label" in df_features.columns else 0
        print(f"   → {n_labeled:,} patients avec label")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    out_path = out_dir / "features.parquet"
    df_features.to_parquet(out_path, index=False)
    print(f"\n✅ Dataset sauvegardé : {out_path}")
    print(f"   Taille : {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Sauvegarde de la liste des features pour référence
    feature_names = [c for c in df_features.columns if c not in ("filename", "label")]
    feat_names_path = out_dir / "feature_names.txt"
    feat_names_path.write_text("\n".join(feature_names))
    print(f"   {len(feature_names)} features → {feat_names_path}")

    # ── Résumé des features ───────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("   RÉSUMÉ DES FEATURES PAR DÉRIVATION :")
    temporal_per_lead = len([
        k for k in results[0].keys()
        if k.startswith("I_") and not k.startswith("I_fft")
    ]) if results else 0
    fft_per_lead = len([
        k for k in results[0].keys() if k.startswith("I_fft")
    ]) if results else 0

    print(f"   • Temporelles × dérivation : {temporal_per_lead}")
    print(f"   • FFT × dérivation         : {fft_per_lead}")
    print(f"   • Inter-dérivations        : 3")
    print(f"   • Total                    : {len(feature_names)}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
