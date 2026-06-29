"""
module2/ecg_image_reader.py — Extraction de signal ECG depuis une image
========================================================================
Pipeline robuste pour convertir une image ECG (JPG / PNG) en signal
numérique exploitable par le moteur de prédiction XGBoost.

Étapes :
    1. Vérification de résolution (DPI) — avertissement si < 200
    2. Crop intelligent (paysage uniquement ; portrait → segmentation auto)
    3. Prétraitement : niveaux de gris → flou gaussien 5×5
       → seuillage adaptatif (blockSize=51, C=15)
       → suppression grille (lignes H/V morphologiques)
       → masquage des marges texte
    4. Segmentation en bandes (leads) + layout multi-colonnes
    5. Extraction par lead : composantes connexes → médiane colonne
       → filtre épaisseur → correction ligne de base → lissage
    6. Calibration d'échelle (détection du carré 1 mV)
    7. Rééchantillonnage à 500 Hz via scipy.signal.resample

Le résultat est un np.ndarray (N_SAMPLES, 12) compatible avec le pipeline
d'extraction de features existant (training/preprocess_ecg.py).

Requirements: opencv-python, Pillow, scipy
"""

from __future__ import annotations

import logging
import os
import sys

import cv2
import numpy as np
from scipy.signal import resample as scipy_resample

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.preprocess_ecg import LEADS, FS, N_SAMPLES

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────
_N_LEADS = len(LEADS)          # 12
_DEFAULT_DPI = 300
_PAPER_SPEED_MM_S = 25         # Vitesse papier standard ECG
_MV_HEIGHT_MM = 10             # 1 mV = 10 mm en standard

# Facteurs d'échelle physiologiques pour reconstituer les 12 leads
# à partir du Lead II uniquement (approximation population-médiane).
# Basés sur les relations d'Einthoven + morphologie précordiale typique.
# Signe négatif = lead typiquement inversé par rapport à Lead II.
_LEAD_II_SCALES = {
    "I":    0.70,   # Lead I  ~ 70 % de Lead II (relation d'Einthoven I+III=II)
    "II":   1.00,   # Lead II — signal réel
    "III":  0.35,   # Lead III = II - I ≈ 35 % de II
    "aVR": -0.65,   # aVR = -(I+II)/2  → fortement négatif quand II > 0
    "aVL":  0.18,   # aVL = (I-III)/2  → faiblement positif
    "aVF":  0.65,   # aVF = (II+III)/2 → proche de II
    "V1":  -0.22,   # V1  précordial droit → souvent négatif
    "V2":   0.12,   # V2  zone de transition
    "V3":   0.40,   # V3  zone de transition → positif modéré
    "V4":   0.65,   # V4  onde R bien formée
    "V5":   0.62,   # V5  similaire à V4
    "V6":   0.52,   # V6  positive, amplitude réduite
}
_LEAD_SYNTH_NOISE_STD = 0.15   # legacy constant; missing leads are no longer synthesized.


def _normalize_brightness_contrast(gray: np.ndarray) -> np.ndarray:
    """Normalize illumination and enhance ECG trace contrast."""
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _correct_skew(img: np.ndarray) -> np.ndarray:
    """Deskew ECG paper using dominant near-horizontal line angles."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, gray.shape[1] // 5),
        maxLineGap=12,
    )
    if lines is None:
        return img

    angles: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if -8.0 <= angle <= 8.0:
            angles.append(float(angle))

    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return img

    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = (255, 255, 255) if len(img.shape) == 3 else 255
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# ═══════════════════════════════════════════════════════════════════════════
#  1. Resolution check
# ═══════════════════════════════════════════════════════════════════════════

def check_resolution(image_path: str) -> str | None:
    """
    Vérifie la résolution DPI de l'image via les métadonnées Pillow.

    Parameters
    ----------
    image_path : str
        Chemin vers le fichier image.

    Returns
    -------
    str | None
        Message d'avertissement si DPI < 200, None sinon.
    """
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as pil_img:
            info = pil_img.info
            dpi = info.get("dpi")
            if dpi is None:
                jfif_density = info.get("jfif_density")
                if jfif_density is not None:
                    dpi = jfif_density
                else:
                    return None

            if isinstance(dpi, (tuple, list)):
                dpi_val = min(float(dpi[0]), float(dpi[1]))
            else:
                dpi_val = float(dpi)

            if dpi_val < 200:
                return (
                    f"⚠ Résolution insuffisante ({int(dpi_val)} DPI). "
                    "Précision réduite sur les intervalles fins (PR, QT). "
                    "Privilégier 300 DPI minimum ou utiliser le fichier CSV."
                )
    except ImportError:
        logger.warning("Pillow non installé — vérification DPI ignorée.")
    except Exception as e:
        logger.debug(f"Impossible de lire les DPI : {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  2. Lead isolation (crop_lead_II)
# ═══════════════════════════════════════════════════════════════════════════

def crop_lead_II(img: np.ndarray) -> np.ndarray:
    """
    Détecte si l'image contient plusieurs leads (format multi-bandes)
    et, le cas échéant, isole la bande centrale (II / aVL / V2 / V5).

    Le crop n'est appliqué que sur les images **paysage** (largeur > hauteur)
    dont la hauteur dépasse le seuil, car les images portrait sont gérées
    par la segmentation automatique (_segment_leads).

    Parameters
    ----------
    img : np.ndarray
        Image couleur ou niveaux de gris.

    Returns
    -------
    np.ndarray
        Image recadrée (ou inchangée si portrait ou single-lead).
    """
    h, w = img.shape[:2]

    # Pour les images portrait (h >= w), la segmentation gère le découpage
    if h >= w:
        return img

    # Image paysage : crop uniquement si la hauteur indique un multi-leads
    threshold = 2 * (w / 4)
    if h > threshold:
        y_start = h // 3
        y_end = 2 * h // 3
        return img[y_start:y_end, :]
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  3. Preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Pipeline de prétraitement robuste :
        1. Conversion en niveaux de gris (si nécessaire)
        2. Flou gaussien (5×5) pour lisser le bruit
        3. Seuillage adaptatif gaussien (blockSize=51, C=15)
        4. Suppression de la grille ECG (lignes horizontales/verticales)
        5. Fermeture morphologique pour reconnecter les fragments
        6. Masquage des marges texte (6 % gauche/droite)

    Parameters
    ----------
    img : np.ndarray
        Image d'entrée (BGR ou déjà en niveaux de gris).

    Returns
    -------
    np.ndarray
        Image binaire uint8 — pixels de courbe = 255, fond = 0.
    """
    # 1. Niveaux de gris
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Brightness normalization + contrast enhancement + denoising
    gray = _normalize_brightness_contrast(gray)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # 3. Seuillage adaptatif — paramètres optimisés pour ECG papier
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,
        C=15,
    )

    # Si > 60 % blanc → probablement inversé
    if np.mean(binary == 255) > 0.60:
        binary = cv2.bitwise_not(binary)

    # 4. Suppression des lignes de grille
    H, W = binary.shape
    #    a) Lignes horizontales (noyau large et plat)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 15, 3), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kh)
    binary = cv2.subtract(binary, horiz_lines)
    #    b) Lignes verticales (noyau étroit et haut)
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 15, 3)))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kv)
    binary = cv2.subtract(binary, vert_lines)

    # 5. Fermeture morphologique pour reconnecter les fragments de trace
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 6. Masquage des marges gauche/droite (annotations de leads)
    margin = max(1, int(W * 0.06))
    binary[:, :margin] = 0
    binary[:, W - margin:] = 0

    return binary


# ═══════════════════════════════════════════════════════════════════════════
#  4. Signal extraction (median + NaN interpolation)
# ═══════════════════════════════════════════════════════════════════════════

def extract_trace(binary_strip: np.ndarray) -> np.ndarray | None:
    """
    Extrait une trace 1-D depuis une bande binaire (un seul lead).

    Pipeline :
        1. Filtrage par composantes connexes — ne garde que les structures
           larges et étendues (supprime texte, bruit ponctuel).
        2. Extraction colonne par colonne (médiane des pixels actifs),
           avec filtre d'épaisseur (rejette les colonnes > 40 % de la hauteur).
        3. Interpolation NaN via np.interp.
        4. Inversion axe Y + centrage sur zéro.
        5. Correction de ligne de base (moyenne glissante, fenêtre ~50 pts).
        6. Lissage léger (moyenne mobile 5 pts).

    Parameters
    ----------
    binary_strip : np.ndarray
        Image binaire (uint8, 0/255) d'une bande contenant un seul lead.

    Returns
    -------
    np.ndarray | None
        Trace 1-D (longueur = largeur de la bande), centrée sur zéro,
        axe Y inversé (haut = positif). None si trop peu de colonnes valides.
    """
    h, w = binary_strip.shape

    # ── 1. Filtrage par composantes connexes ──────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_strip, connectivity=8
    )
    if n_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()

    cleaned = np.zeros_like(binary_strip)
    for lid in range(1, n_labels):
        area_ok = stats[lid, cv2.CC_STAT_AREA] >= max_area * 0.10
        width_ok = stats[lid, cv2.CC_STAT_WIDTH] > w * 0.10
        if area_ok and width_ok:
            cleaned[labels == lid] = 255

    # ── 2. Extraction colonne par colonne avec filtre d'épaisseur ─────────
    trace = np.full(w, np.nan)
    for col in range(w):
        active = np.where(cleaned[:, col] > 0)[0]
        if len(active) > 0:
            thickness = active[-1] - active[0]
            if thickness < h * 0.40:
                trace[col] = np.median(active)

    # ── 3. Vérification minimum de colonnes valides ───────────────────────
    valid_mask = ~np.isnan(trace)
    n_valid = int(np.sum(valid_mask))
    if n_valid / w < 0.20:
        return None

    if not np.any(valid_mask):
        return None

    # Interpolation des NaN via np.interp sur les indices non-NaN
    x_all = np.arange(w)
    x_valid = x_all[valid_mask]
    y_valid = trace[valid_mask]
    trace = np.interp(x_all, x_valid, y_valid)

    # ── 4. Inverser axe Y + centrer ───────────────────────────────────────
    trace = -trace
    trace = trace - np.mean(trace)

    # ── 5. Correction de ligne de base (moyenne glissante) ────────────────
    _BASELINE_WINDOW = 50
    bwin = max(5, min(_BASELINE_WINDOW, len(trace) // 3))
    baseline = np.convolve(trace, np.ones(bwin) / bwin, mode='same')
    trace = trace - baseline

    # ── 6. Lissage léger (moyenne mobile 5 pts) ──────────────────────────
    trace = np.convolve(trace, np.ones(5) / 5, mode='same')

    return trace


# ═══════════════════════════════════════════════════════════════════════════
#  5. Scale calibration
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_scale(img: np.ndarray) -> tuple[float, bool]:
    """
    Détecte le carré de calibration 1 mV dans les 50 premières colonnes
    et estime le ratio pixels_per_second.

    Le carré 1 mV standard mesure 10 mm de haut (1 mV = 10 mm).
    Avec une vitesse papier de 25 mm/s, on en déduit pixels_per_second.

    Parameters
    ----------
    img : np.ndarray
        Image en niveaux de gris ou couleur (BGR).

    Returns
    -------
    tuple[float, bool]
        (pixels_per_second, calibration_found)
    """
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h_img, w_img = gray.shape
    roi_w = min(50, w_img)
    roi = gray[:, :roi_w]

    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cw * ch
        min_area = (h_img * 0.02) ** 2
        max_area = (h_img * 0.25) ** 2

        if 0.5 < aspect < 2.0 and min_area < area < max_area:
            pixels_per_mm = ch / _MV_HEIGHT_MM
            pixels_per_second = pixels_per_mm * _PAPER_SPEED_MM_S
            return pixels_per_second, True

    # Fallback : assumer 300 DPI et 25 mm/s
    logger.warning(
        "Carré de calibration 1 mV non détecté. "
        "Utilisation des valeurs par défaut (300 DPI, 25 mm/s)."
    )
    pixels_per_mm = _DEFAULT_DPI / 25.4  # 300 DPI → ~11.81 px/mm
    pixels_per_second = pixels_per_mm * _PAPER_SPEED_MM_S
    return pixels_per_second, False


# ═══════════════════════════════════════════════════════════════════════════
#  Fonctions internes de segmentation (conservées pour multi-lead)
# ═══════════════════════════════════════════════════════════════════════════

def _segment_leads(binary: np.ndarray) -> list[np.ndarray]:
    """Segmente l'image binaire en bandes horizontales (un lead par bande)."""
    h, w = binary.shape
    h_proj = np.sum(binary, axis=1)
    threshold = 0.05 * w * 255
    active = h_proj > threshold

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for row in range(h):
        if active[row] and not in_band:
            in_band = True
            start = row
        elif not active[row] and in_band:
            in_band = False
            if row - start > h * 0.03:
                bands.append((start, row))
    if in_band and h - start > h * 0.03:
        bands.append((start, h))

    if len(bands) >= 3:
        return [binary[y1:y2, :] for y1, y2 in bands]

    for n_rows in [6, 4, 3, 12]:
        strip_h = h // n_rows
        if strip_h > 20:
            return [binary[i * strip_h:(i + 1) * strip_h, :] for i in range(n_rows)]

    return [binary]


def _handle_multi_column_layout(strips: list[np.ndarray]) -> list[np.ndarray]:
    """
    Gère le format standard ECG papier (4 colonnes × 3 lignes = 12 leads)
    ou (2 colonnes × 6 lignes).
    """
    all_leads: list[np.ndarray] = []

    if len(strips) in (3, 4) and strips[0].shape[1] / max(strips[0].shape[0], 1) > 3:
        n_cols = 12 // len(strips)
        for strip in strips:
            h_s, w_s = strip.shape
            col_w = w_s // n_cols
            for c in range(n_cols):
                sub = strip[:, c * col_w:(c + 1) * col_w]
                all_leads.append(sub)
        return all_leads

    return strips


def _resample_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    """Rééchantillonne une trace 1-D à *target_len* points via scipy.signal.resample."""
    n = len(trace)
    if n == target_len:
        return trace
    return scipy_resample(trace, target_len)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Orchestration — image_to_signal()
# ═══════════════════════════════════════════════════════════════════════════

def image_to_signal(image_path: str) -> dict:
    """
    Pipeline complet : image ECG → signal numérique (N_SAMPLES, 12).

    Étapes :
        1. check_resolution()
        2. Chargement OpenCV
        3. crop_lead_II()
        4. preprocess()
        5. Extraction signal (extract_trace + NaN interp)
        6. calibrate_scale() + rééchantillonnage 500 Hz
        7. Construction matrice 12 leads

    Parameters
    ----------
    image_path : str
        Chemin vers un fichier image JPG / PNG contenant un ECG.

    Returns
    -------
    dict
        {
            "signal": np.ndarray (N_SAMPLES, 12),
            "resolution_warning": str | None,
            "calibration_ok": bool,
            "lead": "II"
        }

    Raises
    ------
    FileNotFoundError / ValueError
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    # 1. Résolution
    resolution_warning = check_resolution(image_path)

    # 2. Chargement
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"Impossible de lire l'image : {image_path}\n"
            "Vérifiez que le fichier est un JPG ou PNG valide."
        )

    img = _correct_skew(img)
    img_original = img.copy()

    # 3. Isolation Lead II
    img_cropped = crop_lead_II(img)

    # 4. Prétraitement
    binary = preprocess(img_cropped)

    # 5. Segmentation et extraction
    strips = _segment_leads(binary)
    strips = _handle_multi_column_layout(strips)

    traces: list[np.ndarray] = []
    for strip in strips:
        trace = extract_trace(strip)
        if trace is not None:
            traces.append(trace)

    if len(traces) == 0:
        raise ValueError(
            "Aucun signal ECG exploitable détecté dans l'image.\n"
            "Assurez-vous que l'image contient un tracé ECG lisible "
            "(courbe sombre sur fond clair ou l'inverse)."
        )

    # 6. Calibration + rééchantillonnage à 500 Hz
    pps, calibration_ok = calibrate_scale(img_original)

    resampled_traces: list[np.ndarray] = []
    for trace in traces:
        resampled_traces.append(_resample_trace(trace, N_SAMPLES))

    # Normalisation Z-score par trace
    for i in range(len(resampled_traces)):
        std = np.std(resampled_traces[i])
        if std > 1e-8:
            resampled_traces[i] = (
                (resampled_traces[i] - np.mean(resampled_traces[i])) / std
            )

    # Construire la matrice (N_SAMPLES, 12)
    signal = np.zeros((N_SAMPLES, _N_LEADS), dtype=np.float64)

    detected_count = min(len(resampled_traces), _N_LEADS)
    if len(resampled_traces) >= _N_LEADS:
        for i in range(_N_LEADS):
            signal[:, i] = resampled_traces[i]
    else:
        for i in range(len(resampled_traces)):
            signal[:, i] = resampled_traces[i]

    return {
        "signal": signal,
        "resolution_warning": resolution_warning,
        "calibration_ok": calibration_ok,
        "lead": "II",
        "detected_leads": LEADS[:detected_count],
        "missing_leads": LEADS[detected_count:],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Backward-compatible wrapper
# ═══════════════════════════════════════════════════════════════════════════

def extract_signal_from_image(image_path: str) -> np.ndarray:
    """
    Legacy wrapper — retourne uniquement le ndarray (N_SAMPLES, 12).
    Utilisé par les appelants existants qui n'attendent pas le dict enrichi.
    """
    result = image_to_signal(image_path)
    return result["signal"]
