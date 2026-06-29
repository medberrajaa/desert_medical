"""Hybrid ECG interpretation pipeline.

Pipeline:
    ECG CSV/image -> signal preprocessing -> clinical measurements
    -> deterministic abnormality detection -> optional classifier
    -> LLM/fallback cardiology report

The original XGBoost classifier is still supported when the model files exist,
but final interpretation no longer depends on it.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from module2.ecg_llm import interpret_ecg_with_llm
from module2.ecg_measurements import analyze_ecg_csv, analyze_ecg_signal, preprocess_measurement_signal
from module2.ecg_reference import match_reference
from module2.ecg_vision import analyze_ecg_image_vision
from training.preprocess_ecg import FS, LEADS, N_SAMPLES, clean_ecg_df, extract_fft_features, extract_temporal_features

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _ROOT / "models"
MODEL_PATH = MODEL_DIR / "ecg_xgboost.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_NAME_CANDIDATES = [
    _ROOT / "data" / "processed_full" / "feature_names.txt",
    _ROOT / "data" / "processed_sample" / "feature_names.txt",
    _ROOT / "data" / "processed" / "feature_names.txt",
]

LABEL_DISPLAY = {
    "NORMAL": "Normal sinus rhythm",
    "AFIB": "Atrial fibrillation",
    "FLUTTER": "Atrial flutter",
    "TACHYCARDIE": "Sinus tachycardia",
    "BRADYCARDIE": "Sinus bradycardia",
    "BAV1": "First-degree AV block",
    "EXTRASYSTOLES": "Premature beats",
    "PACE_VENT": "Ventricular paced rhythm",
    "PACE_AURIC": "Atrial paced rhythm",
    "ARYTHMIE_SINUSALE": "Sinus arrhythmia",
}

PATHOLOGIES = sorted(
    set(LABEL_DISPLAY.values())
    | {
        "Anterior STEMI",
        "Inferior STEMI",
        "Lateral STEMI",
        "Posterior STEMI equivalent",
        "NSTEMI / ischemia pattern",
        "Old infarction pattern",
        "Ventricular tachycardia",
        "Ventricular fibrillation",
        "Left bundle branch block",
        "Right bundle branch block",
        "Hyperkalemia pattern",
        "Pericarditis pattern",
    }
)


def _safe_model_path(path: Path) -> Path:
    resolved = path.resolve()
    model_root = MODEL_DIR.resolve()
    if model_root not in resolved.parents and resolved != model_root:
        raise ValueError(f"Unsafe model path outside model directory: {resolved}")
    return resolved


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any]:
    """Load optional XGBoost classifier and label encoder."""
    model_path = _safe_model_path(MODEL_PATH)
    encoder_path = _safe_model_path(ENCODER_PATH)
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"ECG classifier files not found: {model_path.name}, {encoder_path.name}. "
            "Signal measurements and hybrid interpretation will still run."
        )
    return joblib.load(model_path), joblib.load(encoder_path)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str] | None:
    for path in FEATURE_NAME_CANDIDATES:
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def _interlead_features(cleaned: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    lead_idx = {lead: i for i, lead in enumerate(LEADS)}
    try:
        features["xcorr_II_III"] = float(np.corrcoef(cleaned[:, lead_idx["II"]], cleaned[:, lead_idx["III"]])[0, 1])
        features["xcorr_V1_V6"] = float(np.corrcoef(cleaned[:, lead_idx["V1"]], cleaned[:, lead_idx["V6"]])[0, 1])
        limb_energy = sum(np.sum(cleaned[:, j] ** 2) for j, lead in enumerate(LEADS) if lead in {"I", "II", "III", "aVR", "aVL", "aVF"})
        precord_energy = sum(np.sum(cleaned[:, j] ** 2) for j, lead in enumerate(LEADS) if lead.startswith("V"))
        total_energy = limb_energy + precord_energy + 1e-10
        features["ratio_precord_limb_energy"] = float(precord_energy / (limb_energy + 1e-10))
        features["total_signal_energy"] = float(total_energy / N_SAMPLES)
    except Exception as exc:
        logger.debug("interlead_feature_error", extra={"error": str(exc)})
        features.update(
            {
                "xcorr_II_III": 0.0,
                "xcorr_V1_V6": 0.0,
                "ratio_precord_limb_energy": 0.0,
                "total_signal_energy": 0.0,
            }
        )
    return features


def _extract_features_from_cleaned(cleaned: np.ndarray, filename: str) -> dict[str, float | str]:
    features: dict[str, float | str] = {"filename": filename}
    for idx, lead in enumerate(LEADS):
        sig = cleaned[:, idx]
        features.update(extract_temporal_features(sig, lead, FS))
        features.update(extract_fft_features(sig, lead, FS))
    features.update(_interlead_features(cleaned))
    return features


def extract_features_from_csv(csv_path: str) -> tuple[dict, np.ndarray]:
    """Extract model-compatible features from a 12-lead CSV."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    cleaned = clean_ecg_df(df)
    return _extract_features_from_cleaned(cleaned, os.path.basename(csv_path)), cleaned


def _classify_features(features: dict[str, Any]) -> dict[str, Any]:
    try:
        model, encoder = _load_model()
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}

    feature_names = _load_feature_names()
    if feature_names is None:
        feature_names = [key for key in features if key != "filename"]

    x = np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = str(encoder.classes_[pred_idx])
    probabilities = {
        LABEL_DISPLAY.get(str(cls), str(cls)): round(float(proba[idx]) * 100, 1)
        for idx, cls in enumerate(encoder.classes_)
    }
    probabilities = dict(sorted(probabilities.items(), key=lambda item: -item[1]))
    differentials = [
        {"label": LABEL_DISPLAY.get(str(cls), str(cls)), "confidence": round(float(prob) * 100, 1)}
        for cls, prob in sorted(zip(encoder.classes_, proba), key=lambda item: -item[1])[1:3]
        if float(prob) >= 0.05
    ]
    return {
        "available": True,
        "diagnosis": pred_class,
        "diagnosis_display": LABEL_DISPLAY.get(pred_class, pred_class),
        "confidence": round(float(proba[pred_idx]) * 100, 1),
        "probabilities": probabilities,
        "differentials": differentials,
        "n_features": len(feature_names),
    }


# French limb-lead labels (D1/D2/D3) to match the standard tracing layout.
_FR_LEAD_LABELS = {"I": "D1", "II": "D2", "III": "D3"}


def generate_ecg_plot(cleaned: np.ndarray, fs: int = FS):
    """Generate a 12-lead stacked Matplotlib figure.

    Shared style for CSV and image uploads so both are visualised the same way:
    12 stacked strips, French limb-lead titles (D1/D2/D3), blue trace, sample x-axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_samples = min(int(10.0 * fs), cleaned.shape[0])
    x = np.arange(display_samples)
    fig, axes = plt.subplots(len(LEADS), 1, figsize=(9, 13), sharex=True)
    fig.patch.set_facecolor("white")

    for idx, (ax, lead) in enumerate(zip(axes, LEADS)):
        sig = cleaned[:display_samples, idx] if idx < cleaned.shape[1] else np.zeros(display_samples)
        ax.plot(x, sig, color="#1f77b4", linewidth=0.7)
        ax.set_title(_FR_LEAD_LABELS.get(lead, lead), fontsize=9, pad=2)
        ax.grid(True, color="#cfd8e3", linewidth=0.4, alpha=0.8)
        ax.tick_params(labelsize=7)
        ax.margins(x=0)
    axes[-1].set_xlabel("Échantillons (500 Hz)", fontsize=8)
    plt.tight_layout(h_pad=0.4)
    return fig


def _features_summary(measurements: dict[str, Any], n_features: int) -> dict[str, Any]:
    rhythm = measurements.get("rhythm", {})
    intervals = measurements.get("intervals", {})
    return {
        "hr_mean": rhythm.get("heart_rate_bpm", "N/A"),
        "rr_mean_ms": rhythm.get("rr_mean_ms", "N/A"),
        "rr_cv": rhythm.get("rr_cv", "N/A"),
        "rmssd": "N/A",
        "pr_ms": intervals.get("pr_ms", "N/A"),
        "qrs_ms": intervals.get("qrs_ms", "N/A"),
        "qt_ms": intervals.get("qt_ms", "N/A"),
        "qtc_ms": intervals.get("qtc_ms", "N/A"),
        "lf_hf_ratio": "N/A",
        "n_beats": rhythm.get("organized_r_peaks", 0),
        "n_features": n_features,
    }


def _display_from_report(report: dict[str, Any], classification: dict[str, Any], abnormalities: list[dict[str, Any]]) -> tuple[str, str, float, dict[str, float], list[dict[str, Any]]]:
    critical = [alert for alert in abnormalities if alert.get("severity") == "critical"]
    if critical:
        label = critical[0]["label"]
        return label, label, round(float(report.get("confidence", 0.85)) * 100, 1), {label: round(float(report.get("confidence", 0.85)) * 100, 1)}, []

    if classification.get("available"):
        return (
            classification["diagnosis"],
            classification["diagnosis_display"],
            classification["confidence"],
            classification["probabilities"],
            classification.get("differentials", []),
        )

    diagnoses = report.get("possible_diagnosis", [])
    if diagnoses:
        label = diagnoses[0].get("diagnosis", "ECG interpretation")
        confidence = round(float(diagnoses[0].get("confidence", report.get("confidence", 0.6))) * 100, 1)
        differentials = [
            {"label": diag.get("diagnosis", "Differential"), "confidence": round(float(diag.get("confidence", 0.0)) * 100, 1)}
            for diag in diagnoses[1:3]
        ]
        return label, label, confidence, {label: confidence}, differentials

    return "ECG interpretation", "ECG interpretation", 0.0, {}, []


def _finalize_result(
    *,
    source: str,
    filename: str,
    features: dict[str, Any],
    cleaned_for_plot: np.ndarray,
    measurements: dict[str, Any],
    patient_info: dict[str, Any] | None,
    use_llm: bool,
    image_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classification = _classify_features(features) if features else {"available": False, "error": "feature extraction unavailable"}
    classifier_context = {
        "available": classification.get("available", False),
        "diagnosis": classification.get("diagnosis_display"),
        "confidence": classification.get("confidence"),
        "probabilities": classification.get("probabilities"),
        "error": classification.get("error"),
    }
    interpretation = interpret_ecg_with_llm(
        measurements=measurements,
        patient_info=patient_info,
        classifier_context=classifier_context,
        use_llm=use_llm,
    )
    report = interpretation["data"]
    abnormalities = measurements.get("detected_abnormalities", [])
    diagnosis, diagnosis_display, confidence, probabilities, differentials = _display_from_report(report, classification, abnormalities)
    n_features = int(classification.get("n_features") or max(0, len(features) - 1))
    summary = _features_summary(measurements, n_features)
    plot_signal = cleaned_for_plot if cleaned_for_plot is not None else preprocess_measurement_signal(np.zeros((N_SAMPLES, len(LEADS))), preserve_amplitude=False)

    result = {
        "success": True,
        "source": source,
        "filename": filename,
        "diagnosis": diagnosis,
        "diagnosis_display": diagnosis_display,
        "confidence": confidence,
        "probabilities": probabilities,
        "differentials": differentials,
        "features": summary,
        "features_summary": summary,
        "measurements": measurements,
        "detected_abnormalities": abnormalities,
        "clinical_alerts": abnormalities,
        "classifier": classification,
        "clinical_report": report,
        "interpretation": report,
        "interpretation_source": interpretation.get("source"),
        "interpretation_model": interpretation.get("model"),
        "interpretation_tokens": interpretation.get("tokens_used", 0),
        "ecg_plot": generate_ecg_plot(plot_signal, FS),
        "hr_mean": summary["hr_mean"],
    }
    if image_metadata:
        result.update(image_metadata)
    return result


def predict_ecg(csv_path: str, patient_info: dict[str, Any] | None = None, use_llm: bool = True) -> dict:
    """Analyze ECG CSV with hybrid signal analysis and optional LLM report."""
    try:
        measurements, _raw_signal = analyze_ecg_csv(csv_path, FS)
        try:
            features, cleaned = extract_features_from_csv(csv_path)
        except Exception as exc:
            logger.warning("ecg_feature_extraction_failed", extra={"error": str(exc)})
            features = {"filename": os.path.basename(csv_path)}
            cleaned = preprocess_measurement_signal(_raw_signal, FS, preserve_amplitude=False)
        return _finalize_result(
            source="csv",
            filename=os.path.basename(csv_path),
            features=features,
            cleaned_for_plot=cleaned,
            measurements=measurements,
            patient_info=patient_info,
            use_llm=use_llm,
        )
    except Exception as exc:
        logger.exception("ecg_prediction_failed")
        return {"success": False, "error": f"ECG analysis error: {exc}"}


def predict_from_image(image_path: str, patient_info: dict[str, Any] | None = None, use_llm: bool = True) -> dict:
    """Analyze an ECG image by extracting waveform signal then using the hybrid path."""
    try:
        from module2.ecg_image_reader import image_to_signal

        img_result = image_to_signal(image_path)
        signal = img_result["signal"]
        image_limitations = [
            "Image waveform extraction is approximate; prefer calibrated digital CSV for interval and ST measurements."
        ]
        missing_leads = img_result.get("missing_leads") or []
        if missing_leads:
            image_limitations.append(f"Image extraction did not recover all leads; zero-filled leads: {', '.join(missing_leads)}.")
        measurements = analyze_ecg_signal(
            signal,
            fs=FS,
            source="image",
            amplitude_unit="relative",
            limitations=image_limitations,
        )
        features = _extract_features_from_cleaned(signal, os.path.basename(image_path))
        return _finalize_result(
            source="image",
            filename=os.path.basename(image_path),
            features=features,
            cleaned_for_plot=signal,
            measurements=measurements,
            patient_info=patient_info,
            use_llm=use_llm,
            image_metadata={
                "image_warning": True,
                "resolution_warning": img_result.get("resolution_warning"),
                "calibration_ok": img_result.get("calibration_ok", False),
                "detected_leads": img_result.get("detected_leads", []),
                "missing_leads": missing_leads,
            },
        )
    except Exception as exc:
        logger.exception("ecg_image_prediction_failed")
        return {"success": False, "error": f"ECG image analysis error: {exc}"}


def _deterministic_image_fallback(abnormalities: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Pick a French headline diagnosis from signal abnormalities when no LLM/anchor is available."""
    critical = [a for a in abnormalities if a.get("severity") == "critical"]
    warning = [a for a in abnormalities if a.get("severity") == "warning"]
    if critical:
        return critical[0].get("label", "Anomalie ECG critique"), "élevé", "règles_déterministes"
    if warning:
        return warning[0].get("label", "Anomalie ECG"), "modéré", "règles_déterministes"
    return "Pas d'anomalie aiguë détectée", "faible", "règles_déterministes"


def diagnose_ecg_image(image_path: str, language: str = "Français", use_llm: bool = True) -> dict[str, Any]:
    """Analyze an ECG *image* (PULSE-style): 12-lead visualization + vision-LLM reading.

    Pipeline:
        1. Match the curated reference registry (deterministic anchor for validated cases).
        2. Reconstruct the 12-lead signal for visualization + supplementary measurements.
        3. Read the image with a vision LLM, anchored on the reference when matched.
        4. Build a unified result dict compatible with the UI.
    """
    try:
        reference = match_reference(image_path)

        ecg_fig = None
        measurements: dict[str, Any] = {}
        abnormalities: list[dict[str, Any]] = []
        detected_leads: list[str] = []
        missing_leads: list[str] = []
        try:
            from module2.ecg_image_reader import image_to_signal

            img_res = image_to_signal(image_path)
            signal = img_res["signal"]
            detected_leads = img_res.get("detected_leads", [])
            missing_leads = img_res.get("missing_leads", [])
            ecg_fig = generate_ecg_plot(signal, FS)
            measurements = analyze_ecg_signal(
                signal,
                fs=FS,
                source="image",
                amplitude_unit="relative",
                limitations=["Signal reconstruit depuis l'image ; mesures d'intervalles et de ST approximatives."],
            )
            abnormalities = measurements.get("detected_abnormalities", [])
        except Exception as exc:
            logger.warning("ecg_image_signal_failed", extra={"error": str(exc)})

        vision = (
            analyze_ecg_image_vision(image_path, anchor=reference, language=language)
            if use_llm
            else {"success": False, "error": "LLM désactivé."}
        )

        if reference:
            diagnosis = reference["diagnosis"]
            subtitle = reference.get("subtitle", "")
            territory = reference.get("territory", "")
            urgency = "critique"
            confidence = round(float(reference.get("confidence", 0.95)) * 100, 1)
            source = "reference+vision" if vision.get("success") else "reference"
        elif vision.get("success"):
            diagnosis = vision["pathology"]
            subtitle = ""
            territory = ""
            urgency = vision.get("urgency", "modéré")
            confidence = None
            source = "vision_llm"
        else:
            diagnosis, urgency, source = _deterministic_image_fallback(abnormalities)
            subtitle = ""
            territory = ""
            confidence = None

        if vision.get("success") and vision.get("analysis"):
            analysis = vision["analysis"]
        elif reference:
            analysis = (
                f"Cas de référence validé : **{reference['diagnosis']} ({subtitle})**. "
                "L'analyse visuelle automatique étant indisponible, le diagnostic est ancré sur la référence clinique."
            )
        else:
            analysis = vision.get("error", "Analyse visuelle indisponible.")

        summary = _features_summary(measurements, 0) if measurements else {}

        return {
            "success": True,
            "source": "image",
            "filename": os.path.basename(image_path),
            "diagnosis": diagnosis,
            "diagnosis_display": diagnosis,
            "subtitle": subtitle,
            "territory": territory,
            "confidence": confidence,
            "urgency": urgency,
            "analysis": analysis,
            "interpretation_source": source,
            "interpretation_model": vision.get("model"),
            "vision_raw": vision.get("raw", ""),
            "ecg_plot": ecg_fig,
            "measurements": measurements,
            "features_summary": summary,
            "features": summary,
            "detected_abnormalities": abnormalities,
            "clinical_alerts": abnormalities,
            "reference_case": reference,
            "image_warning": True,
            "detected_leads": detected_leads,
            "missing_leads": missing_leads,
            "hr_mean": summary.get("hr_mean", "N/A"),
        }
    except Exception as exc:
        logger.exception("ecg_image_diagnosis_failed")
        return {"success": False, "error": f"ECG image analysis error: {exc}"}


def apply_clinical_rules(features: dict, cleaned: np.ndarray, fs: int = FS) -> list[dict]:
    """Backward-compatible wrapper returning deterministic ECG alerts."""
    measurements = analyze_ecg_signal(cleaned, fs=fs, source="array", amplitude_unit="relative")
    return measurements.get("detected_abnormalities", [])
