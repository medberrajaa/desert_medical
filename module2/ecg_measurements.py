"""Clinical ECG measurement and deterministic abnormality detection.

The trained classifier in the original project is kept as an optional signal,
but acute safety findings are derived from measurements. This makes STEMI and
other high-risk patterns visible even when the model file is absent or uncertain.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from training.preprocess_ecg import FS, LEADS, N_SAMPLES

logger = logging.getLogger(__name__)

LIMB_LEADS = {"I", "II", "III", "aVR", "aVL", "aVF"}
PRECORDIAL_LEADS = {"V1", "V2", "V3", "V4", "V5", "V6"}

TERRITORIES = {
    "anterior": ["V2", "V3", "V4", "V5"],
    "septal": ["V1", "V2"],
    "inferior": ["II", "III", "aVF"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "posterior": ["V1", "V2", "V3"],
}


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _round_or_none(value: Any, digits: int = 3) -> float | None:
    value = _finite_float(value)
    if value is None:
        return None
    return round(value, digits)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else np.nanstd(x)
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(x)
    return (x - med) / scale


def _pad_or_trim(signal: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == len(LEADS) and arr.shape[1] != len(LEADS):
        arr = arr.T
    if arr.shape[1] < len(LEADS):
        pad = np.zeros((arr.shape[0], len(LEADS) - arr.shape[1]))
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > len(LEADS):
        arr = arr[:, : len(LEADS)]

    if arr.shape[0] < n_samples:
        pad = np.zeros((n_samples - arr.shape[0], arr.shape[1]))
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > n_samples:
        arr = arr[:n_samples]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def read_ecg_csv_signal(csv_path: str) -> tuple[np.ndarray, list[str]]:
    """Read a CSV ECG into a fixed (N_SAMPLES, 12) array.

    Returns the signal plus limitations describing missing or inferred leads.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ECG CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    normalized = {str(col).strip(): col for col in df.columns}
    lead_lookup = {lead.lower(): lead for lead in LEADS}
    selected: dict[str, pd.Series] = {}

    for raw_name, original in normalized.items():
        key = raw_name.replace(" ", "").lower()
        canonical = lead_lookup.get(key)
        if canonical is not None:
            selected[canonical] = pd.to_numeric(df[original], errors="coerce")

    limitations: list[str] = []
    if not selected:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("CSV does not contain recognizable ECG lead columns or numeric columns.")
        limitations.append("Lead names were not recognized; first numeric columns were mapped to standard lead order.")
        for idx, lead in enumerate(LEADS[: numeric.shape[1]]):
            selected[lead] = pd.to_numeric(numeric.iloc[:, idx], errors="coerce")

    missing = [lead for lead in LEADS if lead not in selected]
    if missing:
        limitations.append(f"Missing leads filled with zero signal: {', '.join(missing)}.")

    out = pd.DataFrame({lead: selected.get(lead, pd.Series(dtype=float)) for lead in LEADS})
    out = out.interpolate(limit_direction="both").fillna(0.0)
    return _pad_or_trim(out.to_numpy(dtype=float)), limitations


def preprocess_measurement_signal(raw: np.ndarray, fs: int = FS, preserve_amplitude: bool = True) -> np.ndarray:
    """Denoise while preserving ST/T amplitudes as much as possible."""
    arr = _pad_or_trim(raw)
    cleaned = np.zeros_like(arr, dtype=float)
    nyq = fs / 2.0
    low_sos = sp_signal.butter(3, min(40.0, nyq - 1) / nyq, btype="low", output="sos")
    notch_b, notch_a = sp_signal.iirnotch(50.0 / nyq, Q=30) if fs > 120 else (None, None)

    for idx in range(arr.shape[1]):
        x = np.nan_to_num(arr[:, idx].astype(float), nan=0.0)
        if np.nanstd(x) < 1e-9:
            cleaned[:, idx] = 0.0
            continue

        baseline_window = int(max(101, fs * 0.8))
        if baseline_window % 2 == 0:
            baseline_window += 1
        if baseline_window < len(x):
            baseline = sp_signal.medfilt(x, kernel_size=baseline_window)
            x = x - baseline
        else:
            x = x - np.nanmedian(x)

        try:
            if notch_b is not None and len(x) > 3 * max(len(notch_a), len(notch_b)):
                x = sp_signal.filtfilt(notch_b, notch_a, x)
            if len(x) > 3 * low_sos.shape[0]:
                x = sp_signal.sosfiltfilt(low_sos, x)
        except ValueError:
            logger.debug("measurement_filter_short_signal", extra={"lead": LEADS[idx]})

        if not preserve_amplitude:
            z = _robust_z(x)
            x = z
        cleaned[:, idx] = x

    return cleaned


def detect_r_peaks(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detect organized R peaks in one lead using robust z scores."""
    z = _robust_z(sig)
    min_distance = int(0.22 * fs)
    best: np.ndarray = np.array([], dtype=int)
    best_score = -np.inf

    for candidate in (z, -z, np.abs(z)):
        for height in (2.5, 2.0, 1.5, 1.0):
            peaks, props = sp_signal.find_peaks(
                candidate,
                height=height,
                distance=min_distance,
                prominence=max(0.6, height * 0.35),
            )
            if len(peaks) == 0:
                continue
            bpm = len(peaks) * 60.0 / (len(sig) / fs)
            plausible = 20 <= bpm <= 260
            score = len(peaks) + (5 if plausible else -10) + float(np.nanmedian(props.get("prominences", [0])))
            if score > best_score:
                best = peaks.astype(int)
                best_score = score
    return best


def choose_r_peak_lead(signal: np.ndarray, fs: int = FS) -> tuple[str, np.ndarray]:
    """Prefer lead II when plausible, otherwise choose the cleanest lead."""
    arr = _pad_or_trim(signal)
    lead_candidates: list[tuple[float, str, np.ndarray]] = []
    for idx, lead in enumerate(LEADS):
        peaks = detect_r_peaks(arr[:, idx], fs)
        bpm = len(peaks) * 60.0 / (arr.shape[0] / fs)
        rr = np.diff(peaks) / fs * 1000.0 if len(peaks) > 2 else np.array([])
        rr_cv = float(np.std(rr) / np.mean(rr)) if rr.size and np.mean(rr) > 0 else 1.0
        plausible = 25 <= bpm <= 220
        score = len(peaks) + (4 if plausible else -6) - min(rr_cv, 1.0)
        if lead == "II" and plausible:
            score += 3
        lead_candidates.append((score, lead, peaks))
    _, lead, peaks = max(lead_candidates, key=lambda item: item[0])
    return lead, peaks


def _qrs_bounds_for_peak(sig: np.ndarray, r_peak: int, fs: int = FS) -> tuple[int, int] | None:
    n = len(sig)
    left_limit = max(0, r_peak - int(0.14 * fs))
    right_limit = min(n - 1, r_peak + int(0.16 * fs))
    local = sig[left_limit:right_limit + 1]
    if local.size < int(0.04 * fs):
        return None
    baseline = np.median(sig[max(0, r_peak - int(0.20 * fs)):max(1, r_peak - int(0.08 * fs))])
    local_abs = np.abs(local - baseline)
    peak_amp = np.max(local_abs)
    if not np.isfinite(peak_amp) or peak_amp < 1e-6:
        return None
    threshold = max(0.08 * peak_amp, np.percentile(local_abs, 35))
    center = r_peak - left_limit

    onset = center
    below_needed = max(2, int(0.008 * fs))
    below_count = 0
    for pos in range(center, -1, -1):
        if local_abs[pos] <= threshold:
            below_count += 1
            if below_count >= below_needed:
                onset = min(center, pos + below_needed)
                break
        else:
            below_count = 0

    offset = center
    below_count = 0
    for pos in range(center, len(local_abs)):
        if local_abs[pos] <= threshold:
            below_count += 1
            if below_count >= below_needed:
                offset = max(center, pos - below_needed)
                break
        else:
            below_count = 0

    onset += left_limit
    offset += left_limit
    width_ms = (offset - onset) / fs * 1000.0
    if 30 <= width_ms <= 240:
        return onset, offset
    return None


def _median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(clean)) if clean else None


def _measure_qrs(sig: np.ndarray, peaks: np.ndarray, fs: int = FS) -> tuple[float | None, list[tuple[int, int]]]:
    widths: list[float] = []
    bounds: list[tuple[int, int]] = []
    for peak in peaks:
        bound = _qrs_bounds_for_peak(sig, int(peak), fs)
        if bound is None:
            continue
        onset, offset = bound
        widths.append((offset - onset) / fs * 1000.0)
        bounds.append(bound)
    return _median(widths), bounds


def _measure_pr(sig: np.ndarray, peaks: np.ndarray, qrs_bounds: list[tuple[int, int]], fs: int = FS) -> float | None:
    values: list[float] = []
    for peak, bound in zip(peaks[: len(qrs_bounds)], qrs_bounds):
        onset, _ = bound
        start = max(0, onset - int(0.24 * fs))
        end = max(0, onset - int(0.06 * fs))
        if end <= start + 5:
            continue
        window = sig[start:end]
        z = _robust_z(window)
        pos, pos_props = sp_signal.find_peaks(z, prominence=0.4)
        neg, neg_props = sp_signal.find_peaks(-z, prominence=0.4)
        candidates: list[tuple[float, int]] = []
        if len(pos):
            for p, prom in zip(pos, pos_props["prominences"]):
                candidates.append((float(prom), int(p)))
        if len(neg):
            for p, prom in zip(neg, neg_props["prominences"]):
                candidates.append((float(prom), int(p)))
        if not candidates:
            continue
        _, p_peak = max(candidates, key=lambda item: item[0])
        pr_ms = (peak - (start + p_peak)) / fs * 1000.0
        if 80 <= pr_ms <= 320:
            values.append(pr_ms)
    return _median(values)


def _measure_qt(sig: np.ndarray, peaks: np.ndarray, qrs_bounds: list[tuple[int, int]], rr_ms: float | None, fs: int = FS) -> tuple[float | None, float | None]:
    qt_values: list[float] = []
    for peak, bound in zip(peaks[: len(qrs_bounds)], qrs_bounds):
        onset, _ = bound
        start = min(len(sig) - 1, int(peak + 0.12 * fs))
        end = min(len(sig) - 1, int(peak + 0.55 * fs))
        if end <= start + 10:
            continue
        baseline_start = max(0, onset - int(0.20 * fs))
        baseline_end = max(1, onset - int(0.08 * fs))
        baseline = float(np.median(sig[baseline_start:baseline_end]))
        segment = sig[start:end] - baseline
        if segment.size < 10:
            continue
        t_peak_local = int(np.argmax(np.abs(segment)))
        amp = abs(segment[t_peak_local])
        if amp < 1e-6:
            continue
        threshold = max(0.10 * amp, 0.02)
        t_end = None
        for pos in range(t_peak_local, len(segment)):
            if abs(segment[pos]) <= threshold:
                t_end = start + pos
                break
        if t_end is None:
            continue
        qt_ms = (t_end - onset) / fs * 1000.0
        if 240 <= qt_ms <= 650:
            qt_values.append(qt_ms)
    qt = _median(qt_values)
    qtc = None
    if qt is not None and rr_ms and rr_ms > 0:
        qtc = qt / np.sqrt(rr_ms / 1000.0)
    return qt, qtc


def _measure_st_by_lead(signal: np.ndarray, peaks: np.ndarray, qrs_ms: float | None, fs: int = FS) -> dict[str, float | None]:
    st_by_lead: dict[str, float | None] = {}
    qrs_offset_samples = int(((qrs_ms or 90.0) / 2.0) / 1000.0 * fs)
    st_offset = int(0.060 * fs)
    for idx, lead in enumerate(LEADS):
        sig = signal[:, idx]
        values: list[float] = []
        for peak in peaks:
            baseline_start = int(peak - 0.20 * fs)
            baseline_end = int(peak - 0.08 * fs)
            st_idx = int(peak + qrs_offset_samples + st_offset)
            if baseline_start < 0 or baseline_end <= baseline_start or st_idx >= len(sig):
                continue
            baseline = float(np.median(sig[baseline_start:baseline_end]))
            values.append(float(sig[st_idx] - baseline))
        st_by_lead[lead] = _round_or_none(_median(values), 3)
    return st_by_lead


def _measure_t_waves(signal: np.ndarray, peaks: np.ndarray, fs: int = FS) -> dict[str, dict[str, float | str | None]]:
    out: dict[str, dict[str, float | str | None]] = {}
    for idx, lead in enumerate(LEADS):
        sig = signal[:, idx]
        amps: list[float] = []
        for peak in peaks:
            baseline_start = int(peak - 0.20 * fs)
            baseline_end = int(peak - 0.08 * fs)
            start = int(peak + 0.12 * fs)
            end = int(peak + 0.42 * fs)
            if baseline_start < 0 or baseline_end <= baseline_start or end >= len(sig):
                continue
            baseline = float(np.median(sig[baseline_start:baseline_end]))
            segment = sig[start:end] - baseline
            if segment.size:
                pos_amp = float(np.max(segment))
                neg_amp = float(np.min(segment))
                amps.append(pos_amp if abs(pos_amp) >= abs(neg_amp) else neg_amp)
        amp = _median(amps)
        polarity = "unknown"
        if amp is not None:
            polarity = "positive" if amp > 0.02 else "negative" if amp < -0.02 else "flat"
        out[lead] = {"amplitude": _round_or_none(amp, 3), "polarity": polarity}
    return out


def _measure_qrs_amplitudes(signal: np.ndarray, peaks: np.ndarray, fs: int = FS) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for idx, lead in enumerate(LEADS):
        sig = signal[:, idx]
        r_values: list[float] = []
        s_values: list[float] = []
        q_values: list[float] = []
        for peak in peaks:
            baseline_start = int(peak - 0.20 * fs)
            baseline_end = int(peak - 0.08 * fs)
            start = max(0, int(peak - 0.07 * fs))
            end = min(len(sig), int(peak + 0.09 * fs))
            if baseline_end <= baseline_start or end <= start:
                continue
            baseline = float(np.median(sig[baseline_start:baseline_end]))
            segment = sig[start:end] - baseline
            r_values.append(float(np.max(segment)))
            s_values.append(float(np.min(segment)))
            q_segment = sig[max(0, int(peak - 0.06 * fs)):max(0, int(peak - 0.01 * fs))] - baseline
            if q_segment.size:
                q_values.append(float(np.min(q_segment)))
        out[lead] = {
            "r": _round_or_none(_median(r_values), 3),
            "s": _round_or_none(_median(s_values), 3),
            "q": _round_or_none(_median(q_values), 3),
        }
    return out


def _pathologic_q_waves(qrs_amplitudes: dict[str, dict[str, float | None]]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for lead, values in qrs_amplitudes.items():
        q = values.get("q")
        r = values.get("r")
        out[lead] = bool(q is not None and r is not None and q < -0.1 and abs(q) >= max(0.04, 0.25 * abs(r)))
    return out


def _axis_from_amplitudes(qrs_amplitudes: dict[str, dict[str, float | None]]) -> str:
    def net(lead: str) -> float | None:
        values = qrs_amplitudes.get(lead, {})
        r = values.get("r")
        s = values.get("s")
        if r is None or s is None:
            return None
        return float(r) + float(s)

    lead_i = net("I")
    avf = net("aVF")
    if lead_i is None or avf is None:
        return "indeterminate"
    if lead_i >= 0 and avf >= 0:
        return "normal"
    if lead_i >= 0 and avf < 0:
        return "left_axis_deviation"
    if lead_i < 0 and avf >= 0:
        return "right_axis_deviation"
    return "extreme_axis_deviation"


def _contiguous_elevation(st: dict[str, float | None], leads: list[str], amplitude_unit: str) -> list[str]:
    elevated = []
    for lead in leads:
        value = st.get(lead)
        if value is None:
            continue
        if amplitude_unit == "mV":
            threshold = 0.15 if lead in {"V2", "V3"} else 0.10
        else:
            threshold = 0.25
        if value >= threshold:
            elevated.append(lead)
    return elevated


def _contiguous_depression(st: dict[str, float | None], leads: list[str], amplitude_unit: str) -> list[str]:
    depressed = []
    threshold = -0.10 if amplitude_unit == "mV" else -0.25
    for lead in leads:
        value = st.get(lead)
        if value is not None and value <= threshold:
            depressed.append(lead)
    return depressed


def _alert(rule: str, label: str, detail: str, severity: str, evidence: list[str], value: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {
        "rule": rule,
        "label": label,
        "detail": detail,
        "severity": severity,
        "evidence": evidence,
        "value": value,
        "threshold": threshold,
    }


def detect_abnormalities(measurements: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect clinically important ECG abnormalities with sensitivity bias for MI."""
    alerts: list[dict[str, Any]] = []
    rhythm = measurements["rhythm"]
    intervals = measurements["intervals"]
    st = measurements["st_deviation"]["by_lead"]
    t_waves = measurements["t_waves"]["by_lead"]
    qrs_amps = measurements["qrs_amplitudes"]["by_lead"]
    q_waves = measurements["pathologic_q_waves"]
    amplitude_unit = measurements["st_deviation"]["unit"]

    hr = rhythm.get("heart_rate_bpm")
    rr_cv = rhythm.get("rr_cv")
    qrs_ms = intervals.get("qrs_ms")
    pr_ms = intervals.get("pr_ms")
    qtc_ms = intervals.get("qtc_ms")

    if rhythm.get("organized_r_peaks", 0) < 3 and measurements["signal_quality"].get("global_std", 0) > 0.05:
        alerts.append(_alert("VF", "Possible ventricular fibrillation / chaotic rhythm", "No organized R-peak sequence detected in a noisy ECG signal.", "critical", ["Poor organized R-peak detection."], rhythm.get("organized_r_peaks"), 3))

    if hr is not None and hr > 120 and qrs_ms is not None and qrs_ms >= 120 and rr_cv is not None and rr_cv < 0.12:
        alerts.append(_alert("VT", "Possible ventricular tachycardia", f"Wide-complex regular tachycardia: HR {hr:.0f} bpm, QRS {qrs_ms:.0f} ms.", "critical", [f"HR {hr:.0f} bpm", f"QRS {qrs_ms:.0f} ms", f"RR CV {rr_cv:.2f}"], hr, ">120 bpm with QRS >=120 ms"))

    if rr_cv is not None and rr_cv > 0.18 and rhythm.get("organized_r_peaks", 0) >= 5:
        alerts.append(_alert("AF", "Possible atrial fibrillation", f"Irregular RR intervals with CV {rr_cv:.2f}.", "warning", [f"RR CV {rr_cv:.2f}"], rr_cv, ">0.18"))

    if rhythm.get("premature_beats", 0) > 0:
        count = rhythm["premature_beats"]
        alerts.append(_alert("PVC", "Premature beats detected", f"{count} premature beat(s) detected by RR shortening.", "info" if count < 3 else "warning", [f"{count} RR intervals <80% of median."], count, ">=1"))

    if qrs_ms is not None and qrs_ms >= 120:
        v1_r = qrs_amps.get("V1", {}).get("r") or 0
        v1_s = qrs_amps.get("V1", {}).get("s") or 0
        v6_r = qrs_amps.get("V6", {}).get("r") or 0
        if v1_r > abs(v1_s):
            alerts.append(_alert("RBBB", "Possible right bundle branch block", f"QRS {qrs_ms:.0f} ms with dominant positive V1 complex.", "warning", [f"QRS {qrs_ms:.0f} ms", "Dominant R in V1."], qrs_ms, ">=120 ms"))
        elif v6_r > abs(v1_r):
            alerts.append(_alert("LBBB", "Possible left bundle branch block", f"QRS {qrs_ms:.0f} ms with left-sided dominant R pattern.", "warning", [f"QRS {qrs_ms:.0f} ms", "Dominant lateral R pattern."], qrs_ms, ">=120 ms"))

    for territory, leads in TERRITORIES.items():
        if territory == "posterior":
            continue
        elevated = _contiguous_elevation(st, leads, amplitude_unit)
        if len(elevated) >= 2:
            label = f"Possible acute {territory} STEMI"
            detail = f"ST elevation in contiguous {territory} leads: {', '.join(elevated)}."
            if amplitude_unit != "mV":
                detail += " Image-derived/relative amplitude limits exact millimeter criteria."
            alerts.append(_alert(
                f"{territory.upper()}_STEMI",
                label,
                detail,
                "critical",
                [f"{lead}: ST {st.get(lead)} {amplitude_unit}" for lead in elevated],
                {lead: st.get(lead) for lead in elevated},
                "contiguous ST elevation",
            ))

    posterior_depressed = _contiguous_depression(st, TERRITORIES["posterior"], amplitude_unit)
    if len(posterior_depressed) >= 2:
        alerts.append(_alert(
            "POSTERIOR_STEMI",
            "Possible posterior STEMI equivalent",
            f"Horizontal ST depression in V1-V3 pattern: {', '.join(posterior_depressed)}.",
            "critical",
            [f"{lead}: ST {st.get(lead)} {amplitude_unit}" for lead in posterior_depressed],
            {lead: st.get(lead) for lead in posterior_depressed},
            "ST depression V1-V3",
        ))

    depressed_any = _contiguous_depression(st, [lead for lead in LEADS if lead != "aVR"], amplitude_unit)
    if len(depressed_any) >= 3 and not any("STEMI" in alert["rule"] for alert in alerts):
        alerts.append(_alert(
            "NSTEMI_ISCHEMIA",
            "Possible ischemia / NSTEMI pattern",
            f"ST depression in multiple leads: {', '.join(depressed_any[:6])}. NSTEMI cannot be diagnosed from ECG alone; correlate with symptoms and troponin.",
            "critical",
            [f"{lead}: ST {st.get(lead)} {amplitude_unit}" for lead in depressed_any[:6]],
            {lead: st.get(lead) for lead in depressed_any},
            "multi-lead ST depression",
        ))

    for territory, leads in TERRITORIES.items():
        q_leads = [lead for lead in leads if q_waves.get(lead)]
        if len(q_leads) >= 2 and not any("STEMI" in alert["rule"] for alert in alerts):
            alerts.append(_alert(
                f"OLD_{territory.upper()}_INFARCTION",
                f"Possible old {territory} infarction",
                f"Pathologic Q-wave pattern in {', '.join(q_leads)} without acute STEMI criteria.",
                "warning",
                [f"Pathologic Q wave in {lead}" for lead in q_leads],
                q_leads,
                ">=2 territory leads",
            ))

    diffuse_elevation = [
        lead for lead in [lead for lead in LEADS if lead != "aVR"]
        if st.get(lead) is not None and st[lead] >= (0.08 if amplitude_unit == "mV" else 0.22)
    ]
    reciprocal_depression = [lead for lead in ["I", "aVL", "II", "III", "aVF"] if st.get(lead) is not None and st[lead] <= (-0.08 if amplitude_unit == "mV" else -0.22)]
    if len(diffuse_elevation) >= 6 and len(reciprocal_depression) <= 1:
        alerts.append(_alert(
            "PERICARDITIS",
            "Possible acute pericarditis pattern",
            f"Diffuse ST elevation in {len(diffuse_elevation)} leads with limited reciprocal depression.",
            "warning",
            [f"Diffuse ST elevation: {', '.join(diffuse_elevation[:8])}"],
            len(diffuse_elevation),
            "diffuse ST elevation",
        ))

    tall_t_leads: list[str] = []
    for lead, t_info in t_waves.items():
        t_amp = t_info.get("amplitude")
        r_amp = qrs_amps.get(lead, {}).get("r")
        if t_amp is not None and r_amp is not None and r_amp > 0 and t_amp > max(0.45 if amplitude_unit == "mV" else 0.65, 0.65 * r_amp):
            tall_t_leads.append(lead)
    if len(tall_t_leads) >= 4 or (qrs_ms is not None and qrs_ms >= 130 and pr_ms is not None and pr_ms > 200):
        alerts.append(_alert(
            "HYPERKALEMIA",
            "Possible hyperkalemia pattern",
            f"Tall T waves in {', '.join(tall_t_leads[:6])} and/or conduction widening.",
            "critical" if qrs_ms and qrs_ms >= 130 else "warning",
            [f"Tall T-wave leads: {', '.join(tall_t_leads)}", f"QRS {qrs_ms} ms", f"PR {pr_ms} ms"],
            {"tall_t_leads": tall_t_leads, "qrs_ms": qrs_ms, "pr_ms": pr_ms},
            "diffuse tall T waves or QRS widening",
        ))

    if qtc_ms is not None and qtc_ms >= 500:
        alerts.append(_alert("LONG_QT", "Prolonged QTc", f"QTc approximately {qtc_ms:.0f} ms.", "warning", [f"QTc {qtc_ms:.0f} ms"], qtc_ms, ">=500 ms"))

    return alerts


def analyze_ecg_signal(raw_signal: np.ndarray, *, fs: int = FS, source: str = "csv", amplitude_unit: str = "mV", limitations: list[str] | None = None) -> dict[str, Any]:
    """Measure intervals, ST/T features, axis, and deterministic abnormalities."""
    limitations = list(limitations or [])
    preserve_amplitude = amplitude_unit == "mV"
    signal = preprocess_measurement_signal(raw_signal, fs=fs, preserve_amplitude=preserve_amplitude)
    if amplitude_unit != "mV":
        limitations.append("Amplitude is relative, not calibrated mV; ST/T thresholds are approximate.")

    r_lead, r_peaks = choose_r_peak_lead(signal, fs)
    lead_idx = LEADS.index(r_lead)
    sig = signal[:, lead_idx]
    rr_ms_values = np.diff(r_peaks) / fs * 1000.0 if len(r_peaks) >= 2 else np.array([])
    rr_ms = float(np.mean(rr_ms_values)) if rr_ms_values.size else None
    rr_cv = float(np.std(rr_ms_values) / np.mean(rr_ms_values)) if rr_ms_values.size and np.mean(rr_ms_values) > 0 else None
    heart_rate = 60000.0 / rr_ms if rr_ms and rr_ms > 0 else None
    premature = int(np.sum(rr_ms_values < 0.80 * np.median(rr_ms_values))) if rr_ms_values.size >= 3 else 0

    qrs_ms, qrs_bounds = _measure_qrs(sig, r_peaks, fs)
    pr_ms = _measure_pr(sig, r_peaks, qrs_bounds, fs) if qrs_bounds else None
    qt_ms, qtc_ms = _measure_qt(sig, r_peaks, qrs_bounds, rr_ms, fs) if qrs_bounds else (None, None)
    st_by_lead = _measure_st_by_lead(signal, r_peaks, qrs_ms, fs) if len(r_peaks) else {lead: None for lead in LEADS}
    t_waves = _measure_t_waves(signal, r_peaks, fs) if len(r_peaks) else {lead: {"amplitude": None, "polarity": "unknown"} for lead in LEADS}
    qrs_amplitudes = _measure_qrs_amplitudes(signal, r_peaks, fs) if len(r_peaks) else {lead: {"r": None, "s": None, "q": None} for lead in LEADS}
    q_waves = _pathologic_q_waves(qrs_amplitudes)
    axis = _axis_from_amplitudes(qrs_amplitudes)

    measurements: dict[str, Any] = {
        "source": source,
        "fs": fs,
        "duration_sec": round(signal.shape[0] / fs, 3),
        "signal_quality": {
            "global_std": _round_or_none(float(np.std(signal)), 4),
            "r_peak_lead": r_lead,
        },
        "rhythm": {
            "heart_rate_bpm": _round_or_none(heart_rate, 1),
            "rr_mean_ms": _round_or_none(rr_ms, 1),
            "rr_cv": _round_or_none(rr_cv, 3),
            "organized_r_peaks": int(len(r_peaks)),
            "premature_beats": premature,
        },
        "intervals": {
            "rr_ms": _round_or_none(rr_ms, 1),
            "pr_ms": _round_or_none(pr_ms, 1),
            "qrs_ms": _round_or_none(qrs_ms, 1),
            "qt_ms": _round_or_none(qt_ms, 1),
            "qtc_ms": _round_or_none(qtc_ms, 1),
        },
        "axis": axis,
        "st_deviation": {
            "unit": amplitude_unit,
            "by_lead": st_by_lead,
        },
        "t_waves": {
            "unit": amplitude_unit,
            "by_lead": t_waves,
        },
        "qrs_amplitudes": {
            "unit": amplitude_unit,
            "by_lead": qrs_amplitudes,
        },
        "pathologic_q_waves": q_waves,
        "limitations": limitations,
    }
    measurements["detected_abnormalities"] = detect_abnormalities(measurements)
    return measurements


def analyze_ecg_csv(csv_path: str, fs: int = FS) -> tuple[dict[str, Any], np.ndarray]:
    raw, limitations = read_ecg_csv_signal(csv_path)
    return analyze_ecg_signal(raw, fs=fs, source="csv", amplitude_unit="mV", limitations=limitations), raw


def compact_measurements_for_llm(measurements: dict[str, Any]) -> dict[str, Any]:
    """Keep the LLM prompt small while preserving clinically relevant evidence."""
    st = measurements["st_deviation"]["by_lead"]
    t_waves = measurements["t_waves"]["by_lead"]
    return {
        "source": measurements.get("source"),
        "rhythm": measurements.get("rhythm"),
        "intervals": measurements.get("intervals"),
        "axis": measurements.get("axis"),
        "st_deviation": {
            "unit": measurements["st_deviation"]["unit"],
            "by_lead": {lead: st.get(lead) for lead in LEADS if st.get(lead) not in (None, 0)},
        },
        "t_waves": {
            "unit": measurements["t_waves"]["unit"],
            "abnormal_or_measured": {lead: t_waves.get(lead) for lead in LEADS if t_waves.get(lead, {}).get("amplitude") is not None},
        },
        "pathologic_q_waves": [lead for lead, present in measurements.get("pathologic_q_waves", {}).items() if present],
        "detected_abnormalities": measurements.get("detected_abnormalities", []),
        "limitations": measurements.get("limitations", []),
    }
