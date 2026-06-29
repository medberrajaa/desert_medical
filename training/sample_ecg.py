"""
sample_ecg.py — Stratified sampling of ECG metadata for XGBoost training.

Reads df_meta.pkl, maps diagnoses to simplified clinical classes, applies
stratified sampling (pathology-first, balanced normals), and writes
data/samples/sample_meta.csv.

Usage:
    python training/sample_ecg.py [--seed 42] [--out data/samples/sample_meta.csv]
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — max patients PER CLASS (None = take all available)
# ---------------------------------------------------------------------------
SAMPLING_CONFIG = {
    "NORMAL":             7000,   # was 4000  → ~15 000 total
    "BRADYCARDIE":        2000,   # was 1500
    "EXTRASYSTOLES":      1200,   # was 800  (available ~1830)
    "AFIB":               1200,   # was 800  (available ~1760)
    "BAV1":               None,   # ~1492 → take all (classe difficile)
    "TACHYCARDIE":        1000,   # was 800  (available ~1155)
    "ARYTHMIE_SINUSALE":  None,   # ~305 → take all
    "FLUTTER":            None,   # ~271 → take all
    "PACE_VENT":          None,   # ~228 → take all
    "PACE_AURIC":         None,   # ~138 → take all
    # SINUSAL_AUTRE exclu — trop hétérogène, nuit à la précision
    # BAV2 (23) et BBG (21) exclus — trop peu de samples
}

META_PATH = r"data/ECG AHP/data_ahp_ecg/df_meta.pkl"
ECG_BASE  = r"data/ECG AHP/data_ahp_ecg/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_primary_diagnosis(diag_str: str) -> str:
    """Return the first meaningful diagnosis from the stringified list."""
    skip = {"valid", "aucun ecg", "normal par ailleurs", "userinsert"}
    try:
        lst = ast.literal_eval(diag_str)
        for item in lst:
            s = item.strip()
            sl = s.lower()
            if s and not any(kw in sl for kw in skip):
                # Remove USERINSERT prefix if accidentally kept
                s = re.sub(r"^USERINSERT\s*:\s*", "", s, flags=re.IGNORECASE)
                return s
    except Exception:
        pass
    return str(diag_str)[:80]


def map_to_class(diag: str) -> str:
    """Map a primary diagnosis string to a simplified clinical class label."""
    d = diag.lower()

    if "rythme sinusal normal" in d or "ecg normal" in d:
        return "NORMAL"
    if "fibrillation auriculaire" in d:
        return "AFIB"
    if "flutter auriculaire" in d:
        return "FLUTTER"
    if "tachycardie sinusale" in d:
        return "TACHYCARDIE"
    if "bradycardie sinusale" in d:
        return "BRADYCARDIE"
    if "bloc de branche gauche" in d or "bbg" in d:
        return "BBG"
    if "bloc de branche droit" in d or "bbd" in d or "incomplet droit" in d:
        return "BBD"
    if "bloc a-v du premier" in d:
        return "BAV1"
    if "bloc a-v du deuxi" in d:
        return "BAV2"
    if "bloc a-v du troisi" in d or "bloc a-v complet" in d:
        return "BAV3"
    if (
        "extrasystoles ventriculaires" in d
        or "extrasystoles auriculaires" in d
        or "extrasystoles supraventriculaires" in d
    ):
        return "EXTRASYSTOLES"
    if "rythme ventriculaire entra" in d:
        return "PACE_VENT"
    if "rythme auriculaire entra" in d or "rythme av entra" in d:
        return "PACE_AURIC"
    if "arythmie sinusale" in d:
        return "ARYTHMIE_SINUSALE"
    if "rythme sinusal avec" in d or "rythme sinusal" in d:
        return "SINUSAL_AUTRE"
    return "AUTRE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stratified ECG sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        default="data/samples/sample_meta.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # ── 1. Load metadata ────────────────────────────────────────────────────
    print(f"Loading metadata from: {META_PATH}")
    if not os.path.exists(META_PATH):
        print(f"ERROR: File not found: {META_PATH}", file=sys.stderr)
        sys.exit(1)

    meta = pd.read_pickle(META_PATH)
    print(f"  Loaded {len(meta):,} rows × {meta.shape[1]} columns")

    # ── 2. Map diagnoses ─────────────────────────────────────────────────────
    print("Mapping diagnoses to classes …")
    meta["primary_diag"] = meta["diagnosis"].apply(extract_primary_diagnosis)
    meta["label"] = meta["primary_diag"].apply(map_to_class)

    class_dist = meta["label"].value_counts()
    print("\nFull class distribution:")
    for cls, cnt in class_dist.items():
        mark = "✓" if cls in SAMPLING_CONFIG else "✗ (excluded)"
        print(f"  {cls:<22} {cnt:>6}  {mark}")

    # ── 3. Resolve absolute ECG paths ────────────────────────────────────────
    def resolve_path(rel_path):
        # ecg_file_path is relative to ECG_BASE
        full = os.path.join(ECG_BASE, rel_path)
        # Normalise separators
        return os.path.normpath(full)

    meta["full_ecg_path"] = meta["ecg_file_path"].apply(resolve_path)

    # ── 4. Stratified sampling ───────────────────────────────────────────────
    print("\nApplying stratified sampling …")
    frames = []
    for cls, max_n in SAMPLING_CONFIG.items():
        subset = meta[meta["label"] == cls].copy()
        available = len(subset)
        if available == 0:
            print(f"  {cls:<22}  0 available — SKIP")
            continue
        n = min(available, max_n) if max_n is not None else available
        sampled = subset.sample(n=n, random_state=args.seed)
        frames.append(sampled)
        print(f"  {cls:<22}  {n:>5} / {available:>6} selected")

    if not frames:
        print("ERROR: No classes matched. Check metadata path.", file=sys.stderr)
        sys.exit(1)

    sample_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal selected: {len(sample_df):,} patients")

    # ── 5. Verify files exist ────────────────────────────────────────────────
    missing = sample_df["full_ecg_path"].apply(lambda p: not os.path.exists(p))
    n_missing = missing.sum()
    if n_missing > 0:
        print(f"\nWARNING: {n_missing} ECG files not found on disk. They will be removed.")
        sample_df = sample_df[~missing].reset_index(drop=True)
        print(f"Remaining after filter: {len(sample_df):,}")

    # ── 6. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_cols = ["patient_id", "full_ecg_path", "label", "primary_diag",
                "age", "gender"]
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in sample_df.columns]
    sample_df[out_cols].to_csv(args.out, index=False)
    print(f"\nSaved sample metadata → {args.out}")

    # Final summary
    print("\nFinal class distribution in sample:")
    print(sample_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
