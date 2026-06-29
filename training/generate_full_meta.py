"""
training/generate_full_meta.py — Build full_meta.csv from df_meta.pkl
======================================================================
Uses the same diagnosis-to-class mapping as sample_ecg.py but includes
ALL available patients (no sampling caps) from the 10 training classes.

Usage:
    python training/generate_full_meta.py
    python training/generate_full_meta.py --out data/samples/full_meta.csv
"""

import argparse
import ast
import os
import re
import sys

import pandas as pd

META_PATH = r"data/ECG AHP/data_ahp_ecg/df_meta.pkl"
ECG_BASE  = r"data/ECG AHP/data_ahp_ecg/"

# The 10 classes used for training (same as SAMPLING_CONFIG in sample_ecg.py)
TRAINING_CLASSES = {
    "NORMAL", "BRADYCARDIE", "EXTRASYSTOLES", "AFIB",
    "BAV1", "TACHYCARDIE", "ARYTHMIE_SINUSALE", "FLUTTER",
    "PACE_VENT", "PACE_AURIC",
}

# Under-sampling caps — classer les classes dominantes
SUBSAMPLE_CAPS = {
    "NORMAL":      5000,
    "BRADYCARDIE": 2000,
}


# ---------------------------------------------------------------------------
# Label mapping — identical to sample_ecg.py
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
    parser = argparse.ArgumentParser(
        description="Generate full_meta.csv with all patients from df_meta.pkl"
    )
    parser.add_argument(
        "--out",
        default="data/samples/full_meta.csv",
        help="Output CSV path (default: data/samples/full_meta.csv)",
    )
    args = parser.parse_args()

    # ── 1. Load metadata ──────────────────────────────────────────────────
    print(f"Loading metadata from: {META_PATH}")
    if not os.path.exists(META_PATH):
        print(f"ERROR: File not found: {META_PATH}", file=sys.stderr)
        sys.exit(1)

    meta = pd.read_pickle(META_PATH)
    print(f"  Loaded {len(meta):,} rows × {meta.shape[1]} columns")

    # ── 2. Map diagnoses ──────────────────────────────────────────────────
    print("Mapping diagnoses to classes …")
    meta["primary_diag"] = meta["diagnosis"].apply(extract_primary_diagnosis)
    meta["label"] = meta["primary_diag"].apply(map_to_class)

    class_dist = meta["label"].value_counts()
    print("\nFull class distribution:")
    for cls, cnt in class_dist.items():
        mark = "✓" if cls in TRAINING_CLASSES else "✗ (excluded)"
        print(f"  {cls:<22} {cnt:>6}  {mark}")

    # ── 3. Resolve absolute ECG paths ─────────────────────────────────────
    def resolve_path(rel_path):
        full = os.path.join(ECG_BASE, rel_path)
        return os.path.normpath(full)

    meta["full_ecg_path"] = meta["ecg_file_path"].apply(resolve_path)

    # ── 4. Keep only training classes, check file existence ───────────────
    df = meta[meta["label"].isin(TRAINING_CLASSES)].copy()
    print(f"\nPatients in training classes (before file check): {len(df):,}")

    # Verify the files actually exist
    df = df[df["full_ecg_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Patients with existing ECG files: {len(df):,}")

    # ── 4b. Under-sampling des classes dominantes ─────────────────────────
    if SUBSAMPLE_CAPS:
        print("\nSous-échantillonnage des classes dominantes :")
        frames = []
        for cls in sorted(df["label"].unique()):
            cls_df = df[df["label"] == cls]
            cap = SUBSAMPLE_CAPS.get(cls)
            if cap and len(cls_df) > cap:
                cls_df = cls_df.sample(n=cap, random_state=42)
                print(f"  {cls:<22} {cap:>5} / {df[df['label']==cls].shape[0]:>5}  ✓ sous-échantillonné")
            else:
                print(f"  {cls:<22} {len(cls_df):>5}  (inchangé)")
            frames.append(cls_df)
        df = pd.concat(frames, ignore_index=True)
        print(f"  Total après sous-échantillonnage : {len(df):,}")

    # ── 5. Build output dataframe ─────────────────────────────────────────
    out_df = df[["patient_id", "full_ecg_path", "label", "primary_diag", "age", "gender"]].copy()
    out_df.columns = ["patient_id", "full_ecg_path", "label", "primary_diag", "age", "gender"]

    # ── 6. Save ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n✅ Saved {len(out_df):,} patients → {args.out}")

    print("\nFinal class distribution:")
    for cls, cnt in out_df["label"].value_counts().items():
        print(f"  {cls:<22} {cnt:>6}")


if __name__ == "__main__":
    main()
