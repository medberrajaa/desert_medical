"""
test_ecg_image.py — Teste le pipeline predict_from_image()

Usage :
    python test_ecg_image.py                    # génère une image depuis le 1er CSV trouvé
    python test_ecg_image.py path/to/ecg.png    # teste directement une image existante
"""
import sys
import os
import glob

# ── 1. Si un chemin image est passé en argument, tester directement ───────────
if len(sys.argv) > 1:
    img_path = sys.argv[1]
    print(f"\n=== Test depuis image fournie : {img_path} ===\n")
    from module2.ecg_predictor import predict_from_image

    result = predict_from_image(img_path)
    if result["success"]:
        print(f"✅ Diagnostic   : {result['diagnosis_display']}")
        print(f"   Confiance    : {result['confidence']} %")
        print(f"   Source       : {result.get('source', 'csv')}")
        print(f"   Avertissement: {result.get('image_warning', False)}")
        fs = result["features_summary"]
        print(f"\n   HR           : {fs['hr_mean']} bpm")
        print(f"   RR moyen     : {fs['rr_mean_ms']} ms")
        print(f"   PR           : {fs['pr_ms']} ms")
        print(f"   QRS          : {fs['qrs_ms']} ms")
        if result.get("clinical_alerts"):
            print(f"\n   Alertes ({len(result['clinical_alerts'])}) :")
            for a in result["clinical_alerts"]:
                print(f"     [{a['severity'].upper()}] {a['label']} — {a['detail']}")
        else:
            print("\n   Aucune alerte clinique.")
    else:
        print(f"❌ Erreur : {result['error']}")
    sys.exit(0)


# ── 2. Pas d'argument → générer une image de test depuis un CSV ───────────────
print("=== Génération d'une image ECG de test depuis un CSV ===\n")

# Chercher un CSV ECG dans les dossiers connus
search_dirs = [
    "data/ecg_csv",
    "data/samples",
    "data/processed_sample",
    ".",
]
csv_path = None
for d in search_dirs:
    found = glob.glob(os.path.join(d, "**/*.csv"), recursive=True)
    if found:
        csv_path = found[0]
        break

if csv_path is None:
    print("❌ Aucun fichier CSV ECG trouvé dans data/. "
          "Passez un chemin image en argument : python test_ecg_image.py chemin/image.png")
    sys.exit(1)

print(f"CSV source : {csv_path}")

# Générer le tracé ECG et le sauvegarder en PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from module2.ecg_predictor import predict_ecg, predict_from_image
from module2.ecg_image_reader import extract_signal_from_image
from training.preprocess_ecg import LEADS, FS, N_SAMPLES, clean_ecg_df

# Lire et nettoyer le signal
df = pd.read_csv(csv_path, sep=None, engine="python")
cleaned = clean_ecg_df(df)   # (5000, 12)

# Générer PNG — format "scan de papier ECG" (6 dérivations)
display_leads = ["I", "II", "III", "aVR", "V1", "V5"]
lead_indices = [LEADS.index(l) for l in display_leads if l in LEADS]
n_leads = len(lead_indices)

display_samples = min(int(2.5 * FS), cleaned.shape[0])   # 2.5 s
t = np.arange(display_samples) / FS

fig, axes = plt.subplots(n_leads, 1, figsize=(14, 10), sharex=True)
fig.patch.set_facecolor("white")

for i, (ax, lead_name, lead_idx) in enumerate(
    zip(axes, display_leads, lead_indices)
):
    sig = cleaned[:display_samples, lead_idx]
    ax.plot(t, sig, color="black", linewidth=1.0)
    ax.set_facecolor("white")
    ax.set_ylabel(lead_name, fontsize=9, rotation=0, labelpad=25)
    ax.grid(True, alpha=0.3, color="lightgray", linewidth=0.5)
    ax.axhline(y=0, color="gray", linewidth=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Temps (s)", fontsize=9)
fig.suptitle("ECG test — image générée depuis CSV", fontsize=11)
plt.tight_layout()

img_out = "test_ecg_generated.png"
plt.savefig(img_out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✅ Image sauvegardée : {img_out}\n")

# ── 3. Tester predict_ecg() sur le CSV
print("--- predict_ecg() (CSV de référence) ---")
res_csv = predict_ecg(csv_path)
if res_csv["success"]:
    print(f"  Diagnostic  : {res_csv['diagnosis_display']}")
    print(f"  Confiance   : {res_csv['confidence']} %\n")

# ── 4. Tester predict_from_image() sur l'image générée
print(f"--- predict_from_image() (image PNG reconstituée) ---")
res_img = predict_from_image(img_out)
if res_img["success"]:
    print(f"  Diagnostic  : {res_img['diagnosis_display']}")
    print(f"  Confiance   : {res_img['confidence']} %")
    print(f"  Source      : {res_img.get('source')}")
    print(f"  Warning     : {res_img.get('image_warning')}")
    fs = res_img["features_summary"]
    print(f"\n  HR          : {fs['hr_mean']} bpm")
    print(f"  RR moyen    : {fs['rr_mean_ms']} ms")
    print(f"  PR          : {fs['pr_ms']} ms")
    print(f"  QRS         : {fs['qrs_ms']} ms")
    if res_img.get("clinical_alerts"):
        print(f"\n  Alertes ({len(res_img['clinical_alerts'])}) :")
        for a in res_img["clinical_alerts"]:
            print(f"    [{a['severity'].upper()}] {a['label']} — {a['detail']}")
else:
    print(f"  ❌ {res_img['error']}")

# ── 5. Comparaison confiance CSV vs Image
if res_csv["success"] and res_img["success"]:
    print(f"\n=== Comparaison ===")
    print(f"  Diagnostic CSV   : {res_csv['diagnosis']} ({res_csv['confidence']} %)")
    print(f"  Diagnostic Image : {res_img['diagnosis']} ({res_img['confidence']} %)")
    match = "✅ Même classe" if res_csv["diagnosis"] == res_img["diagnosis"] else "⚠️  Classes différentes (normal dans les cas limites)"
    print(f"  {match}")
