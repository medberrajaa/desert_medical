"""
views/print_sheet.py — Générateur de fiche patient imprimable (HTML A4)
========================================================================
Produit une page HTML autonome (inline CSS, zéro dépendance externe)
optimisée pour l'impression A4 portrait (210 × 297 mm).

Usage :
    from views.print_sheet import build_print_sheet
    html_str = build_print_sheet(data)
"""

from __future__ import annotations
from datetime import datetime

# ── Constantes ────────────────────────────────────────────────────────────
_ESTABLISHMENT = "Centre de santé MedIA"

_URGENCY_COLORS = {
    "critique": ("#dc2626", "#fff"),
    "élevée":   ("#ea580c", "#fff"),
    "modérée":  ("#d97706", "#fff"),
    "faible":   ("#16a34a", "#fff"),
    # Libellés produits par le badge d'urgence ECG
    "urgence absolue": ("#dc2626", "#fff"),
    "urgence élevée":  ("#ea580c", "#fff"),
    "urgence modérée": ("#d97706", "#fff"),
    "non urgent":      ("#16a34a", "#fff"),
}


def build_print_sheet(data: dict) -> str:
    """
    Génère le HTML complet d'une fiche patient imprimable.

    Parameters
    ----------
    data : dict avec les clés suivantes (toutes optionnelles sauf diagnosis) :
        date, age, sexe, motif, diagnosis, differentials (list[dict]),
        confidence, hr, rr, pr_ms, qrs_ms, urgency, actions (list[str]),
        module ("ECG" | "LLM"), metrics_extra (dict)

    Returns
    -------
    str — document HTML complet prêt à ouvrir dans un navigateur.
    """
    now = data.get("date", datetime.now().strftime("%d/%m/%Y %H:%M"))
    age = data.get("age", "—")
    sexe = data.get("sexe", "—")
    motif = data.get("motif", "—")
    diagnosis = data.get("diagnosis", "Non disponible")
    confidence = data.get("confidence", "—")
    urgency = data.get("urgency", "modérée")
    module_used = data.get("module", "—")

    # Différentiels
    diffs = data.get("differentials", [])
    diff_rows = ""
    for i, d in enumerate(diffs[:3], 1):
        diff_rows += (
            f"<tr><td>{i}</td>"
            f"<td>{d.get('label', '—')}</td>"
            f"<td>{d.get('confidence', '—')} %</td></tr>"
        )
    if not diff_rows:
        diff_rows = "<tr><td colspan='3' style='text-align:center;color:#888;'>Aucun différentiel significatif</td></tr>"

    # Métriques 2×2
    hr = data.get("hr", "—")
    rr = data.get("rr", "—")
    pr = data.get("pr_ms", "—")
    qrs = data.get("qrs_ms", "—")

    # Urgence badge
    urg_bg, urg_fg = _URGENCY_COLORS.get(urgency.lower(), ("#6b7280", "#fff"))

    # Actions
    actions = data.get("actions", [
        "Surveillance clinique rapprochée",
        "Recontrôler les constantes dans 30 min",
        "Réévaluation par télémédecin si aggravation",
    ])
    action_items = "".join(f"<li>{a}</li>" for a in actions)

    # Vitaux supplémentaires
    vitals_extra = ""
    if data.get("spo2"):
        vitals_extra += f"<span>SpO₂ : {data['spo2']} %</span>"
    if data.get("temperature"):
        vitals_extra += f"<span>  · T° : {data['temperature']} °C</span>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>MedIA — Fiche diagnostic</title>
<style>
@page {{
  size: A4 portrait;
  margin: 12mm 14mm 12mm 14mm;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
  font-size: 10pt;
  color: #1e293b;
  width: 210mm;
  min-height: 297mm;
  padding: 10mm 12mm;
  background: #fff;
}}
@media print {{
  body {{ padding: 0; }}
  .no-print {{ display: none !important; }}
}}

/* Header */
.header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2.5px solid #1e3a5f;
  padding-bottom: 6px;
  margin-bottom: 10px;
}}
.header h1 {{ font-size: 16pt; color: #1e3a5f; }}
.header .meta {{ text-align: right; font-size: 8.5pt; color: #64748b; }}

/* Section */
.section {{
  margin-bottom: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px 10px;
}}
.section h2 {{
  font-size: 10.5pt;
  color: #1e3a5f;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 3px;
  margin-bottom: 6px;
}}

/* Patient info */
.patient-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 3px 12px;
  font-size: 9.5pt;
}}
.patient-grid b {{ color: #334155; }}

/* Diagnosis */
.diagnosis {{
  font-size: 13pt;
  font-weight: 700;
  color: #1e3a5f;
  margin: 4px 0;
}}
.conf {{ font-size: 10pt; color: #475569; }}

/* Diff table */
table {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
th {{ background: #f1f5f9; text-align: left; padding: 3px 6px; }}
td {{ padding: 3px 6px; border-bottom: 1px solid #e2e8f0; }}

/* Metrics 2×2 */
.metrics-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5px;
}}
.metric-card {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  padding: 5px 8px;
  text-align: center;
}}
.metric-card .val {{ font-size: 14pt; font-weight: 700; color: #1e3a5f; }}
.metric-card .lbl {{ font-size: 8pt; color: #64748b; text-transform: uppercase; }}

/* Urgency badge */
.urgency-badge {{
  display: inline-block;
  padding: 5px 18px;
  border-radius: 20px;
  font-size: 11pt;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: {urg_bg};
  color: {urg_fg};
}}

/* Actions */
.actions ul {{
  margin-left: 16px;
  font-size: 9.5pt;
  line-height: 1.6;
}}

/* Footer */
.footer {{
  margin-top: 10px;
  text-align: center;
  font-size: 7.5pt;
  color: #94a3b8;
  border-top: 1px solid #e2e8f0;
  padding-top: 5px;
}}
</style>
</head>
<body onload="window.print()">

<!-- Header -->
<div class="header">
  <h1>🩺 MedIA — Fiche diagnostic</h1>
  <div class="meta">
    {_ESTABLISHMENT}<br>
    {now}<br>
    Module : {module_used}
  </div>
</div>

<!-- Patient -->
<div class="section">
  <h2>👤 Patient</h2>
  <div class="patient-grid">
    <div><b>Âge :</b> {age} ans</div>
    <div><b>Sexe :</b> {sexe}</div>
    <div><b>Module :</b> {module_used}</div>
    <div style="grid-column: span 3;"><b>Motif :</b> {motif}</div>
    {f'<div style="grid-column: span 3;">{vitals_extra}</div>' if vitals_extra else ''}
  </div>
</div>

<!-- Diagnostic -->
<div class="section">
  <h2>🔬 Diagnostic principal</h2>
  <div class="diagnosis">{diagnosis}</div>
  <div class="conf">Confiance du modèle : <b>{confidence} %</b></div>
</div>

<!-- Différentiels -->
<div class="section">
  <h2>📋 Diagnostics différentiels</h2>
  <table>
    <thead><tr><th>#</th><th>Diagnostic</th><th>Confiance</th></tr></thead>
    <tbody>{diff_rows}</tbody>
  </table>
</div>

<!-- Métriques -->
<div class="section">
  <h2>📊 Métriques clés</h2>
  <div class="metrics-grid">
    <div class="metric-card"><div class="val">{hr}</div><div class="lbl">FC (bpm)</div></div>
    <div class="metric-card"><div class="val">{rr}</div><div class="lbl">RR (ms)</div></div>
    <div class="metric-card"><div class="val">{pr}</div><div class="lbl">PR (ms)</div></div>
    <div class="metric-card"><div class="val">{qrs}</div><div class="lbl">QRS (ms)</div></div>
  </div>
</div>

<!-- Urgence -->
<div class="section" style="text-align:center;">
  <h2>🚨 Niveau d'urgence</h2>
  <div style="margin: 6px 0;">
    <span class="urgency-badge">{urgency}</span>
  </div>
</div>

<!-- Conduite à tenir -->
<div class="section actions">
  <h2>📌 Conduite à tenir immédiate</h2>
  <ul>{action_items}</ul>
</div>

<!-- Footer -->
<div class="footer">
  Document généré par MedIA — À valider par un médecin. Non substituable à un avis médical.
</div>

</body>
</html>"""

    return html
