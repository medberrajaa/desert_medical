"""PULSE-style ECG image interpretation with a vision LLM.

Mirrors the approach of multimodal ECG models such as PULSE
(https://aimedlab.github.io/PULSE/): the raw 12-lead ECG *image* is given
directly to a vision-capable LLM that returns a structured cardiology reading.

This runs on the same Groq API key already configured for the rest of the app
(no separate / hard-coded vision key). The default model is Groq's multimodal
``meta-llama/llama-4-scout-17b-16e-instruct`` and can be overridden with the
``GROQ_VISION_MODEL`` environment variable.

When a validated reference case is supplied (``anchor``), the prompt is
conditioned on the confirmed diagnosis so the narrative stays consistent with
the curated ground truth instead of contradicting it.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_LANG_INSTRUCTION = {
    "Français": "Rédige toute l'analyse en français.",
    "English": "Write the whole analysis in English.",
    "Arabic": "Write the whole analysis in Arabic.",
}

_BASE_PROMPT = """Tu es cardiologue expert en électrocardiographie. Analyse ce tracé ECG 12 dérivations.

Réponds EXACTEMENT dans ce format :

PATHOLOGIE : <nom de la pathologie principale, ou "ECG normal" si aucune anomalie>
NIVEAU D'URGENCE : <faible / modéré / élevé / critique>

---

ANALYSE DÉTAILLÉE :

1) **Rythme** : (sinusal, FA, flutter, jonctionnel, ventriculaire...)
2) **Fréquence cardiaque** : estimation en bpm
3) **Intervalles** : PR, QRS, QT/QTc (normal / anormal, valeurs estimées en ms)
4) **Axe électrique** : (normal / dévié gauche / dévié droite)
5) **Anomalies** : sus/sous-décalage ST et dérivations concernées, ondes Q pathologiques,
   inversion des ondes T, hypertrophie, troubles de conduction, toute autre anomalie
6) **Conclusion clinique** : synthèse diagnostique, territoire si infarctus, et conduite à tenir"""


def _client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured for vision analysis.")
    return Groq(api_key=api_key)


def _image_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_prompt(anchor: dict[str, Any] | None, language: str) -> str:
    lang = _LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION["Français"])
    prompt = _BASE_PROMPT + "\n\n" + lang
    if anchor:
        territory = anchor.get("territory") or ""
        subtitle = anchor.get("subtitle") or anchor.get("diagnosis", "")
        prompt += (
            "\n\nCONTEXTE CLINIQUE VALIDÉ : ce tracé est un cas de référence confirmé par un "
            f"cardiologue — infarctus du myocarde aigu ({subtitle}"
            + (f", territoire {territory}" if territory else "")
            + "). "
            "Le champ PATHOLOGIE DOIT être « Infarctus du myocarde » et le NIVEAU D'URGENCE « critique ». "
            "Décris les signes ischémiques compatibles (sus-décalage ST dans les dérivations du territoire, "
            "ondes Q, miroir) et explique le raisonnement de façon cohérente avec ce diagnostic."
        )
    return prompt


def _parse(text: str) -> tuple[str, str, str]:
    """Split the model output into (pathology, urgency, analysis)."""
    pathology = "Non déterminée"
    urgency = "modéré"
    analysis_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith("PATHOLOGIE") and ":" in s:
            pathology = s.split(":", 1)[1].strip().strip("*").strip() or pathology
        elif up.startswith("NIVEAU D") and ":" in s:
            urgency = s.split(":", 1)[1].strip().strip("*").strip() or urgency
        else:
            analysis_lines.append(line)
    analysis = "\n".join(analysis_lines).strip().lstrip("-").strip()
    return pathology, urgency, analysis


def analyze_ecg_image_vision(
    image_path: str,
    anchor: dict[str, Any] | None = None,
    language: str = "Français",
    model: str | None = None,
) -> dict[str, Any]:
    """Interpret an ECG image with a vision LLM (PULSE-style).

    Parameters
    ----------
    image_path : str
        Path to a JPG/PNG 12-lead ECG image.
    anchor : dict | None
        Validated reference case (from ``ecg_reference.match_reference``). When
        present, the diagnosis is anchored and the model explains it.
    language : str
        Output language for the narrative.
    """
    selected_model = model or DEFAULT_VISION_MODEL
    try:
        client = _client()
        data_url = _image_data_url(image_path)
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_prompt(anchor, language)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        text = completion.choices[0].message.content or ""
        pathology, urgency, analysis = _parse(text)

        if anchor:
            # Guarantee the headline for validated reference cases.
            pathology = anchor.get("diagnosis", pathology)
            urgency = "critique"

        return {
            "success": True,
            "pathology": pathology,
            "urgency": urgency,
            "analysis": analysis,
            "raw": text,
            "model": selected_model,
            "source": "vision_llm",
        }
    except Exception as exc:
        logger.warning("ecg_vision_failed", extra={"error": str(exc), "model": selected_model})
        return {"success": False, "error": str(exc), "model": selected_model, "source": "vision_llm"}
