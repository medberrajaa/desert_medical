"""
module1/voice_engine.py — Moteur de reconnaissance vocale pour MedIA
=====================================================================
Utilise l'API Groq Whisper pour la transcription audio (speech-to-text),
puis le LLM pour extraire les champs patients structurés depuis la transcription.

Flux :
    Microphone → fichier audio (WAV/MP3/WebM) → Whisper (Groq) → texte brut
    → LLM (Groq) → champs patient structurés (age, sexe, motif, signes…)

Whisper model : whisper-large-v3-turbo (rapide, multilingue, excellent en français)
"""
import os
from groq import Groq
from dotenv import load_dotenv
from module1.groq_client import DEFAULT_FAST_MODEL, GroqConfigurationError, GroqJSONError, GroqJsonClient
from module1.schemas import VoicePatientFields

load_dotenv()

# Modèle Whisper disponible via Groq
WHISPER_MODEL = "whisper-large-v3-turbo"

# Modèle LLM pour l'extraction (rapide, déjà utilisé dans prompt_engine)
LLM_MODEL = DEFAULT_FAST_MODEL


def _groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqConfigurationError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


def transcribe_audio(audio_path: str, language: str = "fr") -> dict:
    """
    Transcrit un fichier audio via Groq Whisper.

    Parameters
    ----------
    audio_path : str
        Chemin vers le fichier audio (WAV, MP3, M4A, WebM, OGG…)
    language : str
        Code langue ISO 639-1 (défaut : "fr" pour français)

    Returns
    -------
    dict :
        success : bool
        transcript : str   (texte transcrit)
        error    : str   (si success=False)
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "error": "Aucun fichier audio fourni."}

    try:
        with open(audio_path, "rb") as audio_file:
            transcription = _groq_client().audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language=language,
                response_format="text",
            )
        transcript = str(transcription).strip()
        if not transcript:
            return {"success": False, "error": "Transcription vide — parlez plus fort ou réessayez."}
        return {"success": True, "transcript": transcript}

    except Exception as e:
        return {"success": False, "error": f"Erreur de transcription : {e}"}


# ── Prompt de parsing structuré ───────────────────────────────────────────────

_PARSE_SYSTEM = (
    "Extract structured patient fields from a nurse voice transcript. "
    "Return JSON only. Do not infer absent facts; use null when information is not mentioned. "
    "Convert ages stated in words to integers when clear. Put vital signs in 'signes'. "
    "Use sexe values only: Homme, Femme, Non precise."
)


def parse_transcript_to_patient(transcript: str) -> dict:
    """
    Utilise le LLM pour extraire les champs patient structurés depuis la transcription.

    Parameters
    ----------
    transcript : str
        Texte brut transcrit depuis la voix

    Returns
    -------
    dict :
        success : bool
        fields  : dict avec les champs patient extraits
        raw_json : str (JSON brut retourné par le LLM)
        error   : str (si success=False)
    """
    if not transcript or not transcript.strip():
        return {"success": False, "error": "Transcription vide."}

    prompt = (
        "Voice transcript from a nurse describing a patient. Extract the fields into the required schema.\n\n"
        f"{transcript.strip()}"
    )

    try:
        response = GroqJsonClient().call_json(
            model=LLM_MODEL,
            schema_model=VoicePatientFields,
            schema_name="voice_patient_fields",
            messages=[
                {"role": "system", "content": _PARSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        fields = response.data.model_dump(mode="json")
        return {"success": True, "fields": fields, "raw_json": response.raw_text}

    except (GroqConfigurationError, GroqJSONError, Exception) as e:
        return {"success": False, "error": f"Erreur d'analyse LLM : {e}"}


def transcribe_and_parse(audio_path: str, language: str = "fr") -> dict:
    """
    Pipeline complet : audio → transcription → extraction des champs patient.

    Returns
    -------
    dict :
        success     : bool
        transcript  : str   (texte transcrit)
        fields      : dict  (champs patient extraits)
        error       : str   (si success=False)
    """
    # 1. Transcription Whisper
    trans_result = transcribe_audio(audio_path, language)
    if not trans_result["success"]:
        return trans_result

    transcript = trans_result["transcript"]

    # 2. Parsing LLM
    parse_result = parse_transcript_to_patient(transcript)
    if not parse_result["success"]:
        return {
            "success": False,
            "transcript": transcript,
            "error": parse_result["error"],
        }

    return {
        "success": True,
        "transcript": transcript,
        "fields": parse_result["fields"],
    }
