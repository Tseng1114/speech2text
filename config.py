import os
from dotenv import load_dotenv

load_dotenv()
_base_dir = os.path.dirname(os.path.abspath(__file__))

TRANSCRIPTION_METHOD = "api"      # "api" or "local"
WHISPER_MODEL        = "medium"   # tiny, base, small, medium, large
WHISPER_FP16         = True       # Set to False if running on CPU
WHISPER_DOWNLOAD_PATH = r"path"   # Preferred path for Whisper model storage

AUDIO_LANGUAGE  = "zh"
OUTPUT_LANGUAGE = "en"

AUDIO_FOLDER      = os.path.join(_base_dir, "input_audio")
OUTPUT_FOLDER     = os.path.join(_base_dir, "transcribed_text")
COMPRESSED_FOLDER = os.path.join(_base_dir, "compressed_audio")
GROUND_TRUTH_PATH = os.path.join(_base_dir, "test_model", "ground_truth.txt")

API_TYPE          = "groq"
LLM_API_KEY_gemini = os.getenv("LLM_API_KEY_gemini")
LLM_MODEL_gemini   = "gemini-2.5-flash"
LLM_API_KEY_groq   = os.getenv("LLM_API_KEY_groq")
LLM_MODEL_groq     = "whisper-large-v3-turbo"

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}

_GROQ_VALID_PREFIXES   = ("gsk_",)
_GEMINI_VALID_PREFIXES = ("AIza",)
MAX_UPLOAD_MB = 500


def validate_for_cli() -> None:
    """Hard validation for CLI usage (main.py).

    Raises ValueError immediately so the user sees a clear message.
    Only call this from __main__ entry points, NOT from app.py imports.
    """
    if TRANSCRIPTION_METHOD == "api":
        if API_TYPE == "gemini" and not LLM_API_KEY_gemini:
            raise ValueError(
                "LLM_API_KEY_gemini is not set. "
                "Please add it to your .env file."
            )
        elif API_TYPE == "groq" and not LLM_API_KEY_groq:
            raise ValueError(
                "LLM_API_KEY_groq is not set. "
                "Please add it to your .env file."
            )

    if TRANSCRIPTION_METHOD == "local" and WHISPER_MODEL not in VALID_WHISPER_MODELS:
        raise ValueError(
            f"[config] Invalid WHISPER_MODEL='{WHISPER_MODEL}'. "
            f"Choose from: {sorted(VALID_WHISPER_MODELS)}"
        )


def validate_api_key_format(provider: str, key: str) -> tuple[bool, str]:
    """Return (is_valid, error_message).  Empty key is always invalid."""
    if not key:
        return False, "Key is empty."
    if provider == "groq":
        if not any(key.startswith(p) for p in _GROQ_VALID_PREFIXES):
            return False, "Groq keys should start with 'gsk_'."
    elif provider == "gemini":
        if not any(key.startswith(p) for p in _GEMINI_VALID_PREFIXES):
            return False, "Gemini keys should start with 'AIza'."
    return True, ""


def print_config() -> None:
    print("=" * 50)
    print(f"  Transcription Method : {TRANSCRIPTION_METHOD}")
    if TRANSCRIPTION_METHOD == "api":
        if API_TYPE == "gemini":
            print(f"  LLM Model            : {LLM_MODEL_gemini}")
        elif API_TYPE == "groq":
            print(f"  LLM Model            : {LLM_MODEL_groq}")
    elif TRANSCRIPTION_METHOD == "local":
        print(f"  Whisper Model        : {WHISPER_MODEL}")
        print(f"  Whisper FP16 (GPU)   : {WHISPER_FP16}")
    print(f"  Audio Folder         : {AUDIO_FOLDER}")
    print(f"  Output Folder        : {OUTPUT_FOLDER}")
    print("=" * 50)