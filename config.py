import os
from dotenv import load_dotenv

load_dotenv()
_base_dir = os.path.dirname(os.path.abspath(__file__))

TRANSCRIPTION_METHOD: str = "api"     # change to "api" or "local"
WHISPER_MODEL: str  = "medium"          # change model size if needed, you can choose from: tiny, base, small, medium, large
WHISPER_FP16:  bool = True              # set to False if running on CPU
WHISPER_DOWNLOAD_PATH: str = r"path" # change to your preferred path for storing Whisper models

AUDIO_LANGUAGE:  str | None = "zh"  
OUTPUT_LANGUAGE: str | None = "en"   

AUDIO_FOLDER:      str = os.path.join(_base_dir, "audio_sample")
OUTPUT_FOLDER:     str = os.path.join(_base_dir, "output")
GROUND_TRUTH_PATH: str = os.path.join(_base_dir, "test_model", "ground_truth.txt")

if TRANSCRIPTION_METHOD not in ("api", "local"):
    raise ValueError(
        f"[config] Invalid TRANSCRIPTION_METHOD='{TRANSCRIPTION_METHOD}'. "
        "Please set it to 'api' or 'local'."
    )

LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL:   str = "gemini-2.5-flash"   # change model if needed

if TRANSCRIPTION_METHOD == "api" and not LLM_API_KEY:
    raise ValueError(
        "[config] LLM_API_KEY is not set. "
        "Please add it to your .env file when using TRANSCRIPTION_METHOD=api."
    )

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}
if TRANSCRIPTION_METHOD == "local" and WHISPER_MODEL not in VALID_WHISPER_MODELS:
    raise ValueError(
        f"[config] Invalid WHISPER_MODEL='{WHISPER_MODEL}'. "
        f"Choose from: {sorted(VALID_WHISPER_MODELS)}"
    )


def print_config() -> None:
    print("=" * 50)
    print(f"  Transcription Method : {TRANSCRIPTION_METHOD}")
    if TRANSCRIPTION_METHOD == "api":
        masked_key = LLM_API_KEY[:6] + "..." if LLM_API_KEY else "(not set)"
        print(f"  LLM Model            : {LLM_MODEL}")
        print(f"  LLM API Key          : {masked_key}")
    elif TRANSCRIPTION_METHOD == "local":
        print(f"  Whisper Model        : {WHISPER_MODEL}")
        print(f"  Whisper FP16 (GPU)   : {WHISPER_FP16}")
    print(f"  Audio Folder         : {AUDIO_FOLDER}")
    print(f"  Output Folder        : {OUTPUT_FOLDER}")
    #print(f"  Ground Truth Path    : {GROUND_TRUTH_PATH}")
    print("=" * 50)