import os
from dotenv import load_dotenv

load_dotenv()
_base_dir = os.path.dirname(os.path.abspath(__file__))

TRANSCRIPTION_METHOD = "api"      #Change to "api" or "local"
WHISPER_MODEL = "medium"          #Change model size if needed, you can choose from: tiny, base, small, medium, large
WHISPER_FP16 = True               #Set to False if running on CPU
WHISPER_DOWNLOAD_PATH = r"path"   #Change to your preferred path for storing Whisper models

AUDIO_LANGUAGE = "zh"  
OUTPUT_LANGUAGE = "en"   

AUDIO_FOLDER= os.path.join(_base_dir, "input_audio")
OUTPUT_FOLDER= os.path.join(_base_dir, "transcribed_text")
COMPRESSED_FOLDER=os.path.join(_base_dir,"compressed_audio")
GROUND_TRUTH_PATH= os.path.join(_base_dir, "test_model", "ground_truth.txt")

API_TYPE = "groq"                                                           #Groq or Gemini
LLM_API_KEY_gemini= os.getenv("LLM_API_KEY_gemini")
LLM_MODEL_gemini= "gemini-2.5-flash"                                        #Change model if needed
LLM_API_KEY_groq= os.getenv("LLM_API_KEY_groq")
LLM_MODEL_groq= "whisper-large-v3-turbo"                                    #whisper-large-v3、whisper-large-v3-turbo                                    #Change model if needed
VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"} 

##################### API keys configuration check section #####################
if TRANSCRIPTION_METHOD == "api" :
    if API_TYPE == "gemini" and not LLM_API_KEY_gemini:
        raise ValueError(
            "LLM_API_KEY_gemini is not set. "
            "Please add it to your .env file when using TRANSCRIPTION_METHOD=api."
        )
    elif API_TYPE == "groq" and not LLM_API_KEY_groq:
                raise ValueError(
            "LLM_API_KEY_groq is not set. "
            "Please add it to your .env file when using TRANSCRIPTION_METHOD=api."
        )
##################### API keys configuration check section #####################

############### local whisper models configuration check section ###############
if TRANSCRIPTION_METHOD == "local" and WHISPER_MODEL not in VALID_WHISPER_MODELS:
    raise ValueError(
        f"[config] Invalid WHISPER_MODEL='{WHISPER_MODEL}'. "
        f"Choose from: {sorted(VALID_WHISPER_MODELS)}"
    )
############### local whisper models configuration check section ###############

def print_config() -> None:
    print("=" * 50)
    print(f"  Transcription Method : {TRANSCRIPTION_METHOD}")
    if TRANSCRIPTION_METHOD == "api":
        if API_TYPE == "gemini":
            print(f"  LLM Model            : {LLM_MODEL_gemini}")
        elif API_TYPE == "groq":
            print(f"  LLM Model          : {LLM_MODEL_groq}")
    elif TRANSCRIPTION_METHOD == "local":
        print(f"  Whisper Model        : {WHISPER_MODEL}")
        print(f"  Whisper FP16 (GPU)   : {WHISPER_FP16}")
    print(f"  Audio Folder         : {AUDIO_FOLDER}")
    print(f"  Output Folder        : {OUTPUT_FOLDER}")
    #print(f"  Ground Truth Path    : {GROUND_TRUTH_PATH}")
    print("=" * 50)