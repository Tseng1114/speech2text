import os
import config
from google import genai
from google.genai import types
import whisper
from groq import Groq
import semantic_similarity
from tqdm import tqdm
import threading
import itertools
import time
import audio_compression

#################### Spinner for the transcription process #####################
class Spinner:
    def __init__(self, message: str):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop_event.is_set(): break
            print(f"\r  {ch}  {self.message}", end="", flush=True)
            time.sleep(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        self._thread.join()
        print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)
#################### Spinner for the transcription process #####################

def build_api_prompt() -> str:
    input_info = f"Input audio language: {config.AUDIO_LANGUAGE}." if config.AUDIO_LANGUAGE else ""
    
    if config.OUTPUT_LANGUAGE:
        instruction = f"Transcribe the audio and translate the final text into language code: {config.OUTPUT_LANGUAGE}."
    else:
        instruction = "Transcribe the audio accurately into text."
    return f"{input_info} {instruction}".strip()

def init_transcription_engine():
    client = None
    model = None
    
    if config.TRANSCRIPTION_METHOD == "api":
        if config.API_TYPE == "gemini":
            client = genai.Client(api_key=config.LLM_API_KEY_gemini)
        elif config.API_TYPE == "groq":
            client = Groq(api_key=config.LLM_API_KEY_groq)
    else:
        print(f"Loading local Whisper model: {config.WHISPER_MODEL}...")
        model = whisper.load_model(config.WHISPER_MODEL, download_root=config.WHISPER_DOWNLOAD_PATH)
        
    return client, model


def main():
    config.print_config()
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    client, whisper_model = init_transcription_engine()
    
    audio_files = [f for f in os.listdir(config.AUDIO_FOLDER) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not audio_files:
        print("[-] Error: No audio files found.")
        return

    for filename in audio_files:
        audio_path = os.path.join(config.AUDIO_FOLDER, filename)
        transcript_file = os.path.join(config.OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
        transcript_text = ""

        if config.TRANSCRIPTION_METHOD == "api":
            with Spinner(f"Processing via {config.API_TYPE.upper()}: {filename}"):
                if config.API_TYPE == "gemini":
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    response = client.models.generate_content(
                        model=config.LLM_MODEL_gemini,
                        contents=[
                            build_api_prompt(),
                            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
                        ],
                    )
                    transcript_text = response.text
                elif config.API_TYPE == "groq":
                    with open(audio_path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=(filename, f.read()),
                            model=config.LLM_MODEL_groq,
                            response_format="text",
                            language=config.AUDIO_LANGUAGE or "zh"
                        )
                        transcript_text = transcription
        else:
            task = "translate" if config.OUTPUT_LANGUAGE == "en" else "transcribe"
            result = whisper_model.transcribe(audio_path, task=task, language=config.AUDIO_LANGUAGE)
            transcript_text = result["text"].strip()

        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f" Processed: {filename}")

def test_model():
    if not os.path.exists(config.GROUND_TRUTH_PATH):
        print(f"Similarity Check Skipped: Ground truth file not found at {config.GROUND_TRUTH_PATH}")
        return

    print("\n" + "="*50)
    print("Starting semantic similarity calculate...")
    print("="*50)

    transcript_files = [f for f in os.listdir(config.OUTPUT_FOLDER) if f.endswith(".txt")]
    
    for filename in transcript_files:
        transcript_path = os.path.join(config.OUTPUT_FOLDER, filename)
        similarity = semantic_similarity.compare(transcript_path, config.GROUND_TRUTH_PATH)
        
        if similarity is not None:
            print(f"{filename}")
            print(f"{similarity:.2f}%")
            print("-" * 20)

    print("Similarity calculation completed.")

if __name__ == "__main__":
    main()

    #Use semantic similarity to evaluate model accuracy 
    #test_model()
    #Compress audio if files > 25MB (Groq's limit)
    #After compression, remember move compressed audio to config.AUDIO_FOLDER
    #audio_compression.compress_audio()