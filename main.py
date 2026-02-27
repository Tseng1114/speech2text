import os
import threading
import itertools
import time
import config
from google import genai
from google.genai import types
import whisper
import semantic_similarity
from tqdm import tqdm

config.print_config()
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

if config.TRANSCRIPTION_METHOD == "api":
    client = genai.Client(api_key=config.LLM_API_KEY)

elif config.TRANSCRIPTION_METHOD == "local":
    whisper_model = whisper.load_model(config.WHISPER_MODEL, download_root=config.WHISPER_DOWNLOAD_PATH)


def build_api_prompt() -> str:
    audio_lang  = f"The audio is in language code: {config.AUDIO_LANGUAGE}." if config.AUDIO_LANGUAGE else ""
    if config.OUTPUT_LANGUAGE:
        output_lang = f"Transcribe the speech and translate the output into language code: {config.OUTPUT_LANGUAGE}."
    else:
        output_lang = "Transcribe the speech."
    return f"{audio_lang} {output_lang}".strip()


class Spinner:
    def __init__(self, message: str):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop_event.is_set():
                break
            print(f"\r  {ch}  {self.message}", end="", flush=True)
            time.sleep(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        self._thread.join()
        print("\r" + " " * (len(self.message) + 6) + "\r", end="", flush=True)


audio_files = [
    f for f in os.listdir(config.AUDIO_FOLDER)
    if f.lower().endswith((".mp3", ".wav", ".m4a"))
]

if not audio_files:
    print("No audio files found in audio folder.")
    exit()

for filename in audio_files:

    audio_path      = os.path.join(config.AUDIO_FOLDER, filename)
    transcript_file = os.path.join(config.OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")

    if config.TRANSCRIPTION_METHOD == "api":
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        ext = os.path.splitext(filename)[1].lower()
        mime_map  = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4"}
        mime_type = mime_map.get(ext, "audio/mp3")

        with Spinner(f"[{filename}] Transcribing via API..."):
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=[
                    build_api_prompt(),
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
                ],
            )
        transcript_text = response.text

    elif config.TRANSCRIPTION_METHOD == "local":
        task = "translate" if config.OUTPUT_LANGUAGE == "en" else "transcribe"

        with tqdm(desc=f"  [{filename}]", unit=" seg", leave=False, dynamic_ncols=True) as pbar:
            def progress_callback(seek, total):
                pbar.total = total
                pbar.update(seek - pbar.n)

            result = whisper_model.transcribe(
                audio_path,
                fp16=config.WHISPER_FP16,
                language=config.AUDIO_LANGUAGE,
                task=task,
                verbose=False,
            )
        transcript_text = result["text"].strip()

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"  ✓ [{filename}] saved → {transcript_file}")


#this section compares the transcript with the ground truth text
#if you want to test the model, just remove the "#" from the two lines above.
##################################     test model section    ##################################
#   similarity_percent = semantic_similarity.compare(transcript_file, config.GROUND_TRUTH_PATH)
#   print(f"[{filename}] Semantic Similarity: {similarity_percent:.2f}%\n")
##################################     test model section    ##################################