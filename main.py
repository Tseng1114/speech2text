import os
import threading
import itertools
import time

import config
import audio_compression
import semantic_similarity
from transcriber import run_transcription


# --------------------------------------------------------------------------- #
# Spinner
# --------------------------------------------------------------------------- #
class Spinner:
    def __init__(self, message: str):
        self.message = message
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            print(f"\r  {ch}  {self.message}", end="", flush=True)
            time.sleep(0.1)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)
# --------------------------------------------------------------------------- #


def main():
    config.validate_for_cli()
    config.print_config()

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    audio_files = [
        f for f in os.listdir(config.AUDIO_FOLDER)
        if f.lower().endswith((".mp3", ".wav", ".m4a"))
    ]
    if not audio_files:
        print("[-] Error: No audio files found.")
        return

    for filename in audio_files:
        audio_path      = os.path.join(config.AUDIO_FOLDER, filename)
        transcript_file = os.path.join(
            config.OUTPUT_FOLDER,
            f"{os.path.splitext(filename)[0]}.txt",
        )

        label = f"Processing via {config.API_TYPE.upper() if config.TRANSCRIPTION_METHOD == 'api' else 'Whisper'}: {filename}"
        with Spinner(label):
            transcript_text = run_transcription(audio_path, filename)

        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"  ✓ Saved: {transcript_file}")


def test_model():
    if not os.path.exists(config.GROUND_TRUTH_PATH):
        print(f"Similarity Check Skipped: Ground truth not found at {config.GROUND_TRUTH_PATH}")
        return

    print("\n" + "=" * 50)
    print("Starting semantic similarity calculation...")
    print("=" * 50)

    for filename in os.listdir(config.OUTPUT_FOLDER):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(config.OUTPUT_FOLDER, filename)
        score = semantic_similarity.compare(path, config.GROUND_TRUTH_PATH)
        if score is not None:
            print(f"  {filename}: {score:.2f}%")

    print("Similarity calculation completed.")


if __name__ == "__main__":
    main()

    # Uncomment to evaluate transcription accuracy:
    # test_model()

    # Uncomment to compress audio files > 25 MB (Groq limit):
    # audio_compression.compress_audio()
