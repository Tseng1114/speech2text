import whisper
import config

DOWNLOAD_PATH = config.WHISPER_DOWNLOAD_PATH  
whisper.load_model(config.WHISPER_MODEL, download_root=DOWNLOAD_PATH)

print(f"Download complete! Model saved to {DOWNLOAD_PATH}")