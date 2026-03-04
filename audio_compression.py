import os
import config
from pydub import AudioSegment

def compress_audio():
    audio_files = [f for f in os.listdir(config.AUDIO_FOLDER) 
                   if f.lower().endswith(('.mp3', '.wav', '.m4a'))]

    if not audio_files:
        print(f"No audio files found in: {config.AUDIO_FOLDER}")
        return

    os.makedirs(config.COMPRESSED_FOLDER, exist_ok=True)

    print(f"Found {len(audio_files)} file(s). Starting compression...")

    for filename in audio_files:
        input_path = os.path.join(config.AUDIO_FOLDER, filename)
        output_filename = f"compressed_{os.path.splitext(filename)[0]}.mp3"
        output_path = os.path.join(config.COMPRESSED_FOLDER, output_filename)

        print(f"Processing: {filename}...")

        try:
            audio = AudioSegment.from_file(input_path)

            compressed = audio.set_channels(1).set_frame_rate(16000)

            compressed.export(
                output_path, 
                format="mp3", 
                bitrate="48k",
                codec="libmp3lame"
            )

            original_size = os.path.getsize(input_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"Saved to: {output_filename}")
            print(f"{original_size:.2f} MB -> {new_size:.2f} MB")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("All audio compression completed.")

if __name__ == "__main__":
    compress_audio()