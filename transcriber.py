import os
from pathlib import Path
from typing import Optional

import config

_client       = None
_whisper_model = None
_engine_key: tuple | None = None 


def _current_key() -> tuple:
    return (
        config.TRANSCRIPTION_METHOD,
        config.API_TYPE,
        config.LLM_API_KEY_groq,
        config.LLM_API_KEY_gemini,
        config.WHISPER_MODEL,
        config.WHISPER_DOWNLOAD_PATH,
    )


def reset_engine() -> None:
    global _client, _whisper_model, _engine_key
    _client        = None
    _whisper_model = None
    _engine_key    = None


def _get_engine():
    global _client, _whisper_model, _engine_key

    key = _current_key()
    if key == _engine_key:
        return _client, _whisper_model

    _client        = None
    _whisper_model = None

    if config.TRANSCRIPTION_METHOD == "api":
        if config.API_TYPE == "gemini":
            from google import genai
            _client = genai.Client(api_key=config.LLM_API_KEY_gemini)
        elif config.API_TYPE == "groq":
            from groq import Groq
            _client = Groq(api_key=config.LLM_API_KEY_groq)
    else:
        import whisper
        print(f"Loading local Whisper model: {config.WHISPER_MODEL}...")
        _whisper_model = whisper.load_model(
            config.WHISPER_MODEL,
            download_root=config.WHISPER_DOWNLOAD_PATH,
        )

    _engine_key = key
    return _client, _whisper_model

def build_api_prompt() -> str:
    input_info = f"Input audio language: {config.AUDIO_LANGUAGE}." if config.AUDIO_LANGUAGE else ""
    if config.OUTPUT_LANGUAGE:
        instruction = (
            f"Transcribe the audio and translate the final text into "
            f"language code: {config.OUTPUT_LANGUAGE}."
        )
    else:
        instruction = "Transcribe the audio accurately into text."
    return f"{input_info} {instruction}".strip()



def run_transcription(audio_path: str, filename: str) -> str:
    client, whisper_model = _get_engine()

    if config.TRANSCRIPTION_METHOD == "api":
        if config.API_TYPE == "gemini":
            from google.genai import types as genai_types
            response = client.models.generate_content(
                model=config.LLM_MODEL_gemini,
                contents=[
                    build_api_prompt(),
                    genai_types.Part.from_bytes(
                        data=Path(audio_path).read_bytes(),
                        mime_type="audio/mp3",
                    ),
                ],
            )
            return response.text

        elif config.API_TYPE == "groq":
            with open(audio_path, "rb") as f:
                return client.audio.transcriptions.create(
                    file=(filename, f.read()),
                    model=config.LLM_MODEL_groq,
                    response_format="text",
                    language=config.AUDIO_LANGUAGE or "zh",
                )

    else:
        task = "translate" if config.OUTPUT_LANGUAGE == "en" else "transcribe"
        result = whisper_model.transcribe(
            audio_path,
            task=task,
            language=config.AUDIO_LANGUAGE,
        )
        return result["text"].strip()