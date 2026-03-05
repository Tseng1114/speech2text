import os
import shutil
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config

app = FastAPI(title="Audio Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "templates"), name="static")

jobs: dict[str, dict] = {}

class ConfigUpdate(BaseModel):
    transcription_method: Optional[str] = None
    api_type: Optional[str] = None
    audio_language: Optional[str] = None
    output_language: Optional[str] = None
    whisper_model: Optional[str] = None
    whisper_download_path: Optional[str] = None
    groq_model: Optional[str] = None
    gemini_model: Optional[str] = None

class ApiKeysUpdate(BaseModel):
    groq: Optional[str] = None
    gemini: Optional[str] = None

def _run_transcription(job_id: str, audio_path: str, filename: str):
    jobs[job_id]["status"] = "processing"
    try:
        from main import init_transcription_engine, build_api_prompt
        from google.genai import types

        if config.TRANSCRIPTION_METHOD == "api":
            if config.API_TYPE == "gemini" and not config.LLM_API_KEY_gemini:
                raise ValueError("No Gemini API Key detected. Please enter it in the settings first.")
            if config.API_TYPE == "groq" and not config.LLM_API_KEY_groq:
                raise ValueError("No Groq API Key detected. Please enter it in the settings first.")

        client, whisper_model = init_transcription_engine()
        transcript_text = ""

        if config.TRANSCRIPTION_METHOD == "api":
            if config.API_TYPE == "gemini":
                response = client.models.generate_content(...) 
                transcript_text = response.text
            elif config.API_TYPE == "groq":
                transcript_text = transcription
        else:
            task = "translate" if config.OUTPUT_LANGUAGE == "en" else "transcribe"
            result = whisper_model.transcribe(
                audio_path, task=task, language=config.AUDIO_LANGUAGE
            )
            transcript_text = result["text"].strip()


    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/config")
def get_config():
    return {
        "transcription_method": config.TRANSCRIPTION_METHOD,
        "api_type": config.API_TYPE,
        "audio_language": config.AUDIO_LANGUAGE,
        "output_language": config.OUTPUT_LANGUAGE,
        "whisper_model": config.WHISPER_MODEL,
        "whisper_download_path": config.WHISPER_DOWNLOAD_PATH,
        "valid_whisper_models": sorted(config.VALID_WHISPER_MODELS),
        "llm_model_gemini": config.LLM_MODEL_gemini,
        "llm_model_groq": config.LLM_MODEL_groq,
    }


@app.post("/config")
def update_config(body: ConfigUpdate):
    mapping = {
        "transcription_method": "TRANSCRIPTION_METHOD",
        "api_type": "API_TYPE",
        "audio_language": "AUDIO_LANGUAGE",
        "output_language": "OUTPUT_LANGUAGE",
        "whisper_model": "WHISPER_MODEL",
        "whisper_download_path": "WHISPER_DOWNLOAD_PATH",
        "groq_model": "LLM_MODEL_groq",
        "gemini_model": "LLM_MODEL_gemini",
    }
    for field, attr in mapping.items():
        val = getattr(body, field, None)
        if val is not None:
            setattr(config, attr, val)
    return {"ok": True, "updated": body.model_dump(exclude_none=True)}


@app.get("/apikeys")
def get_apikeys():
    def mask(val: str) -> str:
        if not val:
            return ""
        return val[:4] + "•" * min(len(val) - 4, 20) if len(val) > 4 else "•" * len(val)

    return {
        "groq_masked": mask(config.LLM_API_KEY_groq or ""),
        "gemini_masked": mask(config.LLM_API_KEY_gemini or ""),
        "groq_set": bool(config.LLM_API_KEY_groq),
        "gemini_set": bool(config.LLM_API_KEY_gemini),
    }


@app.post("/apikeys")
def update_apikeys(body: ApiKeysUpdate):
    if body.groq is not None:
        config.LLM_API_KEY_groq = body.groq
    if body.gemini is not None:
        config.LLM_API_KEY_gemini = body.gemini
    return {"ok": True}


@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if config.TRANSCRIPTION_METHOD == "api":
        if config.API_TYPE == "gemini" and not config.LLM_API_KEY_gemini:
            raise HTTPException(status_code=400, detail="Please set your Gemini API Key first.")
        if config.API_TYPE == "groq" and not config.LLM_API_KEY_groq:
            raise HTTPException(status_code=400, detail="Please set your Groq API Key first.")


@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/transcripts")
def list_transcripts():
    folder = config.OUTPUT_FOLDER
    if not os.path.isdir(folder):
        return {"files": []}
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    return {"files": sorted(files)}


@app.get("/transcripts/{filename}")
def get_transcript(filename: str):
    path = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content}


@app.get("/transcripts/{filename}/download")
def download_transcript(filename: str):
    path = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="text/plain", filename=filename)


@app.post("/compress/batch")
async def compress_batch(files: List[UploadFile] = File(...)):
    from pydub import AudioSegment
    from fastapi.responses import StreamingResponse
    import io

    allowed = {".mp3", ".wav", ".m4a"}
    tmp_dir = tempfile.mkdtemp(prefix="audioscript_compress_")

    try:
        compressed_paths = []

        for upload in files:
            ext = Path(upload.filename).suffix.lower()
            if ext not in allowed:
                raise HTTPException(400, f"Unsupported file type '{ext}' in '{upload.filename}'")

            in_path = os.path.join(tmp_dir, f"in_{upload.filename}")
            with open(in_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            stem = Path(upload.filename).stem
            out_filename = f"compressed_{stem}.mp3"
            out_path = os.path.join(tmp_dir, out_filename)

            audio = AudioSegment.from_file(in_path)
            compressed = audio.set_channels(1).set_frame_rate(16000)
            compressed.export(out_path, format="mp3", bitrate="48k", codec="libmp3lame")

            original_mb   = round(os.path.getsize(in_path)  / 1024 / 1024, 2)
            compressed_mb = round(os.path.getsize(out_path) / 1024 / 1024, 2)
            compressed_paths.append({
                "out_path": out_path,
                "out_filename": out_filename,
                "original_filename": upload.filename,
                "original_mb": original_mb,
                "compressed_mb": compressed_mb,
            })

        if len(compressed_paths) == 1:
            item = compressed_paths[0]
            data = open(item["out_path"], "rb").read()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return StreamingResponse(
                io.BytesIO(data),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="{item["out_filename"]}"',
                    "X-Original-MB":   str(item["original_mb"]),
                    "X-Compressed-MB": str(item["compressed_mb"]),
                    "X-Original-Name": item["original_filename"],
                },
            )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in compressed_paths:
                zf.write(item["out_path"], arcname=item["out_filename"])
        zip_buf.seek(0)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="compressed_audio.zip"'},
        )

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.post("/evaluate")
async def evaluate(
    transcript:   UploadFile = File(...),
    ground_truth: UploadFile = File(...),
):
    import semantic_similarity

    for upload in (transcript, ground_truth):
        if not upload.filename.endswith(".txt"):
            raise HTTPException(400, f"'{upload.filename}' is not a .txt file")

    tmp_dir = tempfile.mkdtemp(prefix="audioscript_eval_")
    try:
        t_path  = os.path.join(tmp_dir, "transcript.txt")
        gt_path = os.path.join(tmp_dir, "ground_truth.txt")

        for upload, path in [(transcript, t_path), (ground_truth, gt_path)]:
            with open(path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

        score = semantic_similarity.compare(t_path, gt_path)
        return {
            "transcript":        transcript.filename,
            "ground_truth":      ground_truth.filename,
            "similarity_percent": round(score, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)