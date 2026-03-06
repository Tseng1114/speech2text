import io
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types as genai_types
from pydub import AudioSegment
from pydantic import BaseModel

import config
import semantic_similarity
from main import build_api_prompt, init_transcription_engine

app = FastAPI(title="Audio Transcription API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "frontend"), name="static")

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a"}
jobs: dict[str, dict] = {}


class ConfigUpdate(BaseModel):
    transcription_method:  Optional[str] = None
    api_type:              Optional[str] = None
    audio_language:        Optional[str] = None
    output_language:       Optional[str] = None
    whisper_model:         Optional[str] = None
    whisper_download_path: Optional[str] = None
    groq_model:            Optional[str] = None
    gemini_model:          Optional[str] = None

class ApiKeysUpdate(BaseModel):
    groq:   Optional[str] = None
    gemini: Optional[str] = None


def _run_transcription(job_id: str, audio_path: str, filename: str):
    jobs[job_id]["status"] = "processing"
    try:
        client, whisper_model = init_transcription_engine()

        if config.TRANSCRIPTION_METHOD == "api":
            if config.API_TYPE == "gemini":
                response = client.models.generate_content(
                    model=config.LLM_MODEL_gemini,
                    contents=[
                        build_api_prompt(),
                        genai_types.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/mp3"),
                    ],
                )
                text = response.text
            else:
                with open(audio_path, "rb") as f:
                    text = client.audio.transcriptions.create(
                        file=(filename, f.read()),
                        model=config.LLM_MODEL_groq,
                        response_format="text",
                        language=config.AUDIO_LANGUAGE or "zh",
                    )
        else:
            task = "translate" if config.OUTPUT_LANGUAGE == "en" else "transcribe"
            text = whisper_model.transcribe(audio_path, task=task, language=config.AUDIO_LANGUAGE)["text"].strip()

        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
        out_path = Path(config.OUTPUT_FOLDER) / f"{Path(filename).stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        jobs[job_id].update(status="done", result=text, output_file=str(out_path))

    except Exception as e:
        jobs[job_id].update(status="error", error=_clean_error(e))
    finally:
        Path(audio_path).unlink(missing_ok=True)


def _clean_error(e: Exception) -> str:
    import ast, re as _re
    raw = str(e)

    auth_pat    = _re.compile(r'401|403|invalid.api.key|api.key.not.valid|unauthenticated', _re.I)
    ratelimit_pat = _re.compile(r'429|rate.?limit', _re.I)
    toobig_pat  = _re.compile(r'413|too.large|file.size|exceeds', _re.I)
    timeout_pat = _re.compile(r'timeout|timed.out', _re.I)
    quota_pat   = _re.compile(r'quota|billing|insufficient', _re.I)

    if auth_pat.search(raw):
        return "API key is invalid or incorrect. Please re-enter it in Settings."
    if ratelimit_pat.search(raw):
        return "Rate limit exceeded. Please wait a moment and try again."
    if toobig_pat.search(raw):
        return "File is too large. Please compress the audio first."
    if timeout_pat.search(raw):
        return "Request timed out. Please check your connection and try again."
    if quota_pat.search(raw):
        return "API quota exceeded or billing issue. Please check your account."

    try:
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            obj = ast.literal_eval(m.group())
            msg = (obj.get("error") or {}).get("message") or obj.get("message") or obj.get("detail")
            if msg:
                return msg
    except Exception:
        pass

    return _re.sub(r'^Error code:\s*\d+\s*-\s*', '', raw).strip()


def _mask(val: str) -> str:
    if not val:
        return ""
    return (val[:4] + "•" * min(len(val) - 4, 20)) if len(val) > 4 else "•" * len(val)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse((Path(__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8"))


@app.get("/config")
def get_config():
    return {
        "transcription_method":  config.TRANSCRIPTION_METHOD,
        "api_type":              config.API_TYPE,
        "audio_language":        config.AUDIO_LANGUAGE,
        "output_language":       config.OUTPUT_LANGUAGE,
        "whisper_model":         config.WHISPER_MODEL,
        "whisper_download_path": config.WHISPER_DOWNLOAD_PATH,
        "valid_whisper_models":  sorted(config.VALID_WHISPER_MODELS),
        "llm_model_gemini":      config.LLM_MODEL_gemini,
        "llm_model_groq":        config.LLM_MODEL_groq,
    }

@app.post("/config")
def update_config(body: ConfigUpdate):
    mapping = {
        "transcription_method": "TRANSCRIPTION_METHOD", "api_type": "API_TYPE",
        "audio_language":       "AUDIO_LANGUAGE",       "output_language": "OUTPUT_LANGUAGE",
        "whisper_model":        "WHISPER_MODEL",         "whisper_download_path": "WHISPER_DOWNLOAD_PATH",
        "groq_model":           "LLM_MODEL_groq",        "gemini_model": "LLM_MODEL_gemini",
    }
    for field, attr in mapping.items():
        if (val := getattr(body, field)) is not None:
            setattr(config, attr, val)
    return {"ok": True, "updated": body.model_dump(exclude_none=True)}


@app.get("/apikeys")
def get_apikeys():
    return {
        "groq_masked":   _mask(config.LLM_API_KEY_groq   or ""),
        "gemini_masked": _mask(config.LLM_API_KEY_gemini or ""),
        "groq_set":      bool(config.LLM_API_KEY_groq),
        "gemini_set":    bool(config.LLM_API_KEY_gemini),
    }

@app.post("/apikeys")
def update_apikeys(body: ApiKeysUpdate):
    if body.groq   is not None: config.LLM_API_KEY_groq   = body.groq
    if body.gemini is not None: config.LLM_API_KEY_gemini = body.gemini
    return {"ok": True}


@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    os.makedirs(config.AUDIO_FOLDER, exist_ok=True)
    temp_path = Path(config.AUDIO_FOLDER) / f"_tmp_{int(time.time()*1000)}{ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = f"job_{int(time.time()*1000)}"
    jobs[job_id] = {"status": "queued", "filename": file.filename, "result": None, "error": None}
    background_tasks.add_task(_run_transcription, job_id, str(temp_path), file.filename)
    return {"job_id": job_id}

@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/transcripts")
def list_transcripts():
    folder = Path(config.OUTPUT_FOLDER)
    if not folder.is_dir():
        return {"files": []}
    return {"files": sorted(f.name for f in folder.glob("*.txt"))}

@app.get("/transcripts/{filename}")
def get_transcript(filename: str):
    path = Path(config.OUTPUT_FOLDER) / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}

@app.get("/transcripts/{filename}/download")
def download_transcript(filename: str):
    path = Path(config.OUTPUT_FOLDER) / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="text/plain", filename=filename)


@app.post("/compress/batch")
async def compress_batch(files: List[UploadFile] = File(...)):
    tmp_dir = Path(tempfile.mkdtemp(prefix="compress_"))
    try:
        results = []
        for upload in files:
            ext = Path(upload.filename).suffix.lower()
            if ext not in ALLOWED_AUDIO:
                raise HTTPException(400, f"Unsupported file type '{ext}'")
            in_path  = tmp_dir / f"in_{upload.filename}"
            out_name = f"compressed_{Path(upload.filename).stem}.mp3"
            out_path = tmp_dir / out_name
            with open(in_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            audio = AudioSegment.from_file(in_path)
            audio.set_channels(1).set_frame_rate(16000).export(out_path, format="mp3", bitrate="48k", codec="libmp3lame")
            results.append({
                "out_path": out_path, "out_name": out_name, "orig_name": upload.filename,
                "orig_mb": round(in_path.stat().st_size  / 1024 / 1024, 2),
                "comp_mb": round(out_path.stat().st_size / 1024 / 1024, 2),
            })

        if len(results) == 1:
            r = results[0]
            data = r["out_path"].read_bytes()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg", headers={
                "Content-Disposition": f'attachment; filename="{r["out_name"]}"',
                "X-Original-MB":   str(r["orig_mb"]),
                "X-Compressed-MB": str(r["comp_mb"]),
                "X-Original-Name": r["orig_name"],
            })

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in results:
                zf.write(r["out_path"], arcname=r["out_name"])
        buf.seek(0)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return StreamingResponse(buf, media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="compressed_audio.zip"'})

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.post("/evaluate")
async def evaluate(transcript: UploadFile = File(...), ground_truth: UploadFile = File(...)):
    for upload in (transcript, ground_truth):
        if not upload.filename.endswith(".txt"):
            raise HTTPException(400, f"'{upload.filename}' is not a .txt file")
    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_"))
    try:
        t_path  = tmp_dir / "transcript.txt"
        gt_path = tmp_dir / "ground_truth.txt"
        for upload, path in [(transcript, t_path), (ground_truth, gt_path)]:
            with open(path, "wb") as f:
                shutil.copyfileobj(upload.file, f)
        score = semantic_similarity.compare(str(t_path), str(gt_path))
        return {"transcript": transcript.filename, "ground_truth": ground_truth.filename,
                "similarity_percent": round(score, 2)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)