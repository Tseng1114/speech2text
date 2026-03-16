import ast
import io
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from pydantic import BaseModel

import config
import semantic_similarity
from transcriber import run_transcription, reset_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

app = FastAPI(title="Audio Transcription API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "frontend"),
    name="static",
)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a"}
jobs: dict[str, dict] = {}
JOB_TIMEOUT_SECONDS = 300
MAX_JOBS = 200

class ConfigUpdate(BaseModel):
    transcription_method:  str | None = None
    api_type:              str | None = None
    audio_language:        str | None = None
    output_language:       str | None = None
    whisper_model:         str | None = None
    whisper_download_path: str | None = None
    groq_model:            str | None = None
    gemini_model:          str | None = None


class ApiKeysUpdate(BaseModel):
    groq:   str | None = None
    gemini: str | None = None

def _mask(val: str) -> str:
    if not val:
        return ""
    return (val[:4] + "•" * min(len(val) - 4, 20)) if len(val) > 4 else "•" * len(val)


_ERROR_PATTERNS = [
    (re.compile(r"401|403|invalid.api.key|api.key.not.valid|unauthenticated", re.I),
     "API key is invalid or incorrect. Please re-enter it in Settings."),
    (re.compile(r"429|rate.?limit", re.I),
     "Rate limit exceeded. Please wait a moment and try again."),
    (re.compile(r"413|too.large|file.size|exceeds", re.I),
     "File is too large. Please compress the audio first."),
    (re.compile(r"timeout|timed.out", re.I),
     "Request timed out. Please check your connection and try again."),
    (re.compile(r"quota|billing|insufficient", re.I),
     "API quota exceeded or billing issue. Please check your account."),
]


def _clean_error(e: Exception) -> str:
    raw = str(e)
    for pat, msg in _ERROR_PATTERNS:
        if pat.search(raw):
            return msg
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            obj = ast.literal_eval(m.group())
            msg = (obj.get("error") or {}).get("message") or obj.get("message") or obj.get("detail")
            if msg:
                return msg
    except Exception:
        pass
    return re.sub(r"^Error code:\s*\d+\s*-\s*", "", raw).strip()


def _check_job_timeout(job: dict) -> None:
    if (
        job.get("status") == "processing"
        and job.get("started_at")
        and (time.time() - job["started_at"]) > JOB_TIMEOUT_SECONDS
    ):
        job["status"] = "error"
        job["error"]  = "Transcription timed out. Please try again."


def _prune_jobs() -> None:
    if len(jobs) <= MAX_JOBS:
        return
    finished = [k for k, v in jobs.items() if v.get("status") in ("done", "error")]
    n = len(finished) // 2
    for k in finished[:n]:
        del jobs[k]
    logger.info("Pruned %d old jobs (total now: %d)", n, len(jobs))


def _safe_transcript_path(filename: str) -> Path:
    base = Path(config.OUTPUT_FOLDER).resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base):
        raise HTTPException(400, "Invalid filename.")
    return path

def _run_transcription(job_id: str, audio_path: str, filename: str) -> None:
    jobs[job_id]["status"]     = "processing"
    jobs[job_id]["started_at"] = time.time()
    logger.info("Job %s started  (%s)", job_id, filename)
    try:
        text = run_transcription(audio_path, filename)
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
        out_path = Path(config.OUTPUT_FOLDER) / f"{Path(filename).stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        jobs[job_id].update(status="done", result=text, output_file=str(out_path))
        logger.info("Job %s done  ->  %s", job_id, out_path)
    except Exception as e:
        err_msg = _clean_error(e)
        jobs[job_id].update(status="error", error=err_msg)
        logger.error("Job %s error: %s", job_id, err_msg)
    finally:
        Path(audio_path).unlink(missing_ok=True)
        _prune_jobs()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(
        (Path(__file__).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    )

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

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
        "transcription_method":  "TRANSCRIPTION_METHOD",
        "api_type":              "API_TYPE",
        "audio_language":        "AUDIO_LANGUAGE",
        "output_language":       "OUTPUT_LANGUAGE",
        "whisper_model":         "WHISPER_MODEL",
        "whisper_download_path": "WHISPER_DOWNLOAD_PATH",
        "groq_model":            "LLM_MODEL_groq",
        "gemini_model":          "LLM_MODEL_gemini",
    }
    for field, attr in mapping.items():
        if (val := getattr(body, field)) is not None:
            setattr(config, attr, val)
    reset_engine()
    updated = body.model_dump(exclude_none=True)
    logger.info("Config updated: %s", updated)
    return {"ok": True, "updated": updated}

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
    if body.groq is not None:
        valid, msg = config.validate_api_key_format("groq", body.groq)
        if not valid:
            raise HTTPException(400, f"Groq key rejected: {msg}")
        config.LLM_API_KEY_groq = body.groq
        reset_engine()
        logger.info("Groq API key updated.")
    if body.gemini is not None:
        valid, msg = config.validate_api_key_format("gemini", body.gemini)
        if not valid:
            raise HTTPException(400, f"Gemini key rejected: {msg}")
        config.LLM_API_KEY_gemini = body.gemini
        reset_engine()
        logger.info("Gemini API key updated.")
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    contents = await file.read()
    size_mb  = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            413,
            f"File '{file.filename}' is {size_mb:.1f} MB, "
            f"which exceeds the {config.MAX_UPLOAD_MB} MB upload limit.",
        )
    os.makedirs(config.AUDIO_FOLDER, exist_ok=True)
    temp_path = Path(config.AUDIO_FOLDER) / f"_tmp_{int(time.time() * 1000)}{ext}"
    temp_path.write_bytes(contents)
    job_id = f"job_{int(time.time() * 1000)}"
    jobs[job_id] = {"status": "queued", "filename": file.filename,
                    "result": None, "error": None, "started_at": None}
    background_tasks.add_task(_run_transcription, job_id, str(temp_path), file.filename)
    logger.info("Job %s queued  (%s, %.2f MB)", job_id, file.filename, size_mb)
    return {"job_id": job_id}


@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    _check_job_timeout(job)
    return job

@app.get("/transcripts")
def list_transcripts():
    folder = Path(config.OUTPUT_FOLDER)
    if not folder.is_dir():
        return {"files": []}
    return {"files": sorted(f.name for f in folder.glob("*.txt"))}


@app.get("/transcripts/{filename}")
def get_transcript(filename: str):
    path = _safe_transcript_path(filename)
    if not path.exists():
        raise HTTPException(404, "File not found")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}


@app.get("/transcripts/{filename}/download")
def download_transcript(filename: str):
    path = _safe_transcript_path(filename)
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="text/plain", filename=filename)


@app.delete("/transcripts/{filename}")
def delete_transcript(filename: str):
    path = _safe_transcript_path(filename)
    if not path.exists():
        raise HTTPException(404, "File not found")
    path.unlink()
    logger.info("Deleted transcript: %s", filename)
    return {"ok": True, "deleted": filename}

@app.post("/compress/batch")
async def compress_batch(files: list[UploadFile] = File(...)):
    for upload in files:
        if Path(upload.filename).suffix.lower() not in ALLOWED_AUDIO:
            raise HTTPException(400, f"Unsupported file type '{upload.filename}'")

    tmp_dir = Path(tempfile.mkdtemp(prefix="compress_"))
    try:
        results = []
        for upload in files:
            content = await upload.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > config.MAX_UPLOAD_MB:
                raise HTTPException(
                    413,
                    f"'{upload.filename}' is {size_mb:.1f} MB, "
                    f"exceeds {config.MAX_UPLOAD_MB} MB limit.",
                )
            in_path  = tmp_dir / f"in_{upload.filename}"
            out_name = f"compressed_{Path(upload.filename).stem}.mp3"
            out_path = tmp_dir / out_name
            in_path.write_bytes(content)
            audio      = AudioSegment.from_file(in_path)
            compressed = audio.set_channels(1).set_frame_rate(16000)
            compressed.export(out_path, format="mp3", bitrate="48k", codec="libmp3lame")
            results.append({
                "out_path":  out_path,
                "out_name":  out_name,
                "orig_name": upload.filename,
                "orig_mb":   round(in_path.stat().st_size  / 1024 / 1024, 2),
                "comp_mb":   round(out_path.stat().st_size / 1024 / 1024, 2),
            })
            logger.info(
                "Compressed %s  %.2f MB -> %.2f MB",
                upload.filename, results[-1]["orig_mb"], results[-1]["comp_mb"],
            )

        if len(results) == 1:
            r       = results[0]
            data    = r["out_path"].read_bytes()
            headers = {
                "Content-Disposition": f'attachment; filename="{r["out_name"]}"',
                "X-Original-MB":   str(r["orig_mb"]),
                "X-Compressed-MB": str(r["comp_mb"]),
                "X-Original-Name": r["orig_name"],
            }
            response = StreamingResponse(io.BytesIO(data), media_type="audio/mpeg", headers=headers)
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    zf.write(r["out_path"], arcname=r["out_name"])
            buf.seek(0)
            response = StreamingResponse(
                buf, media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="compressed_audio.zip"'},
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Compression error")
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # single cleanup point

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
            path.write_bytes(await upload.read())
        score = semantic_similarity.compare(str(t_path), str(gt_path))
        logger.info(
            "Evaluate  %s vs %s  ->  %.2f%%",
            transcript.filename, ground_truth.filename, score,
        )
        return {
            "transcript":         transcript.filename,
            "ground_truth":       ground_truth.filename,
            "similarity_percent": round(score, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Evaluation error")
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)