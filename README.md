# Quickstart 
1. Install dependencies
```
pip install -r requirements.txt
```
2. Set up  `.env `
```
LLM_API_KEY_gemini=
LLM_API_KEY_groq=
```
3. Place your audio files

Put your files (.mp3, .wav, .m4a) into the audio_sample folder (or the folder path set in AUDIO_FOLDER).
4. Run  `download_whisper.py ` if you haven't downloaded a model yet
```
python download_whisper.py
```
5. Configure  `config.py `
```
TRANSCRIPTION_METHOD = "api"         # "api" (Gemini) or "local" (Whisper)
WHISPER_MODEL = "medium"             # tiny, base, small, medium, large
AUDIO_LANGUAGE = "zh"               
OUTPUT_LANGUAGE = "en"              

# Path settings
AUDIO_FOLDER= os.path.join(_base_dir, "input_audio")
OUTPUT_FOLDER= os.path.join(_base_dir, "transcribed_text")
COMPRESSED_FOLDER=os.path.join(_base_dir,"compressed_audio")
GROUND_TRUTH_PATH= os.path.join(_base_dir, "test_model", "ground_truth.txt")

API_TYPE = "groq"                                                           #Groq or Gemini
LLM_API_KEY_gemini= os.getenv("LLM_API_KEY_gemini")
LLM_MODEL_gemini= "gemini-2.5-flash"                                        #Change model if needed
LLM_API_KEY_groq= os.getenv("LLM_API_KEY_groq")
LLM_MODEL_groq= "whisper-large-v3-turbo"                                    #Change model if needed
```
6. Run

**Option A — Command line**
```
python main.py
```

**Option B — Web UI**

Start the server and open `http://127.0.0.1:8000` in your browser.
```
uvicorn app:app --reload
```
Settings (method, model, language) can be changed on the Settings page without restarting the server.

> **Note:** If you did not set API keys in `.env`, you can enter them on the Settings page in the web UI. Keys entered via the UI are session-only and will be lost on server restart.
# Notice
## Semantic similarity testing
The code includes a tool to compare your transcription results with a "Ground Truth" text file to calculate accuracy.

1. Place your reference text in test_model/ground_truth.txt.
  
2. To enable testing during the main run, uncomment the "test model section" at the bottom of 'main.py'.

3. Alternatively, run the comparison script directly:
```
python semantic_similarity.py
```
# Architecture diagram
```mermaid
flowchart LR
    subgraph INPUT ["Input audio"]
        direction TB
        i1[".mp3"]
        i2[".wav"]
        i3[".m4a"]
    end

    subgraph TRANSCRIBE ["Transcription"]
        direction LR
        T["Audio loader"]
        subgraph METHOD ["TRANSCRIPTION_METHOD"]
            direction LR
            subgraph API_M ["API"]
                direction TB
                m1["gemini-2.5-flash"]
                m2["gemini-2.5-pro"]
                m3["..."]
            end
            subgraph LOCAL_M ["Local — Whisper"]
                direction TB
                m4["tiny"]
                m5["base"]
                m6["small"]
                m7["medium"]
                m8["large"]
            end
        end
        TR["Transcript text"]
        T --> METHOD --> TR
    end

    subgraph OUTPUT ["Output"]
        direction TB
        o1[".txt"]
    end

    subgraph EVAL ["Evaluation (Optional)"]
        direction LR
        GT["ground_truth.txt"]
        SEM["Embedding model"]
        SCORE["Similarity(cos_sim)"]
        GT --> SEM --> SCORE
    end

    INPUT --> T
    TR --> o1
    o1 --> SEM
```
# Flowchart
```mermaid
flowchart TD
    A([Start]) --> B[Configure relevant settings]
    B --> C{TRANSCRIPTION_METHOD}

    C -->|API| D[client = genai.Client <br/> client = groq]
    C -->|LOCAL| E[whisper_model = whisper.load_model]

    D --> F[Scan AUDIO_FOLDER <br/> .mp3 .wav .m4a]
    E --> F

    F --> G{Files found?}
    G -->|No| Z([Exit])
    G -->|Yes| H[Read audio file]

    H --> I{Method?}
    I -->|API| J[Call API]
    I -->|LOCAL| K[Run Whisper]

    J --> L[Get transcript text]
    K --> L

    L --> M[Save .txt to OUTPUT_FOLDER]
    M --> N{More files?}
    N -->|Yes| H
    N -->|No| O([FINISH])

    style A fill:#4F46E5,color:#fff
    style C fill:#D97706,color:#fff
    style G fill:#D97706,color:#fff
    style I fill:#D97706,color:#fff
    style N fill:#D97706,color:#fff
    style O fill:#DC2626,color:#fff
    style Z fill:#DC2626,color:#fff
```
