# Quiz Generation Using LangChain

Turn any spoken-word YouTube video (lecture, tutorial, interview, podcast) into
a **summary** and a **multiple-choice quiz** - entirely on your own machine.

```
YouTube URL ─▶ audio download ─▶ Whisper transcript ─▶ Mistral GGUF via LangChain ─▶ summary + MCQs ─▶ Streamlit UI
               (yt-dlp)          (speech-to-text)      (llama.cpp, local)
```

No API keys and no cloud inference: the speech model and the language model
both run locally, so nothing about the video or your quiz leaves the computer.

## Features

- **Video & Settings panel** - paste a YouTube URL, choose 0-3 questions for
  each of *Easy*, *Medium* and *Hard*, and set the summary length (150-500 tokens).
- **Summary** of the video, with the full transcript one click away.
- **Interactive quiz** - answer with radio buttons, submit, and get a score
  with the correct answer and an explanation for every question.
- **Works on long videos** - transcripts that do not fit in the model's context
  are summarised chunk by chunk (map-reduce).
- **Export** the summary, the quiz with its answer key, or everything as JSON.
- **GPU accelerated** when an NVIDIA GPU is available, with automatic fallback
  to partial offload or CPU.

## Quick start (Windows)

1. Install [Python 3.10+](https://www.python.org/downloads/) (3.13 is what the
   project was verified with). [FFmpeg](https://ffmpeg.org/) on the `PATH` is
   recommended but optional.
2. Double-click **`setup.bat`**. It creates `.venv`, installs the dependencies
   (the CUDA build of llama.cpp if `nvidia-smi` is found, otherwise the CPU
   build) and downloads the Mistral model (about 4.4 GB) into `models/`.
3. Double-click **`run.bat`**. The app opens at <http://localhost:8501>.

### Manual setup (any OS)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements-gpu.txt   # or requirements-cpu.txt without an NVIDIA GPU
python scripts/download_model.py
streamlit run app.py
```

## Using a different model

The GGUF path is set at the top of [`app.py`](app.py):

```python
GGUF_MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large-v3
```

Any instruct/chat model in GGUF format works; the prompt wrapping (for example
Mistral's `[INST] ... [/INST]`) is taken from the chat template stored inside
the GGUF file. A larger Whisper model (`small`, `medium`) gives a more accurate
transcript at the cost of speed.

Further settings are environment variables (defaults in
[`quizgen/config.py`](quizgen/config.py)):

| Variable | Default | Meaning |
| --- | --- | --- |
| `QUIZGEN_GGUF_PATH` | `models/mistral-7b-instruct-v0.2.Q4_K_M.gguf` | GGUF model file |
| `QUIZGEN_N_CTX` | `4096` | LLM context window in tokens |
| `QUIZGEN_N_GPU_LAYERS` | `-1` | `-1` = as many layers on the GPU as fit, `0` = CPU only |
| `QUIZGEN_TEMPERATURE` | `0.3` | Sampling temperature for summaries |
| `QUIZGEN_WHISPER_MODEL` | `base` | Whisper model size |
| `QUIZGEN_WHISPER_DEVICE` | `cpu` | `cuda` needs cuDNN 9 (`pip install nvidia-cudnn-cu12`) |
| `QUIZGEN_CHUNK_TOKENS` | `2000` | Transcript chunk size for long videos |
| `QUIZGEN_MAX_VIDEO_MINUTES` | `180` | Longest video accepted |
| `QUIZGEN_QUIZ_MAX_RETRIES` | `2` | Extra attempts when the model returns unusable questions |

## Command line

The same pipeline without the web UI:

```bash
python -m quizgen.cli "https://www.youtube.com/watch?v=VIDEO_ID" --easy 1 --medium 1 --hard 1 --summary-tokens 300 --json result.json
```

## Project structure

```
app.py                       Streamlit UI (UIController) - GGUF path at the top
quizgen/
  config.py                  Settings (Pydantic), overridable by environment variables
  models.py                  Pydantic data models + YouTube URL validation
  youtube_handler.py         YouTubeHandler   - audio download with yt-dlp
  audio_processor.py         AudioProcessor   - Whisper transcription
  llm.py                     LLM backend      - Hugging Face download + llama.cpp as a LangChain LLM
  summary_generator.py       SummaryGenerator - LangChain prompt chains (stuff / map-reduce)
  quiz_generator.py          QuizGenerator    - MCQs per difficulty, JSON-schema constrained
  pipeline.py                QuizPipeline     - wires the stages together
  cli.py                     Command-line entry point
scripts/download_model.py    Fetch the GGUF model from the Hugging Face Hub
tests/                       Unit tests (no network, GPU or model needed)
models/                      GGUF + Whisper models (downloaded, git-ignored)
data/                        Downloaded audio and cached transcripts (git-ignored)
```

## How it works

1. **YouTubeHandler** validates the URL and downloads only the audio stream
   with `yt-dlp`, converting it to mono 16 kHz MP3 when FFmpeg is available.
2. **AudioProcessor** transcribes the audio with Whisper (run through
   `faster-whisper`). Voice-activity detection skips music and silence.
   Transcripts are cached per video, so regenerating a quiz with different
   settings does not transcribe again.
3. **SummaryGenerator** builds LangChain chains (`PromptTemplate | LLM |
   StrOutputParser`). A short transcript is summarised in one call; a long one
   is split with `RecursiveCharacterTextSplitter`, each chunk is condensed
   into notes, and the notes are combined into the final summary.
4. **QuizGenerator** asks the model for questions one difficulty level at a
   time. The output is constrained to a JSON schema by llama.cpp, validated
   with Pydantic, de-duplicated, and retried if anything is unusable. The model
   writes the correct answer and three distractors separately and the options
   are shuffled in code, so the answer key can never point at the wrong option.
5. **UIController** (Streamlit) collects the settings, shows live progress and
   renders the summary, the quiz and the score.

## Tests

```bash
python -m pytest
```

## Troubleshooting

- **"Could not download the audio from YouTube"** - YouTube changes often;
  update the downloader with `pip install -U "yt-dlp[default]"`. yt-dlp also
  needs a JavaScript runtime for YouTube: [Node.js](https://nodejs.org/) or
  [Deno](https://deno.com/) on the `PATH`. Age-restricted and private videos
  cannot be downloaded.
- **"llama-cpp-python could not be loaded"** - the GPU build is installed on a
  machine without a working NVIDIA driver. Install the CPU build:
  `pip install --force-reinstall -r requirements-cpu.txt`.
- **Slow generation** - check the sidebar: it shows whether the model is on the
  GPU or CPU. If other programs are using the VRAM, only part of the model is
  offloaded; close them or set `QUIZGEN_N_GPU_LAYERS`.
- **Out of memory** - use a smaller quantisation (for example `Q3_K_M`) or
  lower `QUIZGEN_N_CTX`.

## Limitations

- **Generated content can be wrong.** Mistral 7B is a small local model. It
  sometimes fills in details from its own knowledge instead of the video (for
  example, naming a well-known lecturer of a course when the transcript never
  mentions one), and a question's "correct" answer is only as good as the
  model's reading of the transcript. Review a quiz before using it to assess
  anyone.
- **Transcription errors carry through.** Whisper `base` mishears unusual
  names and jargon, and the summary and quiz inherit those mistakes. Use
  `small` or `medium` for better accuracy.
- **Speech only.** Nothing shown on screen (slides, code, diagrams) is seen;
  videos that are mostly music or visuals do not work.
- **Long videos are quizzed from notes.** Beyond roughly 12 minutes of speech
  the questions are written from the per-part notes rather than the full
  transcript, so fine detail can be lost.
- **Disk space.** The model is 4.4 GB and the GPU build of the environment is
  about 7.7 GB.

## Technology

Python · Streamlit · LangChain (`langchain-core`, `langchain-text-splitters`) ·
llama.cpp (`llama-cpp-python`) · Mistral 7B Instruct v0.2 (GGUF) · Whisper
(`faster-whisper`) · yt-dlp · Pydantic · Hugging Face Hub

---

Capstone project (24AM8PWCP2), Department of Machine Learning, BMS College of
Engineering - Nihal Thakur, Pacharne Kaushal Yogesh, Harshith.
