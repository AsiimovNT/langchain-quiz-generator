"""Central configuration for the quiz generator.

Every value has a sensible default and can be overridden with an environment
variable (``QUIZGEN_*``) or by passing keyword arguments to :class:`Settings`
(``app.py`` does the latter for the GGUF path, so it can be edited at the top
of the script).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

# The model named in the report: Mistral 7B Instruct in GGUF format.
DEFAULT_GGUF_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
DEFAULT_GGUF_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"


def _env(name: str, default):
    """Read ``QUIZGEN_<name>`` from the environment, cast to the default's type."""
    raw = os.getenv(f"QUIZGEN_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw)
    return raw


class Settings(BaseModel):
    """Runtime settings, validated with Pydantic."""

    # --- storage -----------------------------------------------------------
    models_dir: Path = Field(default_factory=lambda: _env("MODELS_DIR", BASE_DIR / "models"))
    data_dir: Path = Field(default_factory=lambda: _env("DATA_DIR", BASE_DIR / "data"))

    # --- LLM (local GGUF through llama.cpp, orchestrated by LangChain) ------
    gguf_repo_id: str = Field(default_factory=lambda: _env("GGUF_REPO", DEFAULT_GGUF_REPO))
    gguf_filename: str = Field(default_factory=lambda: _env("GGUF_FILE", DEFAULT_GGUF_FILE))
    gguf_model_path: Path | None = Field(
        default_factory=lambda: (Path(p) if (p := os.getenv("QUIZGEN_GGUF_PATH")) else None)
    )
    n_ctx: int = Field(default_factory=lambda: _env("N_CTX", 4096), ge=1024)
    # -1 = offload every layer that fits on the GPU, 0 = CPU only.
    n_gpu_layers: int = Field(default_factory=lambda: _env("N_GPU_LAYERS", -1), ge=-1)
    n_threads: int | None = Field(
        default_factory=lambda: (int(t) if (t := os.getenv("QUIZGEN_N_THREADS")) else None)
    )
    temperature: float = Field(default_factory=lambda: _env("TEMPERATURE", 0.3), ge=0.0, le=2.0)

    # --- Whisper (speech-to-text) ------------------------------------------
    whisper_model_size: str = Field(default_factory=lambda: _env("WHISPER_MODEL", "base"))
    whisper_device: str = Field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = Field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "int8"))

    # --- pipeline ----------------------------------------------------------
    # Transcript chunk size (in LLM tokens) for map-reduce summarisation.
    chunk_tokens: int = Field(default_factory=lambda: _env("CHUNK_TOKENS", 2000), ge=200)
    chunk_overlap_tokens: int = Field(default_factory=lambda: _env("CHUNK_OVERLAP", 100), ge=0)
    max_video_minutes: int = Field(default_factory=lambda: _env("MAX_VIDEO_MINUTES", 180), ge=1)
    quiz_max_retries: int = Field(default_factory=lambda: _env("QUIZ_MAX_RETRIES", 2), ge=0)
    use_transcript_cache: bool = Field(default_factory=lambda: _env("TRANSCRIPT_CACHE", True))

    @property
    def model_path(self) -> Path:
        """Absolute path of the GGUF file the app will load."""
        path = self.gguf_model_path or (self.models_dir / self.gguf_filename)
        return path if path.is_absolute() else (BASE_DIR / path)

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def whisper_dir(self) -> Path:
        return self.models_dir / "whisper"

    def ensure_dirs(self) -> None:
        for directory in (self.models_dir, self.audio_dir, self.cache_dir, self.whisper_dir):
            directory.mkdir(parents=True, exist_ok=True)
