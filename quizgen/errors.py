"""Exception hierarchy so the UI can show a precise message for each failure."""


class QuizGenError(Exception):
    """Base class for every error raised by the pipeline."""


class InvalidURLError(QuizGenError, ValueError):
    """The text entered is not a usable YouTube video URL.

    Also a ``ValueError`` so Pydantic validators can raise it directly.
    """


class DownloadError(QuizGenError):
    """The audio could not be downloaded from YouTube."""


class TranscriptionError(QuizGenError):
    """Whisper could not transcribe the audio (or it contained no speech)."""


class ModelNotFoundError(QuizGenError):
    """The GGUF model file is missing."""


class ModelLoadError(QuizGenError):
    """llama.cpp could not load the GGUF model."""


class GenerationError(QuizGenError):
    """The LLM failed to produce usable output."""
