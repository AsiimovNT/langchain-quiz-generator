"""AudioProcessor: speech-to-text with OpenAI's Whisper models.

The models are run with ``faster-whisper`` (the CTranslate2 runtime for the
same Whisper weights): it is several times faster than the reference
implementation on CPU and needs no PyTorch install.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .errors import TranscriptionError
from .models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


class AudioProcessor:
    """Converts an audio file into text.

    Whisper works on 30-second windows internally; voice-activity detection is
    enabled so silence and music are skipped before transcription, which both
    speeds things up and avoids hallucinated text on non-speech audio.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = Path(download_root) if download_root else None
        self._model = None

    def load(self) -> "AudioProcessor":
        """Load the Whisper model (downloads it on first use, ~150 MB for ``base``)."""
        if self._model is not None:
            return self
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper is not installed correctly: {exc}") from exc

        root = str(self.download_root) if self.download_root else None
        if self.download_root:
            self.download_root.mkdir(parents=True, exist_ok=True)
        try:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type, download_root=root
            )
        except Exception as exc:
            if self.device == "cpu":
                raise TranscriptionError(f"Could not load Whisper '{self.model_size}': {exc}") from exc
            logger.warning("Whisper failed to load on %s (%s); falling back to CPU.", self.device, exc)
            self.device, self.compute_type = "cpu", "int8"
            try:
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8", download_root=root)
            except Exception as cpu_exc:
                raise TranscriptionError(f"Could not load Whisper '{self.model_size}': {cpu_exc}") from cpu_exc
        return self

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Transcript:
        """Transcribe ``audio_path``. ``language=None`` auto-detects the spoken language."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        self.load()

        def report(fraction: float, message: str) -> None:
            if progress_callback:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        report(0.0, "Starting transcription")
        try:
            segment_iter, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            duration = float(info.duration or 0.0)
            segments: list[TranscriptSegment] = []
            for segment in segment_iter:  # lazy: transcription happens as we iterate
                text = segment.text.strip()
                if text:
                    segments.append(TranscriptSegment(start=segment.start, end=segment.end, text=text))
                if duration:
                    report(segment.end / duration, f"Transcribed {_clock(segment.end)} of {_clock(duration)}")
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Whisper could not transcribe the audio: {exc}") from exc

        text = " ".join(segment.text for segment in segments).strip()
        if len(text.split()) < 5:
            raise TranscriptionError(
                "No speech was detected in this video, so there is nothing to summarise. "
                "Try a video with spoken content (lecture, tutorial, interview, podcast)."
            )
        report(1.0, "Transcription complete")
        return Transcript(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=duration or None,
            whisper_model=self.model_size,
            segments=segments,
        )


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
