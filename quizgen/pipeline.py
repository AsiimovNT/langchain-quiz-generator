"""The end-to-end flow from the report's flowchart:

    YouTube URL -> audio downloaded -> transcription by Whisper
                -> summary generated -> quizzes generated -> results displayed

Each stage is an independent component (YouTubeHandler, AudioProcessor,
SummaryGenerator, QuizGenerator); this class only wires them together.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from langchain_core.language_models import BaseLanguageModel

from .audio_processor import AudioProcessor
from .config import Settings
from .llm import load_llm
from .models import GeneratedContent, GenerationRequest, Transcript, VideoInfo
from .quiz_generator import QuizGenerator
from .summary_generator import SummaryGenerator
from .youtube_handler import YouTubeHandler

logger = logging.getLogger(__name__)

# (stage, fraction of that stage done 0..1, human-readable message)
PipelineProgress = Callable[[str, float, str], None]

STAGES = ("download", "transcribe", "summarize", "quiz")
STAGE_LABELS = {
    "download": "Downloading audio",
    "transcribe": "Transcribing with Whisper",
    "summarize": "Generating summary",
    "quiz": "Generating quiz",
}


class QuizPipeline:
    """Runs all stages for one request.

    Heavy components (Whisper, the LLM) can be passed in so the UI can cache
    them between runs; anything not supplied is created from ``settings`` on
    first use.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        youtube_handler: YouTubeHandler | None = None,
        audio_processor: AudioProcessor | None = None,
        llm: BaseLanguageModel | None = None,
        llm_loader: Callable[[], BaseLanguageModel] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self._youtube = youtube_handler
        self._audio = audio_processor
        self._llm = llm
        # Called on first use instead of load_llm(settings); lets the UI supply a cached model.
        self._llm_loader = llm_loader

    # -- lazily created components ---------------------------------------------
    @property
    def youtube(self) -> YouTubeHandler:
        if self._youtube is None:
            self._youtube = YouTubeHandler(self.settings.audio_dir, self.settings.max_video_minutes)
        return self._youtube

    @property
    def audio(self) -> AudioProcessor:
        if self._audio is None:
            self._audio = AudioProcessor(
                model_size=self.settings.whisper_model_size,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
                download_root=self.settings.whisper_dir,
            )
        return self._audio

    @property
    def llm(self) -> BaseLanguageModel:
        if self._llm is None:
            self._llm = self._llm_loader() if self._llm_loader else load_llm(self.settings)
        return self._llm

    # -- transcript cache ------------------------------------------------------
    def _transcript_cache_path(self, video: VideoInfo):
        return self.settings.cache_dir / f"{video.video_id}.{self.audio.model_size}.json"

    def _transcribe(self, video: VideoInfo, report: Callable[[float, str], None]) -> Transcript:
        cache_path = self._transcript_cache_path(video)
        if self.settings.use_transcript_cache and cache_path.is_file():
            try:
                transcript = Transcript.model_validate_json(cache_path.read_text(encoding="utf-8"))
                report(1.0, "Using the saved transcript")
                return transcript
            except Exception:
                logger.warning("Ignoring unreadable transcript cache %s", cache_path)
        transcript = self.audio.transcribe(video.audio_path, progress_callback=report)
        if self.settings.use_transcript_cache:
            cache_path.write_text(transcript.model_dump_json(), encoding="utf-8")
        return transcript

    # -- main entry point --------------------------------------------------------
    def run(self, request: GenerationRequest, progress: PipelineProgress | None = None) -> GeneratedContent:
        """Execute every stage and return the summary and quiz for ``request``."""

        def stage_reporter(stage: str) -> Callable[[float, str], None]:
            def report(fraction: float, message: str) -> None:
                if progress:
                    progress(stage, fraction, message)

            return report

        timings: dict[str, float] = {}
        warnings: list[str] = []

        # 1. YouTube URL -> audio file
        started = time.perf_counter()
        video = self.youtube.download_audio(request.url, stage_reporter("download"))
        timings["download"] = time.perf_counter() - started

        # 2. audio -> text (Whisper)
        started = time.perf_counter()
        transcript = self._transcribe(video, stage_reporter("transcribe"))
        timings["transcribe"] = time.perf_counter() - started

        # 3. text -> summary (LangChain + local GGUF model)
        started = time.perf_counter()
        report = stage_reporter("summarize")
        report(0.0, "Loading the language model")
        llm = self.llm
        # How much source material one quiz prompt can hold. The summariser uses
        # the same limit, so whatever it hands on is guaranteed to fit.
        quiz_budget = QuizGenerator.context_budget(self.settings.n_ctx)
        summariser = SummaryGenerator(
            llm,
            n_ctx=self.settings.n_ctx,
            chunk_tokens=self.settings.chunk_tokens,
            chunk_overlap_tokens=self.settings.chunk_overlap_tokens,
            content_budget=quiz_budget,
        )
        result = summariser.summarize(transcript.text, video.title, request.summary_max_tokens, report)
        timings["summarize"] = time.perf_counter() - started

        # 4. text -> quiz
        started = time.perf_counter()
        mcqs = []
        if request.total_mcqs:
            # Short videos: questions come straight from the transcript.
            # Long videos: from the per-part notes, falling back to the summary.
            kind, context = "transcript", transcript.text
            if result.notes:
                kind, context = "notes", "\n\n".join(result.notes)
            if summariser.count_tokens(context) > quiz_budget:
                kind, context = "summary", result.summary
            quiz = QuizGenerator(llm, max_retries=self.settings.quiz_max_retries)
            mcqs = quiz.generate(context, video.title, request.counts, stage_reporter("quiz"), kind=kind)
            if len(mcqs) < request.total_mcqs:
                warnings.append(
                    f"Only {len(mcqs)} of the {request.total_mcqs} requested questions could be generated. "
                    "Very short videos may not contain enough material; try again or lower the counts."
                )
        timings["quiz"] = time.perf_counter() - started

        return GeneratedContent(
            request=request,
            video=video,
            transcript=transcript,
            summary=result.summary,
            mcqs=mcqs,
            warnings=warnings,
            timings=timings,
        )
