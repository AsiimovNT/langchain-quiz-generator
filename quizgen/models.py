"""Pydantic data models: every input and output of the pipeline is validated here."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import InvalidURLError

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_PATH_PREFIXES = ("shorts", "embed", "live", "v")


def extract_video_id(url: str) -> str:
    """Return the 11-character video id from any common YouTube URL form.

    Accepts ``watch?v=``, ``youtu.be/``, ``/shorts/``, ``/embed/``, ``/live/``
    links (with or without ``https://``) and bare video ids.
    Raises :class:`InvalidURLError` for anything else.
    """
    text = (url or "").strip()
    if not text:
        raise InvalidURLError("Please enter a YouTube URL.")
    if _VIDEO_ID_RE.match(text):
        return text

    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    candidate: str | None = None

    if host == "youtu.be":
        candidate = parts[0] if parts else None
    elif host in _YOUTUBE_HOSTS:
        if parts and parts[0] == "watch":
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            candidate = parts[1]
        elif parts and parts[0] == "playlist":
            raise InvalidURLError("That is a playlist link. Please paste the link of a single video.")
    else:
        raise InvalidURLError("That does not look like a YouTube link.")

    if not candidate or not _VIDEO_ID_RE.match(candidate):
        raise InvalidURLError("Could not find a video id in that YouTube link.")
    return candidate


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class GenerationRequest(BaseModel):
    """What the user asked for in the "Video & Settings" panel."""

    url: str
    num_easy: int = Field(1, ge=0, le=3)
    num_medium: int = Field(1, ge=0, le=3)
    num_hard: int = Field(1, ge=0, le=3)
    summary_max_tokens: int = Field(300, ge=150, le=500)

    @field_validator("url")
    @classmethod
    def _must_be_youtube(cls, value: str) -> str:
        extract_video_id(value)  # raises InvalidURLError (a ValueError) if unusable
        return value.strip()

    @property
    def video_id(self) -> str:
        return extract_video_id(self.url)

    @property
    def counts(self) -> dict[Difficulty, int]:
        return {
            Difficulty.EASY: self.num_easy,
            Difficulty.MEDIUM: self.num_medium,
            Difficulty.HARD: self.num_hard,
        }

    @property
    def total_mcqs(self) -> int:
        return self.num_easy + self.num_medium + self.num_hard


class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str | None = None
    duration_seconds: int | None = None
    audio_path: Path


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    whisper_model: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class MCQ(BaseModel):
    """One multiple-choice question with exactly four options."""

    question: str
    options: list[str]
    answer_index: int
    explanation: str = ""
    difficulty: Difficulty

    @field_validator("question", "explanation")
    @classmethod
    def _strip(cls, value: str) -> str:
        return " ".join((value or "").split())

    @field_validator("options")
    @classmethod
    def _clean_options(cls, value: list[str]) -> list[str]:
        return [" ".join(str(option).split()) for option in value]

    @model_validator(mode="after")
    def _check(self) -> "MCQ":
        if len(self.question) < 8:
            raise ValueError("question is too short")
        if len(self.options) != 4:
            raise ValueError("an MCQ needs exactly 4 options")
        if any(not option for option in self.options):
            raise ValueError("options must not be empty")
        if len({option.casefold() for option in self.options}) != 4:
            raise ValueError("options must be distinct")
        if not 0 <= self.answer_index < 4:
            raise ValueError("answer_index must be between 0 and 3")
        return self

    @property
    def answer(self) -> str:
        return self.options[self.answer_index]


class GeneratedContent(BaseModel):
    """Everything produced for one video: shown in the UI and exportable as JSON."""

    request: GenerationRequest
    video: VideoInfo
    transcript: Transcript
    summary: str
    mcqs: list[MCQ] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
