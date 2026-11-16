"""YouTubeHandler: download the audio track of a YouTube video with yt-dlp."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Callable

from .errors import DownloadError
from .models import VideoInfo, canonical_url, extract_video_id

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_AUDIO_EXTENSIONS = (".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".aac", ".flac", ".mp4")

ProgressCallback = Callable[[float, str], None]


class _QuietLogger:
    """Keep yt-dlp's chatter out of the console but remember the last error."""

    def __init__(self) -> None:
        self.last_error: str | None = None

    def debug(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        logger.debug("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        self.last_error = message


class YouTubeHandler:
    """Fetches video metadata and extracts the audio stream to a local file.

    Only the audio is downloaded (the video stream is never fetched). When
    FFmpeg is available the audio is converted to a small mono 16 kHz MP3,
    which is exactly what Whisper resamples to anyway; without FFmpeg the
    original audio container is kept and decoded directly by the transcriber.
    """

    def __init__(self, output_dir: Path, max_duration_minutes: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.max_duration_minutes = max_duration_minutes
        self.ffmpeg_path = shutil.which("ffmpeg")

    # -- helpers -------------------------------------------------------------
    def _base_options(self, quiet_logger: _QuietLogger) -> dict:
        return {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "logger": quiet_logger,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            # YouTube needs a JavaScript runtime to solve its signature
            # challenges; use whichever of these is installed.
            "js_runtimes": {"deno": {}, "node": {}, "bun": {}},
        }

    def _info_path(self, video_id: str) -> Path:
        return self.output_dir / f"{video_id}.info.json"

    def _find_cached_audio(self, video_id: str) -> Path | None:
        for extension in _AUDIO_EXTENSIONS:
            candidate = self.output_dir / f"{video_id}{extension}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _load_cached(self, video_id: str) -> VideoInfo | None:
        audio = self._find_cached_audio(video_id)
        info_path = self._info_path(video_id)
        if audio is None or not info_path.is_file():
            return None
        try:
            meta = json.loads(info_path.read_text(encoding="utf-8"))
            return VideoInfo(**{**meta, "audio_path": audio})
        except Exception:
            return None

    @staticmethod
    def _clean_error(message: str) -> str:
        message = _ANSI_RE.sub("", message or "").strip()
        message = re.sub(r"^ERROR:\s*", "", message)
        message = re.sub(r"^\[[^\]]+\]\s*[\w-]{11}:\s*", "", message)
        return message or "unknown error"

    # -- public API ----------------------------------------------------------
    def download_audio(self, url: str, progress_callback: ProgressCallback | None = None) -> VideoInfo:
        """Download the audio of ``url`` and return its metadata and file path.

        ``progress_callback(fraction, message)`` is called with 0..1 progress.
        A video that was downloaded before is reused from ``output_dir``.
        """
        import yt_dlp

        video_id = extract_video_id(url)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        def report(fraction: float, message: str) -> None:
            if progress_callback:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        cached = self._load_cached(video_id)
        if cached is not None:
            report(1.0, "Audio already downloaded")
            return cached

        quiet_logger = _QuietLogger()

        def hook(status: dict) -> None:
            if status.get("status") == "downloading":
                total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
                done = status.get("downloaded_bytes") or 0
                if total:
                    report(0.05 + 0.85 * done / total, f"Downloading audio ({done / 1e6:.1f} / {total / 1e6:.1f} MB)")
                else:
                    report(0.3, f"Downloading audio ({done / 1e6:.1f} MB)")
            elif status.get("status") == "finished":
                report(0.92, "Converting audio")

        options = self._base_options(quiet_logger)
        options.update(
            {
                "format": "bestaudio/best",
                "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
                "progress_hooks": [hook],
                "overwrites": True,
            }
        )
        if self.ffmpeg_path:
            options["ffmpeg_location"] = self.ffmpeg_path
            options["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}
            ]
            # Mono, 16 kHz: all Whisper needs, and it keeps the file small.
            options["postprocessor_args"] = {"extractaudio": ["-ac", "1", "-ar", "16000"]}

        report(0.02, "Contacting YouTube")
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(canonical_url(video_id), download=False)
                if info.get("is_live"):
                    raise DownloadError("Live streams cannot be processed. Try again once the stream has ended.")
                duration = info.get("duration")
                if duration and self.max_duration_minutes and duration > self.max_duration_minutes * 60:
                    raise DownloadError(
                        f"The video is {duration / 60:.0f} minutes long; the limit is "
                        f"{self.max_duration_minutes} minutes (QUIZGEN_MAX_VIDEO_MINUTES)."
                    )
                ydl.process_ie_result(info, download=True)
        except DownloadError:
            raise
        except yt_dlp.utils.DownloadError as exc:
            detail = self._clean_error(quiet_logger.last_error or str(exc))
            raise DownloadError(f"Could not download the audio from YouTube: {detail}") from exc
        except Exception as exc:
            raise DownloadError(f"Could not download the audio from YouTube: {exc}") from exc

        audio = self._find_cached_audio(video_id)
        if audio is None:
            raise DownloadError("The download finished but no audio file was produced.")

        video = VideoInfo(
            video_id=video_id,
            title=(info.get("title") or video_id).strip(),
            url=canonical_url(video_id),
            channel=info.get("channel") or info.get("uploader"),
            duration_seconds=int(duration) if duration else None,
            audio_path=audio,
        )
        self._info_path(video_id).write_text(
            json.dumps(video.model_dump(mode="json", exclude={"audio_path"}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report(1.0, "Audio downloaded")
        return video
