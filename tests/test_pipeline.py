import os
from pathlib import Path

from quizgen.config import Settings
from quizgen.models import Difficulty, GenerationRequest, Transcript, VideoInfo
from quizgen.pipeline import STAGES, QuizPipeline

from .conftest import make_question, question_json

VIDEO_ID = "dQw4w9WgXcQ"


class StubYouTube:
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.urls = []

    def download_audio(self, url, progress_callback=None):
        self.urls.append(url)
        if progress_callback:
            progress_callback(1.0, "Audio downloaded")
        return VideoInfo(
            video_id=VIDEO_ID, title="Stub video", url=url, channel="Stub", duration_seconds=60, audio_path=self.audio_path
        )


class StubAudio:
    model_size = "stub"

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def transcribe(self, audio_path, language=None, progress_callback=None):
        self.calls += 1
        if progress_callback:
            progress_callback(1.0, "Transcription complete")
        return Transcript(text=self.text, language="en", duration_seconds=60.0, whisper_model="stub")


def _settings(tmp_path: Path) -> Settings:
    return Settings(models_dir=tmp_path / "models", data_dir=tmp_path / "data")


def _responder(prompt, kwargs):
    if "expert quiz writer" in prompt:
        count = kwargs["response_format"]["schema"]["properties"]["questions"]["minItems"]
        base = prompt.count("What is described in point number")  # keep questions unique across calls
        return question_json(*(make_question(base + i) for i in range(1, count + 1)))
    if "Below are notes on consecutive parts" in prompt:
        return "Combined summary."
    if "TRANSCRIPT PART" in prompt:
        return "Notes on a part."
    return "Short summary."


def test_pipeline_runs_all_stages_and_caches_the_transcript(tmp_path, fake_llm):
    fake_llm.responder = _responder
    youtube, audio = StubYouTube(tmp_path / "a.mp3"), StubAudio("The speaker talks about four points. " * 30)
    pipeline = QuizPipeline(_settings(tmp_path), youtube_handler=youtube, audio_processor=audio, llm=fake_llm)
    request = GenerationRequest(url=f"https://youtu.be/{VIDEO_ID}", num_easy=2, num_medium=1, num_hard=1)

    stages_seen = []
    content = pipeline.run(request, lambda stage, fraction, message: stages_seen.append(stage))

    assert content.summary == "Short summary."
    assert [m.difficulty for m in content.mcqs] == [Difficulty.EASY, Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    assert content.warnings == []
    assert set(content.timings) == set(STAGES)
    assert list(dict.fromkeys(stages_seen)) == list(STAGES)  # stages happen in flowchart order
    # short video: the quiz is written from the transcript itself
    quiz_prompt = next(c["prompt"] for c in fake_llm.calls if "expert quiz writer" in c["prompt"])
    assert "The speaker talks about four points." in quiz_prompt
    # ...and opens exactly like the summary prompt, so llama.cpp's prompt cache is reused
    summary_prompt = fake_llm.calls[0]["prompt"]
    shared = os.path.commonprefix([summary_prompt, quiz_prompt])
    assert shared.rstrip().endswith('"""') and "four points." in shared

    # second run reuses the saved transcript instead of running Whisper again
    pipeline.run(request)
    assert audio.calls == 1


def test_pipeline_uses_notes_for_long_videos(tmp_path, fake_llm):
    fake_llm.responder = _responder
    long_text = " ".join(f"Sentence {i} is here." for i in range(2500))  # ~10k words -> map-reduce
    pipeline = QuizPipeline(
        _settings(tmp_path), youtube_handler=StubYouTube(tmp_path / "a.mp3"), audio_processor=StubAudio(long_text), llm=fake_llm
    )
    content = pipeline.run(GenerationRequest(url=VIDEO_ID, num_easy=1, num_medium=0, num_hard=0))

    assert content.summary == "Combined summary."
    quiz_prompt = next(c["prompt"] for c in fake_llm.calls if "expert quiz writer" in c["prompt"])
    assert "Notes on a part." in quiz_prompt and "Sentence 5 is here." not in quiz_prompt


def test_pipeline_without_questions_skips_the_quiz_and_warns_on_shortfall(tmp_path, fake_llm):
    fake_llm.responder = lambda prompt, kwargs: "not json" if "expert quiz writer" in prompt else "Short summary."
    pipeline = QuizPipeline(
        _settings(tmp_path),
        youtube_handler=StubYouTube(tmp_path / "a.mp3"),
        audio_processor=StubAudio("Just enough words to pass as a transcript here."),
        llm=fake_llm,
    )
    none_requested = pipeline.run(GenerationRequest(url=VIDEO_ID, num_easy=0, num_medium=0, num_hard=0))
    assert none_requested.mcqs == [] and none_requested.warnings == []
    assert not any("expert quiz writer" in c["prompt"] for c in fake_llm.calls)

    shortfall = pipeline.run(GenerationRequest(url=VIDEO_ID, num_easy=1, num_medium=0, num_hard=0))
    assert shortfall.mcqs == []
    assert "Only 0 of the 1 requested questions" in shortfall.warnings[0]
