"""UI tests: drive the real Streamlit script headlessly with ``AppTest``.

A finished result is placed in the session, so no model, GPU or network is
needed; the GGUF path is pointed at a dummy file so the app is in its
"model ready" state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from quizgen.models import MCQ, Difficulty, GeneratedContent, GenerationRequest, Transcript, VideoInfo

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _content() -> GeneratedContent:
    mcqs = [
        MCQ(
            question=f"What is the right answer to question {n}?",
            options=[f"Right {n}", f"Wrong A{n}", f"Wrong B{n}", f"Wrong C{n}"],
            answer_index=0,
            explanation=f"Explanation {n}.",
            difficulty=difficulty,
        )
        for n, difficulty in enumerate(Difficulty, start=1)
    ]
    return GeneratedContent(
        request=GenerationRequest(url="https://youtu.be/dQw4w9WgXcQ"),
        video=VideoInfo(
            video_id="dQw4w9WgXcQ",
            title="Costs $5 and uses snake_case",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            channel="Channel",
            duration_seconds=125,
            audio_path=Path("a.mp3"),
        ),
        transcript=Transcript(text="A transcript of several words.", language="en", duration_seconds=125.0),
        summary="The speaker explains a thing.",
        mcqs=mcqs,
        timings={"download": 1.0, "transcribe": 2.0, "summarize": 3.0, "quiz": 4.0},
    )


@pytest.fixture
def app(tmp_path, monkeypatch) -> AppTest:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("QUIZGEN_GGUF_PATH", str(model))
    monkeypatch.setenv("QUIZGEN_DATA_DIR", str(tmp_path / "data"))
    return AppTest.from_file(APP, default_timeout=30)


def _texts(elements) -> str:
    return "\n".join(str(element.value) for element in elements)


def test_welcome_screen_matches_the_original_layout(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "🚀 Lang chain Quiz generator"
    assert "Configure GGUF path at the top of the script" in app.caption[0].value
    assert [s.label for s in app.sidebar.slider] == ["Easy:", "Medium:", "Hard:", "Summary Max Tokens:"]
    assert [(s.min, s.max, s.value) for s in app.sidebar.slider] == [(0, 3, 1)] * 3 + [(150, 500, 300)]
    assert app.sidebar.info[0].value == "Total MCQs to generate: 3"
    assert app.sidebar.button[0].label == "🚀 Generate Content"
    assert "Paste a YouTube link" in app.info[0].value


def test_total_follows_the_sliders(app):
    app.run()
    app.sidebar.slider[0].set_value(3)
    app.sidebar.slider[2].set_value(0).run()
    assert app.sidebar.info[0].value == "Total MCQs to generate: 4"


def test_bad_url_is_reported_without_running_the_pipeline(app):
    app.run()
    app.sidebar.text_input[0].set_value("https://vimeo.com/12345")
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert "does not look like a YouTube link" in app.error[0].value


def test_missing_model_shows_setup_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("QUIZGEN_GGUF_PATH", str(tmp_path / "missing.gguf"))
    monkeypatch.setenv("QUIZGEN_DATA_DIR", str(tmp_path / "data"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "GGUF model not found" in at.warning[0].value
    assert at.sidebar.button[0].disabled


def test_results_are_rendered(app):
    app.session_state["content_json"] = _content().model_dump_json()
    app.run()
    assert not app.exception
    # Markdown/LaTeX characters in the title are escaped, not interpreted
    assert app.header[0].value == r"📝 Summary: Costs \$5 and uses snake\_case"
    assert "Length 2:05" in _texts(app.caption)
    assert "The speaker explains a thing." in _texts(app.markdown)
    assert [r.label for r in app.radio] == [f"Choose your answer for Q{n}" for n in (1, 2, 3)]
    assert all(r.value is None for r in app.radio)  # nothing pre-selected
    assert "Right 1" not in _texts(app.success) + _texts(app.error)  # no answers shown yet


def test_quiz_is_scored_after_submit_and_can_be_retried(app):
    app.session_state["content_json"] = _content().model_dump_json()
    app.run()
    app.radio[0].set_value(0)  # correct
    app.radio[1].set_value(2)  # wrong; Q3 left unanswered
    next(b for b in app.button if b.label == "Submit Answers").click().run()

    assert not app.exception
    assert "Your score: 1 / 3" in _texts(app.subheader)
    assert "Easy: 1/1 · Medium: 0/1 · Hard: 0/1" in _texts(app.caption)
    assert "Correct: Right 1" in app.success[0].value
    assert "Incorrect. Correct answer: **Right 2**" in app.error[0].value
    assert "Not answered. Correct answer: **Right 3**" in app.warning[0].value
    assert "Explanation 2." in _texts(app.caption)
    assert all(r.disabled for r in app.radio)
    assert [r.value for r in app.radio] == [0, 2, None]  # the choices stay visible

    next(b for b in app.button if b.label == "🔁 Try again").click().run()
    assert not app.exception
    assert all(r.value is None and not r.disabled for r in app.radio)
    assert not app.subheader
