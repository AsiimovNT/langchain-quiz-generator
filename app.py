"""Lang chain Quiz generator - Streamlit front end (the report's UIController).

Run with:  streamlit run app.py     (or double-click run.bat)

Flow: YouTube URL -> audio download -> Whisper transcript -> local Mistral GGUF
(through LangChain) -> summary + multiple-choice quiz. Everything runs locally.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from quizgen.audio_processor import AudioProcessor
from quizgen.config import DEFAULT_GGUF_FILE, Settings
from quizgen.errors import QuizGenError
from quizgen.llm import LocalGGUF, download_gguf, load_llm
from quizgen.models import MCQ, Difficulty, GeneratedContent, GenerationRequest
from quizgen.pipeline import STAGE_LABELS, STAGES, QuizPipeline

# =============================================================================
# CONFIGURATION - point GGUF_MODEL_PATH at your local GGUF model.
# A relative path is resolved against this folder. Any instruct/chat GGUF works.
# (Each value can also be set through the environment variable shown.)
# =============================================================================
GGUF_MODEL_PATH = os.getenv("QUIZGEN_GGUF_PATH", f"models/{DEFAULT_GGUF_FILE}")
WHISPER_MODEL_SIZE = os.getenv("QUIZGEN_WHISPER_MODEL", "base")  # tiny | base | small | medium | large-v3
# More options (context size, GPU layers, chunk size ...) live in quizgen/config.py.
# =============================================================================

DIFFICULTY_COLOURS = {Difficulty.EASY: "green", Difficulty.MEDIUM: "orange", Difficulty.HARD: "red"}
# Share of the progress bar given to each pipeline stage.
STAGE_WEIGHTS = {"download": 0.10, "transcribe": 0.35, "summarize": 0.30, "quiz": 0.25}
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_~\[\]$#|])")


def md(text: str) -> str:
    """Escape characters Streamlit would treat as Markdown/LaTeX (e.g. ``$5`` or ``snake_case``)."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text)


def clock(seconds: float | None) -> str:
    if not seconds:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "video"


# -- heavy models are loaded once per server process and shared between reruns --
@st.cache_resource(show_spinner=False)
def get_audio_processor(model_size: str, device: str, compute_type: str, download_root: str) -> AudioProcessor:
    return AudioProcessor(model_size, device, compute_type, Path(download_root)).load()


@st.cache_resource(show_spinner=False)
def get_llm(settings_json: str) -> LocalGGUF:
    return load_llm(Settings.model_validate_json(settings_json))


class UIController:
    """Handles user input, triggers the pipeline and displays the results."""

    def __init__(self) -> None:
        self.settings = Settings(gguf_model_path=Path(GGUF_MODEL_PATH), whisper_model_size=WHISPER_MODEL_SIZE)
        self.settings.ensure_dirs()
        self.input_error: str | None = None  # set when the sidebar input is invalid
        state = st.session_state
        state.setdefault("content_json", None)  # last GeneratedContent, serialised
        state.setdefault("run_id", 0)  # changes per generation -> fresh quiz widgets
        state.setdefault("attempt", 0)  # changes on "Try again" -> cleared answers
        state.setdefault("submitted", False)
        state.setdefault("answers", [])  # chosen option index per question, set on submit
        state.setdefault("llm_device", None)

    # ------------------------------------------------------------------ sidebar
    def render_sidebar(self) -> GenerationRequest | None:
        """Draw the "Video & Settings" panel; return a request when Generate is clicked."""
        model_ready = self.settings.model_path.is_file()
        with st.sidebar:
            st.header("⚙️ Video & Settings")
            url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

            st.markdown("##### Number of MCQs per Difficulty:")
            num_easy = st.slider("Easy:", 0, 3, 1)
            num_medium = st.slider("Medium:", 0, 3, 1)
            num_hard = st.slider("Hard:", 0, 3, 1)
            st.info(f"Total MCQs to generate: {num_easy + num_medium + num_hard}")

            summary_max_tokens = st.slider("Summary Max Tokens:", 150, 500, 300, step=10)
            clicked = st.button("🚀 Generate Content", type="primary", disabled=not model_ready)

            st.divider()
            device = f" · {st.session_state.llm_device}" if st.session_state.llm_device else ""
            st.caption(
                f"**LLM:** {self.settings.model_path.name}{device}  \n"
                f"**Speech-to-text:** Whisper `{self.settings.whisper_model_size}`  \n"
                "Runs fully on this machine - nothing is sent to an API."
            )

        if not clicked:
            return None
        try:
            return GenerationRequest(
                url=url,
                num_easy=num_easy,
                num_medium=num_medium,
                num_hard=num_hard,
                summary_max_tokens=summary_max_tokens,
            )
        except ValidationError as exc:
            self.input_error = exc.errors()[0]["msg"].removeprefix("Value error, ")
            return None

    # -------------------------------------------------------------- model setup
    def render_model_setup(self) -> None:
        """Shown instead of the app when the GGUF file is missing."""
        path = self.settings.model_path
        st.warning(f"**GGUF model not found:** `{path}`")
        is_default = path == self.settings.models_dir / self.settings.gguf_filename
        if not is_default:
            st.write(
                "Put your GGUF file at that location, or change `GGUF_MODEL_PATH` at the top of `app.py`."
            )
            return
        st.write(
            f"The app uses **{self.settings.gguf_filename}** (about 4.4 GB) from "
            f"[{self.settings.gguf_repo_id}](https://huggingface.co/{self.settings.gguf_repo_id}). "
            "Download it once and everything afterwards runs offline. You can also run "
            "`python scripts/download_model.py` in a terminal, or point `GGUF_MODEL_PATH` "
            "(top of `app.py`) at a GGUF you already have."
        )
        if st.button("⬇️ Download model from Hugging Face", type="primary"):
            bar = st.progress(0.0, text="Starting download...")

            def on_progress(done: int, total: int) -> None:
                if total:
                    bar.progress(min(1.0, done / total), text=f"Downloading {done / 1e9:.2f} / {total / 1e9:.2f} GB")

            try:
                download_gguf(self.settings.gguf_repo_id, self.settings.gguf_filename, self.settings.models_dir, on_progress)
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                return
            st.rerun()

    # --------------------------------------------------------------- generation
    def generate(self, request: GenerationRequest) -> None:
        """Run the pipeline with live progress and store the result in the session."""
        settings = self.settings

        def load_cached_llm() -> LocalGGUF:
            llm = get_llm(settings.model_dump_json())
            st.session_state.llm_device = llm.device_description
            return llm

        with st.status("Processing video...", expanded=True) as status:
            bar = st.progress(0.0, text="Starting...")
            log = st.empty()
            finished: list[str] = []

            def on_progress(stage: str, fraction: float, message: str) -> None:
                index = STAGES.index(stage)
                for earlier in STAGES[:index]:
                    if earlier not in finished:
                        finished.append(earlier)
                overall = sum(STAGE_WEIGHTS[s] for s in STAGES[:index]) + STAGE_WEIGHTS[stage] * fraction
                bar.progress(min(1.0, overall), text=f"{STAGE_LABELS[stage]}: {message}")
                status.update(label=f"{STAGE_LABELS[stage]}...", expanded=True)
                lines = [f"✅ {STAGE_LABELS[s]}" for s in finished] + [f"⏳ {STAGE_LABELS[stage]}"]
                log.markdown("  \n".join(lines))

            try:
                pipeline = QuizPipeline(
                    settings,
                    audio_processor=get_audio_processor(
                        settings.whisper_model_size,
                        settings.whisper_device,
                        settings.whisper_compute_type,
                        str(settings.whisper_dir),
                    ),
                    llm_loader=load_cached_llm,
                )
                content = pipeline.run(request, on_progress)
            except QuizGenError as exc:
                status.update(label="Something went wrong", state="error")
                st.error(f"⚠️ {exc}")
                return
            except Exception as exc:  # unexpected: show the traceback to help debugging
                status.update(label="Unexpected error", state="error")
                st.exception(exc)
                return
            bar.progress(1.0, text="Done")
            status.update(label="Done", state="complete", expanded=False)

        state = st.session_state
        state.content_json = content.model_dump_json()
        state.run_id += 1
        state.attempt = 0
        state.submitted = False
        state.answers = []
        st.rerun()

    # ------------------------------------------------------------------ results
    def render_summary(self, content: GeneratedContent) -> None:
        video, transcript = content.video, content.transcript
        st.header(f"📝 Summary: {md(video.title)}")
        details = [
            video.channel,
            f"Length {clock(video.duration_seconds or transcript.duration_seconds)}",
            f"Language: {transcript.language}" if transcript.language else None,
            f"{transcript.word_count:,} words transcribed",
        ]
        st.caption(" · ".join(md(d) for d in details if d) + f" · [Open on YouTube]({video.url})")
        with st.expander("View Summary", expanded=True):
            st.markdown(md(content.summary))
        with st.expander("View Transcript"):
            st.markdown(md(transcript.text))
        for warning in content.warnings:
            st.warning(warning)

    def render_quiz(self, content: GeneratedContent) -> None:
        mcqs = content.mcqs
        if not mcqs:
            if content.request.total_mcqs == 0:
                st.info("No questions were requested. Raise the MCQ sliders and generate again to get a quiz.")
            return

        state = st.session_state
        st.header("🧠 Quiz")
        key_prefix = f"q_{state.run_id}_{state.attempt}"

        # Streamlit ignores the browser's value for a disabled widget, so the
        # choices are copied into `state.answers` on submit and the locked
        # radios shown afterwards are separate widgets preset to those choices.
        answers: list[int | None] = state.answers if state.submitted else [None] * len(mcqs)

        with st.form(f"quiz_form_{key_prefix}"):
            for number, mcq in enumerate(mcqs, start=1):
                colour = DIFFICULTY_COLOURS[mcq.difficulty]
                st.markdown(f"**Q{number}. {md(mcq.question)}** :{colour}-badge[{mcq.difficulty.value}]")
                st.radio(
                    f"Choose your answer for Q{number}",
                    options=range(len(mcq.options)),
                    format_func=lambda i, m=mcq: md(m.options[i]),
                    index=answers[number - 1],
                    key=f"{key_prefix}_{number}" + ("_locked" if state.submitted else ""),
                    disabled=state.submitted,
                )
                if state.submitted:
                    self._render_feedback(mcq, answers[number - 1])
            if st.form_submit_button("Submit Answers", type="primary", disabled=state.submitted):
                state.answers = [state.get(f"{key_prefix}_{n}") for n in range(1, len(mcqs) + 1)]
                state.submitted = True
                st.rerun()

        if state.submitted:
            score = sum(1 for mcq, answer in zip(mcqs, answers) if answer == mcq.answer_index)
            st.subheader(f"📊 Your score: {score} / {len(mcqs)}")
            st.progress(score / len(mcqs))
            by_level = []
            for difficulty in Difficulty:
                level = [(m, a) for m, a in zip(mcqs, answers) if m.difficulty == difficulty]
                if level:
                    correct = sum(1 for m, a in level if a == m.answer_index)
                    by_level.append(f"{difficulty.value}: {correct}/{len(level)}")
            st.caption(" · ".join(by_level))
            if st.button("🔁 Try again"):
                state.submitted = False
                state.answers = []
                state.attempt += 1
                st.rerun()

    @staticmethod
    def _render_feedback(mcq: MCQ, chosen: int | None) -> None:
        if chosen == mcq.answer_index:
            st.success(f"✅ Correct: {md(mcq.answer)}")
        elif chosen is None:
            st.warning(f"Not answered. Correct answer: **{md(mcq.answer)}**")
        else:
            st.error(f"❌ Incorrect. Correct answer: **{md(mcq.answer)}**")
        if mcq.explanation:
            st.caption(f"💡 {md(mcq.explanation)}")

    def render_export(self, content: GeneratedContent) -> None:
        name = slug(content.video.title)
        lines = [f"Summary: {content.video.title}", content.video.url, "", content.summary]
        quiz_lines = [f"Quiz: {content.video.title}", content.video.url, ""]
        for number, mcq in enumerate(content.mcqs, start=1):
            quiz_lines.append(f"Q{number}. [{mcq.difficulty.value}] {mcq.question}")
            quiz_lines += [f"   {letter}) {option}" for letter, option in zip("ABCD", mcq.options)]
            quiz_lines.append("")
        quiz_lines.append("Answer key")
        for number, mcq in enumerate(content.mcqs, start=1):
            quiz_lines.append(f"Q{number}: {'ABCD'[mcq.answer_index]}) {mcq.answer}  -  {mcq.explanation}")

        st.divider()
        columns = st.columns(3)
        columns[0].download_button(
            "⬇️ Summary (.txt)", "\n".join(lines), file_name=f"{name}_summary.txt", mime="text/plain", on_click="ignore"
        )
        if content.mcqs:
            columns[1].download_button(
                "⬇️ Quiz + answer key (.txt)",
                "\n".join(quiz_lines),
                file_name=f"{name}_quiz.txt",
                mime="text/plain",
                on_click="ignore",
            )
        columns[2].download_button(
            "⬇️ Everything (.json)",
            content.model_dump_json(indent=2),
            file_name=f"{name}.json",
            mime="application/json",
            on_click="ignore",
        )
        timings = content.timings
        if timings:
            parts = " · ".join(f"{STAGE_LABELS[s].lower()} {timings[s]:.0f}s" for s in STAGES if s in timings)
            st.caption(f"Generated in {sum(timings.values()):.0f}s ({parts})")

    @staticmethod
    def render_welcome() -> None:
        st.info("👈 Paste a YouTube link in the sidebar, choose how many questions you want and click **Generate Content**.")
        st.markdown(
            """
**How it works**

1. **Audio download** - only the audio track of the video is fetched.
2. **Transcription** - Whisper converts the speech to text.
3. **Summary** - a local Mistral model, driven by LangChain, condenses the transcript.
4. **Quiz** - the same model writes multiple-choice questions at the difficulty levels you chose.

Works best with videos that are mostly speech: lectures, tutorials, interviews and podcasts.
"""
        )

    # --------------------------------------------------------------------- main
    def run(self) -> None:
        st.title("🚀 Lang chain Quiz generator")
        st.caption("Using your local GGUF model. Configure GGUF path at the top of the script.")

        request = self.render_sidebar()
        if not self.settings.model_path.is_file():
            self.render_model_setup()
            return

        progress_area = st.container()  # progress and error messages
        body = st.empty()  # welcome text or the latest result
        if self.input_error:
            progress_area.error(f"⚠️ {self.input_error}")
        if request is not None:
            body.empty()  # hide the previous result while the new one is generated
            with progress_area:
                self.generate(request)  # reruns the script on success

        with body.container():
            if st.session_state.content_json:
                content = GeneratedContent.model_validate_json(st.session_state.content_json)
                self.render_summary(content)
                self.render_quiz(content)
                self.render_export(content)
            else:
                self.render_welcome()


st.set_page_config(page_title="Lang chain Quiz generator", page_icon="🚀", layout="wide")
UIController().run()
