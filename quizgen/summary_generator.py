"""SummaryGenerator: transcript -> concise summary, using LangChain prompt chains.

Short transcripts are summarised in one call ("stuff"). Long ones exceed the
model's context window, so they are split into chunks, each chunk is condensed
into notes ("map"), and the notes are combined into the final summary
("reduce"). The notes are returned too: the quiz generator uses them as its
source material for long videos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .errors import GenerationError
from .prompts import content_header

ProgressCallback = Callable[[float, str], None]

# {header} is the source material (see prompts.content_header).
SUMMARY_PROMPT = PromptTemplate.from_template(
    """{header}

Write a clear, coherent summary of this video for a student who has not watched it.
- Cover the main topic, the key points in the order they are made, and the conclusion.
- Use only the information given above. Never add names, dates, numbers or facts that do not appear in it.
- Write plain prose in short paragraphs, in the third person. If the speaker is not named above, write "the speaker".
- Keep it under {word_limit} words and end with a complete sentence.

SUMMARY:"""
)

NOTES_PROMPT = PromptTemplate.from_template(
    """This is part {index} of {total} of the transcript of a YouTube video titled "{title}".

TRANSCRIPT PART:
\"\"\"
{text}
\"\"\"

Write concise notes on this part in 4 to 6 sentences. Keep the specific facts,
names, numbers, steps and claims that are mentioned. Use only what is in the
text above: never add names or facts that do not appear in it, and if the
speaker is not named, write "the speaker". Write plain prose in the third person.

NOTES:"""
)

_LABEL_RE = re.compile(r"^\s*(?:\*\*)?(?:final\s+)?(?:summary|notes)(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


@dataclass
class SummaryResult:
    summary: str
    notes: list[str] = field(default_factory=list)  # per-chunk notes (empty for short videos)
    strategy: str = "stuff"  # "stuff" or "map_reduce"
    chunks: int = 1


def clean_llm_text(text: str, truncated: bool | None = None) -> str:
    """Drop a leading "Summary:" label and an unfinished final sentence.

    ``truncated`` says whether the model ran into its token limit: ``False``
    means it finished on its own, so the text is left alone; ``True``/``None``
    (unknown) means a dangling half-sentence at the end is cut off.
    """
    text = _LABEL_RE.sub("", (text or "").strip()).strip()
    if truncated is False:
        return text
    if text and text[-1] not in ".!?\"')":
        ends = list(_SENTENCE_END_RE.finditer(text))
        if ends:
            text = text[: ends[-1].end()].rstrip()
    return text


class SummaryGenerator:
    """Summarises transcripts with a LangChain LLM."""

    NOTE_TOKENS = 220  # answer budget for the notes on one chunk
    PROMPT_OVERHEAD = 260  # instructions + chat template around the text
    MAX_SUMMARY_TOKENS = 500  # top of the UI slider

    def __init__(
        self,
        llm: BaseLanguageModel,
        n_ctx: int = 4096,
        chunk_tokens: int = 2000,
        chunk_overlap_tokens: int = 100,
        content_budget: int | None = None,
    ) -> None:
        """
        ``content_budget`` caps how many tokens of source material (the whole
        transcript, or the combined notes) later steps can take in one prompt.
        A transcript within the budget is summarised in a single call; a longer
        one goes through map-reduce and its notes are condensed to fit.
        """
        self.llm = llm
        self.n_ctx = n_ctx
        own_budget = n_ctx - self.PROMPT_OVERHEAD - self.MAX_SUMMARY_TOKENS - 100
        self.content_budget = max(200, min(own_budget, content_budget or own_budget))
        # A chunk must fit in the context together with its prompt and answer.
        self.chunk_tokens = max(200, min(chunk_tokens, n_ctx - self.PROMPT_OVERHEAD - self.NOTE_TOKENS - 100))
        self.chunk_overlap_tokens = min(chunk_overlap_tokens, self.chunk_tokens // 4)

    # -- helpers -------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        try:
            return int(self.llm.get_num_tokens(text))
        except Exception:
            return max(1, len(text) // 4)  # rough fallback: ~4 characters per token

    def _run(self, prompt: PromptTemplate, max_tokens: int, **variables) -> str:
        chain = prompt | self.llm.bind(max_tokens=max_tokens) | StrOutputParser()
        raw = chain.invoke(variables)
        finish_reason = getattr(self.llm, "last_finish_reason", None)
        text = clean_llm_text(raw, truncated=(finish_reason == "length") if finish_reason else None)
        if not text:
            raise GenerationError("The model returned an empty summary.")
        return text

    def _split(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
            keep_separator="end",
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def _write_notes(self, parts: list[str], title: str, report: ProgressCallback | None = None) -> list[str]:
        notes = []
        for index, part in enumerate(parts, start=1):
            if report:
                report(0.85 * (index - 1) / len(parts), f"Summarising part {index} of {len(parts)}")
            notes.append(
                self._run(NOTES_PROMPT, self.NOTE_TOKENS, title=title, index=index, total=len(parts), text=part)
            )
        return notes

    def _condense(self, notes: list[str], title: str) -> list[str]:
        """Merge notes group by group until, joined, they fit in the content budget."""
        for _ in range(4):
            if self.count_tokens("\n\n".join(notes)) <= self.content_budget or len(notes) <= 1:
                break
            groups: list[list[str]] = [[]]
            used = 0
            for note in notes:
                size = self.count_tokens(note)
                if groups[-1] and used + size > self.chunk_tokens:
                    groups.append([])
                    used = 0
                groups[-1].append(note)
                used += size
            if len(groups) >= len(notes):  # nothing merged; stop rather than loop
                break
            notes = self._write_notes(["\n\n".join(group) for group in groups], title)
        return notes

    # -- public API ----------------------------------------------------------
    def summarize(
        self,
        transcript: str,
        title: str,
        max_tokens: int = 300,
        progress_callback: ProgressCallback | None = None,
    ) -> SummaryResult:
        """Summarise ``transcript``; the summary is at most ``max_tokens`` tokens long."""
        transcript = (transcript or "").strip()
        if not transcript:
            raise GenerationError("There is no transcript to summarise.")

        def report(fraction: float, message: str) -> None:
            if progress_callback:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        # ~0.65 words per token leaves room to finish the last sentence.
        word_limit = max(60, int(max_tokens * 0.65))

        if self.count_tokens(transcript) <= self.content_budget:
            report(0.1, "Summarising the transcript")
            summary = self._run(
                SUMMARY_PROMPT,
                max_tokens,
                header=content_header("transcript", title, transcript),
                word_limit=word_limit,
            )
            report(1.0, "Summary ready")
            return SummaryResult(summary=summary, notes=[], strategy="stuff", chunks=1)

        chunks = self._split(transcript)
        notes = self._condense(self._write_notes(chunks, title, report), title)
        report(0.88, "Combining the parts into the final summary")
        summary = self._run(
            SUMMARY_PROMPT,
            max_tokens,
            header=content_header("notes", title, "\n\n".join(notes)),
            word_limit=word_limit,
        )
        report(1.0, "Summary ready")
        return SummaryResult(summary=summary, notes=notes, strategy="map_reduce", chunks=len(chunks))
