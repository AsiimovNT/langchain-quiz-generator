"""The opening block shared by every prompt.

All prompts start with the source material, worded identically, and only then
give their instructions. llama.cpp keeps the evaluated prompt in its KV cache
and reuses the longest common prefix, so the model reads a transcript once and
the summary and each quiz call after it start almost instantly. That matters
most on CPU, where reading a long transcript can take a minute.
"""

from __future__ import annotations

from typing import Literal

ContentKind = Literal["transcript", "notes", "summary"]

_INTRO = {
    "transcript": 'Below is the transcript of a YouTube video titled "{title}".',
    "notes": 'Below are notes on consecutive parts of a YouTube video titled "{title}".',
    "summary": 'Below is a summary of a YouTube video titled "{title}".',
}


def content_header(kind: ContentKind, title: str, text: str) -> str:
    """Return the source-material block that opens a prompt."""
    intro = _INTRO[kind].format(title=title.replace('"', "'"))
    return f'{intro}\n\n{kind.upper()}:\n"""\n{text.strip()}\n"""'
