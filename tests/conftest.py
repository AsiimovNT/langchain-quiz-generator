"""Shared test fixtures. The tests need no network, GPU or model files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from langchain_core.language_models.llms import LLM
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeLLM(LLM):
    """A scripted LangChain LLM: returns queued responses and records every call."""

    responses: list[str] = Field(default_factory=list)
    responder: Callable[[str, dict], str] | None = None
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())  # 1 token per word keeps the arithmetic obvious

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.responder is not None:
            return self.responder(prompt, kwargs)
        return self.responses.pop(0)


def question_json(*questions: dict) -> str:
    return json.dumps({"questions": list(questions)})


def make_question(n: int = 1) -> dict:
    return {
        "question": f"What is described in point number {n}?",
        "correct_answer": f"Right answer {n}",
        "distractors": [f"Wrong A{n}", f"Wrong B{n}", f"Wrong C{n}"],
        "explanation": f"Because point {n} says so.",
    }


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
