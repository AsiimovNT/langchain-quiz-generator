"""QuizGenerator: content -> multiple-choice questions at three difficulty levels.

The model is asked for JSON and, when the backend supports it (llama.cpp
does), its output is constrained to a JSON schema so it is always parseable.
The parser is still tolerant and failed attempts are retried, so a model
without constrained decoding works too.

The model writes the correct answer and three distractors as separate fields;
the options are shuffled here. That removes two classic LLM quiz bugs: an
answer key that points at the wrong option, and the correct answer always
sitting in the same position.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from .errors import GenerationError
from .models import MCQ, Difficulty
from .prompts import ContentKind, content_header

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

DIFFICULTY_GUIDE: dict[Difficulty, str] = {
    Difficulty.EASY: (
        "EASY - factual recall. Ask about a fact that is stated explicitly. Someone who paid "
        'attention can answer immediately. Typical openings: "What is ...", "Which ...", "Who ...", '
        '"How many ...".'
    ),
    Difficulty.MEDIUM: (
        "MEDIUM - understanding. Ask why something happens, how ideas or steps relate, or what "
        "the reasoning behind a point is. Answering requires connecting information, not recalling "
        'a single phrase. Typical openings: "Why does ...", "How does ...", "What is the difference '
        'between ...", "What is the reason ...".'
    ),
    Difficulty.HARD: (
        "HARD - analysis and inference. Ask about implications, comparisons, the overall purpose or "
        "conclusion, or what follows from the ideas presented. Every wrong answer must sound "
        'plausible to someone who only skimmed the video. Typical openings: "What can be inferred '
        '...", "Which statement best explains ...", "What would most likely happen if ...", '
        '"What is the main purpose of ...".'
    ),
}

# {header} is the source material (see prompts.content_header). It comes first
# so llama.cpp reuses its cached evaluation of it across the calls for each
# difficulty level.
QUIZ_PROMPT = PromptTemplate.from_template(
    """{header}

You are an expert quiz writer. Write {count} multiple-choice question(s) about the video above.

Difficulty: {difficulty_guide}

Rules:
- Each question must be answerable from the material above alone and have exactly one correct answer.
- Give the correct answer and exactly 3 wrong answers (distractors) that are plausible but clearly incorrect.
- Every answer is a short phrase of at most 12 words. The correct answer must not be longer or more detailed than the distractors.
- Never use "All of the above", "None of the above" or "Both A and B".
- Ask about the subject itself; do not mention "the transcript", "the notes" or "the passage".
- The explanation is one sentence stating the fact from the video that makes the answer correct.
{avoid_block}
Example of the expected format and style (on an unrelated topic):
{{"questions": [{{"question": "Why do leaves look green?", "correct_answer": "Chlorophyll reflects green light", "distractors": ["Chlorophyll absorbs green light", "Leaves store green pigment in water", "Sunlight contains mostly green light"], "explanation": "Chlorophyll absorbs red and blue light and reflects green light, which is what we see."}}]}}

Respond with ONLY a JSON object in that form containing {count} question(s).

JSON:"""
)

_OPTION_PREFIX_RE = re.compile(r"^\s*(?:\(?[A-Da-d1-4][\)\.:]|[-*•])\s+")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CATCH_ALL_RE = re.compile(r"\b(?:all|none) of the above\b|\bboth [a-d] and [a-d]\b", re.IGNORECASE)
_LETTERS = "ABCD"


def quiz_schema(count: int) -> dict[str, Any]:
    """JSON schema for ``count`` questions (used for constrained decoding).

    Strings are deliberately left without ``maxLength``: llama.cpp expands a
    length cap into a grammar nested one level per character and rejects
    anything deeper than 256. Brevity is asked for in the prompt and checked
    by :func:`style_issue` instead.
    """
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "correct_answer": {"type": "string"},
                        "distractors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["question", "correct_answer", "distractors", "explanation"],
                },
            }
        },
        "required": ["questions"],
    }


def _first_json_value(text: str) -> Any:
    """Parse the first JSON object/array found in ``text`` (tolerates surrounding prose)."""
    text = text.strip()
    fenced = _CODE_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON found in the model output")


def parse_questions(text: str) -> list[dict[str, Any]]:
    """Extract the raw question dicts from the model output."""
    value = _first_json_value(text)
    if isinstance(value, dict):
        for key in ("questions", "mcqs", "quiz", "items"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError("the model output is not a list of questions")
    return [item for item in value if isinstance(item, dict)]


def _clean_option(option: Any) -> str:
    """Normalise an option: no "A)" prefix, single spaces, no lone trailing full stop."""
    text = _OPTION_PREFIX_RE.sub("", " ".join(str(option).split())).strip()
    if text.endswith(".") and text.count(".") == 1:  # keeps "U.S." and "..." intact
        text = text[:-1].rstrip()
    return text


def build_mcq(raw: dict[str, Any], difficulty: Difficulty, rng: random.Random) -> MCQ | None:
    """Turn one raw question dict into a validated :class:`MCQ` (or ``None`` if unusable).

    Understands the requested shape (``correct_answer`` + ``distractors``) and the
    common alternative (``options`` + ``answer`` given as a letter, index or text).
    """
    question = str(raw.get("question") or raw.get("q") or "").strip()
    explanation = str(raw.get("explanation") or raw.get("reason") or "").strip()
    correct: str | None = None
    distractors: list[str] = []

    if raw.get("correct_answer") is not None and isinstance(raw.get("distractors"), list):
        correct = _clean_option(raw["correct_answer"])
        distractors = [_clean_option(d) for d in raw["distractors"]]
    elif isinstance(raw.get("options"), (list, dict)):
        options_raw = raw["options"]
        if isinstance(options_raw, dict):  # {"A": "...", "B": "..."}
            options_raw = [options_raw[key] for key in sorted(options_raw)]
        options = [_clean_option(o) for o in options_raw]
        answer = raw.get("answer", raw.get("correct_option", raw.get("answer_index", raw.get("correct_answer"))))
        index: int | None = None
        if isinstance(answer, bool):
            answer = None
        if isinstance(answer, int):
            # 0-based if it can be, otherwise 1-based
            index = answer if 0 <= answer < len(options) else answer - 1
        elif isinstance(answer, str):
            token = answer.strip()
            folded = [o.casefold() for o in options]
            letter = re.match(r"^\(?([A-Da-d])\)?(?:[\).:]|$)", token)
            if token.casefold() in folded:
                index = folded.index(token.casefold())
            elif letter:
                index = _LETTERS.index(letter.group(1).upper())
            elif _clean_option(token).casefold() in folded:
                index = folded.index(_clean_option(token).casefold())
            elif token.isdigit():
                index = int(token) - 1
        if index is None or not 0 <= index < len(options):
            return None
        correct = options[index]
        distractors = [o for i, o in enumerate(options) if i != index]
    else:
        return None

    seen = {correct.casefold()}
    unique_distractors = []
    for distractor in distractors:
        if distractor and distractor.casefold() not in seen:
            seen.add(distractor.casefold())
            unique_distractors.append(distractor)
    if not correct or len(unique_distractors) < 3:
        return None

    options = [correct, *unique_distractors[:3]]
    rng.shuffle(options)
    try:
        return MCQ(
            question=question,
            options=options,
            answer_index=options.index(correct),
            explanation=explanation,
            difficulty=difficulty,
        )
    except ValidationError:
        return None


def style_issue(mcq: MCQ) -> str | None:
    """Return why ``mcq`` is valid but poorly written, or ``None`` if it is fine.

    A flagged question is only used if a retry does not produce a better one.
    """
    lengths = [len(option.split()) for option in mcq.options]
    correct = lengths[mcq.answer_index]
    others = [n for i, n in enumerate(lengths) if i != mcq.answer_index]
    if any(_CATCH_ALL_RE.search(option) for option in mcq.options):
        return "uses a catch-all option"
    if max(lengths) > 20:
        return "an option is too long"
    if correct >= 8 and correct > 1.6 * max(others):
        return "the correct answer stands out by being the longest"
    return None


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.casefold()).strip()


class QuizGenerator:
    """Generates MCQs with a LangChain LLM."""

    TOKENS_PER_QUESTION = 260  # answer budget per question
    BASE_TOKENS = 120  # JSON scaffolding
    PROMPT_OVERHEAD = 640  # instructions + example + list of existing questions + chat template

    def __init__(self, llm: BaseLanguageModel, max_retries: int = 2, seed: int | None = None) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self._rng = random.Random(seed)

    @classmethod
    def context_budget(cls, n_ctx: int, max_questions_per_call: int = 3) -> int:
        """How many tokens of source content fit next to the prompt and the answer."""
        answer_tokens = cls.BASE_TOKENS + cls.TOKENS_PER_QUESTION * max_questions_per_call
        return n_ctx - cls.PROMPT_OVERHEAD - answer_tokens - 60

    def _ask(self, header: str, difficulty: Difficulty, count: int, avoid: list[MCQ], attempt: int) -> str:
        avoid_block = ""
        if avoid:
            listed = "\n".join(f"  - {mcq.question} (answer: {mcq.answer})" for mcq in avoid)
            avoid_block = (
                "- Each new question must cover a different point and have a different correct answer "
                f"than these existing questions:\n{listed}\n"
            )
        bound = self.llm.bind(
            max_tokens=self.BASE_TOKENS + self.TOKENS_PER_QUESTION * count,
            # a little more randomness on retries so the model does not repeat a bad answer
            temperature=min(1.0, 0.4 + 0.2 * attempt),
            response_format={"type": "json_object", "schema": quiz_schema(count)},
        )
        chain = QUIZ_PROMPT | bound | StrOutputParser()
        return chain.invoke(
            {
                "header": header,
                "count": count,
                "difficulty_guide": DIFFICULTY_GUIDE[difficulty],
                "avoid_block": avoid_block,
            }
        )

    def generate_for_difficulty(
        self,
        header: str,
        difficulty: Difficulty,
        count: int,
        existing: list[MCQ] | None = None,
    ) -> list[MCQ]:
        """Generate up to ``count`` questions of one difficulty (fewer only if every retry fails).

        ``header`` is the source-material block from :func:`prompts.content_header`.
        """
        existing = list(existing or [])
        taken = {_normalise(mcq.question) for mcq in existing}
        answers = {_normalise(mcq.answer) for mcq in existing}
        produced: list[MCQ] = []
        spare: list[MCQ] = []  # valid but flawed; used only if retries do not do better
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            missing = count - len(produced)
            if missing <= 0:
                break
            try:
                raw_text = self._ask(header, difficulty, missing, existing + produced, attempt)
                raw_questions = parse_questions(raw_text)
            except GenerationError:
                raise
            except Exception as exc:  # unparseable output: try again
                last_error = exc
                logger.warning("Quiz attempt %d (%s) failed: %s", attempt + 1, difficulty.value, exc)
                continue
            for raw in raw_questions:
                if len(produced) >= count:
                    break
                mcq = build_mcq(raw, difficulty, self._rng)
                if mcq is None or _normalise(mcq.question) in taken:
                    continue
                issue = style_issue(mcq)
                if issue is None and _normalise(mcq.answer) in answers:
                    # same answer as an earlier question: almost always the same point reworded
                    issue = "repeats the answer of an earlier question"
                if issue:
                    logger.info("Setting aside a %s question (%s): %s", difficulty.value, issue, mcq.question)
                    spare.append(mcq)
                    continue
                taken.add(_normalise(mcq.question))
                answers.add(_normalise(mcq.answer))
                produced.append(mcq)

        for mcq in spare:  # better an imperfect question than a missing one
            if len(produced) >= count:
                break
            if _normalise(mcq.question) not in taken:
                taken.add(_normalise(mcq.question))
                produced.append(mcq)

        if not produced and last_error is not None:
            logger.warning("No %s questions could be generated: %s", difficulty.value, last_error)
        return produced

    def generate(
        self,
        context: str,
        title: str,
        counts: dict[Difficulty, int],
        progress_callback: ProgressCallback | None = None,
        kind: ContentKind = "transcript",
    ) -> list[MCQ]:
        """Generate MCQs for every difficulty in ``counts`` (ordered Easy -> Medium -> Hard).

        ``context`` is the source material and ``kind`` says what it is (the
        transcript itself, notes on its parts, or the summary).
        """
        context = (context or "").strip()
        total = sum(max(0, n) for n in counts.values())
        if total == 0:
            return []
        if not context:
            raise GenerationError("There is no content to write questions about.")

        def report(fraction: float, message: str) -> None:
            if progress_callback:
                progress_callback(max(0.0, min(1.0, fraction)), message)

        header = content_header(kind, title, context)
        mcqs: list[MCQ] = []
        done = 0
        for difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
            count = counts.get(difficulty, 0)
            if count <= 0:
                continue
            report(done / total, f"Writing {count} {difficulty.value.lower()} question{'s' if count != 1 else ''}")
            mcqs.extend(self.generate_for_difficulty(header, difficulty, count, existing=mcqs))
            done += count
        report(1.0, "Quiz ready")
        return mcqs
