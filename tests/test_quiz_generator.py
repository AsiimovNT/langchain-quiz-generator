import json
import random

import pytest

from quizgen.errors import GenerationError
from quizgen.models import Difficulty
from quizgen.quiz_generator import QuizGenerator, build_mcq, parse_questions, quiz_schema, style_issue

from .conftest import make_question, question_json


# -- parsing ---------------------------------------------------------------
def test_parse_plain_json():
    assert len(parse_questions(question_json(make_question(1), make_question(2)))) == 2


def test_parse_json_with_prose_and_code_fence():
    text = "Sure! Here are your questions:\n```json\n" + question_json(make_question()) + "\n```\nHope it helps."
    assert parse_questions(text)[0]["correct_answer"] == "Right answer 1"


def test_parse_bare_list_and_single_object():
    assert len(parse_questions(json.dumps([make_question(1), make_question(2)]))) == 2
    assert len(parse_questions(json.dumps(make_question(1)))) == 1


def test_parse_json_after_leading_text():
    assert len(parse_questions("JSON: " + question_json(make_question()) + " trailing")) == 1


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_questions("I could not think of any questions.")


# -- building MCQs -----------------------------------------------------------
def test_build_mcq_shuffles_but_keeps_the_right_answer():
    positions = set()
    for seed in range(40):
        mcq = build_mcq(make_question(), Difficulty.MEDIUM, random.Random(seed))
        assert mcq is not None
        assert mcq.answer == "Right answer 1"
        assert sorted(mcq.options) == sorted(["Right answer 1", "Wrong A1", "Wrong B1", "Wrong C1"])
        positions.add(mcq.answer_index)
    assert positions == {0, 1, 2, 3}  # the correct option is not stuck in one slot


@pytest.mark.parametrize(
    "answer",
    ["B", "b", "B)", "(B)", "B.", "Paris", "paris", "B) Paris", 1, "2"],
)
def test_build_mcq_from_options_and_answer_key(answer):
    raw = {
        "question": "What is the capital of France?",
        "options": ["A) London", "B) Paris", "C) Rome", "D) Madrid"],
        "answer": answer,
    }
    mcq = build_mcq(raw, Difficulty.EASY, random.Random(0))
    assert mcq is not None
    assert mcq.answer == "Paris"
    assert all(not option.startswith(("A)", "B)", "C)", "D)")) for option in mcq.options)


def test_build_mcq_from_lettered_dict():
    raw = {
        "question": "Which planet is known as the Red Planet?",
        "options": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"},
        "answer": "B",
    }
    assert build_mcq(raw, Difficulty.EASY, random.Random(0)).answer == "Mars"


@pytest.mark.parametrize(
    "raw",
    [
        {"question": "What is described here exactly?", "correct_answer": "X", "distractors": ["Y", "Z"]},
        {"question": "What is described here exactly?", "correct_answer": "X", "distractors": ["x", "Y", "Z"]},
        {"question": "What is described here exactly?", "correct_answer": "", "distractors": ["W", "Y", "Z"]},
        {"question": "", "correct_answer": "X", "distractors": ["W", "Y", "Z"]},
        {"question": "What is described here exactly?", "options": ["W", "X", "Y", "Z"], "answer": "E"},
        {"question": "What is described here exactly?"},
    ],
)
def test_build_mcq_rejects_unusable(raw):
    assert build_mcq(raw, Difficulty.HARD, random.Random(0)) is None


def test_schema_pins_the_question_count():
    schema = quiz_schema(2)["properties"]["questions"]
    assert schema["minItems"] == schema["maxItems"] == 2


# -- generation --------------------------------------------------------------
def test_generate_orders_by_difficulty_and_passes_settings(fake_llm):
    fake_llm.responses = [
        question_json(make_question(1), make_question(2)),
        question_json(make_question(3)),
        question_json(make_question(4)),
    ]
    quiz = QuizGenerator(fake_llm, seed=1)
    counts = {Difficulty.HARD: 1, Difficulty.EASY: 2, Difficulty.MEDIUM: 1}
    mcqs = quiz.generate("Some content about four points.", "Demo video", counts)

    assert [m.difficulty for m in mcqs] == [Difficulty.EASY, Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
    assert len(fake_llm.calls) == 3
    first = fake_llm.calls[0]
    assert "EASY" in first["prompt"] and "Demo video" in first["prompt"]
    assert first["kwargs"]["response_format"]["schema"]["properties"]["questions"]["minItems"] == 2
    # later calls are told which questions already exist
    assert "What is described in point number 1?" in fake_llm.calls[1]["prompt"]


def test_generate_skips_zero_counts_without_calling_the_model(fake_llm):
    quiz = QuizGenerator(fake_llm)
    assert quiz.generate("content", "t", {Difficulty.EASY: 0, Difficulty.MEDIUM: 0, Difficulty.HARD: 0}) == []
    assert fake_llm.calls == []


def test_generate_retries_after_garbage(fake_llm):
    fake_llm.responses = ["this is not json at all", question_json(make_question(1))]
    mcqs = QuizGenerator(fake_llm, max_retries=2).generate("content here", "t", {Difficulty.EASY: 1})
    assert len(mcqs) == 1
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["kwargs"]["temperature"] > fake_llm.calls[0]["kwargs"]["temperature"]


def test_generate_tops_up_when_a_question_is_invalid_or_duplicate(fake_llm):
    broken = {"question": "What is broken about this one?", "correct_answer": "A", "distractors": ["B"]}
    fake_llm.responses = [
        question_json(make_question(1), broken, make_question(1)),  # 1 usable out of 3
        question_json(make_question(2), make_question(3)),
    ]
    mcqs = QuizGenerator(fake_llm, max_retries=2).generate("content here", "t", {Difficulty.MEDIUM: 3})
    assert [m.question for m in mcqs] == [f"What is described in point number {n}?" for n in (1, 2, 3)]
    assert fake_llm.calls[1]["kwargs"]["response_format"]["schema"]["properties"]["questions"]["minItems"] == 2


def test_generate_returns_partial_result_when_retries_run_out(fake_llm):
    fake_llm.responder = lambda prompt, kwargs: "nope"
    mcqs = QuizGenerator(fake_llm, max_retries=1).generate("content here", "t", {Difficulty.HARD: 2})
    assert mcqs == []
    assert len(fake_llm.calls) == 2


def _wordy_question(n: int) -> dict:
    """Valid, but the correct answer gives itself away by being far longer than the rest."""
    return {
        "question": f"What is the wordy point number {n} about?",
        "correct_answer": "It is the one that goes on and on with far more detail than any other option here",
        "distractors": ["Short one", "Another short", "Third short"],
        "explanation": "Because.",
    }


@pytest.mark.parametrize(
    "raw,flagged",
    [
        (make_question(), False),
        (_wordy_question(1), True),
        ({**make_question(), "distractors": ["Wrong A", "Wrong B", "All of the above"]}, True),
        ({**make_question(), "distractors": ["x " * 25, "Wrong B", "Wrong C"]}, True),
    ],
)
def test_style_issue(raw, flagged):
    mcq = build_mcq(raw, Difficulty.EASY, random.Random(0))
    assert (style_issue(mcq) is not None) == flagged


def test_poorly_styled_question_is_replaced_by_a_retry(fake_llm):
    fake_llm.responses = [question_json(_wordy_question(1)), question_json(make_question(2))]
    mcqs = QuizGenerator(fake_llm, max_retries=2).generate("content here", "t", {Difficulty.EASY: 1})
    assert [m.question for m in mcqs] == ["What is described in point number 2?"]
    assert len(fake_llm.calls) == 2


def test_poorly_styled_question_is_kept_when_retries_do_no_better(fake_llm):
    fake_llm.responder = lambda prompt, kwargs: question_json(_wordy_question(prompt.count("wordy point") + 1))
    mcqs = QuizGenerator(fake_llm, max_retries=1).generate("content here", "t", {Difficulty.HARD: 1})
    assert len(mcqs) == 1 and "wordy point" in mcqs[0].question
    assert len(fake_llm.calls) == 2


def test_question_repeating_an_earlier_answer_is_replaced(fake_llm):
    reworded = {**make_question(1), "question": "Which thing does the first point describe?"}
    fake_llm.responses = [
        question_json(make_question(1)),  # easy
        question_json(reworded),  # medium, first try: same answer as the easy question
        question_json(make_question(2)),  # medium, retry
    ]
    mcqs = QuizGenerator(fake_llm, max_retries=2).generate(
        "content here", "t", {Difficulty.EASY: 1, Difficulty.MEDIUM: 1}
    )
    assert [m.answer for m in mcqs] == ["Right answer 1", "Right answer 2"]
    # the retry prompt lists the existing question together with its answer
    assert "(answer: Right answer 1)" in fake_llm.calls[2]["prompt"]


def test_options_are_normalised():
    raw = {
        "question": "Which country hosted the event that year?",
        "correct_answer": "The U.S.",
        "distractors": ["France.", "  Japan  ", "B) Brazil."],
    }
    mcq = build_mcq(raw, Difficulty.EASY, random.Random(0))
    assert sorted(mcq.options) == ["Brazil", "France", "Japan", "The U.S."]


def test_prompt_states_what_the_source_material_is(fake_llm):
    fake_llm.responder = lambda prompt, kwargs: question_json(make_question(1))
    QuizGenerator(fake_llm).generate("Notes about things.", 'A "quoted" title', {Difficulty.EASY: 1}, kind="notes")
    prompt = fake_llm.calls[0]["prompt"]
    assert prompt.startswith("Below are notes on consecutive parts of a YouTube video titled \"A 'quoted' title\".")
    assert 'NOTES:\n"""\nNotes about things.\n"""' in prompt


def test_generate_requires_content(fake_llm):
    with pytest.raises(GenerationError):
        QuizGenerator(fake_llm).generate("   ", "t", {Difficulty.EASY: 1})


def test_context_budget_leaves_room_for_three_questions():
    assert 2000 < QuizGenerator.context_budget(4096) < 4096 - 3 * QuizGenerator.TOKENS_PER_QUESTION
