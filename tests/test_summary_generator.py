import pytest

from quizgen.errors import GenerationError
from quizgen.summary_generator import SummaryGenerator, clean_llm_text


def test_short_transcript_uses_a_single_call(fake_llm):
    fake_llm.responses = ["The speaker explains photosynthesis."]
    result = SummaryGenerator(fake_llm, n_ctx=4096).summarize("Plants make food from light. " * 20, "Bio 101", 300)

    assert result.strategy == "stuff"
    assert result.notes == []
    assert result.summary == "The speaker explains photosynthesis."
    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call["kwargs"]["max_tokens"] == 300
    assert "Bio 101" in call["prompt"] and "Plants make food from light." in call["prompt"]
    assert "under 195 words" in call["prompt"]  # 300 tokens * 0.65


def test_long_transcript_is_mapped_then_reduced(fake_llm):
    def responder(prompt, kwargs):
        return "Final summary of everything." if "Below are notes on consecutive parts" in prompt else "Notes for one part."

    fake_llm.responder = responder
    transcript = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(900))  # ~6300 words
    progress = []
    result = SummaryGenerator(fake_llm, n_ctx=4096, chunk_tokens=2000).summarize(
        transcript, "Long lecture", 400, lambda fraction, message: progress.append((fraction, message))
    )

    assert result.strategy == "map_reduce"
    assert result.chunks >= 4
    assert result.notes == ["Notes for one part."] * result.chunks
    assert result.summary == "Final summary of everything."
    map_calls = [c for c in fake_llm.calls if "TRANSCRIPT PART" in c["prompt"]]
    reduce_calls = [c for c in fake_llm.calls if "Below are notes on consecutive parts" in c["prompt"]]
    assert len(map_calls) == result.chunks and len(reduce_calls) == 1
    assert reduce_calls[0]["kwargs"]["max_tokens"] == 400
    # every chunk fits the window with its prompt
    assert all(fake_llm.get_num_tokens(c["prompt"]) < 4096 - 220 for c in map_calls)
    # nothing is lost between chunks
    joined = " ".join(c["prompt"] for c in map_calls)
    assert "Sentence number 0 " in joined and "Sentence number 899 " in joined
    assert progress[-1] == (1.0, "Summary ready")
    assert [f for f, _ in progress] == sorted(f for f, _ in progress)


def test_content_budget_decides_between_one_call_and_map_reduce(fake_llm):
    fake_llm.responder = lambda prompt, kwargs: "Some text."
    transcript = " ".join(f"Word{i}." for i in range(3000))  # 3000 "tokens"

    assert SummaryGenerator(fake_llm, n_ctx=4096).summarize(transcript, "t").strategy == "stuff"
    limited = SummaryGenerator(fake_llm, n_ctx=4096, content_budget=2500)
    assert limited.content_budget == 2500
    assert limited.summarize(transcript, "t").strategy == "map_reduce"


def test_notes_are_collapsed_when_too_many_to_fit(fake_llm):
    # each note is 300 "tokens"; 20 of them (6000) exceed the content budget
    def responder(prompt, kwargs):
        if "Below are notes on consecutive parts" in prompt:
            return "Final."
        return "note " * 300

    fake_llm.responder = responder
    transcript = " ".join(f"Word{i}." for i in range(20 * 1900))
    generator = SummaryGenerator(fake_llm, n_ctx=4096, chunk_tokens=2000)
    result = generator.summarize(transcript, "Marathon", 300)

    assert result.summary == "Final."
    assert len(result.notes) < 20
    assert generator.count_tokens("\n\n".join(result.notes)) <= generator.content_budget
    reduce_prompt = [c["prompt"] for c in fake_llm.calls if "Below are notes on consecutive parts" in c["prompt"]][0]
    assert generator.count_tokens(reduce_prompt) < 4096 - 300


def test_empty_transcript_and_empty_answer_raise(fake_llm):
    with pytest.raises(GenerationError):
        SummaryGenerator(fake_llm).summarize("   ", "t")
    fake_llm.responses = ["   "]
    with pytest.raises(GenerationError):
        SummaryGenerator(fake_llm).summarize("Some real words here.", "t")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Summary: The talk covers X.", "The talk covers X."),
        ("**Summary:** The talk covers X.", "The talk covers X."),
        ("The talk covers X. It then explains Y. Finally the speaker uses a ram rider to take out the la", "The talk covers X. It then explains Y."),
        ("One complete sentence.", "One complete sentence."),
        ("No punctuation at all so nothing to trim", "No punctuation at all so nothing to trim"),
        ('He said "done."', 'He said "done."'),
    ],
)
def test_clean_llm_text(raw, expected):
    assert clean_llm_text(raw) == expected
