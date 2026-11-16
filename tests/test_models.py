import pytest
from pydantic import ValidationError

from quizgen.errors import InvalidURLError
from quizgen.models import MCQ, Difficulty, GenerationRequest, extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s&list=PL123",
        f"https://youtube.com/watch?feature=share&v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?si=abcdef",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}?feature=share",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"  https://www.youtube.com/watch?v={VIDEO_ID}  ",
        VIDEO_ID,
    ],
)
def test_extract_video_id_accepts_common_forms(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "https://vimeo.com/123456789",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@somechannel",
    ],
)
def test_extract_video_id_rejects_non_videos(url):
    with pytest.raises(InvalidURLError):
        extract_video_id(url)


def test_request_defaults_match_the_ui():
    request = GenerationRequest(url=f"https://youtu.be/{VIDEO_ID}")
    assert (request.num_easy, request.num_medium, request.num_hard) == (1, 1, 1)
    assert request.summary_max_tokens == 300
    assert request.total_mcqs == 3
    assert request.video_id == VIDEO_ID
    assert request.counts == {Difficulty.EASY: 1, Difficulty.MEDIUM: 1, Difficulty.HARD: 1}


@pytest.mark.parametrize(
    "field,value",
    [("num_easy", 4), ("num_hard", -1), ("summary_max_tokens", 149), ("summary_max_tokens", 501)],
)
def test_request_enforces_slider_ranges(field, value):
    with pytest.raises(ValidationError):
        GenerationRequest(url=f"https://youtu.be/{VIDEO_ID}", **{field: value})


def test_request_rejects_bad_url():
    with pytest.raises(ValidationError):
        GenerationRequest(url="https://example.com/video")


def _mcq(**overrides):
    data = dict(
        question="What colour is the sky on a clear day?",
        options=["Blue", "Green", "Red", "Yellow"],
        answer_index=0,
        explanation="Rayleigh scattering.",
        difficulty=Difficulty.EASY,
    )
    data.update(overrides)
    return MCQ(**data)


def test_mcq_valid():
    mcq = _mcq()
    assert mcq.answer == "Blue"


@pytest.mark.parametrize(
    "overrides",
    [
        {"options": ["Blue", "Green", "Red"]},
        {"options": ["Blue", "Green", "Red", "Yellow", "Pink"]},
        {"options": ["Blue", "blue", "Red", "Yellow"]},
        {"options": ["Blue", "", "Red", "Yellow"]},
        {"answer_index": 4},
        {"answer_index": -1},
        {"question": "Why?"},
    ],
)
def test_mcq_rejects_malformed(overrides):
    with pytest.raises(ValidationError):
        _mcq(**overrides)
