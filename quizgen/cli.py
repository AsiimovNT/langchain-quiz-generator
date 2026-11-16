"""Command-line interface: run the whole pipeline without the web UI.

    python -m quizgen.cli https://www.youtube.com/watch?v=VIDEO_ID
    python -m quizgen.cli URL --easy 2 --medium 1 --hard 0 --summary-tokens 400 --json out.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .errors import QuizGenError
from .models import GenerationRequest
from .pipeline import STAGE_LABELS, QuizPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quizgen", description="YouTube video -> summary + MCQ quiz, fully local.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--easy", type=int, default=1, help="number of easy MCQs (0-3)")
    parser.add_argument("--medium", type=int, default=1, help="number of medium MCQs (0-3)")
    parser.add_argument("--hard", type=int, default=1, help="number of hard MCQs (0-3)")
    parser.add_argument("--summary-tokens", type=int, default=300, help="summary max tokens (150-500)")
    parser.add_argument("--json", type=Path, help="also write the full result to this JSON file")
    parser.add_argument("--verbose", action="store_true", help="show debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    for stream in (sys.stdout, sys.stderr):  # video titles may contain non-ASCII characters
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        request = GenerationRequest(
            url=args.url,
            num_easy=args.easy,
            num_medium=args.medium,
            num_hard=args.hard,
            summary_max_tokens=args.summary_tokens,
        )
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"])
            print(f"Invalid {field}: {error['msg']}", file=sys.stderr)
        return 2

    last = {"key": None}

    def progress(stage: str, fraction: float, message: str) -> None:
        key = (stage, message)
        if key != last["key"]:
            last["key"] = key
            print(f"[{STAGE_LABELS[stage]:<26} {fraction:4.0%}] {message}", file=sys.stderr)

    try:
        content = QuizPipeline(Settings()).run(request, progress)
    except QuizGenError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== Summary: {content.video.title} ===\n")
    print(content.summary)
    if content.mcqs:
        print("\n=== MCQs ===")
    for number, mcq in enumerate(content.mcqs, start=1):
        print(f"\nQ{number}. [{mcq.difficulty.value}] {mcq.question}")
        for letter, option in zip("ABCD", mcq.options):
            print(f"   {letter}) {option}")
        print(f"   Answer: {'ABCD'[mcq.answer_index]}  -  {mcq.explanation}")
    for warning in content.warnings:
        print(f"\nWarning: {warning}", file=sys.stderr)
    print("\nTimings: " + ", ".join(f"{stage} {seconds:.1f}s" for stage, seconds in content.timings.items()), file=sys.stderr)

    if args.json:
        args.json.write_text(content.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
