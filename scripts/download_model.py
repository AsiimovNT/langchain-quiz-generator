"""Download the GGUF model from the Hugging Face Hub into ``models/``.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
        --file mistral-7b-instruct-v0.2.Q5_K_M.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quizgen.config import Settings  # noqa: E402
from quizgen.llm import download_gguf  # noqa: E402


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--repo", default=settings.gguf_repo_id, help="Hugging Face repo id")
    parser.add_argument("--file", default=settings.gguf_filename, help="GGUF file name in the repo")
    parser.add_argument("--dest", default=str(settings.models_dir), help="Destination folder")
    args = parser.parse_args()

    print(f"Downloading {args.file} from {args.repo} -> {args.dest}")
    path = download_gguf(args.repo, args.file, Path(args.dest))
    print(f"Model ready: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
