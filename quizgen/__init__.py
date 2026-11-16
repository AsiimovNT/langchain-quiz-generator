"""Quiz Generation Using LangChain.

YouTube video -> audio -> Whisper transcript -> local Mistral GGUF (via
LangChain) -> summary + multiple-choice quiz.
"""

import os

# Download models from the Hugging Face Hub over plain resumable HTTP. The
# default Xet transfer backend stalled at 97% of the 4.4 GB GGUF during
# development; plain HTTP was fast and resumes cleanly. Set
# HF_HUB_DISABLE_XET=0 to switch Xet back on. (Must be set before
# huggingface_hub is first imported.)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "1.0.0"
