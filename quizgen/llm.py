"""LLM backend: a local GGUF model run by llama.cpp, exposed as a LangChain LLM.

The report's "LLM Backend Configuration" has two halves, both implemented here:

* **Hugging Face integration** - :func:`download_gguf` fetches the GGUF file
  from the Hugging Face Hub.
* **Local model setup** - :class:`LocalGGUF` loads that file with llama.cpp and
  plugs it into LangChain, so prompts/chains/parsers compose with ``|``.

Nothing is sent to an external API: inference runs entirely on this machine.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict, PrivateAttr

from .config import Settings
from .errors import GenerationError, ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)

_cuda_dirs_registered = False


def register_cuda_dll_dirs() -> list[str]:
    """Make pip-installed CUDA runtime DLLs visible to llama.cpp (Windows only).

    The CUDA build of llama-cpp-python links against ``cublas64_12.dll`` and
    ``cudart64_12.dll``. Without a system-wide CUDA Toolkit these come from the
    ``nvidia-*-cu12`` wheels, which unpack into ``site-packages/nvidia/*/bin``;
    that folder is not on the DLL search path by default.
    """
    global _cuda_dirs_registered
    if sys.platform != "win32" or _cuda_dirs_registered:
        return []
    _cuda_dirs_registered = True

    import site

    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:  # pragma: no cover - user site can be disabled
        pass

    found: list[str] = []
    for root in dict.fromkeys(roots):
        for bin_dir in Path(root, "nvidia").glob("*/bin"):
            if not any(bin_dir.glob("*.dll")):
                continue
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
            found.append(str(bin_dir))
    return found


def _import_llama_cpp():
    register_cuda_dll_dirs()
    try:
        import llama_cpp
    except Exception as exc:  # ImportError or a DLL load failure
        raise ModelLoadError(
            "llama-cpp-python could not be loaded. If this machine has no NVIDIA GPU, "
            "install the CPU build instead:\n"
            "  pip install --force-reinstall llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
            f"Original error: {exc}"
        ) from exc
    return llama_cpp


# --------------------------------------------------------------------------
# Hugging Face integration
# --------------------------------------------------------------------------
def download_gguf(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download ``filename`` from the Hugging Face repo ``repo_id`` into ``dest_dir``.

    ``progress_callback(downloaded_bytes, total_bytes)`` is called as data arrives.
    Interrupted downloads resume where they stopped.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {}
    if progress_callback is not None:
        from tqdm.auto import tqdm

        class _CallbackBar(tqdm):
            def update(self, n=1):
                result = super().update(n)
                try:
                    progress_callback(int(self.n), int(self.total or 0))
                except Exception:
                    pass
                return result

        kwargs["tqdm_class"] = _CallbackBar

    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir), **kwargs)
    return Path(path)


# --------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------
def free_vram_mb() -> int | None:
    """Free memory on the first NVIDIA GPU in MiB, or ``None`` if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _block_count(llama_cpp, model_path: Path) -> int:
    """Number of transformer blocks, read from the GGUF metadata."""
    try:
        probe = llama_cpp.Llama(model_path=str(model_path), vocab_only=True, verbose=False)
        try:
            for key, value in probe.metadata.items():
                if key.endswith(".block_count"):
                    return int(value)
        finally:
            probe.close()
    except Exception:
        pass
    return 32


def resolve_gpu_layers(llama_cpp, model_path: Path, n_ctx: int, requested: int) -> int:
    """Decide how many layers to put on the GPU.

    ``requested`` is the user's setting: ``0`` forces CPU, a positive number is
    used as-is, and ``-1`` means "as many as fit". For ``-1`` the free VRAM is
    checked first so a GPU that is busy with other apps gets a partial offload
    instead of spilling into (very slow) shared system memory.
    """
    if requested == 0 or not llama_cpp.llama_supports_gpu_offload():
        return 0
    if requested > 0:
        return requested

    free = free_vram_mb()
    if free is None:
        return -1
    size_mb = model_path.stat().st_size / (1024 * 1024)
    overhead_mb = 700 + n_ctx * 0.15  # KV cache + compute buffers + safety margin
    if free >= size_mb + overhead_mb:
        return -1
    blocks = _block_count(llama_cpp, model_path)
    fraction = max(0.0, (free - overhead_mb) / size_mb)
    layers = int(fraction * (blocks + 1))
    logger.warning(
        "Only %d MiB VRAM free (model needs ~%d MiB): offloading %d/%d layers.",
        free,
        int(size_mb + overhead_mb),
        layers,
        blocks + 1,
    )
    return layers


# --------------------------------------------------------------------------
# LangChain wrapper
# --------------------------------------------------------------------------
class LocalGGUF(LLM):
    """A llama.cpp model as a LangChain ``LLM``.

    Each prompt is sent as a single user turn through the model's own chat
    template (read from the GGUF metadata), so the ``[INST] ... [/INST]``
    wrapping Mistral expects is applied automatically and other instruct models
    work without changing the prompts.

    Per-call overrides can be bound the LangChain way, e.g.
    ``llm.bind(max_tokens=300)`` or ``llm.bind(response_format={...})`` for
    JSON-schema-constrained output.
    """

    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: int | None = None
    temperature: float = 0.3
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 512

    _client: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)
    _gpu_layers_used: int = PrivateAttr(default=0)
    _last_finish_reason: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(protected_namespaces=())

    # -- loading -----------------------------------------------------------
    def load(self) -> "LocalGGUF":
        """Load the model into memory (idempotent). Falls back to CPU if the GPU load fails."""
        if self._client is not None:
            return self
        path = Path(self.model_path)
        if not path.is_file():
            raise ModelNotFoundError(f"GGUF model not found: {path}")

        llama_cpp = _import_llama_cpp()
        layers = resolve_gpu_layers(llama_cpp, path, self.n_ctx, self.n_gpu_layers)

        def _create(gpu_layers: int):
            return llama_cpp.Llama(
                model_path=str(path),
                n_ctx=self.n_ctx,
                n_gpu_layers=gpu_layers,
                n_threads=self.n_threads,
                n_batch=512,
                verbose=False,
            )

        try:
            self._client = _create(layers)
            self._gpu_layers_used = layers
        except Exception as gpu_error:
            if layers == 0:
                raise ModelLoadError(f"Could not load {path.name}: {gpu_error}") from gpu_error
            logger.warning("GPU load failed (%s); retrying on CPU.", gpu_error)
            try:
                self._client = _create(0)
                self._gpu_layers_used = 0
            except Exception as cpu_error:
                raise ModelLoadError(f"Could not load {path.name}: {cpu_error}") from cpu_error
        return self

    @property
    def device_description(self) -> str:
        if self._client is None:
            return "not loaded"
        if self._gpu_layers_used == 0:
            return "CPU"
        if self._gpu_layers_used < 0:
            return "GPU (all layers)"
        return f"GPU ({self._gpu_layers_used} layers) + CPU"

    @property
    def last_finish_reason(self) -> str | None:
        """``"stop"`` if the last answer ended naturally, ``"length"`` if it hit ``max_tokens``."""
        return self._last_finish_reason

    # -- LangChain interface -------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "llama-cpp-gguf"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": Path(self.model_path).name,
            "n_ctx": self.n_ctx,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def get_num_tokens(self, text: str) -> int:
        """Exact token count using the model's own tokenizer."""
        self.load()
        return len(self._client.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.load()
        max_tokens = int(kwargs.get("max_tokens") or self.max_tokens)

        # Keep prompt + answer inside the context window instead of crashing.
        prompt_tokens = self.get_num_tokens(prompt) + 16  # chat-template tokens
        room = self.n_ctx - prompt_tokens
        if room < 64:
            raise GenerationError(
                f"Prompt is too long for the model's context window "
                f"({prompt_tokens} tokens, n_ctx={self.n_ctx})."
            )
        max_tokens = min(max_tokens, room)

        params: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if stop:
            params["stop"] = stop
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        try:
            with self._lock:  # a llama.cpp context is not thread-safe
                result = self._client.create_chat_completion(**params)
        except Exception as exc:
            if "response_format" not in params:
                raise GenerationError(f"The local model failed to generate: {exc}") from exc
            # llama.cpp rejected the schema-derived grammar. The model itself is
            # unaffected, so answer free-form and let the caller's parser cope.
            logger.warning("Constrained decoding failed (%s); retrying without a schema.", exc)
            del params["response_format"]
            try:
                with self._lock:
                    result = self._client.create_chat_completion(**params)
            except Exception as retry_exc:
                raise GenerationError(f"The local model failed to generate: {retry_exc}") from retry_exc

        choice = result["choices"][0]
        self._last_finish_reason = choice.get("finish_reason")
        return (choice["message"].get("content") or "").strip()


def load_llm(settings: Settings) -> LocalGGUF:
    """Build and load the LLM described by ``settings``."""
    path = settings.model_path
    if not path.is_file():
        raise ModelNotFoundError(
            f"GGUF model not found at {path}.\n"
            f"Download it with:  python scripts/download_model.py\n"
            f"(source: https://huggingface.co/{settings.gguf_repo_id})"
        )
    return LocalGGUF(
        model_path=str(path),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        n_threads=settings.n_threads,
        temperature=settings.temperature,
    ).load()
