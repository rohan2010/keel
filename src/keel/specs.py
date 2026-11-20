"""Swappable configuration objects. A run is fully described by (ModelSpec, SamplingSpec, task name, problem ids)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str                      # short label used in run folders, e.g. "qwen3.5-0.8b"
    hf_repo: str                   # e.g. "Qwen/Qwen3.5-0.8B"
    revision: str                  # pinned commit, so weight hashes are reproducible
    local_dir: str                 # where the weights live: "models/<name>" (-> keel.paths.models_dir()) or absolute
    chat_template_kwargs: dict = field(default_factory=dict)   # e.g. {"enable_thinking": True}

    @property
    def path(self) -> Path:
        from .modelstore import resolve  # models folder, then the Hugging Face cache (pinned revision)
        return resolve(self)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5  # vLLM semantics: applied once to every token generated so far (not the prompt)
    max_tokens: int = 8192
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---- run samplers -------------------------------------------------------------------------------------

# keel's non-thinking sampler (unchanged since the research) and Qwen3.5's recommended thinking-mode sampler. The
# temperature is always given on the command line (`keel-core eval` makes runs greedy unless one is given).
PLAIN_SAMPLER = {"top_p": 0.8, "top_k": 20, "presence_penalty": 0.0}
THINKING_SAMPLER = {"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def add_thinking_arguments(ap) -> None:
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen thinking mode: the template's think block on, Qwen's thinking sampler (top-p 0.95, "
                         "presence 1.5); give the temperature with --temperature (Qwen: 0.6)")
    ap.add_argument("--top-p", type=float, default=None, help="override the mode's top-p (0.8 plain, 0.95 thinking)")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="override the mode's presence penalty (0.0 plain, 1.5 thinking)")


def run_spec_and_sampling(base: ModelSpec, a) -> tuple[ModelSpec, "SamplingSpec"]:
    """The model spec and sampler a run command uses. Without --thinking / overrides: exactly the old values."""
    thinking = bool(getattr(a, "thinking", False))
    preset = THINKING_SAMPLER if thinking else PLAIN_SAMPLER
    top_p = preset["top_p"] if getattr(a, "top_p", None) is None else a.top_p
    presence = preset["presence_penalty"] if getattr(a, "presence_penalty", None) is None else a.presence_penalty
    spec = ModelSpec(base.name, base.hf_repo, base.revision, base.local_dir, {"enable_thinking": thinking})
    return spec, SamplingSpec(temperature=a.temperature, top_p=top_p, top_k=preset["top_k"],
                              presence_penalty=presence, max_tokens=a.max_tokens, seed=a.seed)


# ---- presets ---------------------------------------------------------------------------------------------------------

QWEN35_08B = ModelSpec(
    name="qwen3.5-0.8b",
    hf_repo="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    local_dir="models/Qwen3.5-0.8B",
    chat_template_kwargs={"enable_thinking": True},
)

# 8-bit MLX conversion of Qwen/Qwen3.5-4B: the default model.
QWEN35_4B_8BIT = ModelSpec(
    name="qwen3.5-4b-8bit",
    hf_repo="mlx-community/Qwen3.5-4B-8bit",
    revision="1704583a86670ba531b7f2c92530a2f725cfd45b",
    local_dir="models/Qwen3.5-4B-8bit",
    chat_template_kwargs={"enable_thinking": True},
)

# Model-swap test: same family, 9B at 4-bit (~6 GB on disk). Not a fine-tune.
QWEN35_9B_4BIT = ModelSpec(
    name="qwen3.5-9b-4bit",
    hf_repo="mlx-community/Qwen3.5-9B-MLX-4bit",
    revision="938d8919941c6e7efd3c7150eff7fe9d12afa631",
    local_dir="models/Qwen3.5-9B-4bit",
    chat_template_kwargs={"enable_thinking": True},
)
