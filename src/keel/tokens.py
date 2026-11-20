"""Token counting without MLX.

The trimmer (`trim.py`) and a server's context check count tokens. In-process MLX counts with the model's tokenizer through
mlx-lm; a server backend has no MLX loaded, so it reads the same `tokenizer.json` from the local model folder with
the `tokenizers` library (no download, no transformers import). Same file, same ids: "8k tokens" means the same on
every backend (checked against the mlx-lm tokenizer on every tool output in the research transcripts:
`tests/test_tokens.py`).

With no tokenizer file, counts fall back to characters / 2.0, and the counter says so (`approximate`), so a run that
trimmed by estimate is marked in its config. 2.0, not the usual ~4 for prose: tool output is JSON, numbers and paths
(median 2.2 characters per token over 13k research tool outputs), and an estimate that undercounts lets outputs past
the trim limit. The plan said 3.5; that undercounted the median output by 38%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CHARS_PER_TOKEN = 2.0          # measured on 13k research tool outputs: median 2.2 chars / token (P4)


def approx_count(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class TokenCounter:
    count: Callable[[str], int]
    method: str                          # "tokenizer.json" | "chars/2.0"
    source: str | None = None            # the tokenizer file used

    @property
    def approximate(self) -> bool:
        return self.method != "tokenizer.json"

    def __call__(self, text: str) -> int:
        return self.count(text)

    def record(self) -> dict:
        """For the run config."""
        return {"method": self.method, "approximate": self.approximate, "source": self.source}


def load_counter(model_dir: str | Path | None) -> TokenCounter:
    """The model's own tokenizer from `<model_dir>/tokenizer.json`; characters / 2.0 if that is missing or unreadable
    (or the `tokenizers` library isn't installed)."""
    path = Path(model_dir) / "tokenizer.json" if model_dir else None
    if path is not None and path.is_file():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(path))
        except Exception:
            tok = None
        if tok is not None:
            return TokenCounter(lambda s: len(tok.encode(s, add_special_tokens=False).ids), "tokenizer.json",
                                str(path))
    return TokenCounter(approx_count, f"chars/{CHARS_PER_TOKEN}")
