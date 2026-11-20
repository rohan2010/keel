"""Choosing a backend from command-line flags. Shared by `keel-core run`, `bfcl-mt` and `acp`.

    --backend mlx           in-process MLX (the validated path; default)
    --backend server        any local server: --url, --flavour openai|ollama|llama, --mode raw|chat
    --backend mlx-server    shortcut: server, openai flavour, http://127.0.0.1:8080, raw   (mlx_lm.server)
    --backend ollama        shortcut: server, ollama flavour, http://127.0.0.1:11434, raw  (needs --server-model)

Raw mode renders keel's own prompt, so it needs the model's tokenizer + chat template: `--model-dir` (default: the
model preset's local folder), loaded with transformers (no MLX, no weights). That folder also gives the token counter
for trimming and the context check, and the `<tool_call>` token id for the loop breaker's logit_bias rung. Chat mode
works without one (trimming then counts approximately, and the run says so).

Every run records what it ran on: `capabilities_record()` -> `capabilities.json` in the run folder / the ACP session.
Server results are a different backend from in-process MLX: never `validated`, noise from repeats (decision 5).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..specs import ModelSpec, SamplingSpec

SERVER_BACKENDS = ("server", "mlx-server", "ollama")
SHORTCUTS = {"mlx-server": ("openai", "http://127.0.0.1:8080"), "ollama": ("ollama", "http://127.0.0.1:11434")}
MLX_ONLY_FLAGS = ("stepwise", "state_saves", "shared_prefix", "probe_gate")      # scheduler features a server doesn't have


def add_backend_arguments(ap: argparse.ArgumentParser, extra_choices: tuple[str, ...] = ()) -> None:
    g = ap.add_argument_group("model backend")
    g.add_argument("--backend", choices=("mlx", *SERVER_BACKENDS, *extra_choices), default="mlx",
                   help="mlx = in-process (default, validated); server / mlx-server / ollama = a local server")
    g.add_argument("--url", default=None, help="server address (default: the shortcut's, or http://127.0.0.1:8080)")
    g.add_argument("--flavour", choices=("openai", "ollama", "llama"), default=None,
                   help="--backend server: wire format (default openai)")
    g.add_argument("--mode", choices=("raw", "chat"), default="raw",
                   help="raw = keel renders the prompt and parses calls (compares with MLX); chat = the server does")
    g.add_argument("--server-model", default="", help="the model's name on the server (Ollama needs it)")
    g.add_argument("--model-dir", default=None,
                   help="tokenizer + chat template folder (default: the model preset's local folder)")
    g.add_argument("--ban-rungs", default=None,
                   help="loop-breaker rungs the server honours, comma-separated (logit_bias,tool_choice,stop; "
                        "'none' for none); default per server type")


@dataclass
class ServerSetup:
    backend: Any                                  # ServerBackend
    tok: Any                                      # HF tokenizer (raw mode, or chat mode when the folder exists)
    trimmer: Callable | None
    trim_record: dict
    model_dir: str | None
    extra: dict = field(default_factory=dict)

    def capabilities_record(self) -> dict:
        b = self.backend
        return {"backend": b.name, "kind": "server", "mode": b.mode, "flavour": b.flavour, "url": b.url,
                "server_model": b.model, "model_dir": self.model_dir, "capabilities": b.capabilities().to_dict(),
                "ban_rung": b.ban_rung(), "ban_rungs": list(b.ban_rungs), "sampling": b.sampling_record(),
                "trim": self.trim_record, "noise": "repeats (server: no per-episode paired streams)", **self.extra}


def mlx_capabilities_record(trim_outputs: bool) -> dict:
    from .mlx import CAPABILITIES  # imports MLX; only called on the MLX path, where it is loaded anyway
    return {"backend": "mlx", "kind": "in-process", "capabilities": CAPABILITIES.to_dict(), "ban_rung": "logits mask",
            "trim": {"method": "mlx tokenizer", "approximate": False} if trim_outputs else None,
            "noise": "paired (per-episode random streams)"}


def add_gate_argument(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--probe-gate", action="store_true",
                    help="experimental, in-process MLX only: a frozen probe decides call / no-call before the first "
                         "token (helped BFCL single-turn only; no effect measured on multi-turn or When2Call)")


def load_gate(a: argparse.Namespace, spec: ModelSpec):
    """The probe gate for `--probe-gate`, or None. Refused (SystemExit) on any backend without hidden states, with
    the MLX-only scheduler options it isn't combined with, and on any model other than the one it was fit on."""
    if not getattr(a, "probe_gate", False):
        return None
    if a.backend != "mlx":
        raise SystemExit(f"--probe-gate needs the model's hidden states: in-process MLX only, not --backend {a.backend}")
    clash = [f"--{f.replace('_', '-')}" for f in ("stepwise", "state_saves", "shared_prefix") if getattr(a, f, False)]
    if clash:
        raise SystemExit(f"--probe-gate is not combined with {', '.join(clash)}")
    from .mlx_gate import GateRefused, ProbeGate
    try:
        gate = ProbeGate.load(spec.path)
    except GateRefused as err:
        raise SystemExit(f"--probe-gate refused: {err}") from err
    import sys
    print("WARNING --probe-gate is experimental: validated on BFCL single-turn only. On multi-turn and agent tasks it "
          "did not transfer (AUROC 0.53) and can block calls a task needs (agent task B1 blocked at p=0.083 and "
          "failed).", file=sys.stderr, flush=True)
    return gate


def check_server_flags(a: argparse.Namespace) -> None:
    """Refuse flags that only the in-process scheduler has, instead of silently ignoring them."""
    used = [f"--{f.replace('_', '-')}" for f in MLX_ONLY_FLAGS if getattr(a, f, False)]
    if used:
        raise SystemExit(f"{', '.join(used)}: in-process MLX only (not with --backend {a.backend})")
    if a.backend == "ollama" and not a.server_model:
        raise SystemExit("--backend ollama needs --server-model (the model's name in `ollama list`)")


def load_tokenizer(model_dir: str | Path):
    """The model's tokenizer + chat template, without MLX or weights."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir))


def server_setup(a: argparse.Namespace, spec: ModelSpec, sampling: SamplingSpec, *, trim_outputs: bool = True,
                 stream: bool = False, tokenizer_loader=load_tokenizer) -> ServerSetup:
    from ..tokens import load_counter
    from ..trim import trim
    from .server import ServerBackend
    check_server_flags(a)
    flavour, url = SHORTCUTS.get(a.backend, (a.flavour or "openai", None))
    if a.backend != "server" and a.flavour and a.flavour != flavour:
        raise SystemExit(f"--backend {a.backend} is the {flavour} flavour; use --backend server --flavour {a.flavour}")
    model_dir = a.model_dir or spec.local_dir      # as given: what the run records (no absolute paths)
    model_path = Path(a.model_dir) if a.model_dir else spec.path   # spec default: keel.paths.models_dir(), not cwd
    have_dir = bool(model_dir) and (model_path / "tokenizer_config.json").is_file()
    if a.mode == "raw" and not have_dir:
        raise SystemExit(f"raw mode renders keel's prompt and needs the model's tokenizer folder ({model_dir!r} not "
                         "found); pass --model-dir, or use --mode chat")
    tok = tokenizer_loader(model_path) if have_dir else None
    counter = load_counter(model_path if have_dir else None)
    tool_call_id = None
    if tok is not None:
        tid = tok.convert_tokens_to_ids("<tool_call>")
        tool_call_id = tid if isinstance(tid, int) and tid != getattr(tok, "unk_token_id", None) else None
    rungs, rungs_source = None, "default for this server type"
    if a.ban_rungs is not None:
        rungs = () if a.ban_rungs.strip() in ("", "none") else tuple(r.strip() for r in a.ban_rungs.split(","))
        rungs_source = "--ban-rungs"
    else:                                          # what `keel doctor` measured on this server, if it was probed
        from ..agent.sandbox import default_state_dir
        from ..doctor import measured_rungs
        from .server import DEFAULT_URLS
        state = Path(getattr(a, "state_dir", None) or default_state_dir())
        found = measured_rungs(state, (a.url or url or DEFAULT_URLS[flavour]).rstrip("/"), flavour, a.mode)
        if found is not None:
            rungs, rungs_source = found
    decode = (lambda ids: tok.decode(ids)) if (tok is not None and a.mode == "raw" and flavour == "openai") else None
    try:
        backend = ServerBackend(url=a.url or url or "", flavour=flavour, mode=a.mode, sampling=sampling,
                                model=a.server_model, decode=decode, tool_call_id=tool_call_id,
                                count_tokens=counter if (a.mode == "raw" and not counter.approximate) else None,
                                stream=stream, ban_rungs=rungs)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    trimmer = (lambda text: trim(text, counter)) if trim_outputs else None
    return ServerSetup(backend, tok, trimmer, counter.record() if trim_outputs else {"off": True},
                       str(model_dir) if have_dir else None, {"ban_rungs_source": rungs_source})


def agent_prompt_fn(setup: ServerSetup, spec: ModelSpec):
    """The agent loop's model input for this server: rendered text (raw) or messages + tools (chat)."""
    from ..agent.loop import chat_prompt, prompt_text
    if setup.backend.mode == "chat":
        return chat_prompt
    tok = setup.tok
    return lambda e: prompt_text(tok, spec, e.messages, e.toolset)
