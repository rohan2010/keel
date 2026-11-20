"""Probe gate: an experimental call / no-call decision before the first token. In-process MLX only.

Ported from the research gate (exp-weights `wsp/gate.py` / X1). A frozen linear probe reads the hidden
state of the last prompt token at layer 16 and decides, before the model writes anything:
    p = sigmoid(((h - mu) / sd) . w + b);   |p - 0.5| < margin -> "free" (the model decides as usual)
                                             p >= 0.5          -> "force": the first token is <tool_call>
                                             p <  0.5          -> "block": <tool_call> is banned for the whole reply

What it can claim (from the research):
- validated on BFCL **single-turn**: 80.6% -> 86.5% (p = 7e-15), live, frozen probe, margin 0.27;
- **not** on multi-turn: AUROC 0.53 on BFCL multi-turn (it reads intent, not whether a call is right), so it is not
  expected to help the agent tasks; it is off by default and labelled experimental;
- only for the exact model it was fit on: any other model is refused (fingerprint = sha1 of config.json and every
  safetensors header), and the probe file itself is checked against its frozen sha1.

How it runs (`decide`): the caller's cache (fresh, or an episode's reused cache) gets the new prompt tokens except
the last, in chunks; then the last token alone, with a hook on the probe layer; that one forward gives both the
probe's input and the first token's logits, so there is no second prefill. The scheduler (`run_scheduler(...,
gate=)`) then picks the first token and hands cache + first token to the batch generator.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import mlx.core as mx
import numpy as np

TOOL_CALL = "<tool_call>"
DEFAULT_GATE = "qwen3.5-4b-8bit_L16"


class GateRefused(RuntimeError):
    """The gate can't be used here (other model, changed probe file, backend without hidden states)."""


def model_fingerprint(model_dir: str | Path) -> str:
    """sha1 of config.json + every safetensors header (tensor names, shapes, dtypes, offsets). Cheap: no weights."""
    d = Path(model_dir).resolve()
    h = hashlib.sha1()
    h.update((d / "config.json").read_bytes())
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise GateRefused(f"no safetensors files in {d}")
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h.update(fh.read(n))
    return h.hexdigest()


def gate_files(name: str = DEFAULT_GATE) -> tuple[Path, Path]:
    base = resources.files("keel") / "data" / "gates"
    return Path(str(base / f"{name}.npz")), Path(str(base / f"{name}.json"))


@dataclass
class Decision:
    p_call: float
    action: str                      # "force" | "block" | "free"


class ProbeGate:
    def __init__(self, npz: Path, meta: dict, margin: float | None = None):
        g = np.load(npz)
        self.path, self.meta = str(npz), meta
        self.layer = int(g["layer"])
        self.margin = float(g["margin"]) if margin is None else margin
        self.mu = mx.array(g["mu"].astype(np.float32))
        self.sd = mx.array(g["sd"].astype(np.float32))
        self.w = mx.array(g["w"].astype(np.float32))
        self.b = float(g["b"])

    @classmethod
    def load(cls, model_dir: str | Path, name: str = DEFAULT_GATE, margin: float | None = None) -> "ProbeGate":
        """The packaged gate, after checking the probe file and that `model_dir` is the model it was fit on."""
        npz, js = gate_files(name)
        meta = json.loads(js.read_text())
        sha = hashlib.sha1(npz.read_bytes()).hexdigest()
        if sha != meta["sha1_npz"]:
            raise GateRefused(f"probe file {npz.name} changed (sha1 {sha}, frozen {meta['sha1_npz']})")
        want = meta["model"]["fingerprint"]
        got = model_fingerprint(model_dir)
        if got != want:
            raise GateRefused(f"the probe gate only works with the model it was fit on "
                              f"({meta['model']['hf_repo']} @ {meta['model']['revision'][:8]}); "
                              f"{model_dir} has a different fingerprint")
        return cls(npz, meta, margin)

    def prob(self, h: mx.array) -> float:
        z = ((h.astype(mx.float32) - self.mu) / self.sd) @ self.w + self.b
        return float(mx.sigmoid(z).item())

    def action(self, p: float) -> str:
        if abs(p - 0.5) < self.margin:
            return "free"
        return "force" if p >= 0.5 else "block"

    def record(self) -> dict:
        return {"gate": Path(self.path).stem, "layer": self.layer, "margin": self.margin,
                "sha1_npz": self.meta["sha1_npz"], "model": self.meta["model"]["hf_repo"],
                "validated": self.meta.get("validated"), "not_for": self.meta.get("not_for"),
                "experimental": True}


class _Hook:
    """Wraps one decoder layer and keeps its last output (a lazy array; cheap)."""

    def __init__(self, layer):
        self._layer = layer
        self.out = None

    def __call__(self, *a, **k):
        self.out = self._layer(*a, **k)
        return self.out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class Hooked:
    """Context manager: the probe layer hooked for the duration (restored even on errors)."""

    def __init__(self, model, layer: int):
        self.layers = model.language_model.model.layers
        self.i = layer
        self.hook = _Hook(self.layers[layer])

    def __enter__(self) -> _Hook:
        self.layers[self.i] = self.hook
        return self.hook

    def __exit__(self, *exc) -> None:
        self.layers[self.i] = self.hook._layer


def decide(model, gate: ProbeGate, hook: _Hook, new_ids: list[int], cache, prefill_step: int = 1024
           ) -> tuple[Decision, mx.array]:
    """Feed `new_ids` into `cache` (all but the last in chunks, then the last alone); return the gate's decision and
    the first token's logits [1, vocab]. `cache` is filled in place and ready for decoding."""
    for c0 in range(0, len(new_ids) - 1, prefill_step):
        model(mx.array(new_ids[c0:min(c0 + prefill_step, len(new_ids) - 1)])[None], cache=cache)
        mx.eval([c.state for c in cache])
    logits = model(mx.array(new_ids[-1:])[None], cache=cache)[:, -1, :]
    p = gate.prob(hook.out[0, -1])
    return Decision(p, gate.action(p)), logits
