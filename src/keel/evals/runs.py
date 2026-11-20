"""Run folders -> one shape. Read-only: old research run folders are read, never rewritten.

Every suite keeps its own `outputs.jsonl` / `config.json` layout (agent, BFCL multi-turn, When2Call, across code
versions since 2026-11-06). `load_run(dir)` maps any of them onto:

    RunHeader   what the run was: suite, model, backend, batch size, sampler, split / tier, item set, code hash,
                layers on. `keel compare` checks two headers before it compares anything.
    ItemRecord  one line per item and try: correct, score, false-success label, tokens, seconds.

Rules:
- `outputs.jsonl` is the current scoring. `outputs_scored_<time>.jsonl` files are backups of EARLIER scorings
  (the research tool rescore_agent.py, not part of this repo, wrote the old one there before rewriting outputs.jsonl). The research false-success
  script read the latest backup instead ("latest rescore wins"), which in 23 of 29 research agent runs is the scoring
  before the last rescore (found in P0, 2026-11-16).
- A run that can't be read raises `RunUnreadable` with the reason; nothing is skipped silently.
- An incomplete run (fewer outputs than `config.items`) loads, with `header.missing` listing what never finished.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUITES = {"agent": "agent", "bfcl_multi_turn": "bfcl-mt", "when2call": "when2call"}
# split name -> tier (dev: build and tune; validation: confirm; holdout: report once)
TIERS = {"dev": "dev", "test": "validation", "val": "validation", "holdout": "holdout", "held_out": "holdout",
         "seen": "seen"}
# config keys that switch a harness layer or change what the model sees (absent in older runs = off / default)
LAYER_KEYS = ("toolset", "verify", "recovery", "reminder", "loop_break", "trim_outputs", "probe_gate", "m5",
              "reissue", "loop_note", "long_note", "state_saves", "shared_prefix", "rendering", "check_requests",
              "user", "scheduler", "done_gate", "gnu_tools", "examples", "send_note", "search_hint")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "presence_penalty", "max_tokens", "seed")


class RunUnreadable(ValueError):
    """The folder is not a run of a known suite, or its files can't be read."""


@dataclass
class ItemRecord:
    suite: str
    split: str
    item_id: str
    try_: int
    correct: bool
    score: float | None            # agent: the scorer's partial score; others: None
    fs_label: str | None           # pass | false_success | honest_fail | no_answer; None = outside the measure
    prompt_tokens: int | None
    processed_tokens: int | None   # prompt tokens actually read (after cache reuse)
    output_tokens: int | None
    seconds: float | None
    group: str = ""                # agent task, BFCL category, When2Call label: the per-group breakdown
    stop: str | None = None
    kind: str = ""                 # agent task kind (state / answer / limits / safety); research pass rates leave
                                   # out "safety" (overall_useful)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["try"] = d.pop("try_")
        return d


@dataclass
class RunHeader:
    dir: str
    suite: str
    name: str                      # the run's folder name
    started: str | None
    model: dict                    # name, hf_repo, revision, chat_template_kwargs
    backend: str                   # mlx | server-raw:... | server-chat:...
    batch_size: int | None
    sampling: dict
    rng: str | None
    split: str
    tier: str
    items: list[str]               # item ids the run was meant to cover (config)
    items_sha1: str
    code: str | None               # code hash at run time (runs since 2026-11-08), else None
    env: dict
    layers: dict                   # LAYER_KEYS present in the config
    limits: dict                   # max_steps, max_context, context_budget
    missing: list[str] = field(default_factory=list)   # items in the config with no output (unfinished run)
    prompt: str | None = None      # sha1[:12] of the recorded system prompt (agent runs), so a changed prompt shows

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        return {**asdict(self), "complete": self.complete}


def _read_jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as err:
        raise RunUnreadable(f"{p}: {err}") from err


def _backend(c: dict) -> str:
    srv = c.get("server")
    if isinstance(srv, dict) and srv.get("backend"):
        return str(srv["backend"])
    return str(c.get("backend") or "mlx")


def header(run_dir: str | Path) -> RunHeader:
    d = Path(run_dir)
    try:
        c = json.loads((d / "config.json").read_text())
    except FileNotFoundError as err:
        raise RunUnreadable(f"{d}: no config.json") from err
    except (OSError, json.JSONDecodeError) as err:
        raise RunUnreadable(f"{d}: config.json unreadable ({err})") from err
    kind = c.get("kind")
    if kind not in SUITES:
        raise RunUnreadable(f"{d}: kind {kind!r} is not an eval suite keel compares "
                            f"({', '.join(SUITES)})")
    items = [str(x) for x in c.get("items") or []]
    s = c.get("sampling") or {}
    split = str(c.get("split") or "?")
    return RunHeader(
        dir=str(d), suite=SUITES[kind], name=d.name, started=c.get("started"),
        model={k: (c.get("model") or {}).get(k) for k in ("name", "hf_repo", "revision", "chat_template_kwargs")},
        backend=_backend(c), batch_size=c.get("batch_size"),
        sampling={k: s.get(k) for k in SAMPLING_KEYS}, rng=c.get("rng"),
        split=split, tier=TIERS.get(split, split), items=items,
        items_sha1=hashlib.sha1("\n".join(sorted(items)).encode()).hexdigest()[:12],
        code=(c["code"].get("combined") if isinstance(c.get("code"), dict) else c.get("code")),
        env=c.get("env") or {}, layers={k: c[k] for k in LAYER_KEYS if k in c},
        limits={k: c.get(k) for k in ("max_steps", "max_context", "context_budget") if k in c},
        prompt=hashlib.sha1(c["system"].encode()).hexdigest()[:12] if isinstance(c.get("system"), str) else None)


# ---- per-suite item readers --------------------------------------------------------------------------------------

def _agent(r: dict, split: str) -> ItemRecord:
    from ..false_success import agent_label
    return ItemRecord("agent", split, r["id"], int(r.get("try", 0)), bool(r["correct"]), float(r["score"]),
                      agent_label(r), r.get("prompt_tokens"), r.get("processed_tokens"), r.get("output_tokens"),
                      r.get("seconds"), group=r.get("task", ""), stop=r.get("stop"), kind=r.get("kind", ""))


_BFCL_ENTRIES: dict = {}


def _bfcl_fs(r: dict) -> str:
    """bfcl_label on the first failing turn; the turn comes from the checker's message, else from re-running the
    official checker on growing prefixes (needs bfcl-eval; without it, the last turn is used)."""
    from ..false_success import bfcl_label, first_failing_turn
    if r.get("valid"):
        return "pass"
    turn = first_failing_turn(r)
    if turn is None:
        try:
            if not _BFCL_ENTRIES:
                from . import bfcl_mt as M
                for cat in M.CATEGORIES:
                    _BFCL_ENTRIES.update({e["id"]: e for e in M.load(cat)})
            turn = first_failing_turn(r, _BFCL_ENTRIES.get(r["id"]))
        except ImportError:
            turn = None
    return bfcl_label(r, turn)


def _bfcl(r: dict, split: str, fs: bool) -> ItemRecord:
    return ItemRecord("bfcl-mt", split, r["id"], int(r.get("try", 0)), bool(r["valid"]), None,
                      _bfcl_fs(r) if fs else None, r.get("prompt_tokens"), r.get("processed_tokens"),
                      r.get("output_tokens"), r.get("seconds"), group=r.get("category", ""), stop=r.get("stop"))


def _when2call(r: dict, split: str) -> ItemRecord:
    return ItemRecord("when2call", split, r["id"], int(r.get("try", 0)), bool(r["correct"]), None, None,
                      r.get("prompt_tokens"), None, r.get("output_tokens"), r.get("seconds"),
                      group=r.get("label", ""), stop=r.get("finish_reason"))


def load_run(run_dir: str | Path, *, fs: bool = True) -> tuple[RunHeader, list[ItemRecord]]:
    """Header + one record per output line. `fs=False` skips the BFCL false-success labels (they may re-run the
    official checker on turn prefixes, which is the slow part)."""
    h = header(run_dir)
    out = Path(run_dir) / "outputs.jsonl"
    if not out.exists():
        raise RunUnreadable(f"{run_dir}: no outputs.jsonl")
    rows = _read_jsonl(out)
    if not rows:
        raise RunUnreadable(f"{run_dir}: outputs.jsonl is empty (the run produced nothing)")
    recs: list[ItemRecord] = []
    meta = eval_meta(run_dir)                       # written by keel-core eval: which try / repeat this run is
    for r in rows:
        if "try" not in r:
            r = {**r, "try": meta.get("try", 0)}
        try:
            if h.suite == "agent":
                recs.append(_agent(r, h.split))
            elif h.suite == "bfcl-mt":
                recs.append(_bfcl(r, h.split, fs))
            else:
                recs.append(_when2call(r, h.split))
        except (KeyError, TypeError, ValueError) as err:
            raise RunUnreadable(f"{run_dir}: item {r.get('id')!r} lacks a field ({err!r})") from err
    done = {x.item_id for x in recs}
    h.missing = [i for i in h.items if i not in done]
    return h, recs


EVAL_META = "eval.json"


def eval_meta(run_dir: str | Path) -> dict:
    """`eval.json` next to a run made by keel-core eval (arm, suite, try, repeat, eval id); {} for other runs."""
    p = Path(run_dir) / EVAL_META
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def find_runs(*bases: str | Path) -> list[Path]:
    """Every folder directly under the bases that looks like a run (has a config.json)."""
    return sorted(p for b in bases for p in Path(b).glob("*/") if (p / "config.json").exists())


def records_dicts(recs: list[ItemRecord]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in recs]
