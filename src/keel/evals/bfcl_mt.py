"""BFCL v4 multi-turn (base, miss_func, miss_param, long_context), run through our continuous scheduler and scored
with the official checker (`bfcl-eval`, UC Berkeley, Apache-2.0).

Protocol = BFCL's own FC protocol for local Qwen models (`base_handler.inference_multi_turn_FC`, `qwen_fc.py`):
- the model's chat template with the tools, no extra system prompt ("FC models use its own system prompt");
- each turn starts with the dataset's user message; the model calls tools, results come back as tool messages; a
  reply without a tool call ends the turn (also an unparseable reply, as in BFCL);
- more than MAXIMUM_STEP_LIMIT (20) steps in one turn force-quits the entry (later turns are not run; the checker then
  fails the entry because the turn count does not match, unless it was the last turn: same as BFCL);
- miss_func: at the hold-out turn the missing function docs are added to the tools and BFCL's fixed user message
  ("I have updated some more functions ...") is sent;
- tools run in BFCL's stateful Python classes (`execute_multi_turn_func_call`).

Scoring: `_evaluate_single_multi_turn_entry` from `eval_checker/eval_runner.py` copied (that module needs tree_sitter,
which is not installed): force-terminated check, decoding (failed / empty steps skipped), then `multi_turn_checker`
(per turn: state of the classes vs ground truth, and the ground truth's results must appear in the model's results).
BFCL v4 imports `multi_turn_irrelevance_checker` but never calls it, so calling tools in a turn whose ground truth is
empty is not penalised. We report the official score as primary and `strict` (official + irrelevance check) as
secondary.

Splits: dev / val / holdout by fixed hashes of the scenario number (see split_of). The four categories reuse the same 200 scenarios (index i is the
same task in each), so hashing the full id would put the same scenario in both halves.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC, MAXIMUM_STEP_LIMIT  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker, multi_turn_irrelevance_checker  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.utils import populate_test_cases_with_predefined_functions  # noqa: E402

from ..agent.verify import BUDGET_NOTE, CHECK, CHECK_RECENT, claims_change
from ..core.arg_source import ArgSourceCheck
from ..core.events import Event, Sink, null_sink
from ..core.layers import Pipeline, ResultContext, build_pipeline
from ..nudges import LOOP_NOTE, LoopBreaker, TodoList  # noqa: F401
from . import bfcl  # noqa: F401  (registers the model_config stand-in before bfcl_eval imports)
from .bfcl import DATA, parse_calls, tools_for

CATEGORIES = ["base", "miss_func", "miss_param", "long_context"]


def _read(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def scenario(entry_id: str) -> int:
    return int(entry_id.rsplit("_", 1)[1])


def half(entry_id: str) -> str:
    return "dev" if int(hashlib.sha1(f"mt{scenario(entry_id)}".encode()).hexdigest(), 16) % 2 == 0 else "eval"


def split_of(entry_id: str) -> str:
    """dev (design, read freely) | val (validation: one comparison per module, totals only) | holdout (final, once).
    The eval half is cut in two by a second fixed hash of the scenario number; dev is unchanged."""
    if half(entry_id) == "dev":
        return "dev"
    return "val" if scenario(entry_id) in _val_scenarios() else "holdout"


@lru_cache(maxsize=None)
def _val_scenarios() -> frozenset:
    """Balanced cut: eval-half scenarios sorted by a fixed hash; the first half (rounded up) is validation."""
    ev = sorted((n for n in range(200) if half(f"x_{n}") == "eval"),
                key=lambda n: hashlib.sha1(f"mtv{n}".encode()).hexdigest())
    return frozenset(ev[:(len(ev) + 1) // 2])


@lru_cache(maxsize=None)
def _load(category: str) -> tuple:
    rows = _read(DATA / f"BFCL_v4_multi_turn_{category}.json")
    rows = populate_test_cases_with_predefined_functions(rows)     # adds "function"; moves held-out docs
    gt = {r["id"]: r["ground_truth"] for r in _read(DATA / "possible_answer" / f"BFCL_v4_multi_turn_{category}.json")}
    return tuple({**r, "category": category, "ground_truth": gt[r["id"]]} for r in rows)


def load(category: str, split: str | None = None) -> list[dict]:
    """Fresh deep copies (episodes mutate their entry's function list). split: dev | val | holdout | None (all)."""
    assert split in (None, "dev", "val", "holdout"), split
    return [copy.deepcopy(r) for r in _load(category) if split is None or split_of(r["id"]) == split]


def to_call_strings(calls: list[dict]) -> list[str]:
    """[{name: {arg: value}}] -> ["name(arg=value,...)"], as BFCL's convert_to_function_call / decode_execute."""
    return [f"{name}({','.join(f'{k}={v!r}' for k, v in (args if isinstance(args, dict) else {}).items())})"
            for c in calls for name, args in c.items()]


# --- state report for BFCL -----------------------------------------------------------------------------------
# What the official checker compares: each class instance's public attributes (vars() without "_" names). We show the
# model how those changed during the current turn, flattened to readable paths. Generic: no class is special-cased,
# except that object back-links named "parent" are skipped (the file system tree would loop).

def _plain(o, depth: int = 0):
    if depth > 30:
        return "..."
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _plain(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v, depth + 1) for v in o]
    if isinstance(o, set):
        return sorted((_plain(v, depth + 1) for v in o), key=repr)
    if hasattr(o, "__dict__"):
        return {k: _plain(v, depth + 1) for k, v in vars(o).items() if not k.startswith("_") and k != "parent"}
    return repr(o)


def _flatten(o, prefix: str, out: dict) -> None:
    if isinstance(o, dict) and o:
        for k, v in o.items():
            _flatten(v, prefix if k == "contents" else f"{prefix}/{k}", out)   # "contents" adds nothing to a path
    elif isinstance(o, list) and o:
        for i, v in enumerate(o):
            _flatten(v, f"{prefix}/{i}", out)
    else:
        parent, _, leaf = prefix.rpartition("/")
        if leaf == "name" and o == parent.rpartition("/")[2]:
            return                                       # a "name" that only repeats its own path segment
        out[prefix] = o


def state_snapshot(instances: dict) -> dict[str, object]:
    out: dict = {}
    for cls, inst in instances.items():
        _flatten({k: _plain(v) for k, v in vars(inst).items() if not k.startswith("_")}, cls, out)
    return out


def _short(v, n: int = 120) -> str:
    t = repr(v)
    return t if len(t) <= n else t[:n] + "..."


def state_report(before: dict, after: dict, errors: list[str] | None = None, answer: str | None = None,
                 max_lines: int = 40) -> str:
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    changed = [k for k in after if k in before and after[k] != before[k]]
    lines = ["STATE REPORT (computed from the apps' real state this turn, not from your messages)"]
    if not (added or removed or changed):
        lines.append("Nothing in the apps' state changed during this turn.")
    budget = max_lines
    for title, keys, fmt in (("Added", added, lambda k: f"  + {k} = {_short(after[k])}"),
                             ("Removed", removed, lambda k: f"  - {k} (was {_short(before[k])})"),
                             ("Changed", changed, lambda k: f"  ~ {k}: {_short(before[k], 60)} -> {_short(after[k])}")):
        if not keys or budget <= 0:
            continue
        lines.append(f"{title} ({len(keys)}):")
        lines.extend(fmt(k) for k in keys[:budget])
        if len(keys) > budget:
            lines.append(f"  ... and {len(keys) - budget} more")
        budget -= min(len(keys), budget)
    if errors:
        lines.append(f"Tool calls this turn that returned errors ({len(errors)}):")
        lines.extend(f"  {e[:200]}" for e in errors[-5:])
    if answer is not None and not (added or removed or changed) and claims_change(answer):
        lines.append("NOTE: your answer describes changes, but nothing in the apps' state changed this turn.")
    return "\n".join(lines)


def _is_error(result: str) -> bool:
    head = result[:40].lower()
    return result.startswith("Error during execution") or '"error"' in head or "'error'" in head


# --- episode -------------------------------------------------------------------------------------------------------

@dataclass
class MTEpisode:
    entry: dict
    exec_name: str                     # executor instance namespace; must be unique per run and try
    messages: list[dict] = field(default_factory=list)
    turn: int = 0
    step: int = 0                      # steps in the current turn
    current: list = field(default_factory=list)          # this turn's step responses
    result: list = field(default_factory=list)           # per turn: list of step responses (call strings | text)
    tool_results: list = field(default_factory=list)     # per turn: list of steps' execution results
    exec_errors: int = 0
    # (one check per turn; recovery per turn, BFCL's 20-step limit counts as usual)
    verify: str = "none"               # "none" | "self" (self-check) | "state" (state report)
    recovery: bool = False
    check_requests: str = "current"    # "current": quote this turn's user message; "recent": previous + latest turn
    user_turns: list = field(default_factory=list)
    instances: dict = field(default_factory=dict)
    turn_start_state: dict = field(default_factory=dict)
    turn_request: str = ""
    turn_errors: list = field(default_factory=list)
    turn_checked: bool = False
    notes: list = field(default_factory=list)
    #; off = unchanged behaviour
    reminder: bool = False             # to-do reminder: keel-kept list of user requests
    loop_break: bool = False           # loop breaker: 3rd identical (call, result) in a turn -> note + text-only reply
    arg_source: bool = False           # hold calls whose argument values have no source (core.arg_source)
    argcheck: ArgSourceCheck | None = None
    todo: TodoList = field(default_factory=TodoList)
    todo_log: list = field(default_factory=list)
    loops: LoopBreaker = field(default_factory=LoopBreaker)
    ban_tool_call: bool = False        # read by the scheduler for the next reply
    # trims what the model sees of long tool results; None = off
    trimmer: object = None
    thinking: bool = False             # the prompt opens a think block; replies start inside it
    done: bool = False
    stop: str = ""                     # "done" | "force_quit" | "context_limit"
    steps: int = 0
    # scheduler fields
    prompt_tokens: int = 0
    processed_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    t_start: float = 0.0
    cache: list | None = None
    seq: list[int] | None = None
    events: Sink = field(default=null_sink, repr=False)  # event stream (core.events); off = discarded
    pipeline: Pipeline = field(init=False, repr=False)   # trim -> repeat -> loop_break (core.layers)

    def __post_init__(self):
        self.argcheck = ArgSourceCheck() if self.arg_source else None
        self.pipeline = build_pipeline(trimmer=self.trimmer, repeat=self.recovery, loop_break=self.loop_break,
                                       loops=self.loops)

    @property
    def long_context(self) -> bool:
        return "long_context" in self.entry["id"]

    def tools(self) -> list[dict]:
        return tools_for(self.entry)

    def start(self) -> "MTEpisode":
        # instantiate the classes once (as BFCL does before turn 0), then open turn 0
        _, self.instances = execute_multi_turn_func_call([], self.entry["initial_config"],
                                                         self.entry["involved_classes"], self.exec_name,
                                                         self.entry["id"], long_context=self.long_context)
        self._open_turn()
        return self

    def _open_turn(self) -> None:
        msgs = self.entry["question"][self.turn]
        holdout = self.entry.get("missed_function", {})
        if str(self.turn) in holdout:
            self.entry["function"].extend(holdout[str(self.turn)])
            assert len(msgs) == 0, "holdout turn should not have a user message"
            msgs = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
        new = [{"role": m["role"], "content": m["content"]} for m in msgs]
        if self.reminder:                                # to-do reminder: open items from earlier turns, then add this one
            note = self.todo.note_for_turn()
            users = [m for m in new if m["role"] == "user"]
            if note and users:
                users[-1]["content"] += note
                self.notes.append(f"reminder@t{self.turn}:{len(self.todo.shown)}")
            if str(self.turn) not in holdout:
                self.todo.add(self.turn, " ".join(m["content"] for m in msgs if m["role"] == "user"))
        self.pipeline.new_turn()                         # repeat + loop breaker count per turn
        self.messages.extend(new)
        self.current, self.step = [], 0
        self.tool_results.append([])
        self.turn_request = " ".join(m["content"] for m in msgs)
        self.user_turns.append(self.turn_request)
        self.turn_errors, self.turn_checked = [], False
        if self.verify == "state":
            self.turn_start_state = state_snapshot(self.instances)

    def _close_turn(self) -> None:
        if self.reminder:
            final = next((x for x in reversed(self.current) if isinstance(x, str)), "")
            made = any(isinstance(x, list) for x in self.current)
            self.todo_log.append(self.todo.close_turn(self.turn, final, made))
        self.result.append(self.current)
        self.turn += 1
        if self.turn >= len(self.entry["question"]):
            self.done, self.stop = True, "done"
        else:
            self._open_turn()

    def apply(self, raw: str, finish_reason: str) -> None:
        """One model reply (runs in a worker thread)."""
        self.steps += 1
        self.ban_tool_call = False                       # the loop breaker's ban covered only the reply just received
        unfinished = self.thinking and "</think>" not in raw     # still thinking when the reply ended
        text = "" if unfinished else raw.split("</think>")[-1].strip()
        msg = {"role": "assistant", "content": text}
        if self.thinking:                                        # kept for the template (see agent.loop)
            msg["reasoning_content"] = raw.split("</think>")[0].split("<think>")[-1].strip()
            if unfinished:
                self.notes.append(f"thinking_unfinished@{self.turn}.{self.step}")
        self.messages.append(msg)
        sid, where = self.entry["id"], {"turn": self.turn, "step": self.step}
        if text:
            self.events(Event("text", sid, {**where, "text": text}))
        calls = [] if unfinished else parse_calls(raw, self.entry["function"])
        if not calls:
            self.current.append(text)                    # text step: skipped by the checker's decoding
            if self.verify != "none" and not self.turn_checked and self.step < MAXIMUM_STEP_LIMIT:
                self.turn_checked = True                 # one check per turn, before the turn closes
                if self.check_requests == "recent" and len(self.user_turns) >= 2:
                    msg = CHECK_RECENT.format(earlier=self.user_turns[-2], latest=self.user_turns[-1])
                else:
                    msg = CHECK.format(request=self.turn_request)
                if self.verify == "state":
                    msg += "\n\n" + state_report(self.turn_start_state, state_snapshot(self.instances),
                                                  self.turn_errors, text)
                self.messages.append({"role": "user", "content": msg})
                self.notes.append(f"check@t{self.turn}s{self.step}")
                return
            self._close_turn()
            return
        if self.argcheck is not None:                    # no call in this reply runs if a value has no source
            hold = self.argcheck.check(calls, self.entry["function"],
                                       [m["content"] for m in self.messages[:-1]
                                        if m["role"] in ("system", "user", "tool") and isinstance(m["content"], str)])
            if hold is not None:
                self.notes.append(f"argsrc@t{self.turn}s{self.step}")
                self.events(Event("layer", sid, {"note": f"argsrc@t{self.turn}s{self.step}"}))
                for _ in calls:
                    self.messages.append({"role": "tool", "content": hold})
                self.step += 1
                if self.step > MAXIMUM_STEP_LIMIT:
                    self.result.append(self.current)
                    self.done, self.stop = True, "force_quit"
                return
        strs = to_call_strings(calls)
        self.current.append(strs)
        for c in strs:                                   # BFCL tools are simulated: no file kind, no delete guard
            self.events(Event("tool_started", sid, {**where, "name": c.split("(")[0], "call": c, "kind": None}))
        results, _ = execute_multi_turn_func_call(strs, self.entry["initial_config"], self.entry["involved_classes"],
                                                  self.exec_name, self.entry["id"], long_context=self.long_context)
        self.exec_errors += sum(r.startswith("Error during execution") for r in results)
        self.turn_errors += [f"{c} -> {r}" for c, r in zip(strs, results, strict=False) if _is_error(r)]
        self.tool_results[-1].append(results)
        for c, r in zip(strs, results, strict=False):
            res = self.pipeline.tool_result(ResultContext(         # what the model sees; layers read the raw r
                label=f"@t{self.turn}s{self.step}", trim_name=c.split("(")[0], repeat_sig=(c, r[:2000]),
                loop_key=c, loop_label=c[:60]), r)
            view = res.view
            self.notes.extend(res.notes)
            if res.ban_tool_call:                        # <tool_call> banned for the next reply
                self.ban_tool_call = True
            self.events(Event("tool_finished", sid, {**where, "name": c.split("(")[0], "output": r,
                                                     "shown": None if view == r else view, "error": _is_error(r)}))
            for n in res.notes:
                self.events(Event("layer", sid, {"note": n}))
            self.messages.append({"role": "tool", "content": view})
        self.step += 1
        if self.step > MAXIMUM_STEP_LIMIT:
            self.result.append(self.current)
            self.done, self.stop = True, "force_quit"
        elif self.recovery and MAXIMUM_STEP_LIMIT - self.step == 3:   # budget warning (17 of 20)
            self.messages[-1]["content"] += BUDGET_NOTE.format(left=3)
            self.notes.append(f"budget@t{self.turn}s{self.step}")


# --- scoring (official) ----------------------------------------------------------------------------------------------

def _decode_execute(step) -> list[str]:
    """Our stored step: a list of call strings, or the reply text when it had no parseable call (BFCL's FC decoder
    fails on text, and the checker skips failed decodes)."""
    if isinstance(step, list) and all(isinstance(s, str) for s in step):
        return step
    raise ValueError("no function call")


def score(entry: dict, result: list, model_name: str) -> dict:
    """Copy of eval_runner._evaluate_single_multi_turn_entry (bfcl-eval 2026.3.23), plus the strict variant.
    `model_name` namespaces the checker's executor instances (must be unique per run and try)."""
    gt = entry["ground_truth"]
    out = {"id": entry["id"], "category": entry["category"]}
    if not isinstance(result, list):
        return {**out, "valid": False, "strict": False, "error_type": "multi_turn:inference_error"}
    if len(result) != len(gt):
        return {**out, "valid": False, "strict": False, "error_type": "multi_turn:force_terminated"}
    decoded: list[list[list[str]]] = []
    for turn in result:
        d = []
        for step in turn:
            try:
                calls = _decode_execute(step)
                if is_empty_execute_response(calls):
                    continue
                d.append(calls)
            except Exception:
                continue
        decoded.append(d)
    prompt_entry = {k: v for k, v in entry.items() if k not in ("function", "ground_truth", "category")}
    r = multi_turn_checker(decoded, gt, prompt_entry, entry["id"].rsplit("_", 1)[0], model_name)
    valid = bool(r["valid"])
    irr = multi_turn_irrelevance_checker(decoded, gt)
    return {**out, "valid": valid, "strict": valid and bool(irr["valid"]),
            "error_type": None if valid else r.get("error_type"),
            "error": None if valid else r.get("error_message"),
            "irrelevance_error": None if irr["valid"] else irr.get("error_message")}


def safe_name(s: str) -> str:
    """A valid Python identifier prefix: BFCL names its tool objects f"{name}_{id}_{class}_instance" and runs the calls
    with eval(), so a name starting with a digit (like a run folder "20261108-...") makes every call a syntax error."""
    return "r_" + re.sub(r"[^A-Za-z0-9_]", "_", s)
