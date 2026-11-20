"""Bare agent loop (the checkup baseline; harness layers get added on top in later modules).

One tool, `run_shell`, executed in the OS sandbox through the delete guard (always on: no file is deleted without
the user's confirmation; in checkups a simulated user answers, see guard.approve_all). The model either calls the tool (we run the command and return
its output) or replies in plain text, which ends the episode and counts as the final answer to the user.
Two schedulers, same episodes and scoring:
- `run_episodes` (default): episodes run independently in one continuous batch. When an episode's
  reply finishes, its tool calls run and it goes straight back into the batch with its own KV cache, so only the new
  tokens (tool output + next turn header) are processed; the conversation is never re-read. Checked: the re-rendered
  chat template extends the cached tokens exactly, and reused-cache logits differ from a fresh prefill only by the
  same rounding as two fresh prefills with different chunk sizes.
- `run_episodes_stepwise`: all episodes advance one step at a time and every step re-reads the
  whole conversation. Kept for comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import native
from ..core.backend import RunHooks, RunLimits
from ..core.events import Event, Sink, null_sink
from ..core.layers import Pipeline, ResultContext, build_pipeline
from ..core.parse import parse_calls
from ..core.stream import VisibleText
from ..core.tools import REGISTRIES, reporting
from ..nudges import LOOP_NOTE, LoopBreaker  # noqa: F401
from ..specs import ModelSpec, SamplingSpec
from . import done_gate, gnu_shims
from .examples import EXAMPLES
from .filetools import FileTools, ToolError
from .guard import GuardedShell, _files, approve_all, careful_user
from .sandbox import Sandbox
from .verify import BUDGET_NOTE, CHECK, REPEAT_NOTE, change_report  # noqa: F401

if TYPE_CHECKING:                       # eval tasks are only a type here; the loop itself imports no eval code
    from ..evals.tasks import Item

SYSTEM = ("You are a helpful assistant working on the user's computer (macOS). You can run shell commands with the "
          "run_shell tool. You can only see and change files in the current folder, and there is no internet. "
          "Work step by step and look at the files before changing them. When you are finished, reply to the user "
          "in plain text without calling a tool: answer their question or say briefly what you did.")


# File tools + shell. The system prompt differs only in naming the tools' purpose.
SYSTEM_FILES = ("You are a helpful assistant working on the user's computer (macOS). You can only see and change files "
                "in the current folder, and there is no internet. Use the file tools to look at, edit, move and delete "
                "files; use run_shell only when no file tool fits (for example calculations or scripts). Work step by "
                "step and look at the files before changing them. When you are finished, reply to the user in plain "
                "text without calling a tool: answer their question or say briefly what you did.")
# run_python is for regular data only (narrowed after a small run, 2026-11-17). The first line ("counting, sums,
# dates, parsing files and changing many files at once") sent messy text to code too: C1 4 -> 3, E3 2 -> 0.
SEND_NOTE = (" You cannot send email or messages from here: if the user asks you to, say so at once without running "
             "any command, and tell them how to do it themselves.")
SYSTEM_FILES_PY = SYSTEM_FILES + (" Use run_python for counting, adding up numbers, comparing file contents (hashes) and "
                                  "moving or renaming many files at once. Read messy documents yourself, and do not "
                                  "write a parser for them." + SEND_NOTE)
# ^ last sentence added 2026-11-17 after F1 (0/4: all four ran mail/sendmail first; thinking 3/4 by checking
# its abilities before acting). The send guard (guard.looks_like_send) refuses a send in every arm either way.
# `--no-send-note` removes SEND_NOTE to measure the rest of the stack without it.
# toolset -> (system prompt, tool schemas, parser funcs); the tools themselves are declared once, in core.tools
_REG = {name: make() for name, make in REGISTRIES.items()}
TOOLSETS = {name: (prompt, _REG[name].schemas, _REG[name].funcs)
            for name, prompt in (("shell", SYSTEM), ("files", SYSTEM_FILES), ("files_py", SYSTEM_FILES_PY))}
TOOLS, FUNCS = TOOLSETS["shell"][1], TOOLSETS["shell"][2]
_VIEWS: dict[str, tuple] = {}


def toolset_view(name: str) -> tuple:
    """(system prompt, schemas, parser funcs) the model sees. "files@qwen" = the files toolset with Qwen's native tool
    names (`--native-tools`); a plain name is exactly TOOLSETS[name]."""
    if name in TOOLSETS:
        return TOOLSETS[name]
    if name not in _VIEWS:
        base, _, fam = name.partition("@")
        prompt, schemas, funcs = TOOLSETS[base]
        _VIEWS[name] = (native.text(prompt, fam), [native.schema(x, fam) for x in schemas],
                        [native.func(x, fam) for x in funcs])
    return _VIEWS[name]


def base_toolset(name: str) -> str:
    return name.partition("@")[0]


def tool_family(name: str) -> str | None:
    return name.partition("@")[2] or None

@dataclass
class SessionItem:
    """What an interactive session's episode carries instead of an eval task (no files to build, nothing to score)."""
    task: str
    request: str = ""
    seed: int = 0
    kind: str = "session"
    files: dict = field(default_factory=dict)
    mtimes: dict = field(default_factory=dict)
    gold: dict = field(default_factory=dict)
    steps: str = "M"

    @property
    def id(self) -> str:
        return self.task


@dataclass
class Episode:
    item: Item | SessionItem
    sandbox: Sandbox
    shell: GuardedShell
    messages: list[dict]
    toolset: str = "shell"
    ftools: FileTools | None = None
    steps: int = 0
    done: bool = False
    stop: str = ""                 # "answer" | "step_limit" | "context_limit" | "length" | "backend_error"
    backend_error: str = ""        # the server's message when stop == "backend_error"
    answer: str = ""
    commands: list[str] = field(default_factory=list)
    prompt_tokens: int = 0         # tokens in all prompts (what a no-reuse loop would process)
    processed_tokens: int = 0      # prompt tokens actually processed (after cache reuse)
    output_tokens: int = 0
    seconds: float = 0.0
    t_start: float = 0.0
    cache: list | None = None      # KV cache after the last reply (owned while the episode runs its tools)
    seq: list[int] | None = None   # tokens covered by `cache`
    # verify / recovery (experimental)
    verify: str = "none"           # "none" | "self" (self-check) | "state" (state report)
    recovery: bool = False         # repeat detector + step-budget warning
    start_dir: Path | None = None  # the task's start state (template folder), for the change report
    checked: bool = False
    answer_before_check: str | None = None
    correct_before_check: bool | None = None   # measurement only (scored at check time, never shown to the model)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)       # which layer fired when
    # layers
    loop_break: bool = False       # 3rd identical (call, result) -> note + <tool_call> banned for one reply
    loops: LoopBreaker = field(default_factory=LoopBreaker)
    ban_tool_call: bool = False    # read by the scheduler for the next reply
    trimmer: object = None         # trims what the model sees of long tool results; None = off
    events: Sink = field(default=null_sink, repr=False)  # event stream (core.events): TUI / JSONL; off = discarded
    thinking: bool = False         # the prompt opens a think block; replies start inside it
    # concrete-signal checks hold back "done" (agent.done_gate); off = unchanged
    done_gate: bool = False
    scorer: object = None          # eval runs only: the task's scorer, for measurement (never shown to the model)
    gate_rounds: int = 0
    gate_before: dict | None = None            # the folder before the first tool call
    gate_om_seen: int = 0                      # FileTools.overmatches already reported
    gate_log: list = field(default_factory=list)
    answer_before_gate: str | None = None
    correct_before_gate: bool | None = None    # measurement only (blocked-correct), never shown to the model
    last_tool_error: str | None = None         # the last tool call failed and nothing succeeded after it
    pipeline: Pipeline = field(init=False, repr=False)   # trim -> repeat -> loop_break (core.layers)

    def __post_init__(self):
        if self.verify == "state" and self.start_dir is None:   # the change report needs a start state
            raise ValueError("verify='state' needs start_dir (the task's start state); use 'self' or 'none' instead")
        self.pipeline = build_pipeline(trimmer=self.trimmer, repeat=self.recovery, loop_break=self.loop_break,
                                       loops=self.loops)


def prompt_text(tok, spec: ModelSpec, messages: list[dict], toolset: str = "shell") -> str:
    """The rendered prompt: what the model reads. Raw-mode server backends send this text."""
    return tok.apply_chat_template(messages, tools=toolset_view(toolset)[1], add_generation_prompt=True, tokenize=False,
                                   **spec.chat_template_kwargs)


def chat_prompt(e: "Episode") -> dict:
    """The model input for chat-mode server backends: the transcript, the tool schemas, and the parser's view of the
    tools (to turn the transcript's call text into structured calls). See backends.chat."""
    _, schemas, funcs = toolset_view(e.toolset)
    return {"messages": [dict(m) for m in e.messages], "tools": schemas, "funcs": funcs}


def prompt_ids(tok, spec: ModelSpec, messages: list[dict], toolset: str = "shell") -> list[int]:
    return tok.encode(prompt_text(tok, spec, messages, toolset), add_special_tokens=False)


def _new_episodes(items, out_dir, confirmer, toolset: str = "shell", verify: str = "none", recovery: bool = False,
                  loop_break: bool = False, trimmer=None, events: Sink = null_sink):
    """`confirmer`: a function, or "approve_all" / "careful" (simulated users for checkups)."""
    from ..evals import tasks as T  # eval runs: build each task's folder, score it at the end
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = []
    for it in items:
        tmpl = T.materialize(it, out_dir / "templates" / it.id)
        sb = Sandbox(out_dir / "sandboxes" / it.id).create(tmpl)
        if confirmer == "careful":
            conf = careful_user(sb.work, _files(sb.work))
        elif confirmer in ("approve_all", None):
            conf = approve_all
        else:
            conf = confirmer
        sh = GuardedShell(sb, reporting(conf, events, it.id),   # every delete question + answer is an event
                          backup_dir=out_dir / "backups" / it.id)   # silent backups, outside the sandbox
        eps.append(Episode(it, sb, sh, [{"role": "system", "content": toolset_view(toolset)[0]},
                                         {"role": "user", "content": it.request}],
                           toolset=toolset, ftools=FileTools(sh) if base_toolset(toolset) in ("files", "files_py") else None,
                           verify=verify, recovery=recovery, start_dir=tmpl, loop_break=loop_break,
                           trimmer=trimmer, events=events, scorer=T.score))
    return eps


def _forms(p: Path) -> set[str]:
    """Every spelling of a folder a tool may print: as given, resolved, and without macOS's /private prefix."""
    out = {str(p), str(Path(p).resolve())}
    return out | {x[len("/private"):] for x in out if x.startswith("/private/")}


def _relative(text: str, work: Path, root: Path | None = None) -> str:
    """Rewrite absolute sandbox paths into equivalent paths relative to the task folder (the agent's working folder):
    the task folder -> `.`, the sandbox root (holds .tmp, where mktemp writes) -> `..`. An absolute path leaks the
    user's folder layout to the model and makes runs depend on where keel is installed (found by the A7 gate,
    2026-11-13); the rewritten paths still work in the next command, because they are relative to the same folder."""
    if root is not None and Path(work).resolve().parent != Path(root).resolve():
        root = None              # in place: the state folder is not the task folder's parent, so `..` would be wrong
    pairs = [(f, "work") for f in _forms(work)] + ([(f, "root") for f in _forms(root)] if root is not None else [])
    for f, kind in sorted(pairs, key=lambda x: len(x[0]), reverse=True):
        if kind == "work":
            text = text.replace(f + "/", "").replace(f, ".")
        else:
            text = text.replace(f + "/", "../").replace(f, "..")
    return text


def _is_error(out: str) -> bool:
    """Text rule, only for keel's own messages (guard notes). Tool results use what happened: exit code / tool error
    (`head errors.log` used to count as a failed call)."""
    return out.startswith("Error") or "[exit code " in out


def _heredoc(code: str) -> str:
    """A delimiter that cannot occur as a line of `code` (audit #14: a line `KEEL_PY` ended the script early and ran
    the rest as shell). Derived from the code, so the same call gives the same command in every run."""
    tag = "KEEL_PY_" + hashlib.sha1(code.encode()).hexdigest()[:12]
    while tag in code.splitlines():
        tag += "_"
    return tag


# A reply cut at the token limit inside a <tool_call> used to run the half-written call
# (the parser keeps the arguments that were closed), so the model was told e.g. "missing argument 'content'" and
# retried the same too-long call (29 + 47 times in the research runs, every one at max_tokens). Now the unfinished
# call is not run and the model is told why. The parser is unchanged (BFCL and When2Call scoring use it).
CUT_CALL = ("Error: your reply reached the length limit before this tool call was finished, so it was not run. "
            "Make each tool call shorter: write a long file in several smaller steps.")


def _cut_calls(text: str, funcs, finish_reason: str) -> int:
    """How many of the reply's parsed calls come from a <tool_call> block the length limit cut off (0 if none)."""
    if finish_reason != "length":
        return 0
    head, sep, tail = text.rpartition("<tool_call>")
    if not sep or "</tool_call>" in tail:
        return 0
    return len(parse_calls(text, funcs)) - len(parse_calls(head, funcs))


GATE_ROUNDS = 2                  # at most this many held-back answers per episode; then the answer stands
NAME = re.compile(r"[\w.#$@:-]{1,60}")   # done check, over-match: a single name, like 'btn-primary' or '.btn-primary'


def _gate_problems(e: Episode, answer: str) -> list[str]:
    """The done check's findings for this episode right now (generic: files changed since the start, the harness's own records)."""
    from ..false_success import admits
    work = e.sandbox.work
    problems = done_gate.file_problems(work, done_gate.changed_files(work, e.gate_before or {}))
    if e.ftools is not None:
        for f, old, names, replace_all in e.ftools.overmatches[e.gate_om_seen:]:
            # only an accidental hit counts: replace_all on one short name (no spaces). Whole-line edits that
            # spell out the longer name are a choice, not an accident (a small run: they caused both blocked-correct
            # answers and never named the real mistake).
            if replace_all and NAME.fullmatch(old):
                problems.append(f"edit_file with replace_all also changed '{old}' inside longer names in {f}: "
                                f"{', '.join(names)}. Those can be different names; if they should have stayed as "
                                "they were, change them back")
        e.gate_om_seen = len(e.ftools.overmatches)
    if e.last_tool_error and not admits(answer):     # an honest "this failed" answer is not held back for it
        problems.append(f"your last tool call failed and nothing after it succeeded: {e.last_tool_error}")
    return problems


def _apply_reply(e: Episode, raw: str, finish_reason: str, max_steps: int) -> None:
    """Record one model reply: run its tool calls, or handle a plain-text reply (final answer; with a verify round the first
    final answer is intercepted once by the check). The check gets no extra steps: if the step limit is reached
    during the check, the answer given before the check stands."""
    e.steps += 1
    fam = tool_family(e.toolset)
    e.ban_tool_call = False                              # the loop breaker's ban covered only the reply just received
    if e.done_gate and e.gate_before is None:           # the folder before any of the model's tool calls
        e.gate_before = done_gate.snapshot(e.sandbox.work)
    reasoning = None
    if e.thinking and "</think>" not in raw:             # still thinking when the reply ended (length):
        reasoning, text, calls, n_cut = raw.split("<think>")[-1].strip(), "", [], 0   # no answer, no calls
        e.notes.append(f"thinking_unfinished@{e.steps}")
    else:
        text = raw.split("</think>")[-1].strip()
        calls = parse_calls(raw, toolset_view(e.toolset)[2])
        n_cut = _cut_calls(raw.split("</think>")[-1], toolset_view(e.toolset)[2], finish_reason)
        if fam:                                          # native names back to keel's, right after parsing
            calls = native.to_internal(calls, fam)
        if e.thinking:
            reasoning = raw.split("</think>")[0].split("<think>")[-1].strip()
    first_cut = len(calls) - n_cut                       # calls from this index on were cut off: not run
    if n_cut:
        e.notes.append(f"cut_call@{e.steps}")
    msg = {"role": "assistant", "content": text}
    if reasoning is not None:                            # kept for the template: Qwen renders it back for replies
        msg["reasoning_content"] = reasoning             # after the latest user turn (so the cache prefix holds)
    e.messages.append(msg)
    if text:
        e.events(Event("text", e.item.id, {"step": e.steps, "text": text}))
    if not calls:
        if e.done_gate and finish_reason != "length" and e.gate_rounds < GATE_ROUNDS and e.steps < max_steps:
            problems = _gate_problems(e, text)
            if problems:                                 # hold back "done" once, with the exact problems
                if e.gate_rounds == 0:
                    e.answer_before_gate = text
                    try:                                 # read-only scorer: blocked-correct, measurement only
                        e.correct_before_gate = bool(e.scorer(e.item, e.sandbox.work, text, list(e.commands),
                                                              [c.__dict__ for c in e.shell.log])["correct"]) \
                            if e.scorer else None
                    except Exception:
                        e.correct_before_gate = None
                e.gate_rounds += 1
                e.gate_log.append(problems)
                e.messages.append({"role": "user", "content": native.text(done_gate.message(problems), fam)})
                e.notes.append(f"gate@{e.steps}:{len(problems)}")
                e.events(Event("layer", e.item.id, {"note": f"gate@{e.steps}", "problems": problems}))
                return
        if e.verify != "none" and not e.checked and finish_reason != "length" and e.steps < max_steps:
            e.checked, e.answer_before_check = True, text
            try:                                         # read-only scorer: what the check could still change
                e.correct_before_check = bool(e.scorer(e.item, e.sandbox.work, text, list(e.commands),
                                                       [c.__dict__ for c in e.shell.log])["correct"]) \
                    if e.scorer else None
            except Exception:
                e.correct_before_check = None
            msg = CHECK.format(request=e.item.request)
            if e.verify == "state":
                assert e.start_dir is not None                   # refused at construction (Episode.__post_init__)
                msg += "\n\n" + change_report(e.start_dir, e.sandbox.work, e.errors, text)
            e.messages.append({"role": "user", "content": msg})
            e.notes.append(f"check@{e.steps}")
            return
        e.done, e.stop, e.answer = True, ("length" if finish_reason == "length" else "answer"), text
        return
    for i, c in enumerate(calls):
        name, args = next(iter(c.items()))
        cmd = args.get("command") if isinstance(args, dict) else None
        e.events(Event("tool_started", e.item.id, {"step": e.steps, "name": name, "args": args,
                                                   "kind": _REG[base_toolset(e.toolset)].kind(name)}))
        if name == "run_python" and isinstance(args, dict) and isinstance(args.get("code"), str):
            # a script runs through the same guarded shell as run_shell (sandbox + delete guard)
            tag = _heredoc(args["code"])
            name, cmd = "run_shell", f"{gnu_shims.interpreter()} - <<'{tag}'\n{args['code']}\n{tag}"
        failed = True                                    # set from what happened, never from the output's text
        if i >= first_cut:
            out = CUT_CALL
        elif e.ftools is not None and name != "run_shell":
            e.commands.append(f"{name} {json.dumps(args, ensure_ascii=False)[:300]}")
            try:
                out = e.ftools.call(name, args if isinstance(args, dict) else {})
                failed = False
            except ToolError as err:
                out = f"Error: {native.text(str(err), fam)}"
            except Exception as err:                     # a tool bug must not end the run
                out = f"Error: {type(err).__name__}: {_relative(str(err), e.sandbox.work, e.sandbox.root)}"
        elif name != "run_shell" or not isinstance(cmd, str) or not cmd.strip():
            out = native.text("Error: unknown tool or missing arguments." if e.ftools is not None else
                              "Error: the only tool is run_shell with a 'command' string.", fam)
        else:
            e.commands.append(cmd)
            r, note = e.shell.run(cmd)
            out = note if r is None else ((r.output.strip() or "(no output)") +
                                          ("" if r.exit_code == 0 else f"\n[exit code {r.exit_code}]") + note)
            failed = _is_error(note) if r is None else r.exit_code != 0     # not run: keel's own note decides
            out = _relative(out, e.sandbox.work, e.sandbox.root)   # e.g. `pwd`, `find $PWD`, mktemp paths
        e.sandbox.settle_clock(e.steps)                  # logical clock: no wall-clock times reach the model
        if failed:
            e.errors.append(out)
        e.last_tool_error = out.strip().splitlines()[0][:200] if failed and out.strip() else None
        res = e.pipeline.tool_result(ResultContext(               # what the model sees; layers read the raw result
            label=f"@{e.steps}", trim_name=name,
            repeat_sig=json.dumps([name, args, out[:2000]], sort_keys=True, default=str),
            loop_key=json.dumps([name, args], sort_keys=True, default=str), loop_label=name), out)
        view = res.view
        e.notes.extend(res.notes)
        if res.ban_tool_call:                            # <tool_call> banned for the next reply
            e.ban_tool_call = True
        e.events(Event("tool_finished", e.item.id, {"step": e.steps, "name": name, "output": out,
                                                    "shown": None if view == out else view, "error": failed}))
        for n in res.notes:
            e.events(Event("layer", e.item.id, {"note": n}))
        e.messages.append({"role": "tool", "content": view, "_step": e.steps,
                           "_call": f"{name}({json.dumps(args, ensure_ascii=False, default=str)[:80]})"})
    if e.steps >= max_steps:
        e.done, e.stop = True, "step_limit"
        if e.answer_before_check is not None:            # limit hit during the check: the earlier answer stands
            e.stop, e.answer = "answer", e.answer_before_check
            e.notes.append("check_cut_by_step_limit")
        return
    if e.recovery and max_steps - e.steps == 3:          # step-budget warning on the last tool message
        e.messages[-1]["content"] += BUDGET_NOTE.format(left=3)
        e.notes.append(f"budget@{e.steps}")


def _record(e: Episode, out_dir: Path, results: list, on_episode) -> None:
    e.events(Event("stop", e.item.id, {"reason": e.stop, "steps": e.steps}))
    confs = [c.__dict__ for c in e.shell.log]
    assert e.scorer is not None, "_record is for eval episodes (they carry their task's scorer)"
    s = e.scorer(e.item, e.sandbox.work, e.answer, e.commands, confs)
    rec = {"id": e.item.id, "task": e.item.task, "kind": e.item.kind, "seed": e.item.seed, "request": e.item.request,
           "toolset": e.toolset,
           "correct": bool(s["correct"]), "score": s["score"], "labels": s.get("labels", []), "stop": e.stop,
           "steps": e.steps, "answer": e.answer, "commands": e.commands, "prompt_tokens": e.prompt_tokens,
           "processed_tokens": e.processed_tokens, "output_tokens": e.output_tokens, "seconds": round(e.seconds, 1),
           "confirmations": confs, "detail": {k: v for k, v in s.items() if k not in ("correct", "score", "labels")},
           "verify": e.verify, "recovery": e.recovery, "checked": e.checked,
           "answer_before_check": e.answer_before_check, "correct_before_check": e.correct_before_check,
           "notes": e.notes, "n_errors": len(e.errors), "loop_break": e.loop_break, "trim": e.trimmer is not None,
           "edit": dict(e.ftools.stats) if e.ftools is not None else None}
    if e.done_gate:                                      # done-check records (absent with the flag off: outputs unchanged)
        rec.update(done_gate=True, gate_rounds=e.gate_rounds, gate_problems=e.gate_log,
                   answer_before_gate=e.answer_before_gate, correct_before_gate=e.correct_before_gate)
    results.append(rec)
    (out_dir / "transcripts").mkdir(exist_ok=True)
    (out_dir / "transcripts" / f"{e.item.id}.json").write_text(json.dumps({**rec, "messages": e.messages}, indent=1))
    e.cache = e.seq = None
    if on_episode:
        on_episode(rec)


def episode_hooks(prompt_fn, max_steps: int, out_dir: Path, results: list, on_episode=None, *, seeds: bool = True,
                  stream: bool = False, finish=None) -> RunHooks:
    """The agent side of a run, for any backend: record replies (and run their tools), stream what a person sees,
    record finished episodes. One set of functions for MLX and every other backend. `finish(e)` replaces the task
    scorer (interactive sessions save a transcript instead). An episode with `cancelled` set stops before the
    reply's tools run (the user pressed cancel while the model was writing)."""
    views: dict[int, VisibleText] = {}                       # stream=True: what a person sees of the current reply

    def on_token(e, text):
        vis = views.setdefault(id(e), VisibleText()).feed(text)
        if vis:
            e.events(Event("text_delta", e.item.id, {"step": e.steps + 1, "text": vis}))

    def apply(e, raw, fr):
        v = views.pop(id(e), None)                           # the reply is complete: flush its held-back tail
        if v is not None and (tail := v.flush()):
            e.events(Event("text_delta", e.item.id, {"step": e.steps + 1, "text": tail}))
        if getattr(e, "cancelled", False):
            e.done, e.stop = True, "cancelled"
            return
        _apply_reply(e, raw, fr, max_steps)

    return RunHooks(prompt_fn=prompt_fn, apply_fn=apply,
                    finish_fn=finish or (lambda e: _record(e, out_dir, results, on_episode)),
                    seed_fn=(lambda e: e.item.id) if seeds else None, token_fn=on_token if stream else None)


def _queue(items, out_dir, confirmer, toolset, verify, recovery, loop_break, trimmer, events) -> list[Episode]:
    eps = _new_episodes(items, out_dir, confirmer, toolset, verify, recovery, loop_break, trimmer, events)
    order = {"L": 0, "M": 1, "S": 2}
    return sorted(eps, key=lambda e: order.get(e.item.steps, 1))      # stable: long first


def configure(e: "Episode", *, thinking: bool = False, done_gate: bool = False, search_hint: bool = False,
              gnu_tools: bool = False, send_note: bool = True, examples: bool = False) -> None:
    """Apply a stack's per-episode settings (the one place they are set: eval runners and TUI sessions).
    thinking: the model may think before answering; done_gate: check the changed files before "done";
    search_hint: say so when a literal search looks like a regular expression; gnu_tools: GNU forms of stat / uniq /
    date on macOS; send_note: keep the "you can't send email" sentence; examples: worked examples after the prompt."""
    e.thinking = thinking
    e.done_gate = done_gate
    if search_hint and e.ftools is not None:
        e.ftools.search_hint = True
    if gnu_tools:
        gnu_shims.install(e.sandbox.bin)
    if not send_note:
        e.messages[0]["content"] = e.messages[0]["content"].replace(SEND_NOTE, "")
    if examples:
        e.messages[0]["content"] += "\n\n" + EXAMPLES


def run_episodes(model, tok, spec: ModelSpec, items: list[Item], out_dir: Path, sampling: SamplingSpec, *,
                 max_steps: int = 20, max_context: int = 16000, batch_size: int = 6, prefill_step: int = 1024,
                 on_episode=None, confirmer="approve_all", reuse_cache: bool = True,
                 progress_every: float = 60.0, context_budget: int = 48000, prefill_batch: int = 4,
                 toolset: str = "shell", verify: str = "none", recovery: bool = False,
                 episode_seeds: bool = True, loop_break: bool = False, trim_outputs: bool = False,
                 events: Sink = null_sink, state_saves: bool = False, shared_prefix: bool = False,
                 stream: bool = False, gate=None, done_gate: bool = False, gnu_tools: bool = False,
                 examples: bool = False, send_note: bool = True, search_hint: bool = False) -> list[dict]:
    """Agent tasks through the continuous scheduler (`run_scheduler`). Long tasks start first, so they do not run
    alone at the end with the batch mostly empty."""
    from ..trim import make_trimmer
    view = make_trimmer(tok) if trim_outputs else None
    queue = _queue(items, out_dir, confirmer, toolset, verify, recovery, loop_break, view, events)
    for e in queue:
        configure(e, thinking=bool(spec.chat_template_kwargs.get("enable_thinking")), done_gate=done_gate,
                  search_hint=search_hint, gnu_tools=gnu_tools, send_note=send_note, examples=examples)
    results: list[dict] = []
    from ..backends.mlx import run_scheduler  # MLX: loaded only when running
    h = episode_hooks(lambda e: prompt_ids(tok, spec, e.messages, e.toolset), max_steps, out_dir, results, on_episode,
                      seeds=episode_seeds, stream=stream)
    run_scheduler(model, tok, sampling, queue, prompt_fn=h.prompt_fn, apply_fn=h.apply_fn, finish_fn=h.finish_fn,
                  max_context=max_context, batch_size=batch_size, prefill_step=prefill_step,
                  prefill_batch=prefill_batch, context_budget=context_budget, reuse_cache=reuse_cache,
                  progress_every=progress_every, seed_fn=h.seed_fn,
                  state_saves=state_saves, shared_prefix=shared_prefix, on_token=h.token_fn, gate=gate)
    return results


def run_episodes_on(backend, items: list[Item], out_dir: Path, *, max_steps: int = 20, on_episode=None,
                    confirmer="approve_all", toolset: str = "files", verify: str = "none", recovery: bool = False,
                    loop_break: bool = True, trimmer=None, events: Sink = null_sink, stream: bool = False,
                    limits: RunLimits | None = None, prompt_fn=None, stats: dict | None = None,
                    handle=None, thinking: bool = False, done_gate: bool = False,
                    gnu_tools: bool = False, examples: bool = False, send_note: bool = True,
                    search_hint: bool = False) -> list[dict]:
    """Agent tasks through any `Backend` (scripted for tests, OpenAI-compatible servers). The model input is
    `prompt_fn(e)`: by default the chat messages; a raw-mode server gets the rendered text (`prompt_text`), a
    chat-mode server the messages + tools (`chat_prompt`). Same
    loop, tools, layers and records as `run_episodes`."""
    queue = _queue(items, out_dir, confirmer, toolset, verify, recovery, loop_break, trimmer, events)
    for e in queue:
        configure(e, thinking=thinking, done_gate=done_gate, search_hint=search_hint, gnu_tools=gnu_tools,
                  send_note=send_note, examples=examples)
    results: list[dict] = []
    h = episode_hooks(prompt_fn or (lambda e: [dict(m) for m in e.messages]), max_steps, out_dir, results,
                      on_episode, stream=stream)
    h.handle = handle                                        # episodes may join this run (swarm groundwork)
    st = backend.run(queue, h, limits or RunLimits())
    if stats is not None:                                    # the backend's run stats (tokens, cache, ban rungs...)
        stats.update(st or {})
    return results


def run_episodes_stepwise(model, tok, spec: ModelSpec, items: list[Item], out_dir: Path, sampling: SamplingSpec, *,
                          max_steps: int = 20, max_context: int = 16000, batch_size: int = 4, prefill_step: int = 1024,
                          on_episode=None, confirmer="approve_all") -> list[dict]:
    """The first scheduler: lockstep, the whole conversation re-read every step (kept for comparison)."""
    from ..generate import generate_batch  # MLX: loaded only when running
    eps = _new_episodes(items, out_dir, confirmer)
    results: list[dict] = []
    step = 0
    while active := [e for e in eps if not e.done]:
        step += 1
        pairs = []
        for e in active:
            ids = prompt_ids(tok, spec, e.messages)
            if len(ids) > max_context:
                e.done, e.stop = True, "context_limit"
            else:
                pairs.append((e, ids))
        if pairs:
            t0 = time.perf_counter()
            gens, _ = generate_batch(model, tok, [p for _, p in pairs], sampling, batch_size=batch_size,
                                     prefill_step=prefill_step, progress_every=0)
            dt = time.perf_counter() - t0
            for (e, p), g in zip(pairs, gens, strict=True):
                e.prompt_tokens += len(p)
                e.processed_tokens += len(p)
                e.output_tokens += g.n_tokens
                e.seconds += dt / len(pairs)
                _apply_reply(e, g.text, g.finish_reason, max_steps)
        for e in [e for e in active if e.done]:
            _record(e, out_dir, results, on_episode)
        print(f"  step {step}: {len(active)} active, {sum(e.done for e in eps)}/{len(eps)} done", flush=True)
    return results
