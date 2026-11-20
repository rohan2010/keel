"""Tool registry and the approver.

Every tool is declared once with its kind, schema and output limit. The kind says what a tool can do to the user's
folder, which the front end shows and the delete guard keys on:
    read    looks only (list_files, read_file, search)
    write   changes or creates files (write_file, edit_file, move)
    delete  removes files: always goes through the delete guard
    shell   anything (run_shell): runs in the sandbox, and any delete it would make goes through the guard

The approver answers the delete guard's question "may this command delete these files?". Who answers varies (a
person in the TUI, a simulated user in an eval, recorded in the run config); whether the question is asked never
does. There is deliberately no setting that skips the guard, and `tests/checks/check_guard.py` + `tests/test_core.py`
check that none exists.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Protocol

from ..agent.filetools import FILE_TOOLS
from .events import Event, Sink, null_sink

KINDS = ("read", "write", "delete", "shell")

RUN_SHELL_SCHEMA = {"type": "function", "function": {
    "name": "run_shell",
    "description": "Run a bash command in the current folder and return its output (stdout and stderr).",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command."}},
                   "required": ["command"]}}}

RUN_PYTHON_SCHEMA = {"type": "function", "function": {
    "name": "run_python",
    "description": "Run a Python 3 script in the current folder and return what it prints. Use it for counting, sums, "
                   "dates, parsing or writing CSV / JSON, and changing many files at once (os, shutil, csv, json, "
                   "hashlib, datetime are available). Print the result you need.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code."}},
                   "required": ["code"]}}}

_KIND = {"list_files": "read", "read_file": "read", "search": "read", "write_file": "write", "edit_file": "write",
         "move": "write", "delete": "delete", "run_shell": "shell", "run_python": "shell"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str
    schema: dict                 # OpenAI-style function schema (what the chat template renders)
    output_limit: int | None     # characters the tool itself returns at most (None = unbounded; trim handles it)
    parse_func: dict | None = None   # parameter types for the call parser, when they differ from the schema's

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown tool kind {self.kind!r}")

    @property
    def func(self) -> dict:
        """The shape the call parser wants (name + parameter types)."""
        return self.parse_func or {"name": self.name, "parameters": self.schema["function"]["parameters"]}


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        self.specs = list(specs)
        self._by = {s.name: s for s in specs}

    def __contains__(self, name: str) -> bool:
        return name in self._by

    def __getitem__(self, name: str) -> ToolSpec:
        return self._by[name]

    @property
    def schemas(self) -> list[dict]:
        return [s.schema for s in self.specs]

    @property
    def funcs(self) -> list[dict]:
        return [s.func for s in self.specs]

    def kind(self, name: str) -> str | None:
        s = self._by.get(name)
        return s.kind if s else None


# the research parser saw run_shell with only its argument type; kept so parsing stays byte-identical
RUN_SHELL_FUNC = {"name": "run_shell", "parameters": {"properties": {"command": {"type": "string"}}}}


# file text the parser must keep byte for byte (its own final newline included); see core.parse._coerce
RAW_TEXT = {"write_file": ("content",), "edit_file": ("old_text", "new_text"), "run_python": ("code",)}


def _parse_func(schema: dict) -> dict | None:
    name = schema["function"]["name"]
    if name == "run_shell":
        return RUN_SHELL_FUNC
    if name not in RAW_TEXT:
        return None
    params = copy.deepcopy(schema["function"]["parameters"])
    for p in RAW_TEXT[name]:
        params["properties"][p]["x-keel-raw"] = True
    return {"name": name, "parameters": params}


def _spec(schema: dict, limit: int | None) -> ToolSpec:
    name = schema["function"]["name"]
    return ToolSpec(name, _KIND[name], schema, limit, _parse_func(schema))


def shell_registry() -> ToolRegistry:
    """The plain stack: run_shell only."""
    return ToolRegistry([_spec(RUN_SHELL_SCHEMA, None)])


def files_registry() -> ToolRegistry:
    """The default stack: file tools + run_shell, in the order the research prompt rendered them."""
    from ..agent.filetools import MAX_OUT
    return ToolRegistry([_spec(s, MAX_OUT) for s in FILE_TOOLS] + [_spec(RUN_SHELL_SCHEMA, None)])


def files_py_registry() -> ToolRegistry:
    """: the default file tools + run_shell + run_python (code for computation; `--toolset files_py`)."""
    from ..agent.filetools import MAX_OUT
    return ToolRegistry([_spec(s, MAX_OUT) for s in FILE_TOOLS] + [_spec(RUN_SHELL_SCHEMA, None),
                                                                   _spec(RUN_PYTHON_SCHEMA, None)])


REGISTRIES = {"shell": shell_registry, "files": files_registry, "files_py": files_py_registry}


# --- approver -----------------------------------------------------------------------------------------------------

class Approver(Protocol):
    def __call__(self, command: str, files: list[str]) -> bool: ...


def reporting(approver: Callable[[str, list[str]], bool], sink: Sink = null_sink, session: str = "") -> Approver:
    """Wrap an approver so every question and answer is also an event (the TUI shows it; evals log it).
    A crash or any non-True answer counts as a refusal."""

    def ask(command: str, files: list[str]) -> bool:
        try:
            allowed = approver(command, files) is True
        except Exception:
            allowed = False
        sink(Event("permission", session, {"command": command, "files": list(files), "allowed": allowed}))
        return allowed

    return ask
