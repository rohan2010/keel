""": native tool vocabulary (`--native-tools <family>`).

The model sees the tool and argument names its family's own agent uses; keel's code keeps its own names. Translation
happens at the edge only:
    out  schemas + parser view renamed (`toolset_view`), keel-authored text renamed (`text`)
    in   parsed calls renamed back right after parsing (`to_internal`), so scorers, the done gate, the delete guard,
         trim, the loop breaker and the logs see keel's names and arguments exactly as before.
The call format is already the family's own (the chat template), so only names change. Tools with no counterpart in
the family's agent keep keel's names. Flag off = nothing here runs.

Qwen: names from Qwen Code (qwenlm.github.io/qwen-code-docs, developers/tools: file-system, shell; read 2026-11-17).
read_file's `offset` is 0-based there (keel's start_line is 1-based), so it is converted; every other argument keeps
keel's meaning.
"""

from __future__ import annotations

import copy
import re

# family -> keel tool -> (native name, {keel arg: native arg})
TABLES: dict[str, dict[str, tuple[str, dict[str, str]]]] = {
    "qwen": {
        "run_shell": ("run_shell_command", {}),
        "list_files": ("list_directory", {}),
        "read_file": ("read_file", {"path": "file_path", "start_line": "offset", "max_lines": "limit"}),
        "write_file": ("write_file", {"path": "file_path"}),
        "edit_file": ("edit", {"path": "file_path", "old_text": "old_string", "new_text": "new_string"}),
        "search": ("grep_search", {}),
    },
}
FAMILIES = tuple(TABLES)

# descriptions that must change with a renamed argument's meaning (keyed by native tool, native arg)
_DESC = {("qwen", "read_file", "offset"): "First line to read, counted from 0. Default 0."}


def _check(family: str) -> dict:
    if family not in TABLES:
        raise ValueError(f"unknown tool family {family!r}; known: {', '.join(FAMILIES)}")
    return TABLES[family]


def schema(s: dict, family: str) -> dict:
    """One OpenAI-style schema with the family's names (a copy; the input is not changed)."""
    t = _check(family)
    s = copy.deepcopy(s)
    f = s["function"]
    name, args = t.get(f["name"], (f["name"], {}))
    f["name"] = name
    p = f["parameters"]
    p["properties"] = {args.get(k, k): v for k, v in p["properties"].items()}
    if "required" in p:
        p["required"] = [args.get(k, k) for k in p["required"]]
    for k, v in p["properties"].items():
        if (family, name, k) in _DESC:
            v["description"] = _DESC[(family, name, k)]
    if "description" in f:
        f["description"] = text(f["description"], family)
    return s


def func(fn: dict, family: str) -> dict:
    """The parser's view of one tool (name + parameter types, raw-text flags kept) with the family's names."""
    t = _check(family)
    fn = copy.deepcopy(fn)
    name, args = t.get(fn["name"], (fn["name"], {}))
    fn["name"] = name
    props = fn.get("parameters", {}).get("properties")
    if props is not None:
        fn["parameters"]["properties"] = {args.get(k, k): v for k, v in props.items()}
    return fn


def to_internal(calls: list[dict], family: str) -> list[dict]:
    """Parsed calls ([{name: args}]) back to keel's names and argument meanings. Unknown names pass through unchanged
    (the loop answers them with its usual error), so a keel name the model remembers is not silently accepted twice."""
    t = _check(family)
    back = {native: (keel, {v: k for k, v in args.items()}) for keel, (native, args) in t.items()}
    out = []
    for c in calls:
        name, args = next(iter(c.items()))
        if name not in back:
            out.append(c)
            continue
        keel, amap = back[name]
        if isinstance(args, dict):
            args = {amap.get(k, k): v for k, v in args.items()}
            if family == "qwen" and keel == "read_file" and "start_line" in args:
                try:                                     # Qwen Code's offset is 0-based; keel's start_line 1-based
                    args["start_line"] = int(args["start_line"]) + 1
                except (TypeError, ValueError):
                    pass                                 # the file tool reports the bad value as usual
        out.append({keel: args})
    return out


def _words(family: str) -> list[tuple[re.Pattern, str]]:
    t = _check(family)
    pairs = {keel: native for keel, (native, _) in t.items() if keel != native}
    for _, (_, args) in t.items():
        pairs.update({k: v for k, v in args.items() if k != "path"})   # "path" is an ordinary English word too
    return [(re.compile(rf"(?<![\w-]){re.escape(k)}(?![\w-])"), v) for k, v in
            sorted(pairs.items(), key=lambda kv: -len(kv[0]))]


def text(s: str, family: str | None) -> str:
    """keel-authored text (system prompt, tool errors and warnings, the done-gate message) with the family's names.
    Never applied to file contents or command output."""
    if not family:
        return s
    for pat, new in _words(family):
        s = pat.sub(new, s)
    return s
