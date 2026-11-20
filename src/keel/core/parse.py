"""Tool-call parser for the Qwen3.5 chat template (XML `<tool_call><function=..>`), with a JSON fallback.

Moved out of `evals/bfcl.py` (which re-exports it) so the harness can parse calls without importing bfcl_eval.
Behaviour is unchanged for every call except file text: a parameter marked `x-keel-raw` in the parser's view of a
tool (write_file content, edit_file old/new text; see core.tools) keeps its own leading / trailing whitespace. The
model writes `<parameter=content>\n...text\n\n</parameter>`: the regex removes the format's one newline on each side,
and `.strip()` used to remove the text's own final newline too, so every written file lost it (2026-11-14).
"""

from __future__ import annotations

import ast
import json
import re

_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(raw: str, schema: dict | None):
    if (schema or {}).get("x-keel-raw"):
        return raw                 # file text: exactly what the model wrote (only the format's own newline is removed)
    t = (schema or {}).get("type", "string")
    v = raw.strip()
    try:
        if t == "integer":
            return int(float(v)) if re.fullmatch(r"-?\d+(\.0+)?", v) else int(v)
        if t == "float" or t == "number":
            return float(v)
        if t == "boolean":
            return {"true": True, "false": False}.get(v.lower(), v)
        if t in ("array", "tuple", "dict", "object"):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return ast.literal_eval(v)
        if t == "any":
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
    except (ValueError, SyntaxError):
        return v
    return v  # string


def parse_calls(text: str, functions: list[dict]) -> list[dict]:
    """Return [{name: {param: value}}] in call order. Accepts the template's XML format and a JSON fallback."""
    text = text.split("</think>")[-1]
    specs = {f["name"]: f.get("parameters", {}).get("properties", {}) for f in functions}
    calls = []
    for block in _CALL.findall(text):
        found = _FUNC.findall(block)
        if found:
            for name, body in found:
                name = name.strip()
                props = specs.get(name, {})
                calls.append({name: {p.strip(): _coerce(v, props.get(p.strip())) for p, v in _PARAM.findall(body)}})
            continue
        try:                                   # JSON fallback: {"name": ..., "arguments": {...}}
            obj = json.loads(block.strip())
            calls.append({obj["name"]: obj.get("arguments", obj.get("parameters", {}))})
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return calls
