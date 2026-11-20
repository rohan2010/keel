""": every argument has a source.

A small model asked to "post a comment" with no comment text, or to "buy shares" with no amount, often makes the
value up and calls the tool (When2Call: a tool is called on 61.8% of ask-for-info items; BFCL miss_param 0/5 in the
 smoke). This check holds such a call before it runs: every argument value must appear in something the
model was given (the user's messages, tool outputs). If one doesn't, the call is not run and the model is told which
values have no source and to ask the user. Deterministic, generic, no model call.

Scope: tools whose arguments come from the user or the world (BFCL's APIs, When2Call's tools, a user's own tools).
Not keel's own file tools or shell: there the model legitimately computes new values (a file renamed to its date,
a command), so the check would misfire.

What counts as sourced (lenient on purpose; a held call costs one extra step, a wrong hold can cost the task):
- a value listed in the parameter's `enum`, or equal to its `default`; booleans; empty values;
- short text (up to 4 words): found in the sources after normalising case, quotes and spaces; a path when every part
  is found; a file name when its stem (without extension) is found;
- longer text: at least half of its words of 4+ letters are found;
- a number: written the same way somewhere in the sources (5, 5.0, 1,000 ...);
- lists and objects: every element.
Each call (name + arguments) is held at most once per episode: if the model sends the same call again, it runs.
"""

from __future__ import annotations

import json
import re

HOLD = ("Not run: {items} {verb} not in the user's request or in any tool output so far. Do not guess missing "
        "values: ask the user for them (or look them up with a tool), then make the call.")


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().strip("\"'`.,;:!?()[]{}")


def _words(s: str) -> set[str]:
    return set(re.findall(r"[\w']+", s))


class Sources:
    """The text the model was given, normalised once."""

    def __init__(self, texts: list[str]):
        self.text = " \n ".join(_norm(t) for t in texts if isinstance(t, str))
        self.words = _words(self.text)

    def has_text(self, v: str) -> bool:
        nv = _norm(v)
        if not nv:
            return True
        words = nv.split()
        if len(words) <= 4:
            if nv in self.text:
                return True
            parts = [p for p in re.split(r"[\\/]", nv) if p and p not in (".", "..", "~")]
            if len(parts) > 1 and all(p in self.text for p in parts):
                return True
            stem = nv.rsplit(".", 1)[0] if "." in nv else ""
            return len(stem) >= 3 and stem in self.text
        long_words = [w for w in _words(nv) if len(w) >= 4]
        return not long_words or sum(w in self.words for w in long_words) / len(long_words) >= 0.5

    def has_number(self, v: float) -> bool:
        forms = {str(v), f"{v:g}", f"{v:,}"}
        if float(v).is_integer():
            forms |= {str(int(v)), f"{int(v):,}"}
        return any(re.search(rf"(?<![\w.]){re.escape(f)}(?![\w]|\.\d)", self.text) for f in forms)


def unsourced(args: dict, sources: Sources, params: dict | None) -> list[tuple[str, object]]:
    """(argument, value) pairs with no source. `params`: the function's parameter schema ({"properties": ...})."""
    props = (params or {}).get("properties", {}) or {}
    out: list[tuple[str, object]] = []

    def visit(name: str, v, spec: dict) -> None:
        if v is None or isinstance(v, bool) or v == "" or v == [] or v == {}:
            return
        if v in (spec.get("enum") or []) or ("default" in spec and v == spec.get("default")):
            return
        if isinstance(v, (int, float)):
            if not sources.has_number(v):
                out.append((name, v))
        elif isinstance(v, str):
            if not sources.has_text(v):
                out.append((name, v))
        elif isinstance(v, list):
            item = spec.get("items") or {}
            for x in v:
                visit(name, x, item if isinstance(item, dict) else {})
        elif isinstance(v, dict):
            sub = spec.get("properties") or {}
            for k, x in v.items():
                visit(f"{name}.{k}", x, sub.get(k) or {})

    for k, v in (args or {}).items():
        visit(k, v, props.get(k) or {})
    return out


class ArgSourceCheck:
    """Per-episode state: which calls were already held once."""

    def __init__(self):
        self.held: set[str] = set()
        self.holds = 0

    def check(self, calls: list[dict], functions: list[dict], texts: list[str]) -> str | None:
        """calls: [{name: {arg: value}}] from one reply. Returns the message for the model if the reply is held
        (no call in it runs), else None."""
        src = Sources(texts)
        schemas = {f.get("name"): f.get("parameters") for f in functions}
        missing, sigs = [], []
        for c in calls:
            name, args = next(iter(c.items()))
            sig = json.dumps([name, args], sort_keys=True, default=str)
            if sig in self.held:
                continue                                  # held once already: the model insisted, let it run
            bad = unsourced(args if isinstance(args, dict) else {}, src, schemas.get(name))
            if bad:
                missing += [(name, k, v) for k, v in bad]
                sigs.append(sig)
        if not missing:
            return None
        self.held.update(sigs)
        self.holds += 1
        items = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)} in {n}" for n, k, v in missing[:6])
        return HOLD.format(items=items, verb="is" if len(missing) == 1 else "are")
