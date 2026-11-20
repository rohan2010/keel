"""When2Call (NVIDIA, CC-BY-4.0): should the model call a tool, ask for missing info, or say it can't help?

Data: `data/external/when2call/` (fetched by `scripts/fetch_when2call.py`, pinned revision, git-ignored). Each item
is rewritten from a BFCL v2 Live question (`source_id`) into up to three versions, labelled `tool_call`,
`request_for_info` or `cannot_answer`. Attribution: Ross et al., "When2Call: When (not) to Call Tools", NVIDIA 2025,
https://huggingface.co/datasets/nvidia/When2Call (CC-BY-4.0).

Splits (pre-registration):
- `half`: the research rule that split BFCL into practice / eval (`sha1(id) % 2`). The probe gate was fit on the
  practice half, so only eval-half (`unseen`) source questions count for the headline; practice-half = `seen`.
- `dev` / `held_out` inside the unseen items, by question (all versions of a question on one side).

Scoring (generation): the primary metric is the call decision. A reply is correct when it contains a parsed tool
call iff the label is `tool_call`. The MCQ reference (log-prob of each candidate answer) is in `mcq_candidates`.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path

from .. import paths
from ..core.parse import parse_calls
from ..core.schema import to_json_schema

DATA = paths.data_dir() / "external" / "when2call"
MCQ_FILE = DATA / "test" / "when2call_test_mcq.jsonl"
MCQ_SHA256 = "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c"
LABELS = ("tool_call", "request_for_info", "cannot_answer")
SPLITS = ("dev", "held_out", "seen")
ATTRIBUTION = ("When2Call (Ross et al., NVIDIA 2025), https://huggingface.co/datasets/nvidia/When2Call, "
               "CC-BY-4.0, revision 0582f774")


def bfcl_half(source_id: str) -> str:
    """The research's BFCL practice / eval rule (exp-weights wsp/tasks/tools.py); the probe was fit on practice."""
    return "practice" if int(hashlib.sha1(source_id.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def split_of(source_id: str) -> str:
    if bfcl_half(source_id) == "practice":
        return "seen"
    return "dev" if int(hashlib.sha1(f"when2call-dev:{source_id}".encode()).hexdigest(), 16) % 10 == 0 else "held_out"


def _parse_obj(x):
    return json.loads(x) if isinstance(x, str) else x


def load(split: str | None = None, path: Path = MCQ_FILE, check_sha: bool = True) -> list[dict]:
    """Items as dicts: id, source_id, split, label, question, functions (BFCL dialect), target (name or None), answers."""
    if not path.exists():
        raise SystemExit(f"When2Call data not found at {path}; fetch it with: uv run python scripts/fetch_when2call.py")
    raw = path.read_bytes()
    if check_sha and hashlib.sha256(raw).hexdigest() != MCQ_SHA256:
        raise SystemExit(f"{path} does not match the pinned sha256 (re-fetch with scripts/fetch_when2call.py)")
    out = []
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        target = _parse_obj(r["target_tool"]) if r.get("target_tool") else None
        item = {"id": r["uuid"], "source_id": r["source_id"], "split": split_of(r["source_id"]),
                "label": r["correct_answer"], "question": r["question"],
                "functions": [_parse_obj(t) for t in r["tools"]],
                "target": target["name"] if target else None, "answers": _parse_obj(r["answers"])}
        if split is None or item["split"] == split:
            out.append(item)
    return out


def tools_for(item: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                              "parameters": to_json_schema(f.get("parameters", {}))}}
            for f in item["functions"]]


def messages_for(item: dict) -> list[dict]:
    return [{"role": "user", "content": item["question"]}]


def score(item: dict, text: str) -> dict:
    calls = parse_calls(text, item["functions"])
    called = len(calls) > 0
    want = item["label"] == "tool_call"
    first = next(iter(calls[0])) if called else None
    return {"called": called, "correct": called == want, "n_calls": len(calls), "first_call": first,
            "right_tool": (first == item["target"]) if want and called else None}


def _rate(n: int, d: int):
    return round(n / d, 4) if d else None


def summarize(rows: list[dict]) -> dict:
    """rows: per-item records with label, called, correct, right_tool and (gate runs) gate action."""
    n = len(rows)
    by = {lab: [r for r in rows if r["label"] == lab] for lab in LABELS}
    non = [r for r in rows if r["label"] != "tool_call"]
    tc = by["tool_call"]
    out = {"n": n, "correct": sum(r["correct"] for r in rows),
           "accuracy": _rate(sum(r["correct"] for r in rows), n),
           "over_call": _rate(sum(r["called"] for r in non), len(non)),
           "over_call_request_for_info": _rate(sum(r["called"] for r in by["request_for_info"]),
                                               len(by["request_for_info"])),
           "over_call_cannot_answer": _rate(sum(r["called"] for r in by["cannot_answer"]), len(by["cannot_answer"])),
           "under_call": _rate(sum(not r["called"] for r in tc), len(tc)),
           "right_tool_when_called": _rate(sum(bool(r["right_tool"]) for r in tc if r["called"]),
                                           sum(r["called"] for r in tc)),
           "right_tool_all_tool_call_items": _rate(sum(bool(r["right_tool"]) for r in tc), len(tc)),
           "n_by_label": {lab: len(v) for lab, v in by.items()}}
    if any(r.get("gate") for r in rows):
        out["gate"] = {lab: {a: sum(1 for r in v if r.get("gate") == a) for a in ("force", "block", "free")}
                       for lab, v in by.items()}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b = right only in A, c = right only in B (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def compare(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """Paired comparison of two runs on the same items (A = gate off, B = gate on)."""
    a = {r["id"]: r for r in a_rows}
    b = {r["id"]: r for r in b_rows}
    ids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError(f"runs cover different items ({len(set(a) ^ set(b))} not shared)")
    only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
    only_b = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
    same_text = sum(a[i].get("text") == b[i].get("text") for i in ids)
    return {"n": len(ids), "a_correct": sum(a[i]["correct"] for i in ids), "b_correct": sum(b[i]["correct"] for i in ids),
            "only_a_correct": only_a, "only_b_correct": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
            "delta_points": round(100 * (only_b - only_a) / max(len(ids), 1), 2), "identical_replies": same_text}


# --- MCQ reference (log-prob of each candidate answer) ----------------------------------------------------------------
def qwen_call_text(call: dict) -> str:
    """A When2Call tool-call answer ({"name", "arguments"}) in Qwen3.5's XML call format."""
    parts = [f"<tool_call>\n<function={call['name']}>"]
    for k, v in (call.get("arguments") or {}).items():
        parts.append(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>")
    parts.append("</function>\n</tool_call>")
    return "\n".join(parts)


def mcq_candidates(item: dict) -> dict[str, str]:
    """The four candidate replies keyed by answer type (`direct` is a distractor, never the label)."""
    ans = item["answers"]
    out = {}
    for k in ("direct", "tool_call", "request_for_info", "cannot_answer"):
        v = ans.get(k)
        if v is None:
            continue
        if k == "tool_call":
            try:
                v = qwen_call_text(_parse_obj(v))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass                                    # malformed: score the raw text as given
        out[k] = v
    return out
