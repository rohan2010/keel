"""BFCL (Berkeley Function-Calling Leaderboard) single-turn tool calling, scored with the official AST checker.

Data and checker come from the `bfcl-eval` package (UC Berkeley, Apache-2.0), installed with --no-deps (approved by
the user). Only its AST checker is used; its model-config module imports API clients we don't install, so a minimal
stand-in providing the one flag the checker reads (`underscore_to_dot`, False for local models) is registered first.

Qwen3.5 calls tools in an XML-style format defined by its chat template:
  <tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>
Values arrive as text; they are converted to typed Python values using the function's parameter schema before the
official checker compares them with the possible answers.
"""

from __future__ import annotations

import json
import sys
import types
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# --- minimal stand-in for bfcl_eval.constants.model_config (pulls in anthropic/openai clients otherwise) ----------
if "bfcl_eval.constants.model_config" not in sys.modules:
    _stub = types.ModuleType("bfcl_eval.constants.model_config")
    _stub.MODEL_CONFIG_MAPPING = defaultdict(lambda: types.SimpleNamespace(underscore_to_dot=False))
    sys.modules["bfcl_eval.constants.model_config"] = _stub

import bfcl_eval  # noqa: E402
from bfcl_eval.constants.enums import Language  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402

DATA = Path(bfcl_eval.__file__).parent / "data"
# single-turn categories scored by AST; *irrelevance = correct when no call is made; live_relevance = >= 1 call
AST_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple",
                  "live_parallel", "live_parallel_multiple"]
NO_CALL_CATEGORIES = ["irrelevance", "live_irrelevance"]
CATEGORIES = AST_CATEGORIES + NO_CALL_CATEGORIES


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


@lru_cache(maxsize=None)
def load(category: str) -> list[dict]:
    rows = _read(DATA / f"BFCL_v4_{category}.json")
    ans_file = DATA / "possible_answer" / f"BFCL_v4_{category}.json"
    answers = {r["id"]: r["ground_truth"] for r in _read(ans_file)} if ans_file.exists() else {}
    return [{**r, "category": category, "ground_truth": answers.get(r["id"])} for r in rows]


# --- tool schema: BFCL's dialect -> JSON schema for the chat template -------------------------------------------------
from ..core.schema import _TYPE, to_json_schema  # noqa: E402,F401  (moved to core/schema.py: no bfcl_eval needed)


def tools_for(problem: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                              "parameters": to_json_schema(f.get("parameters", {}))}}
            for f in problem["function"]]


def messages_for(problem: dict) -> list[dict]:
    turn = problem["question"][0]           # single-turn categories: one turn of messages
    return [{"role": m["role"], "content": m["content"]} for m in turn]


# --- parsing the model's calls (moved to core/parse.py; re-exported for old imports) ---------------------
from ..core.parse import _CALL, _FUNC, _PARAM, _coerce, parse_calls  # noqa: E402,F401


def score(problem: dict, text: str) -> dict:
    calls = parse_calls(text, problem["function"])
    cat = problem["category"]
    if cat in NO_CALL_CATEGORIES:
        ok = len(calls) == 0
        return {"correct": ok, "n_calls": len(calls), "error_type": None if ok else "irrelevance:called_a_tool"}
    if not calls:
        return {"correct": False, "n_calls": 0, "error_type": "no_call"}
    res = ast_checker(problem["function"], calls, problem["ground_truth"], Language.PYTHON, cat, "local")
    return {"correct": bool(res["valid"]), "n_calls": len(calls), "error_type": res.get("error_type"),
            "error": (res.get("error") or [None])[0] if not res["valid"] else None}
