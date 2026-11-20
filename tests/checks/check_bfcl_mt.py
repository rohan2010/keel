"""Checks for the BFCL multi-turn adapter (no model needed).

  uv run python tests/checks/check_bfcl_mt.py

1. Ground truth straight into the scorer: 100% valid (and strict) on all 800 entries.
2. Ground truth rendered as Qwen tool-call replies and pushed through MTEpisode (parse -> call strings -> BFCL
   executor -> next turn -> score): 100% valid. Tests our parser, value types and turn handling end to end.
3. Empty replies (model never calls a tool): 0% valid.
4. Force quit: a model that keeps calling forever is cut after 21 steps in turn 0 and the entry fails.
5. Split: dev/eval by scenario number, identical across the four categories, roughly half each.
"""

import ast
import json
import sys
from collections import Counter

from keel.evals import bfcl_mt as M


def _param_names(entry: dict) -> dict[str, list[str]]:
    """Method name -> parameter names in signature order (ground truth sometimes passes arguments positionally)."""
    import importlib
    import inspect

    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
    out = {}
    for cls in entry["involved_classes"]:
        c = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[cls]), cls)
        for n, f in inspect.getmembers(c, predicate=inspect.isfunction):
            out[n] = [p for p in inspect.signature(f).parameters if p != "self"]
    return out


def render_reply(call_strings: list[str], entry: dict) -> str:
    """Ground-truth call strings -> the XML tool-call text Qwen3.5 produces (always named parameters)."""
    names = _param_names(entry)
    blocks = []
    for s in call_strings:
        node = ast.parse(s, mode="eval").body
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        args = [(names[node.func.id][i], a) for i, a in enumerate(node.args)] + [(k.arg, k.value) for k in node.keywords]
        params = []
        for k, a in args:
            v = ast.literal_eval(a)
            txt = v if isinstance(v, str) else json.dumps(v)
            params.append(f"<parameter={k}>\n{txt}\n</parameter>")
        blocks.append(f"<tool_call>\n<function={node.func.id}>\n" + "\n".join(params) + "\n</function>\n</tool_call>")
    return "\n".join(blocks)


def run_episode(entry, name, replies_for_turn):
    e = M.MTEpisode(entry, name).start()
    while not e.done:
        t = e.turn
        for r in replies_for_turn(t, e):
            e.apply(r, "stop")
            if e.done or e.turn != t:
                break
    return e


def main():
    fails = 0
    for cat in M.CATEGORIES:
        rows = M.load(cat)
        # 1. ground truth straight in
        ok = strict = 0
        for r in rows:
            res = [[g] if g else [] for g in r["ground_truth"]]
            s = M.score(r, res, f"t1_{cat}")
            ok += s["valid"]
            strict += s["strict"]
            if not s["valid"]:
                print("  GT direct FAIL", r["id"], s["error_type"], str(s["error"])[:200])
        # 2. ground truth as model replies through the episode
        ok2, bad2 = 0, []
        for r in M.load(cat):
            gt = r["ground_truth"]
            e = run_episode(r, M.safe_name(f"20261108-000000_t2_{cat}"), lambda t, e, gt=gt, r=r: ([render_reply(gt[t], r)] if gt[t] else []) + ["Done."])
            s = M.score(r, e.result, M.safe_name(f"20261108-000000_t2_{cat}") + "_score")
            ok2 += s["valid"]
            if not s["valid"]:
                bad2.append((r["id"], s["error_type"], str(s["error"])[:200]))
        for b in bad2[:5]:
            print("  GT via episode FAIL", *b)
        # 3. empty replies
        ok3 = 0
        for r in M.load(cat)[:50]:
            e = run_episode(r, f"t3_{cat}", lambda t, e: ["I can't help with that."])
            ok3 += M.score(r, e.result, f"t3s_{cat}")["valid"]
        n = len(rows)
        line = f"{cat:13s} GT direct {ok}/{n} (strict {strict}) | GT via episode {ok2}/{n} | empty replies {ok3}/50"
        print(line)
        fails += (ok != n) + (ok2 != n) + (ok3 != 0)

    # 4. force quit
    r = M.load("base")[0]
    call = render_reply([r["ground_truth"][0][0]], r)
    e = M.MTEpisode(r, "t4").start()
    while not e.done:
        e.apply(call, "stop")
    s = M.score(r, e.result, "t4s")
    print(f"force quit: stop={e.stop} steps={e.steps} turns recorded={len(e.result)} valid={s['valid']} "
          f"({s['error_type']})")
    fails += not (e.stop == "force_quit" and e.steps == M.MAXIMUM_STEP_LIMIT + 1 and not s["valid"])

    # 5. split
    halves = {c: {M.scenario(r["id"]): M.half(r["id"]) for r in M.load(c)} for c in M.CATEGORIES}
    same = all(halves[c] == halves["base"] for c in M.CATEGORIES)
    cnt = Counter(halves["base"].values())
    print(f"split: same across categories={same}, scenarios per half {dict(cnt)}")
    fails += not same
    three = Counter(M.split_of(r["id"]) for r in M.load("base"))
    dev_same = all((M.split_of(r["id"]) == "dev") == (M.half(r["id"]) == "dev") for r in M.load("base"))
    print(f"3-way split (scenarios): {dict(three)}; dev unchanged: {dev_same}")
    fails += not dev_same or min(three.values()) < 40

    # 6. Module 6 layers
    m6 = 0
    for cat in M.CATEGORIES:                     # V1 on, ground truth + "Done." twice per turn -> still all valid
        ok = 0
        for r in M.load(cat):
            gt = r["ground_truth"]
            e = M.MTEpisode(r, M.safe_name(f"9_v1_{cat}"), verify="state").start()
            while not e.done:
                t = e.turn
                for rep in ([render_reply(gt[t], r)] if gt[t] else []) + ["Done.", "Done.", "Done."]:
                    e.apply(rep, "stop")
                    if e.done or e.turn != t:
                        break
            ok += M.score(r, e.result, M.safe_name(f"9_v1s_{cat}"))["valid"]
            checks = sum("STATE REPORT" in m["content"] for m in e.messages if m["role"] == "user")
            m6 += checks != len(gt)              # exactly one check per turn
        print(f"V1 + ground truth {cat:13s}: {ok}/{len(M.load(cat))} valid")
        fails += ok != len(M.load(cat))
    print(f"one check per turn: {'yes' if m6 == 0 else f'NO ({m6} entries off)'}")
    fails += m6 != 0

    r = M.load("base")[0]                        # report shows the turn's real change (a move in the file system)
    e = M.MTEpisode(r, M.safe_name("9_rep"), verify="state").start()
    e.apply(render_reply(r["ground_truth"][0], r), "stop")
    e.apply("I have successfully moved the file.", "stop")
    rep = e.messages[-1]["content"]
    print("report sample:\n  " + "\n  ".join(rep.splitlines()[:8]))
    ok = "STATE REPORT" in rep and ("Added" in rep or "Changed" in rep or "Removed" in rep) and "NOTE" not in rep
    fails += not ok
    print("report shows the change:", ok)

    e = M.MTEpisode(M.load("base")[0], M.safe_name("9_claim"), verify="state").start()
    e.apply("I have successfully moved the file to the archive.", "stop")
    ok = "nothing in the apps' state changed" in e.messages[-1]["content"]
    fails += not ok
    print("false-claim note:", ok)

    e = M.MTEpisode(M.load("base")[0], M.safe_name("9_self"), verify="self").start()
    e.apply("Done.", "stop")
    ok = e.messages[-1]["role"] == "user" and "Automatic check" in e.messages[-1]["content"] \
        and "STATE REPORT" not in e.messages[-1]["content"] and not e.done and e.turn == 0
    fails += not ok
    print("V0 check (no report), turn still open:", ok)

    e = M.MTEpisode(M.load("base")[0], M.safe_name("9_r1"), recovery=True).start()
    for _ in range(17):
        e.apply(render_reply(["pwd()"], r), "stop")
    tools = [m["content"] for m in e.messages if m["role"] == "tool"]
    ok = ("same result 3 times" in tools[2] and "same result 3 times" not in tools[3]
          and "3 steps left" in tools[16] and "3 steps left" not in tools[15])
    fails += not ok
    print("R1 repeat note on 3rd call, budget note after step 17:", ok, e.notes)

    # recent-requests variant: at the holdout turn the check quotes the earlier request too
    r = [x for x in M.load("miss_func") if x["id"] == "multi_turn_miss_func_84"][0]
    ht = int(next(iter(r["missed_function"])))
    for mode in ("state", "self"):
        e = M.MTEpisode(r, M.safe_name(f"9_recent_{mode}"), verify=mode, check_requests="recent").start()
        while e.turn < ht:
            e.apply("ok", "stop")
            e.apply("ok", "stop")
        e.apply("Please tell me what you would like to do.", "stop")
        msg = e.messages[-1]["content"]
        ok = ("Earlier: \"" + r["question"][ht - 1][0]["content"][:40]) in msg and "Latest: \"I have updated" in msg \
            and (("STATE REPORT" in msg) == (mode == "state"))
        fails += not ok
        print(f"recent requests ({mode}): check quotes earlier + latest turn:", ok)
    e = M.MTEpisode(M.load("base")[0], M.safe_name("9_recent_t0"), verify="state", check_requests="recent").start()
    e.apply("Done.", "stop")
    ok = "Latest:" not in e.messages[-1]["content"] and "user's request:" in e.messages[-1]["content"]
    fails += not ok
    print("recent requests, turn 0: falls back to the current form:", ok)
    okc = 0
    for x in M.load("miss_func"):                 # still scores ground truth correctly
        gt = x["ground_truth"]
        e = M.MTEpisode(x, M.safe_name("9_recent_gt"), verify="state", check_requests="recent").start()
        while not e.done:
            t = e.turn
            for rep in ([render_reply(gt[t], x)] if gt[t] else []) + ["Done.", "Done.", "Done."]:
                e.apply(rep, "stop")
                if e.done or e.turn != t:
                    break
        okc += M.score(x, e.result, M.safe_name("9_recent_gts"))["valid"]
    print(f"recent requests + V1 + ground truth miss_func: {okc}/200 valid")
    fails += okc != 200

    print("ALL CHECKS PASS" if not fails else f"{fails} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
