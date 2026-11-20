"""Pre-registration: write the comparison down before the runs exist, freeze it, and only call a
verdict "pre-registered" when the runs match what was frozen.

A pre-registration is a short YAML file:

    name: reminder-keep            # unique; letters, digits, - and _
    suite: bfcl-mt                 # agent | bfcl-mt | when2call
    split: val
    hypothesis: the reminder raises pass@1 on miss_func
    arms:                          # how to recognise each arm's runs (every given key must hold for every run)
      a: {arm: noreminder}         #   arm:    the `keel-core eval --arm` name (eval.json)
      b: {arm: full, layers: {reminder: true}}   # layers: values the run header must show
                                   #   runs:   folder-name fragments, one per run, in try order
    primary: pass@1                # the metric the verdict is about (pass@1 only, for now)
    pair: try                      # item | try
    sided: greater                 # two | greater (B > A)
    alpha: 0.05
    group: miss_func               # optional: one BFCL category / agent task
    test: task-clustered           # optional: item (default; exact McNemar / paired bootstrap) or task-clustered
                                   #   (sign-flip permutation over tasks; the item test is shown beside it)
    family: [h1-name, h2-name]     # optional: every pre-registration tested together (this one included); Holm
                                   #   over the family once each member has a pre-registered result

`keel-core prereg <file>` freezes it: `experiments/prereg/<name>.json` keeps the text, its sha256 and the time.
Refreezing a name with different text is refused. `keel-core compare A B --prereg <name>` then checks:
the file (if given as a path) still has the frozen hash; suite and split match; each arm's runs match its spec; every
run started after the freeze. Then it compares with the frozen pair / sided / alpha / group. Anything that fails is
listed, and the verdict stays "exploratory".
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import yaml

from .compare import Arm
from .runs import SUITES, eval_meta

KEYS = {"name", "suite", "split", "hypothesis", "arms", "primary", "pair", "sided", "alpha", "group", "notes",
        "test", "family", "gate", "allow"}
TESTS = ("item", "task-clustered")   # item: exact McNemar / paired bootstrap; task-clustered: sign-flip over tasks
PRIMARY = ("pass@1", "false_success")   # false_success: B's false-"done" rate lower than A's (sided greater = lower)


def _dir() -> Path:
    from ..runlog import EXPERIMENTS
    return EXPERIMENTS / "prereg"


def parse(text: str) -> dict:
    """Validate a pre-registration; raises ValueError with every problem."""
    try:
        d = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ValueError(f"not valid YAML: {err}") from err
    if not isinstance(d, dict):
        raise ValueError("a pre-registration is a YAML mapping")
    errs = []
    unknown = set(d) - KEYS
    if unknown:
        errs.append(f"unknown keys: {', '.join(sorted(unknown))}")
    for k in ("name", "suite", "split", "arms", "primary", "sided", "alpha"):
        if k not in d:
            errs.append(f"missing {k}")
    if "name" in d and not re.fullmatch(r"[A-Za-z0-9_\-]+", str(d["name"])):
        errs.append("name: letters, digits, - and _ only")
    if "suite" in d and d["suite"] not in SUITES.values():
        errs.append(f"suite must be one of {', '.join(SUITES.values())}")
    if "primary" in d and d["primary"] not in PRIMARY:
        errs.append(f"primary must be one of {', '.join(PRIMARY)}")
    if d.get("sided", "two") not in ("two", "greater"):
        errs.append("sided: two or greater")
    if d.get("pair", "item") not in ("item", "try"):
        errs.append("pair: item or try")
    if d.get("test", "item") not in TESTS:
        errs.append(f"test: {' or '.join(TESTS)}")
    fam = d.get("family")
    if fam is not None and not (isinstance(fam, list) and len(fam) >= 2 and d.get("name") in fam
                                and all(isinstance(x, str) for x in fam)):
        errs.append("family: a list of at least two pre-registration names, this one included")
    if d.get("allow") is not None and not (isinstance(d["allow"], list) and all(isinstance(x, str) for x in d["allow"])):
        errs.append("allow: a list of run-header keys whose differences are accepted (frozen with the plan)")
    if d.get("gate") is not None:
        if not isinstance(d["gate"], str) or d["gate"] == d.get("name"):
            errs.append("gate: the name of the pre-registration that must be confirmed first")
        if d.get("family"):
            errs.append("gate and family are two different corrections: use one")
    if "alpha" in d and not (isinstance(d["alpha"], (int, float)) and 0 < d["alpha"] < 1):
        errs.append("alpha: a number between 0 and 1")
    arms = d.get("arms")
    if "arms" in d and not (isinstance(arms, dict) and set(arms) == {"a", "b"}):
        errs.append("arms: exactly a and b")
    elif isinstance(arms, dict):
        for k, spec in arms.items():
            if not isinstance(spec, dict) or not set(spec) & {"arm", "layers", "runs"} or set(spec) - {"arm", "layers",
                                                                                                    "runs"}:
                errs.append(f"arms.{k}: give arm, layers and / or runs")
    if errs:
        raise ValueError("; ".join(errs))
    d.setdefault("pair", "item")
    d.setdefault("test", "item")
    return d


def freeze(path: str | Path, base: Path | None = None) -> dict:
    text = Path(path).read_text()
    d = parse(text)
    base = base or _dir()
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{d['name']}.json"
    sha = hashlib.sha256(text.encode()).hexdigest()
    if out.exists():
        old = json.loads(out.read_text())
        if old["sha256"] != sha:
            raise ValueError(f"{d['name']} is already frozen with different text (frozen {old['frozen']}); a changed "
                             "plan needs a new name")
        return old
    rec = {"name": d["name"], "frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256": sha, "text": text,
           "source": str(path), "plan": d}
    out.write_text(json.dumps(rec, indent=1))
    return rec


def load(name_or_path: str, base: Path | None = None) -> dict:
    """A frozen pre-registration by name, or by its file (which must still match the frozen hash)."""
    base = base or _dir()
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        text = p.read_text()
        d = parse(text)
        rec_path = base / f"{d['name']}.json"
        if not rec_path.exists():
            raise ValueError(f"{d['name']} was never frozen (keel-core prereg {p})")
        rec = json.loads(rec_path.read_text())
        if rec["sha256"] != hashlib.sha256(text.encode()).hexdigest():
            rec = {**rec, "edited_after_freeze": True}
        return rec
    rec_path = base / f"{name_or_path}.json"
    if not rec_path.exists():
        raise ValueError(f"no frozen pre-registration named {name_or_path!r}")
    return json.loads(rec_path.read_text())


def _arm_problems(arm: Arm, spec: dict, side: str) -> list[str]:
    out = []
    for h in arm.headers:
        if "arm" in spec and eval_meta(h.dir).get("arm") != spec["arm"]:
            out.append(f"{side}: {h.name} is not eval arm {spec['arm']!r}")
        for k, v in (spec.get("layers") or {}).items():
            if h.layers.get(k) != v:
                out.append(f"{side}: {h.name} has {k} = {h.layers.get(k)!r}, pre-registered {v!r}")
    if "runs" in spec:
        want = list(spec["runs"])
        if len(want) != len(arm.headers) or any(w not in h.name for w, h in zip(want, arm.headers, strict=True)):
            out.append(f"{side}: runs {[h.name for h in arm.headers]} are not {want}")
    return out


def check(rec: dict, A: Arm, B: Arm) -> list[str]:
    """Every way these runs differ from the frozen plan (empty = the label may be used)."""
    p, problems = rec["plan"], []
    if rec.get("edited_after_freeze"):
        problems.append("the file was edited after it was frozen")
    for side, arm in (("A", A), ("B", B)):
        for h in arm.headers:
            if h.suite != p["suite"]:
                problems.append(f"{side}: suite {h.suite}, pre-registered {p['suite']}")
            if h.split != p["split"]:
                problems.append(f"{side}: split {h.split}, pre-registered {p['split']}")
            if not h.started or h.started < rec["frozen"]:
                problems.append(f"{side}: {h.name} started {h.started}, before the freeze ({rec['frozen']})")
        problems += _arm_problems(arm, p["arms"]["a" if side == "A" else "b"], side)
    return problems


def test_settings(rec: dict) -> dict:
    """The comparison settings a pre-registration fixes."""
    p = rec["plan"]
    return {"pair": p.get("pair", "item"), "sided": p["sided"], "alpha": float(p["alpha"]), "group": p.get("group"),
            "cluster": "task" if p.get("test") == "task-clustered" else None, "primary": p.get("primary", "pass@1"),
            "allow": tuple(p.get("allow") or ())}


# ---- Holm over a family ------------------------------------------------------------------------------------------

def record_result(rec: dict, c, base: Path | None = None) -> None:
    """Keep a pre-registered verdict's deciding p next to the frozen plan (`<name>.result.json`). The first one is
    kept: a pre-registered test is run once, so a later compare of the same plan cannot replace it."""
    base = base or _dir()
    out = base / f"{rec['name']}.result.json"
    if out.exists():
        return
    t = c.pass_test
    out.write_text(json.dumps({"name": rec["name"], "p": t["p"], "delta": t["delta"], "test": t["test"],
                               "a": c.runs.get("a"), "b": c.runs.get("b"),
                               "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=str))


def gate_status(rec: dict, base: Path | None = None, _seen: frozenset = frozenset()) -> dict | None:
    """Fixed-sequence testing: a plan with `gate: X` is tested only if X was confirmed. Returns None (no gate),
    or {"gate": X, "state": "confirmed" | "not confirmed" | "pending"}; X's own gate is followed too."""
    g = rec["plan"].get("gate")
    if not g:
        return None
    base = base or _dir()
    if g in _seen:
        return {"gate": g, "state": "not confirmed", "why": "gate cycle"}
    res, frz = base / f"{g}.result.json", base / f"{g}.json"
    if not frz.exists():
        return {"gate": g, "state": "pending", "why": f"{g} was never frozen"}
    if not res.exists():
        return {"gate": g, "state": "pending", "why": f"{g} has no pre-registered result yet"}
    grec = json.loads(frz.read_text())
    up = gate_status(grec, base, _seen | {rec["name"]})
    if up and up["state"] != "confirmed":
        return {"gate": g, "state": up["state"], "why": f"{g} itself waits on {up['gate']}"}
    r = json.loads(res.read_text())
    ok = r["p"] < float(grec["plan"]["alpha"]) and (r["delta"] > 0 if grec["plan"].get("primary", "pass@1") ==
                                                     "pass@1" else r["delta"] < 0)
    return {"gate": g, "state": "confirmed" if ok else "not confirmed", "p": r["p"]}


def holm(rec: dict, base: Path | None = None) -> dict | None:
    """Holm step-down over the plan's family, once every member has a recorded pre-registered result. Returns
    {name: {"p", "threshold", "confirmed"}} plus "pending" (members still without a result), or None if the plan
    has no family."""
    fam = rec["plan"].get("family")
    if not fam:
        return None
    base = base or _dir()
    alpha = float(rec["plan"]["alpha"])
    got = {}
    for n in fam:
        f = base / f"{n}.result.json"
        if f.exists():
            got[n] = json.loads(f.read_text())["p"]
    pending = [n for n in fam if n not in got]
    if pending:
        return {"pending": pending}
    out, m, still = {}, len(fam), True
    for rank, (n, p) in enumerate(sorted(got.items(), key=lambda kv: kv[1])):
        thr = alpha / (m - rank)
        still = still and p < thr
        out[n] = {"p": p, "threshold": thr, "confirmed": still}
    return {"pending": [], "results": out}
