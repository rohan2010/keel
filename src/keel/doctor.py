"""keel doctor: what this Mac and a model server can really do.

    keel-core doctor                                Mac checks only (seconds, no model loaded)
    keel-core doctor --backend mlx                  + load the model in-process and ask for one tool call
    keel-core doctor --backend mlx-server           + server conformance (also: --backend server --url ... / ollama)

Mac checks: Apple Silicon, macOS, RAM (with a suggested memory budget), Python + uv, MLX + Metal, tokenizers /
transformers, the model (models folder or Hugging Face cache; never downloaded: `keel download`), bfcl-eval, the state folder, the temp folder's volume.

Server conformance: about ten small greedy requests on a probe task ("Create notes.txt containing 'hello
keel', then list the folder."), in raw mode (keel renders the prompt; needs the model folder's tokenizer) and chat
mode. Per mode: are tool calls kept / parsed, does a multi-call history round trip, which loop-breaker rungs the server
honours (logit_bias, tool_choice, stop), is greedy output repeatable, is the prompt cache reused. The result is a table
on stdout and `<state>/servers/<host>_<port>_<flavour>.json` (`--out` for another path). Later `--backend` runs read
the measured rungs from that file instead of the defaults, and record that they did.

Exit status: 1 if a required check fails (server: unreachable, or no mode where tool calls work; `--backend mlx`:
not Apple Silicon, MLX missing, no model, or the model gives no tool call), else 0. Warnings never fail.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends.choose import SHORTCUTS, add_backend_arguments, load_tokenizer
from .specs import QWEN35_4B_8BIT, ModelSpec, SamplingSpec

PROBE = "Create notes.txt containing 'hello keel', then list the folder."
OK, WARN, FAIL, SKIP, INFO = "ok", "warn", "FAIL", "skip", "info"
WEIGHTS_GB = 4.5                     # Qwen3.5-4B 8-bit resident
GPU_SHARE = 0.68                     # macOS lets the GPU use ~68% of RAM (~10.9 GB of 16, context.md)
SLOTS_PER_GB = 32000 / 6.4           # measured: 16 GB Mac -> 32k token slots (episodes x longest context)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    required: bool = False


# ---- the Mac ------------------------------------------------------------------------------------------------------

def _ram_gb() -> float | None:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()) / 2**30
    except Exception:
        return None


def memory_advice(ram_gb: float) -> dict:
    """An estimate from the research measurements, not a guarantee: token slots = episodes generating x longest
    context that fit next to the weights in the GPU's share of RAM."""
    budget = max(0, round((ram_gb * GPU_SHARE - WEIGHTS_GB) * SLOTS_PER_GB / 8000) * 8000)   # 16 GB -> 32000
    max_ctx = 16000 if budget >= 16000 else max(budget, 0)
    return {"gpu_gb": round(ram_gb * GPU_SHARE, 1), "context_budget": budget,
            "slots_at_8k": budget // 8000, "max_context": max_ctx}


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _models(models_dir: Path) -> list[tuple[str, float]]:
    out = []
    if models_dir.is_dir():
        for p in sorted(models_dir.iterdir()):
            q = p.resolve()
            if (q / "config.json").is_file():
                size = sum(f.stat().st_size for f in q.rglob("*") if f.is_file())
                out.append((p.name, size / 1e9))
    return out


def mac_checks(state_dir: Path, models_dir: Path, need_mlx: bool) -> list[Check]:
    c: list[Check] = []
    arm = sys.platform == "darwin" and platform.machine() == "arm64"
    c.append(Check("Apple Silicon", OK if arm else (FAIL if need_mlx else WARN),
                   platform.machine() + ("" if arm else ": in-process MLX needs an Apple Silicon Mac"), need_mlx))
    mac = platform.mac_ver()[0]
    major = int(mac.split(".")[0]) if mac else 0
    c.append(Check("macOS", OK if major >= 14 else WARN, mac or "not macOS"))
    ram = _ram_gb()
    if ram:
        adv = memory_advice(ram)
        c.append(Check("RAM", OK if adv["context_budget"] >= 16000 else WARN,
                       f"{ram:.0f} GB; GPU ~{adv['gpu_gb']} GB -> suggested --context-budget {adv['context_budget']}, "
                       f"{adv['slots_at_8k']} episodes at 8k tokens, max context {adv['max_context']} (estimate)"))
    else:
        c.append(Check("RAM", WARN, "could not read hw.memsize"))
    py = sys.version_info
    c.append(Check("Python", OK if py >= (3, 12) else FAIL, platform.python_version(), True))
    c.append(Check("uv", OK if shutil.which("uv") else WARN, shutil.which("uv") or "not on PATH (needed to set up)"))
    try:
        import mlx.core as mx  # noqa: F401  (the import is the check)
        metal = bool(getattr(mx, "metal", None) and mx.metal.is_available())
        c.append(Check("MLX", OK if metal else (FAIL if need_mlx else WARN),
                       f"mlx {_version('mlx')}, mlx-lm {_version('mlx-lm')}, Metal {'yes' if metal else 'no'}",
                       need_mlx))
    except Exception as err:
        c.append(Check("MLX", FAIL if need_mlx else WARN, f"not importable ({type(err).__name__})", need_mlx))
    tk, tf = _version("tokenizers"), _version("transformers")
    c.append(Check("tokenizers / transformers", OK if tk and tf else WARN,
                   f"tokenizers {tk or 'missing'}, transformers {tf or 'missing'} (servers: token counts, raw mode)"))
    from .modelstore import DEFAULT, find
    have = find(DEFAULT)
    others = [f"{n} ({g:.1f} GB)" for n, g in _models(models_dir) if not (have and (models_dir / n).resolve() == have)]
    if have:
        size = sum(f.stat().st_size for f in have.rglob("*") if f.is_file()) / 1e9
        detail = f"{DEFAULT.name} at {have} ({size:.1f} GB)" + (f"; also {', '.join(others)}" if others else "")
    else:
        detail = (f"{DEFAULT.name} not found (looked in {models_dir} and the Hugging Face cache): run `keel download`"
                  + (f"; found {', '.join(others)}" if others else ""))
    c.append(Check("model", OK if have else (FAIL if need_mlx else WARN), detail, need_mlx))
    bf = _version("bfcl-eval")
    c.append(Check("bfcl-eval", OK if bf else INFO, bf or      # optional: only the BFCL benchmark needs it
                   "not installed; only needed for the BFCL benchmark (keel-core install-bfcl)"))
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=state_dir, prefix=".doctor-", delete=True):
            pass
        c.append(Check("state folder", OK, str(state_dir), True))
    except OSError as err:
        c.append(Check("state folder", FAIL, f"{state_dir}: {err}", True))
    try:
        same = os.stat(tempfile.gettempdir()).st_dev == os.stat(state_dir).st_dev
        c.append(Check("temp folder volume", OK if same else INFO,
                       "same volume as the state folder" if same else
                       "different volume from the state folder: session scratch is copied, not moved"))
    except OSError:
        c.append(Check("temp folder volume", SKIP, ""))
    return c


# ---- in-process MLX -------------------------------------------------------------------------------------------------

def _probe_messages():
    from .agent.loop import TOOLSETS
    return [{"role": "system", "content": TOOLSETS["files"][0]}, {"role": "user", "content": PROBE}]


def mlx_check(spec: ModelSpec) -> Check:
    """Load the model and ask for one tool call through the real scheduler (greedy)."""
    from .agent.loop import TOOLSETS, prompt_ids
    from .backends.mlx import MLXBackend
    from .core.parse import parse_calls
    from .model import load_model
    t = time.perf_counter()
    try:
        model, tok = load_model(spec)
        be = MLXBackend(model, tok, SamplingSpec(temperature=0.0, presence_penalty=0.0, max_tokens=256))
        with contextlib.redirect_stdout(sys.stderr):          # the scheduler's progress line stays out of the table
            text, _, _ = be.complete(prompt_ids(tok, spec, _probe_messages(), "files"))
    except Exception as err:
        return Check("in-process model", FAIL, f"{type(err).__name__}: {err}", True)
    calls = parse_calls(text, TOOLSETS["files"][2])
    names = [next(iter(c)) for c in calls]
    return Check("in-process model", OK if calls else FAIL,
                 f"tool calls {names} in {time.perf_counter() - t:.1f} s (incl. load)" if calls else
                 f"no tool call in the reply: {text[:120]!r}", True)


# ---- a server ------------------------------------------------------------------------------------------------------

def _reachable(url: str, flavour: str, timeout: float) -> tuple[bool, str]:
    path = {"ollama": "/api/tags", "llama": "/health"}.get(flavour, "/v1/models")
    try:
        with urllib.request.urlopen(url + path, timeout=timeout) as r:
            body = r.read(20000).decode(errors="replace")
        try:                                                   # name the served models (folder names only)
            o = json.loads(body)
            names = [Path(str(m.get("id") or m.get("name") or "")).name for m in (o.get("data") or o.get("models") or [])]
            return True, "serving " + (", ".join(n for n in names if n) or "(no model list)")
        except (ValueError, AttributeError):
            return True, "answers"
    except urllib.error.HTTPError as err:
        return err.code < 500, f"HTTP {err.code}"
    except (urllib.error.URLError, OSError) as err:
        return False, str(getattr(err, "reason", err))


class ServerProbe:
    """Runs the conformance requests. `make(mode, **kw)` builds a ServerBackend for one probe."""

    def __init__(self, url: str, flavour: str, server_model: str, tok, spec: ModelSpec, timeout: float):
        self.url, self.flavour, self.server_model, self.tok, self.spec = url, flavour, server_model, tok, spec
        self.timeout = timeout
        self.tid = None
        if tok is not None:
            tid = tok.convert_tokens_to_ids("<tool_call>")
            self.tid = tid if isinstance(tid, int) and tid != getattr(tok, "unk_token_id", None) else None
        from .agent.loop import TOOLSETS
        self.schemas, self.funcs = TOOLSETS["files"][1], TOOLSETS["files"][2]
        self.requests = 0

    def make(self, mode: str, **kw):
        from .backends.server import ServerBackend
        greedy = SamplingSpec(temperature=0.0, presence_penalty=0.0, max_tokens=256)
        return ServerBackend(url=self.url, flavour=self.flavour, mode=mode, sampling=greedy, model=self.server_model,
                             timeout=self.timeout, **kw)

    def ask(self, be, prompt, forbid: bool = False):
        self.requests += 1
        return be.request(prompt, forbid_tools=forbid)

    def calls(self, text: str) -> list[str]:
        from .core.parse import parse_calls
        return [next(iter(c)) for c in parse_calls(text, self.funcs)]


def _try(checks: list[Check], name: str, fn, required: bool = False):
    from .backends.server import BackendError
    try:
        return fn()
    except BackendError as err:
        checks.append(Check(name, FAIL if required else WARN, f"server error: {err}"[:200], required))
    except Exception as err:                                   # a probe bug must not hide the other results
        checks.append(Check(name, WARN, f"{type(err).__name__}: {err}"[:200]))
    return None


def _ban_checks(p: ServerProbe, mode: str, rungs: tuple[str, ...], prompt, base_kw: dict, checks: list[Check],
                label: str) -> list[str]:
    honoured = []
    for rung in rungs:
        if rung == "logit_bias" and p.tid is None:
            checks.append(Check(f"{label}: ban logit_bias", SKIP, "no tokenizer, so no <tool_call> token id"))
            continue
        extra = {"tool_call_id": p.tid} if rung == "logit_bias" else {}
        be = p.make(mode, ban_rungs=(rung,), **base_kw, **extra)
        r = _try(checks, f"{label}: ban {rung}", lambda be=be: p.ask(be, prompt, forbid=True))
        if r is None:
            continue
        if r.ban_cut:
            checks.append(Check(f"{label}: ban {rung}", WARN, "ignored: the reply still called a tool"))
        else:
            honoured.append(rung)
            checks.append(Check(f"{label}: ban {rung}", OK, f"honoured (finish {r.finish}, {r.output_tokens} tokens)"))
    return honoured


def probe_raw(p: ServerProbe, checks: list[Check]) -> dict:
    from .agent.loop import prompt_text
    from .backends.server import DEFAULT_RUNGS
    if p.tok is None:
        checks.append(Check("raw: tool calls", SKIP, "no model folder: raw mode needs the tokenizer + chat template"))
        return {"usable": False, "reason": "no tokenizer"}
    prompt = prompt_text(p.tok, p.spec, _probe_messages(), "files")
    out: dict = {"usable": False}
    r = _try(checks, "raw: call text kept", lambda: p.ask(p.make("raw", token_ids=False), prompt))
    kw: dict = {"token_ids": False}
    if r is not None:
        names = p.calls(r.text)
        out["call_text_kept"] = bool(names)
        checks.append(Check("raw: call text kept", OK if names else WARN,
                            f"calls {names}" if names else "the server removes tool-call text from raw replies"))
        if names:
            out["usable"], out["tool_calls"] = True, "text"
    if not out["usable"] and p.flavour == "openai":
        be = p.make("raw", decode=lambda ids: p.tok.decode(ids), token_ids=True)
        r = _try(checks, "raw: calls via token ids", lambda: p.ask(be, prompt))
        if r is not None:
            names = p.calls(r.text)
            checks.append(Check("raw: calls via token ids", OK if names else WARN,
                                f"calls {names} (logprobs ids decoded; replies can't stream)" if names else
                                "no calls in the decoded ids"))
            if names:
                out["usable"], out["tool_calls"] = True, "token_ids"
                kw = {"decode": lambda ids: p.tok.decode(ids), "token_ids": True}
    if not out["usable"]:
        return out
    out["ban_rungs"] = _ban_checks(p, "raw", DEFAULT_RUNGS[("raw", p.flavour)], prompt, kw, checks, "raw")
    out.update(_repeat_and_cache(p, p.make("raw", **kw), prompt, checks, "raw"))
    return out


def _repeat_and_cache(p: ServerProbe, be, prompt, checks: list[Check], label: str) -> dict:
    a = _try(checks, f"{label}: greedy repeat", lambda: p.ask(be, prompt))
    b = _try(checks, f"{label}: greedy repeat", lambda: p.ask(be, prompt))
    if a is None or b is None:
        return {}
    same = a.text == b.text
    checks.append(Check(f"{label}: greedy repeat", OK if same else WARN,
                        "same reply twice" if same else "different replies at temperature 0"))
    cached = b.cached_tokens
    checks.append(Check(f"{label}: prompt cache", OK if cached > 0 else WARN,
                        f"{cached} / {b.prompt_tokens} prompt tokens reused on a repeat" if cached > 0 else
                        "no cached tokens reported (every request re-reads the whole prompt, or the server doesn't say)"))
    return {"greedy_repeatable": same, "prefix_cache": cached > 0}


def probe_chat(p: ServerProbe, checks: list[Check]) -> dict:
    from .backends.server import DEFAULT_RUNGS
    prompt = {"messages": _probe_messages(), "tools": p.schemas, "funcs": p.funcs}
    out: dict = {"usable": False}
    be = p.make("chat")
    r = _try(checks, "chat: tool calls parsed", lambda: p.ask(be, prompt))
    if r is None:
        return out
    names = p.calls(r.text)
    checks.append(Check("chat: tool calls parsed", OK if names else WARN,
                        f"calls {names}" + (f", {r.bad_args} with bad arguments" if r.bad_args else "") if names else
                        f"no tool calls in the reply: {r.text[:80]!r}"))
    if not names:
        return out
    out["usable"] = True
    history = prompt["messages"] + [{"role": "assistant", "content": r.text}] + [
        {"role": "tool", "content": "ok" if n != "list_files" else "notes.txt  11 bytes"} for n in names]
    r2 = _try(checks, "chat: multi-call history", lambda: p.ask(be, {**prompt, "messages": history}))
    if r2 is not None:
        checks.append(Check("chat: multi-call history", OK,
                            f"{len(names)} call(s) + results sent back; reply: {(r2.text.split('</think>')[-1].strip() or '(calls)')[:60]!r}"))
        out["history_round_trip"] = True
    out["ban_rungs"] = _ban_checks(p, "chat", DEFAULT_RUNGS[("chat", p.flavour)], prompt, {}, checks, "chat")
    out.update(_repeat_and_cache(p, be, prompt, checks, "chat"))
    return out


def server_checks(a: argparse.Namespace, spec: ModelSpec) -> tuple[list[Check], dict]:
    from .backends.server import DEFAULT_URLS
    flavour, default_url = SHORTCUTS.get(a.backend, (a.flavour or "openai", None))
    url = (a.url or default_url or DEFAULT_URLS[flavour]).rstrip("/")
    checks: list[Check] = []
    ok, detail = _reachable(url, flavour, min(a.timeout, 10))
    checks.append(Check("server reachable", OK if ok else FAIL, f"{url}: " + (detail[:100] if ok else detail), True))
    rec: dict = {"url": url, "flavour": flavour, "server_model": a.server_model, "probed": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not ok:
        return checks, rec
    if flavour == "ollama" and not a.server_model:
        checks.append(Check("server model", FAIL, "Ollama needs --server-model (a name from `ollama list`)", True))
        return checks, rec
    model_dir = a.model_dir or spec.local_dir      # as given, for the record
    model_path = Path(a.model_dir) if a.model_dir else spec.path   # spec default: keel.paths.models_dir(), not cwd
    tok = load_tokenizer(model_path) if (model_path / "tokenizer_config.json").is_file() else None
    rec["model_dir"] = str(model_dir) if tok is not None else None
    p = ServerProbe(url, flavour, a.server_model, tok, spec, a.timeout)
    modes = {"raw": probe_raw(p, checks), "chat": probe_chat(p, checks)}
    usable = [m for m, v in modes.items() if v.get("usable")]
    checks.append(Check("tool calls work", OK if usable else FAIL,
                        f"in {' and '.join(usable)} mode" if usable else "in neither raw nor chat mode", True))
    rec.update({"modes": modes, "requests": p.requests,
                "recommended": {"mode": "raw" if "raw" in usable else ("chat" if usable else None),
                                "why": "raw: keel renders and parses, so results compare with in-process MLX"}})
    return checks, rec


# ---- the record later runs read ------------------------------------------------------------------------------------

def record_path(state_dir: Path, url: str, flavour: str) -> Path:
    host = url.split("://", 1)[-1].replace(":", "_").replace("/", "_").strip("_")
    return Path(state_dir) / "servers" / f"{host}_{flavour}.json"


def measured_rungs(state_dir: Path, url: str, flavour: str, mode: str) -> tuple[tuple[str, ...], str] | None:
    """The rungs `keel doctor` found honoured for this server and mode, and when; None if never probed."""
    f = record_path(state_dir, url, flavour)
    try:
        rec = json.loads(f.read_text())
        m = rec["modes"][mode]
        if not m.get("usable") or "ban_rungs" not in m:
            return None
        return tuple(m["ban_rungs"]), f"keel doctor {rec.get('probed', '?')}"
    except (OSError, KeyError, ValueError, TypeError):
        return None


# ---- command ------------------------------------------------------------------------------------------------------

def add_arguments(ap: argparse.ArgumentParser) -> None:
    add_backend_arguments(ap)
    ap.set_defaults(backend=None)
    ap.add_argument("--state-dir", default=None, help="keel state folder (default ~/Library/Application Support/keel)")
    ap.add_argument("--models-dir", default=None, help="model folders to list (default: keel.paths.models_dir())")
    ap.add_argument("--out", default=None, help="write the server record here (default <state>/servers/...)")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per server request")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON instead of the table")


def _table(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks)
    mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip", INFO: "info"}
    return "\n".join(f"  {mark[c.status]}  {c.name:<{w}}  {c.detail}" for c in checks)


def run(a: argparse.Namespace) -> None:
    from .agent.sandbox import default_state_dir
    state = Path(a.state_dir) if a.state_dir else default_state_dir()
    b = QWEN35_4B_8BIT
    spec = ModelSpec(b.name, b.hf_repo, b.revision, b.local_dir, {"enable_thinking": False})
    server = a.backend not in (None, "mlx")
    from . import paths
    mac = mac_checks(state, Path(a.models_dir) if a.models_dir else paths.models_dir(), need_mlx=a.backend == "mlx")
    print("keel doctor: this Mac\n" + _table(mac), flush=True)
    checks = list(mac)
    record: dict = {"mac": [asdict(c) for c in mac]}
    if a.backend == "mlx":
        c = mlx_check(spec)
        checks.append(c)
        print("\nin-process MLX\n" + _table([c]), flush=True)
    if server:
        sc, rec = server_checks(a, spec)
        checks += sc
        record["server"] = rec
        print(f"\nserver {rec['url']} ({rec['flavour']})\n" + _table(sc), flush=True)
        if "modes" in rec:
            out = Path(a.out) if a.out else record_path(state, rec["url"], rec["flavour"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({**rec, "checks": [asdict(c) for c in sc]}, indent=1))
            rc = rec["recommended"]
            rungs = {m: v.get("ban_rungs") for m, v in rec["modes"].items() if v.get("usable")}
            print(f"\n  recommended: --mode {rc['mode']}; loop-breaker rungs honoured: {rungs}\n"
                  f"  wrote {out} ({rec['requests']} requests); later --backend runs on {rec['url']} use these rungs")
    failed = [c for c in checks if c.required and c.status == FAIL]
    if a.json:
        print(json.dumps({**record, "failed": [c.name for c in failed]}, indent=1))
    print(f"\n{'FAILED: ' + ', '.join(c.name for c in failed) if failed else 'all required checks passed'}")
    raise SystemExit(1 if failed else 0)
