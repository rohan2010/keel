"""Off-the-shelf agent baselines: goose and opencode on keel's own tasks, same model, keel's scorers.

Each episode: the task folder is materialized into a fresh eval sandbox (as for keel's own arms), the agent runs its
headless mode in that folder against `mlx_lm.server` serving keel's Qwen3.5-4B-8bit weights (keel's sampling defaults
set on the server: thinking off, T 0.7, top-p 0.8, top-k 20, 8,192 tokens), and keel's scorer grades the folder and
the final answer. Run folders use keel's agent format, so `keel compare` can put a baseline next to a keel arm.

Safety: these agents have no sandbox or delete guard of their own, so they run under macOS `sandbox-exec` with keel's
profile: writes only in the task copy, its temp folder and a scratch home; reads of /Users only inside the episode
folder (plus the agent's own install folder); network only to the local model server. Mail programs are replaced
by logging stand-ins that refuse to send (F1's scorer reads the attempts as its command log). F3 (safety) is scored
with no confirmations: these agents never ask before deleting.

Configuration, stated so a reader can judge fairness: goose as shipped minus `summon` (sub-agents; 40-113 min per
task on tau2-bench in the Mingbird paper) and `apps` (GUI windows); opencode as shipped with web tools and
sub-agents denied. Both get a per-task wall-clock cap.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import paths
from ..agent.sandbox import PATH, Sandbox, _profile
from . import tasks as T

SERVER = "http://127.0.0.1:8080"
MODEL_ID = "models/Qwen3.5-4B-8bit"
SEND_SHIMS = ("mail", "mailx", "sendmail", "mutt", "msmtp", "ssmtp", "postdrop")
SHIM = """#!/bin/bash
echo "$(basename "$0") $*" >> "{log}"
echo "$(basename "$0"): cannot send: there is no mail setup on this computer (nothing was sent)" >&2
exit 1
"""
OPENCODE_BIN = paths.baselines_dir() / "node_modules" / ".bin" / "opencode"


@dataclass
class AgentSpec:
    name: str
    version: str
    config: dict                                    # recorded in the run header


def _profile_with_server(root: Path, writable: list[Path], extra_read: tuple[str, ...], port: int) -> str:
    p = _profile(str(root), [str(x) for x in writable], extra_read)
    # localhost only: the model server, and an agent's own local server (opencode starts one); no internet
    return p.replace("(deny network*)", '(deny network*)\n'
                                        '(allow network-bind (local ip "localhost:*"))\n'
                                        '(allow network-inbound (local ip "localhost:*"))\n'
                                        '(allow network-outbound (remote ip "localhost:*"))')


def _version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1]
    except Exception:
        return "?"


# ---- goose ---------------------------------------------------------------------------------------------------------

def goose_setup(home: Path, max_turns: int) -> tuple[list[str], dict]:
    cfg_dir = home / ".config" / "goose"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    defaults = subprocess.run(["goose", "info", "-v"], capture_output=True, text=True,
                              env={**os.environ, "HOME": str(home), "GOOSE_DISABLE_KEYRING": "1"}).stdout
    ext = yaml.safe_load(defaults.split("goose Configuration:", 1)[1].replace("\n  ", "\n"))["extensions"]
    for off in ("summon", "apps"):
        if off in ext:
            ext[off]["enabled"] = False
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump({"extensions": ext}))
    env = {"GOOSE_PROVIDER": "openai", "GOOSE_MODEL": MODEL_ID, "OPENAI_HOST": SERVER, "OPENAI_API_KEY": "local",
           "GOOSE_MODE": "auto", "GOOSE_TEMPERATURE": "0.7", "GOOSE_CONTEXT_LIMIT": "32000",
           "GOOSE_DISABLE_KEYRING": "1", "GOOSE_DISABLE_SESSION_NAMING": "true", "GOOSE_TELEMETRY_ENABLED": "false"}
    cmd = ["goose", "run", "--no-session", "-q", "--output-format", "json", "--max-turns", str(max_turns), "-t"]
    return cmd, env


# ---- opencode ------------------------------------------------------------------------------------------------------

def opencode_setup(home: Path, max_turns: int) -> tuple[list[str], dict]:
    cfg_dir = home / ".config" / "opencode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"$schema": "https://opencode.ai/config.json", "model": "local/qwen",
           "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "Local MLX",
                                  "options": {"baseURL": f"{SERVER}/v1", "apiKey": "local"},
                                  "models": {"qwen": {"id": MODEL_ID, "name": "Qwen3.5-4B-8bit",
                                                      "limit": {"context": 32000, "output": 8192}}}}},
           "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
           "share": "disabled", "autoupdate": False}
    (cfg_dir / "opencode.json").write_text(json.dumps(cfg, indent=1))
    env = {"OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_LSP_DOWNLOAD": "true"}
    return [str(OPENCODE_BIN), "run", "--pure", "--auto", "--format", "json", "--model", "local/qwen"], env


def parse_goose(out: str) -> dict:
    """goose `--output-format json`: the final assistant text, the shell commands it ran, steps, tokens."""
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return {"answer": out.strip(), "commands": [], "steps": None, "tokens": {}, "parsed": False}
    msgs = d.get("messages", [])
    texts = [c.get("text", "") for m in msgs if m.get("role") == "assistant" for c in m.get("content", [])
             if c.get("type") == "text"]
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    answer = "".join(c.get("text", "") for c in (last or {}).get("content", []) if c.get("type") == "text").strip()
    cmds = []
    for m in msgs:
        for c in m.get("content", []):
            if c.get("type") == "toolRequest":
                v = (c.get("toolCall") or {}).get("value") or {}
                a = v.get("arguments") or {}
                cmds.append(a.get("command") if isinstance(a, dict) and "command" in a
                            else f"{v.get('name')} {json.dumps(a)[:300]}")
    meta = d.get("metadata") or {}
    return {"answer": answer or (texts[-1].strip() if texts else ""), "commands": cmds,
            "steps": sum(m.get("role") == "assistant" for m in msgs),
            "tokens": {"input": meta.get("input_tokens"), "output": meta.get("output_tokens")}, "parsed": True}


def parse_opencode(out: str) -> dict:
    """opencode `run --format json`: one JSON event per line. Answer = the text of the last message that has text;
    commands = bash tool inputs (other tools by name); steps = step_finish events; tokens summed over steps."""
    ev = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not ev:
        return parse_plain(out)
    texts = [(e["part"].get("messageID"), e["part"].get("text", "")) for e in ev if e.get("type") == "text"]
    last_msg = texts[-1][0] if texts else None
    answer = "".join(t for m, t in texts if m == last_msg).strip()
    cmds = []
    for e in ev:
        if e.get("type") == "tool_use":
            part = e.get("part", {})
            inp = (part.get("state") or {}).get("input") or {}
            cmds.append(inp["command"] if part.get("tool") == "bash" and "command" in inp
                        else f"{part.get('tool')} {json.dumps(inp)[:300]}")
    fins = [e["part"].get("tokens", {}) for e in ev if e.get("type") == "step_finish"]
    return {"answer": answer, "commands": cmds, "steps": len(fins),
            "tokens": {"input": sum((t.get("input") or 0) + ((t.get("cache") or {}).get("read") or 0) for t in fins),
                       "output": sum(t.get("output") or 0 for t in fins)}, "parsed": True}


def parse_plain(out: str) -> dict:
    return {"answer": out.strip(), "commands": [], "steps": None, "tokens": {}, "parsed": False}


SETUPS = {"goose": goose_setup, "opencode": opencode_setup}
PARSERS = {"goose": parse_goose, "opencode": parse_opencode}


def agent_spec(name: str, max_turns: int, timeout_s: int) -> AgentSpec:
    if name == "goose":
        return AgentSpec("goose", _version(["goose", "--version"]),
                         {"extensions_off": ["summon", "apps"], "mode": "auto", "max_turns": max_turns,
                          "timeout_s": timeout_s})
    return AgentSpec("opencode", _version([str(OPENCODE_BIN), "--version"]),
                     {"permission": "edit/bash allow, webfetch deny", "timeout_s": timeout_s})


# ---- one episode ---------------------------------------------------------------------------------------------------

def run_episode(agent: str, item: T.Item, out_dir: Path, *, max_turns: int = 30, timeout_s: int = 900,
                port: int = 8080) -> dict:
    tmpl = T.materialize(item, out_dir / "templates" / item.id)
    sb = Sandbox(out_dir / "sandboxes" / item.id).create(tmpl)
    home = sb.root / ".home"
    home.mkdir()
    send_log = home / "sends.log"
    for prog in SEND_SHIMS:                               # logging stand-ins, first on PATH
        p = sb.bin / prog
        p.write_text(SHIM.format(log=send_log))
        p.chmod(0o755)
    cmd, env = SETUPS[agent](home, max_turns)
    extra = (str(sb.work),) + ((str(OPENCODE_BIN.resolve().parent.parent),) if agent == "opencode" else ())
    profile = _profile_with_server(sb.root, [sb.work, sb.tmp, home], extra, port)
    full_env = {"PATH": f"{sb.bin}:{PATH}", "HOME": str(home), "TMPDIR": str(sb.tmp) + "/", "LANG": "en_US.UTF-8",
                "TERM": "dumb", "NO_COLOR": "1", **env}
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(["sandbox-exec", "-p", profile] + cmd + [item.request], cwd=sb.work, env=full_env,
                              capture_output=True, text=True, timeout=timeout_s)
        answer, err, code = proc.stdout.strip(), proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        answer = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")).strip()
        err, code = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")), -1
    secs = time.time() - t0
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "logs" / f"{item.id}.stderr.txt").write_text(err[-200_000:])
    (out_dir / "logs" / f"{item.id}.stdout.txt").write_text(answer[-2_000_000:])     # raw agent output, for audit
    sends = send_log.read_text().splitlines() if send_log.exists() else []
    parsed = PARSERS[agent](answer)
    answer = parsed["answer"]
    commands = parsed["commands"] + sends                 # F1 reads send attempts from the command log
    s = T.score(item, sb.work, answer, commands, [])
    shutil.rmtree(home / ".local", ignore_errors=True)    # agent caches: not part of the result
    return {"id": item.id, "task": item.task, "kind": item.kind, "seed": item.seed, "request": item.request,
            "toolset": agent, "correct": bool(s["correct"]), "score": s.get("score"), "labels": s.get("labels", []),
            "stop": "timeout" if timed_out else ("answer" if code == 0 else f"exit {code}"),
            "steps": parsed["steps"], "answer": answer, "commands": commands,
            "prompt_tokens": parsed["tokens"].get("input"), "processed_tokens": parsed["tokens"].get("input"),
            "output_tokens": parsed["tokens"].get("output"), "output_parsed": parsed["parsed"], "seconds": round(secs, 1), "confirmations": [], "detail": {}}
