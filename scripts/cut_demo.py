"""Turn a raw demo cast (scripts/record_demo.py) into a short captioned GIF + MP4 a stranger can follow.

    uv run python scripts/cut_demo.py <take.cast> <out_stem> [--speed 6]

What it does, all from the real recording (no model output is changed or invented):
- a title card: what keel is, the model, the task, the folder before (from <take>.meta.json);
- the session sped up (`--speed`, shown on screen), with short holds at the moments that matter;
- a caption strip under the terminal saying what is happening, triggered by what the TUI actually shows
  (a move, the delete question, the answer), so a re-take keeps its captions in sync;
- an end card: the folder after, and the command to try it.
Needs `agg` and `ffmpeg` (brew install agg ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]")
W, H, CAP_H = 100, 30, 3
DIM, BOLD, YEL, GRN, RED, CYAN, RST = "\x1b[2m", "\x1b[1m", "\x1b[33m", "\x1b[32m", "\x1b[31m", "\x1b[36m", "\x1b[0m"
FOLLOW = "ok, you can delete"
# the recording folder, whole or as the TUI banner shortens it ("…<tail>/keel-demo-xxxx/Downloads")
TEMP_DIR = re.compile(r"(?:(?:/private)?/var/folders/|…)[^\s\x1b]*?/keel-demo-[^/\s\x1b]+/Downloads")


def load(path):
    lines = Path(path).read_text().splitlines()
    return json.loads(lines[0]), [json.loads(l) for l in lines[1:]]


def write_cast(path, w, h, events):
    with open(path, "w") as f:
        f.write(json.dumps({"version": 2, "width": w, "height": h}) + "\n")
        for e in events:
            f.write(json.dumps([round(e[0], 3), "o", e[1]]) + "\n")


def captions_for(chunk, state):
    """Return (caption, hold seconds) when this chunk of screen output crosses a milestone, else None.

    Keyed on chunks, not on the whole screen text: the TUI redraws, so a phrase can be on screen many times.
    The status line says "waiting for you" exactly while a delete question is open."""
    status = re.search(r"· (waiting for you|working|thinking|starting|ready)", chunk)
    opened = bool(status) and status.group(1) == "waiting for you" and state.get("status") != "waiting for you"
    if status:
        state["status"] = status.group(1)
    if FOLLOW in chunk:
        state["followed"] = True
    if opened:
        state["asks"] = state.get("asks", 0) + 1
        if state.get("followed"):
            return (f"{YEL}It asks again before deleting.{RST} This time we press {BOLD}y{RST}.", 3.0)
        if state["asks"] == 1:
            return (f"{YEL}It wants to delete files. keel stops and asks{RST}: every delete, no off switch.", 3.5)
        return (f"The model tries again. {YEL}The guard still asks.{RST} We say {BOLD}n{RST} again.", 3.0)
    if not state.get("moved") and re.search(r"(?<!re)move \S", chunk):     # a move call, not "remove" in the prompt
        state["moved"] = True
        return (f"It looks at the files, makes folders, moves them: {BOLD}real tool calls{RST}.", 1.5)
    if "kept " in chunk and not state.get("kept"):
        state["kept"] = True
        return (f"We pressed {BOLD}n{RST}: {GRN}nothing deleted{RST}, and the model is told so.", 2.5)
    if "> ok" in chunk and not state.get("typed"):
        state["typed"] = True
        return ("Now we tell it to go ahead.", 0.5)
    if "Deleted " in chunk and not state.get("deleted"):
        state["deleted"] = True
        return (f"{GRN}Deleted, only after a yes.{RST}", 2.5)
    return None


def cut_session(events, speed):
    """Speed the session up and hold at milestones. Returns (session events, caption events), both retimed."""
    first = (f"You ask in plain English. {BOLD}Qwen3.5-4B runs on this laptop{RST}: no cloud, no API key.", 3.0)
    out, caps, state = [(0.0, "\x1b[?25l")], [(0.0, first[0])], {}
    t_new, t_prev = first[1], 0.0
    for t, _, data in events:
        t_new += min((t - t_prev) / speed, 1.0)        # and no screen sits still for more than 1 s
        data = data.replace("\x1b[?25h", "")           # hide the cursor block (it only shows in the GIF)
        data = TEMP_DIR.sub("~/Downloads\x1b[K", data)      # the recorder's temp folder: a local path, and long
        t_prev = t
        out.append((t_new, data))
        cap = captions_for(ANSI.sub(" ", data), state)
        if cap:
            caps.append((t_new, cap[0]))
            t_new += cap[1]
    return out, caps, t_new


def caption_events(caps, speed, end):
    tag = f" ▶ real run, {speed:g}× speed "
    rule = f"{DIM}{'─' * (W - len(tag) - 2)}{tag}──{RST}"
    ev = []
    for t, text in caps:
        line = f"\x1b[?25l\x1b[H\x1b[2J{rule}\r\n\r\n  {text}"
        ev.append((t, line))
    ev.append((end, ""))
    return ev


def card(lines):
    top = max(0, (H + CAP_H - len(lines)) // 2 - 1)                          # roughly centred, a bit high
    return "\x1b[?25l\x1b[H\x1b[2J" + "\r\n" * top + "\r\n".join(lines)


def title_card(meta, prompt):
    dup = {"IMG_2041 (1).jpg", "invoice_march copy.pdf"}
    files = [f"    {n}" + (f"   {YEL}← duplicate{RST}" if n in dup else "") for n in meta["before"]]
    task = textwrap.wrap(f"\"{prompt}\"", W - 14)
    return card([f"  {BOLD}{CYAN}keel{RST}", "",
                 "  A small AI that runs on your own Mac does real jobs in your folders,",
                 "  and it always asks before it deletes anything.", "",
                 f"  {DIM}model{RST}   Qwen3.5-4B (8-bit), running locally on a 16 GB Mac",
                 f"  {DIM}task{RST}    {task[0]}", *[f"          {x}" for x in task[1:]], "",
                 f"  {DIM}~/Downloads, before:{RST}", *files])


def end_card(meta):
    tree = meta["after"]
    left = [f"    {p}" + (f"   {DIM}(not sorted by the model){RST}" if "/" not in p else "") for p in tree]
    return card([f"  {BOLD}{CYAN}keel{RST}  {DIM}result{RST}", "",
                 f"  {DIM}~/Downloads, after:{RST}", *left, "",
                 f"  {GRN}✓{RST} sorted into folders   {GRN}✓{RST} duplicates deleted only after a yes", "",
                 f"  {DIM}try it:{RST}  keel run --dir ~/Downloads \"sort these files by type\""])


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        sys.exit(f"{cmd[0]} failed:\n{r.stderr[-1500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("out_stem")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--prompt", default="Sort these files into folders by type (Images, Documents, Music) "
                                        "and remove duplicate copies.")
    a = ap.parse_args()
    _, events = load(a.cast)
    meta = json.loads(Path(a.cast).with_suffix(".meta.json").read_text())
    session, caps, end = cut_session(events, a.speed)
    print("captions:", *[f"{t:5.1f}s  {ANSI.sub('', c)}" for t, c in caps], f"session {end:.1f}s", sep="\n  ")

    tmp = Path(tempfile.mkdtemp(prefix="keel-cut-"))
    print("work files:", tmp)
    write_cast(tmp / "session.cast", W, H, session + [(end, "")])
    write_cast(tmp / "caps.cast", W, CAP_H, caption_events(caps, a.speed, end))
    write_cast(tmp / "title.cast", W, H + CAP_H, [(0, title_card(meta, a.prompt)), (6.5, "")])
    write_cast(tmp / "end.cast", W, H + CAP_H, [(0, end_card(meta)), (6.0, "")])
    agg = ["agg", "--fps-cap", "15", "--font-size", "16", "--theme", "monokai", "--last-frame-duration", "0.1",
           "--idle-time-limit", "60"]
    for n in ("session", "caps"):
        run([*agg, str(tmp / f"{n}.cast"), str(tmp / f"{n}.gif")])
    for n, secs in (("title", "7"), ("end", "6")):            # agg ignores a final frame that changes nothing
        run([*agg[:-4], "--last-frame-duration", secs, str(tmp / f"{n}.cast"), str(tmp / f"{n}.gif")])
    stem = Path(a.out_stem)
    size = lambda n: [int(x) for x in subprocess.check_output(  # noqa: E731
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(tmp / f"{n}.gif")],
        text=True).split(",")]
    w, h = size("session")[0], size("session")[1] + size("caps")[1]
    pad = f"pad={w}:{h}:0:0:color=0x272822"                                    # monokai background
    graph = (f"[0]fps=15,setsar=1[s];[1]fps=15,setsar=1[c];[s][c]vstack=shortest=0[m];"
             f"[2]fps=15,setsar=1,{pad}[t];[3]fps=15,setsar=1,{pad}[e];[t][m][e]concat=n=3:v=1[v]")
    inputs = [x for n in ("session", "caps", "title", "end") for x in ("-i", str(tmp / f"{n}.gif"))]
    run(["ffmpeg", "-y", *inputs, "-filter_complex", graph + ";[v]split[a][b];[a]palettegen=max_colors=64[p];"
         "[b][p]paletteuse=dither=none", str(stem.with_suffix(".gif"))])
    run(["ffmpeg", "-y", *inputs, "-filter_complex", graph + ";[v]pad=ceil(iw/2)*2:ceil(ih/2)*2[o]", "-map", "[o]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(stem.with_suffix(".mp4"))])
    for ext in (".gif", ".mp4"):
        p = stem.with_suffix(ext)
        print(f"{p}  {p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
