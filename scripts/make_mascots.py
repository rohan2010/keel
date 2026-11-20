"""Draw keel's mascots as SVGs from the pixel grids in crates/keel-tui/src/logo.rs (the one source of truth).

Writes docs-keel/images/mascots/<name>.svg (each mascot alone, transparent, for marketing), mascots/all.svg (the
whole set on one sheet) and docs-keel/images/mascot.svg (the README banner: the main mascot beside the name).

    uv run python scripts/make_mascots.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "crates/keel-tui/src/logo.rs"
OUT = ROOT / "docs-keel/images"

# the 256-colour palette entries logo.rs uses, as hex (k = the screen, filled in pictures)
HEX = {"#": "#5fd7d7", "h": "#afffff", "d": "#008787", "a": "#ffd75f", "k": "#262626", "w": "#ffffff",
       "S": "#ffffff", "p": "#ff87af", "g": "#5fff87", "m": "#ff5fd7", "v": "#af5fff", "r": "#ffd787",
       "R": "#ff5f5f", "l": "#bcbcbc"}


def mascots() -> list[tuple[str, str, list[str]]]:
    """(name, title, pixel rows) for every mascot, main first, in logo.rs order."""
    src = LOGO.read_text()
    found = re.findall(r'name: "([^"]+)",\s*title: "([^"]+)",\s*pixels: &\[(.*?)\]', src, re.S)
    out = [(n, t, re.findall(r'"([^"]*)"', body)) for n, t, body in found]
    if not out or out[0][0] != "screen":
        raise SystemExit(f"could not read the mascots from {LOGO}")
    return out


def rects(rows: list[str], px: int, x0: int = 0, y0: int = 0) -> list[str]:
    """One <rect> per run of same-coloured pixels in a row."""
    out = []
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            c = row[x]
            if c == ".":
                x += 1
                continue
            end = x
            while end + 1 < len(row) and row[end + 1] == c:
                end += 1
            out.append(f'<rect x="{x0 + x * px}" y="{y0 + y * px}" width="{(end - x + 1) * px}" height="{px}" '
                       f'fill="{HEX[c]}"/>')
            x = end + 1
    return out


def svg(width: int, height: int, title: str, body: list[str], style: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-labelledby="t">\n  <title id="t">{title}</title>\n{style}'
            f'  <g shape-rendering="crispEdges">\n    ' + "\n    ".join(body) + "\n  </g>\n</svg>\n")


TEXT_STYLE = """  <style>
    .ink { fill: #0b0b0b; } .sub { fill: #52514e; }
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    @media (prefers-color-scheme: dark) { .ink { fill: #f0efec; } .sub { fill: #c3c2b7; } }
  </style>
"""


def main() -> None:
    ms = mascots()
    (OUT / "mascots").mkdir(parents=True, exist_ok=True)
    px = 16
    for name, title, rows in ms:
        w, h = len(rows[0]) * px, len(rows) * px
        (OUT / "mascots" / f"{name}.svg").write_text(svg(w, h, f"keel mascot: {title}", rects(rows, px)))

    # the sheet: every mascot on a grid, its title underneath
    px, cell_w, cell_h, cols = 8, 190, 150, 5
    body = []
    for i, (_name, title, rows) in enumerate(ms):
        cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
        x0 = cx + (cell_w - len(rows[0]) * px) // 2
        body += rects(rows, px, x0, cy + 16)
        body.append(f'<text class="sub" x="{cx + cell_w // 2}" y="{cy + 16 + len(rows) * px + 22}" font-size="14" '
                    f'text-anchor="middle">{title}</text>')
    n_rows = -(-len(ms) // cols)
    (OUT / "mascots" / "all.svg").write_text(
        svg(cols * cell_w, n_rows * cell_h, "keel's mascots", body, TEXT_STYLE))

    # the README banner: the main mascot, then the name and the tagline
    rows = ms[0][2]
    px, x0, y0 = 7, 14, 6
    tx = x0 + len(rows[0]) * px + 21
    body = rects(rows, px, x0, y0) + [
        f'<text class="ink" x="{tx}" y="50" font-size="34" font-weight="700">keel</text>',
        f'<text class="sub" x="{tx + 2}" y="76" font-size="15">a steady harness for small local models</text>',
    ]
    (OUT / "mascot.svg").write_text(
        svg(490, 96, "keel, a steady harness for small local models (mascot: the screen bot)", body, TEXT_STYLE))
    print(f"wrote {len(ms)} mascots, mascots/all.svg and mascot.svg to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
