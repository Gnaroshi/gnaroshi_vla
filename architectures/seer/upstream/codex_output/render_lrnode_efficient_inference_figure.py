#!/usr/bin/env python3
"""Render a high-resolution raster PNG for the LR-NODE inference figure.

The script uses plain PostScript drawing commands and Ghostscript only as a
rasterizer. No SVG/vector artifact is written to the repository.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path


OUT = Path("codex_output/figures/lrnode_efficient_seer_inference.png")
PAGE_W = 1152
PAGE_H = 648
RASTER_W = 3840
RASTER_H = 2160
DPI = 240

BLUE = (0.86, 0.92, 1.00)
BLUE_STROKE = (0.15, 0.39, 0.92)
BLUE_DARK = (0.10, 0.26, 0.62)
GREEN = (0.86, 0.97, 0.89)
GREEN_STROKE = (0.09, 0.64, 0.29)
GREEN_DARK = (0.08, 0.45, 0.20)
ORANGE = (1.00, 0.92, 0.82)
ORANGE_STROKE = (0.94, 0.45, 0.08)
ORANGE_DARK = (0.76, 0.25, 0.05)
GRAY = (0.94, 0.95, 0.97)
GRAY_STROKE = (0.60, 0.64, 0.70)
TEXT = (0.08, 0.11, 0.18)
MUTED = (0.31, 0.35, 0.42)
RED = (0.86, 0.15, 0.15)
WHITE = (1.0, 1.0, 1.0)


class PS:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, s: str) -> None:
        self.lines.append(s)

    def rgb(self, color: tuple[float, float, float]) -> str:
        return f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} setrgbcolor"

    def xy(self, x: float, y: float) -> tuple[float, float]:
        return x, PAGE_H - y

    def rect_path(self, x: float, y: float, w: float, h: float, r: float = 8) -> None:
        yb = PAGE_H - y - h
        self.add("newpath")
        self.add(f"{x + r:.2f} {yb:.2f} moveto")
        self.add(f"{x + w - r:.2f} {yb:.2f} lineto")
        self.add(f"{x + w - r:.2f} {yb + r:.2f} {r:.2f} 270 360 arc")
        self.add(f"{x + w:.2f} {yb + h - r:.2f} lineto")
        self.add(f"{x + w - r:.2f} {yb + h - r:.2f} {r:.2f} 0 90 arc")
        self.add(f"{x + r:.2f} {yb + h:.2f} lineto")
        self.add(f"{x + r:.2f} {yb + h - r:.2f} {r:.2f} 90 180 arc")
        self.add(f"{x:.2f} {yb + r:.2f} lineto")
        self.add(f"{x + r:.2f} {yb + r:.2f} {r:.2f} 180 270 arc")
        self.add("closepath")

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: tuple[float, float, float],
        stroke: tuple[float, float, float],
        lw: float = 1.6,
        r: float = 8,
        dash: bool = False,
    ) -> None:
        self.rect_path(x, y, w, h, r)
        self.add(f"gsave {self.rgb(fill)} fill grestore")
        self.add(f"{self.rgb(stroke)} {lw:.2f} setlinewidth")
        self.add("[6 4] 0 setdash" if dash else "[] 0 setdash")
        self.add("stroke")
        self.add("[] 0 setdash")

    def escape(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 10,
        color: tuple[float, float, float] = TEXT,
        font: str = "Helvetica",
        align: str = "left",
    ) -> None:
        tx, ty = self.xy(x, y)
        esc = self.escape(s)
        self.add(f"/{font} findfont {size:.2f} scalefont setfont")
        self.add(self.rgb(color))
        if align == "center":
            self.add(f"{tx:.2f} {ty:.2f} moveto ({esc}) dup stringwidth pop 2 div neg 0 rmoveto show")
        elif align == "right":
            self.add(f"{tx:.2f} {ty:.2f} moveto ({esc}) dup stringwidth pop neg 0 rmoveto show")
        else:
            self.add(f"{tx:.2f} {ty:.2f} moveto ({esc}) show")

    def multi(
        self,
        x: float,
        y: float,
        lines: list[str],
        size: float = 9.5,
        color: tuple[float, float, float] = TEXT,
        font: str = "Helvetica",
        leading: float | None = None,
        align: str = "left",
    ) -> None:
        if leading is None:
            leading = size * 1.25
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, size=size, color=color, font=font, align=align)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float],
        lw: float = 1.8,
        dashed: bool = False,
        arrow: bool = True,
    ) -> None:
        x1p, y1p = self.xy(x1, y1)
        x2p, y2p = self.xy(x2, y2)
        self.add(self.rgb(color))
        self.add(f"{lw:.2f} setlinewidth")
        self.add("[5 4] 0 setdash" if dashed else "[] 0 setdash")
        self.add(f"newpath {x1p:.2f} {y1p:.2f} moveto {x2p:.2f} {y2p:.2f} lineto stroke")
        self.add("[] 0 setdash")
        if arrow:
            self.arrowhead(x1, y1, x2, y2, color, size=6.5)

    def arrowhead(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: tuple[float, float, float],
        size: float = 6.5,
    ) -> None:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        tip = (x2, y2)
        base = (x2 - ux * size, y2 - uy * size)
        p1 = (base[0] + px * size * 0.45, base[1] + py * size * 0.45)
        p2 = (base[0] - px * size * 0.45, base[1] - py * size * 0.45)
        pts = [tip, p1, p2]
        pts_ps = [self.xy(a, b) for a, b in pts]
        self.add(self.rgb(color))
        self.add(
            "newpath "
            + f"{pts_ps[0][0]:.2f} {pts_ps[0][1]:.2f} moveto "
            + f"{pts_ps[1][0]:.2f} {pts_ps[1][1]:.2f} lineto "
            + f"{pts_ps[2][0]:.2f} {pts_ps[2][1]:.2f} lineto closepath fill"
        )

    def cross(self, cx: float, cy: float, size: float = 18) -> None:
        self.line(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, RED, lw=3.2, arrow=False)
        self.line(cx - size / 2, cy + size / 2, cx + size / 2, cy - size / 2, RED, lw=3.2, arrow=False)


def obs_box(ps: PS, x: float, y: float, w: float = 62, h: float = 68, compact: bool = False) -> None:
    ps.box(x, y, w, h, WHITE, GRAY_STROKE, lw=1.2, r=7)
    ps.text(x + w / 2, y + 13, "Observation", size=7.5 if compact else 8.0, font="Helvetica-Bold", align="center")
    chip_h = 13 if compact else 14
    labels = [("RGB", BLUE), ("q", GREEN), ("lang", ORANGE)]
    for i, (label, fill) in enumerate(labels):
        cy = y + 23 + i * (chip_h + 3)
        ps.box(x + 8, cy, w - 16, chip_h, fill, GRAY_STROKE, lw=0.7, r=4)
        ps.text(x + w / 2, cy + chip_h - 3.3, label, size=6.7, align="center", color=TEXT, font="Helvetica-Bold")


def full_block(ps: PS, x: float, y: float, w: float = 108, h: float = 70, gray: bool = False) -> None:
    fill = GRAY if gray else BLUE
    stroke = GRAY_STROKE if gray else BLUE_STROKE
    color = MUTED if gray else BLUE_DARK
    ps.box(x, y, w, h, fill, stroke, lw=1.8, r=8, dash=gray)
    ps.text(x + w / 2, y + 19, "Full Seer Forward", size=9.5, color=color, font="Helvetica-Bold", align="center")
    ps.text(x + w / 2, y + 37, "Vision Encoder +", size=7.8, color=color, align="center")
    ps.text(x + w / 2, y + 51, "Perceiver + Transformer", size=7.8, color=color, align="center")


def action_head(ps: PS, x: float, y: float, w: float = 70, h: float = 43) -> None:
    ps.box(x, y, w, h, BLUE, BLUE_STROKE, lw=1.5, r=7)
    ps.text(x + w / 2, y + 18, "Existing", size=8.0, color=BLUE_DARK, font="Helvetica-Bold", align="center")
    ps.text(x + w / 2, y + 32, "Action Head", size=8.0, color=BLUE_DARK, font="Helvetica-Bold", align="center")


def draw_legend(ps: PS) -> None:
    x, y = 742, 30
    ps.box(x, y, 378, 42, WHITE, GRAY_STROKE, lw=1.0, r=8)
    entries = [
        (BLUE, BLUE_STROKE, "Original Seer modules reused"),
        (GREEN, GREEN_STROKE, "LR-NODE cheap update"),
        (ORANGE, ORANGE_STROKE, "Cached action latent"),
        (GRAY, GRAY_STROKE, "Skipped full query"),
    ]
    for i, (fill, stroke, label) in enumerate(entries):
        ex = x + 14 + (i % 2) * 178
        ey = y + 12 + (i // 2) * 16
        ps.box(ex, ey, 10, 8, fill, stroke, lw=0.8, r=2, dash=(label == "Skipped full query"))
        ps.text(ex + 15, ey + 7.2, label, size=6.8, color=MUTED)


def draw_baseline_row(ps: PS) -> None:
    ps.box(28, 82, 1096, 224, WHITE, BLUE_STROKE, lw=1.4, r=13)
    ps.text(48, 106, "Baseline Seer: Full Policy Query Every Step", size=15, color=BLUE_DARK, font="Helvetica-Bold")
    starts = [46, 316, 586, 856]
    steps = ["t", "t+1", "t+2", "t+3"]
    actions = ["Action a_t", "Action a_t+1", "Action a_t+2", "Action a_t+3"]
    for x, step, action in zip(starts, steps, actions):
        ps.text(x + 125, 130, step, size=12, color=TEXT, font="Helvetica-Bold", align="center")
        obs_box(ps, x, 150)
        full_block(ps, x + 76, 150)
        action_head(ps, x + 198, 163)
        ps.box(x + 282, 169, 58, 31, WHITE, BLUE_STROKE, lw=1.2, r=15)
        ps.text(x + 311, 188.5, action, size=7.5, color=BLUE_DARK, font="Helvetica-Bold", align="center")
        ps.line(x + 62, 184, x + 76, 184, BLUE_STROKE, lw=1.8)
        ps.line(x + 184, 184, x + 198, 184, BLUE_STROKE, lw=1.8)
        ps.line(x + 268, 184, x + 282, 184, BLUE_STROKE, lw=1.8)
    ps.box(352, 262, 450, 26, (0.98, 0.99, 1.0), BLUE_STROKE, lw=1.0, r=12)
    ps.text(
        577,
        280,
        "Expensive full visual-transformer query at every control step",
        size=10,
        color=BLUE_DARK,
        font="Helvetica-Bold",
        align="center",
    )


def skip_column(ps: PS, x: float, step: str, z_label: str, action_label: str) -> None:
    ps.text(x + 118, 386, step, size=12, color=TEXT, font="Helvetica-Bold", align="center")
    obs_box(ps, x, 412, w=58, h=54, compact=True)
    full_block(ps, x + 74, 405, w=95, h=42, gray=True)
    ps.cross(x + 185, 425, size=17)
    ps.line(x + 58, 438, x + 74, 427, GRAY_STROKE, lw=1.5, dashed=True)
    ps.text(x + 122, 456, "full query skipped", size=6.7, color=MUTED, align="center")

    ps.box(x, 484, 66, 42, WHITE, GRAY_STROKE, lw=1.0, r=6)
    ps.text(x + 33, 500, "current obs", size=6.8, align="center", font="Helvetica-Bold")
    ps.text(x + 33, 513, "+ cached prev", size=6.8, align="center")
    ps.box(x + 78, 484, 80, 42, GREEN, GREEN_STROKE, lw=1.3, r=7)
    ps.text(x + 118, 499, "Fast Visual/", size=7.0, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.text(x + 118, 512, "Proprio Delta", size=7.0, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.box(x + 170, 494, 50, 22, WHITE, GREEN_STROKE, lw=1.0, r=11)
    ps.text(x + 195, 509, "u_delta", size=7.2, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.line(x + 66, 505, x + 78, 505, GREEN_STROKE, lw=1.5)
    ps.line(x + 158, 505, x + 170, 505, GREEN_STROKE, lw=1.5)

    ps.box(x + 78, 543, 88, 44, GREEN, GREEN_STROKE, lw=1.3, r=7)
    ps.text(x + 122, 559, "Controlled", size=7.4, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.text(x + 122, 573, "Latent NODE", size=7.4, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.line(x + 195, 516, x + 165, 543, GREEN_STROKE, lw=1.5)
    ps.box(x + 178, 548, 72, 34, ORANGE, ORANGE_STROKE, lw=1.2, r=6)
    ps.text(x + 214, 561.5, z_label, size=6.7, color=ORANGE_DARK, font="Helvetica-Bold", align="center")
    ps.text(x + 214, 574.5, "= z + gate*dt*dz", size=5.9, color=ORANGE_DARK, align="center")
    ps.line(x + 166, 565, x + 178, 565, ORANGE_STROKE, lw=1.4)

    action_head(ps, x + 96, 602, w=74, h=34)
    ps.box(x + 184, 604, 55, 29, WHITE, BLUE_STROKE, lw=1.0, r=14)
    ps.text(x + 211.5, 622.5, action_label, size=6.8, color=BLUE_DARK, font="Helvetica-Bold", align="center")
    ps.line(x + 214, 582, x + 155, 602, ORANGE_STROKE, lw=1.2)
    ps.line(x + 170, 619, x + 184, 619, BLUE_STROKE, lw=1.4)


def draw_ours_row(ps: PS) -> None:
    ps.box(28, 330, 1096, 292, WHITE, GREEN_STROKE, lw=1.4, r=13)
    ps.text(48, 354, "Ours: Full Refresh Every K Steps + LR-NODE Skip Updates", size=15, color=GREEN_DARK, font="Helvetica-Bold")

    # Full refresh at t.
    ps.text(134, 386, "t", size=12, color=TEXT, font="Helvetica-Bold", align="center")
    obs_box(ps, 48, 412, w=58, h=58, compact=True)
    full_block(ps, 120, 404, w=110, h=64)
    ps.line(106, 441, 120, 436, BLUE_STROKE, lw=1.7)
    ps.box(132, 489, 100, 34, ORANGE, ORANGE_STROKE, lw=1.3, r=7)
    ps.text(182, 502.5, "cached action", size=7.3, color=ORANGE_DARK, font="Helvetica-Bold", align="center")
    ps.text(182, 516.0, "latent z_t", size=7.3, color=ORANGE_DARK, font="Helvetica-Bold", align="center")
    ps.line(175, 468, 175, 489, ORANGE_STROKE, lw=1.4)
    action_head(ps, 244, 414, w=74, h=42)
    ps.line(230, 436, 244, 436, BLUE_STROKE, lw=1.7)
    ps.box(332, 419, 48, 31, WHITE, BLUE_STROKE, lw=1.0, r=14)
    ps.text(356, 438.5, "a_t", size=8.0, color=BLUE_DARK, font="Helvetica-Bold", align="center")
    ps.line(318, 435, 332, 435, BLUE_STROKE, lw=1.4)

    skip_column(ps, 318, "t+1", "z_t+1", "a_t+1")
    skip_column(ps, 584, "t+2", "z_t+2", "a_t+2")

    # Full refresh at t+K.
    x = 858
    ps.text(x + 128, 386, "t+K", size=12, color=TEXT, font="Helvetica-Bold", align="center")
    obs_box(ps, x, 412, w=58, h=58, compact=True)
    full_block(ps, x + 72, 404, w=112, h=64)
    ps.line(x + 58, 441, x + 72, 436, BLUE_STROKE, lw=1.7)
    ps.box(x + 85, 489, 110, 34, ORANGE, ORANGE_STROKE, lw=1.3, r=7)
    ps.text(x + 140, 502.5, "full refresh", size=7.3, color=ORANGE_DARK, font="Helvetica-Bold", align="center")
    ps.text(x + 140, 516.0, "re-anchors latent", size=7.3, color=ORANGE_DARK, font="Helvetica-Bold", align="center")
    ps.line(x + 128, 468, x + 128, 489, ORANGE_STROKE, lw=1.4)
    action_head(ps, x + 202, 414, w=74, h=42)
    ps.line(x + 184, 436, x + 202, 436, BLUE_STROKE, lw=1.7)
    ps.box(x + 290, 419, 51, 31, WHITE, BLUE_STROKE, lw=1.0, r=14)
    ps.text(x + 315.5, 438.5, "a_t+K", size=7.1, color=BLUE_DARK, font="Helvetica-Bold", align="center")
    ps.line(x + 276, 435, x + 290, 435, BLUE_STROKE, lw=1.4)

    # Concept labels.
    ps.box(48, 577, 288, 30, (0.98, 0.99, 1.0), BLUE_STROKE, lw=1.0, r=14)
    ps.text(192, 596, "Action head is not replaced", size=8.8, color=BLUE_DARK, font="Helvetica-Bold", align="center")
    ps.box(356, 577, 288, 30, (0.96, 1.00, 0.97), GREEN_STROKE, lw=1.0, r=14)
    ps.text(500, 596, "LR-NODE updates latent, not actions", size=8.8, color=GREEN_DARK, font="Helvetica-Bold", align="center")
    ps.box(664, 577, 420, 30, (0.98, 0.99, 1.0), GRAY_STROKE, lw=1.0, r=14)
    ps.text(874, 596, "Full Seer query rate is approximately 1/K of baseline", size=8.6, color=TEXT, font="Helvetica-Bold", align="center")
    ps.text(874, 613, "Same action decoder/head reused", size=8.0, color=BLUE_DARK, font="Helvetica-Bold", align="center")

    # Dashed temporal cache arrows.
    ps.line(232, 506, 318, 506, ORANGE_STROKE, lw=1.2, dashed=True)
    ps.text(276, 498, "cached z", size=6.5, color=ORANGE_DARK, align="center")
    ps.line(568, 565, 584, 565, ORANGE_STROKE, lw=1.2, dashed=True)
    ps.text(576, 555, "cache", size=5.8, color=ORANGE_DARK, align="center")
    ps.line(834, 565, 858, 506, ORANGE_STROKE, lw=1.2, dashed=True)


def build_ps() -> str:
    ps = PS()
    ps.add("%!PS-Adobe-3.0")
    ps.add(f"<< /PageSize [{PAGE_W} {PAGE_H}] >> setpagedevice")
    ps.add("1 setlinejoin 1 setlinecap")
    ps.add("1 1 1 setrgbcolor clippath fill")
    ps.text(
        PAGE_W / 2,
        36,
        "Latent-Reactive NODE for Efficient Seer Inference",
        size=23,
        color=TEXT,
        font="Helvetica-Bold",
        align="center",
    )
    draw_legend(ps)
    draw_baseline_row(ps)
    draw_ours_row(ps)
    ps.add("showpage")
    return "\n".join(ps.lines) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ps", delete=False) as f:
        ps_path = Path(f.name)
        f.write(build_ps())
    try:
        cmd = [
            "gs",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{DPI}",
            f"-g{RASTER_W}x{RASTER_H}",
            f"-sOutputFile={OUT}",
            str(ps_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        ps_path.unlink(missing_ok=True)
    print(OUT)


if __name__ == "__main__":
    main()
