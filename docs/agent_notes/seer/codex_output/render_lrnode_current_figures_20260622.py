#!/usr/bin/env python3
"""Render current-code LR-NODE paper figures as high-resolution raster PNGs.

This script writes PNG files only. It does not create SVG/vector artifacts.
The content reflects the current implementation in:
- models/seer_model.py
- models/lrnode_modules.py
- utils/train_utils.py
- utils/eval_utils_libero.py
"""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "codex_output" / "figures" / "lrnode_raster_current_20260622"

BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
BLUE_FILL = "#EAF2FF"
GREEN = "#059669"
GREEN_DARK = "#047857"
GREEN_FILL = "#EAFBF3"
ORANGE = "#EA580C"
ORANGE_FILL = "#FFF4E8"
RED = "#DC2626"
RED_FILL = "#FEF2F2"
GRAY = "#64748B"
GRAY_LIGHT = "#CBD5E1"
GRAY_FILL = "#F8FAFC"
BLACK = "#0F172A"
MUTED = "#475569"
WHITE = "#FFFFFF"

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(size, bold=False, mono=False):
    if mono:
        return ImageFont.truetype(FONT_MONO_BOLD if bold else FONT_MONO, size)
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def tsize(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw, text, fnt, max_w):
    lines = []
    for raw in text.split("\n"):
        words = raw.split()
        line = ""
        for word in words:
            cand = f"{line} {word}".strip()
            if not line or tsize(draw, cand, fnt)[0] <= max_w:
                line = cand
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def text_box(draw, box, text, fnt, fill=BLACK, align="center", spacing=6):
    x0, y0, x1, y1 = box
    lines = wrap(draw, text, fnt, max(1, x1 - x0 - 28))
    heights = [tsize(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + spacing * max(0, len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, h in zip(lines, heights):
        w, _ = tsize(draw, line, fnt)
        x = x0 + 18 if align == "left" else x0 + (x1 - x0 - w) / 2
        draw.text((x, y), line, font=fnt, fill=fill)
        y += h + spacing


def round_box(draw, box, fill, outline, width=3, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def module(draw, box, text, stroke, fill, fnt, txt=None, width=3, radius=18):
    round_box(draw, box, fill, stroke, width=width, radius=radius)
    text_box(draw, box, text, fnt, fill=txt or stroke)


def dashed_line(draw, p0, p1, color=GRAY, width=3, dash=20, gap=13):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    d = 0
    while d < length:
        e = min(d + dash, length)
        draw.line((x0 + ux * d, y0 + uy * d, x0 + ux * e, y0 + uy * e), fill=color, width=width)
        d += dash + gap


def arrow(draw, p0, p1, color=BLACK, width=4, head=18, dashed=False):
    if dashed:
        dashed_line(draw, p0, p1, color=color, width=width)
    else:
        draw.line((p0, p1), fill=color, width=width)
    x0, y0 = p0
    x1, y1 = p1
    angle = math.atan2(y1 - y0, x1 - x0)
    left = (x1 - head * math.cos(angle - math.pi / 6), y1 - head * math.sin(angle - math.pi / 6))
    right = (x1 - head * math.cos(angle + math.pi / 6), y1 - head * math.sin(angle + math.pi / 6))
    draw.polygon([p1, left, right], fill=color)


def elbow(draw, pts, color=BLACK, width=4, head=18, dashed=False):
    for p0, p1 in zip(pts, pts[1:]):
        if dashed:
            dashed_line(draw, p0, p1, color=color, width=width)
        else:
            draw.line((p0, p1), fill=color, width=width)
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    angle = math.atan2(y1 - y0, x1 - x0)
    left = (x1 - head * math.cos(angle - math.pi / 6), y1 - head * math.sin(angle - math.pi / 6))
    right = (x1 - head * math.cos(angle + math.pi / 6), y1 - head * math.sin(angle + math.pi / 6))
    draw.polygon([pts[-1], left, right], fill=color)


def cross(draw, cx, cy, s=32):
    draw.line((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), fill=RED, width=7)
    draw.line((cx - s / 2, cy + s / 2, cx + s / 2, cy - s / 2), fill=RED, width=7)


def title(draw, w, text, fnt):
    tw, _ = tsize(draw, text, fnt)
    draw.text(((w - tw) / 2, 42), text, font=fnt, fill=BLACK)


def latent(draw, box, text, fnt):
    module(draw, box, text, ORANGE, ORANGE_FILL, fnt, width=4, radius=20)


def obs(draw, box, title_text, items, f_title, f_item):
    round_box(draw, box, WHITE, GRAY_LIGHT, width=2, radius=18)
    x0, y0, x1, _ = box
    draw.text((x0 + 20, y0 + 16), title_text, font=f_title, fill=BLACK)
    y = y0 + 62
    for item in items:
        round_box(draw, (x0 + 24, y, x1 - 24, y + 40), GRAY_FILL, GRAY_LIGHT, width=1, radius=11)
        draw.text((x0 + 42, y + 9), item, font=f_item, fill=MUTED)
        y += 50


def legend(draw, box, entries, fnt):
    round_box(draw, box, WHITE, GRAY_LIGHT, width=2, radius=16)
    x0, y0, _, _ = box
    x = x0 + 22
    for fill, stroke, label, dashed in entries:
        if dashed:
            dashed_line(draw, (x, y0 + 38), (x + 70, y0 + 38), color=stroke, width=5)
            draw.text((x + 86, y0 + 25), label, font=fnt, fill=MUTED)
            x += 390
        else:
            draw.rounded_rectangle((x, y0 + 22, x + 34, y0 + 56), radius=7, fill=fill, outline=stroke, width=3)
            draw.text((x + 48, y0 + 25), label, font=fnt, fill=MUTED)
            x += 310


def figure_pipeline():
    W, H = 5600, 2160
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    F_TITLE = font(60, bold=True)
    F_ROW = font(34, bold=True)
    F_LABEL = font(25, bold=True)
    F_TEXT = font(20)
    F_SMALL = font(17)
    F_MONO = font(21, bold=True, mono=True)

    title(d, W, "Latent-Reactive NODE for Efficient Seer Inference", F_TITLE)
    legend(
        d,
        (1700, 125, 3900, 205),
        [
            (BLUE_FILL, BLUE, "original Seer modules", False),
            (GREEN_FILL, GREEN, "LR-NODE update", False),
            (ORANGE_FILL, ORANGE, "cached latent", False),
            (WHITE, GRAY, "gray dashed = skipped full query", True),
        ],
        F_SMALL,
    )

    round_box(d, (90, 265, 5510, 905), WHITE, BLUE, width=3, radius=26)
    d.text((130, 300), "Baseline Seer: Full Policy Query Every Step", font=F_ROW, fill=BLUE_DARK)
    xcols = [130, 1460, 2790, 4120]
    steps = ["t", "t+1", "t+2", "t+3"]
    for x, step in zip(xcols, steps):
        d.text((x + 320, 365), step, font=F_LABEL, fill=BLACK)
        obs(d, (x, 430, x + 260, 610), "Observation", ["RGB images", "proprio q", "language"], F_SMALL, F_SMALL)
        module(d, (x + 330, 420, x + 650, 620), "Full Seer Forward:\nVision Encoder + Perceiver\n+ Transformer", BLUE, BLUE_FILL, F_TEXT, width=4)
        module(d, (x + 700, 465, x + 885, 575), "Existing\nAction Head", BLUE, BLUE_FILL, F_TEXT, width=3)
        round_box(d, (x + 925, 490, x + 1070, 550), WHITE, BLUE, width=2, radius=26)
        text_box(d, (x + 925, 490, x + 1070, 550), f"Action a_{step}", F_SMALL, BLUE_DARK)
        arrow(d, (x + 260, 520), (x + 325, 520), color=BLUE, width=4)
        arrow(d, (x + 650, 520), (x + 695, 520), color=BLUE, width=4)
        arrow(d, (x + 885, 520), (x + 920, 520), color=BLUE, width=4)
    round_box(d, (1690, 785, 3910, 850), "#F8FBFF", BLUE, width=2, radius=24)
    text_box(d, (1690, 785, 3910, 850), "Expensive full visual-transformer query at every control step", F_TEXT, BLUE_DARK)

    round_box(d, (90, 1000, 5510, 1955), WHITE, GREEN, width=3, radius=26)
    d.text((130, 1035), "Ours: Full Refresh Every K Steps + LR-NODE Skip Updates", font=F_ROW, fill=GREEN_DARK)
    xcols = [130, 1460, 2790, 4120]
    labels = ["t", "t+1", "t+2", "t+K"]
    for x, step in zip(xcols, labels):
        d.text((x + 300, 1105), step, font=F_LABEL, fill=BLACK)
        obs(d, (x, 1170, x + 250, 1350), "Observation", ["RGB images", "proprio q", "language"], F_SMALL, F_SMALL)

    # Refresh at t.
    x = xcols[0]
    module(d, (x + 315, 1160, x + 635, 1345), "Full Seer Forward", BLUE, BLUE_FILL, F_TEXT, width=4)
    latent(d, (x + 690, 1188, x + 930, 1292), "cached action\nlatent z_t", F_MONO)
    module(d, (x + 985, 1190, x + 1180, 1290), "Existing\nAction Head", BLUE, BLUE_FILL, F_TEXT)
    arrow(d, (x + 250, 1260), (x + 310, 1260), color=BLUE, width=4)
    arrow(d, (x + 635, 1250), (x + 685, 1240), color=BLUE, width=4)
    arrow(d, (x + 930, 1240), (x + 980, 1240), color=BLUE, width=4)

    def skip_step(x, ztext):
        module(d, (x + 300, 1138, x + 560, 1250), "Full Seer\nForward", GRAY, GRAY_FILL, F_TEXT, width=3)
        cross(d, x + 585, 1194, s=42)
        dashed_line(d, (x + 250, 1230), (x + 300, 1195), color=GRAY, width=4)
        obs(d, (x + 20, 1395, x + 300, 1525), "Cached previous obs", ["prev RGB", "prev q"], F_SMALL, F_SMALL)
        module(d, (x + 360, 1370, x + 650, 1535), "Fast Visual/Proprio\nDelta Encoder", GREEN, GREEN_FILL, F_TEXT, width=4)
        module(d, (x + 710, 1370, x + 1015, 1535), "Controlled\nLatent NODE", GREEN, GREEN_FILL, F_TEXT, width=4)
        latent(d, (x + 1065, 1390, x + 1305, 1515), ztext, F_MONO)
        module(d, (x + 1125, 1590, x + 1325, 1695), "Existing\nAction Head", BLUE, BLUE_FILL, F_TEXT)
        arrow(d, (x + 250, 1260), (x + 358, 1420), color=GREEN, width=4)
        arrow(d, (x + 300, 1460), (x + 355, 1460), color=GREEN, width=4)
        arrow(d, (x + 650, 1452), (x + 705, 1452), color=GREEN, width=4)
        arrow(d, (x + 1015, 1452), (x + 1060, 1452), color=GREEN, width=4)
        arrow(d, (x + 1185, 1515), (x + 1215, 1585), color=BLUE, width=4)

    skip_step(xcols[1], "z_{t+1}\nupdated")
    skip_step(xcols[2], "z_{t+2}\nupdated")

    # Refresh at t+K.
    x = xcols[3]
    module(d, (x + 315, 1160, x + 635, 1345), "Full Seer Forward\nrefresh / re-anchor", BLUE, BLUE_FILL, F_TEXT, width=4)
    latent(d, (x + 690, 1188, x + 930, 1292), "fresh latent\nz_{t+K}", F_MONO)
    module(d, (x + 985, 1190, x + 1180, 1290), "Existing\nAction Head", BLUE, BLUE_FILL, F_TEXT)
    arrow(d, (x + 250, 1260), (x + 310, 1260), color=BLUE, width=4)
    arrow(d, (x + 635, 1250), (x + 685, 1240), color=BLUE, width=4)
    arrow(d, (x + 930, 1240), (x + 980, 1240), color=BLUE, width=4)

    notes = [
        ((560, 1800, 1550, 1880), "Action head is not replaced"),
        ((1800, 1800, 2850, 1880), "LR-NODE updates latent, not actions"),
        ((3100, 1800, 4150, 1880), "Full Seer query reduction: approximately 1/K"),
        ((4400, 1800, 5250, 1880), "Same action decoder/head reused"),
    ]
    for box, note in notes:
        round_box(d, box, WHITE, GRAY_LIGHT, width=2, radius=18)
        text_box(d, box, note, F_TEXT, MUTED)

    out = OUT_DIR / "01_pipeline_idea.png"
    img.save(out, "PNG", optimize=True)


def figure_main_idea():
    W, H = 3840, 2160
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    F_TITLE = font(56, bold=True)
    F_SECTION = font(30, bold=True)
    F_LABEL = font(23, bold=True)
    F_TEXT = font(20)
    F_SMALL = font(17)
    F_MONO = font(21, mono=True)
    F_MONO_B = font(23, bold=True, mono=True)

    title(d, W, "Main Idea: Learn a Cheap Transition in Seer's Action-Latent Space", F_TITLE)
    legend(
        d,
        (1060, 120, 2780, 200),
        [
            (BLUE_FILL, BLUE, "teacher / original Seer", False),
            (GREEN_FILL, GREEN, "LR-NODE student", False),
            (ORANGE_FILL, ORANGE, "latent tensor", False),
            (WHITE, GRAY, "dashed = teacher-only", True),
        ],
        F_SMALL,
    )

    obs(d, (140, 330, 640, 560), "Context C_t", ["RGB history", "proprio history", "language"], F_TEXT, F_SMALL)
    module(d, (110, 660, 690, 820), "Full Seer Teacher\nat context C_t", BLUE, BLUE_FILL, F_LABEL, width=4)
    latent(d, (175, 920, 625, 1035), "action latent\nz_t", F_MONO_B)
    arrow(d, (390, 560), (390, 655), color=GRAY, width=4, dashed=True)
    arrow(d, (390, 820), (390, 915), color=GRAY, width=4, dashed=True)

    obs(d, (3200, 330, 3700, 560), "Shifted context C_{t+1}", ["next RGB history", "next proprio history", "language"], F_TEXT, F_SMALL)
    module(d, (3170, 660, 3750, 820), "Full Seer Teacher\nat shifted context C_{t+1}", BLUE, BLUE_FILL, F_LABEL, width=4)
    latent(d, (3235, 920, 3685, 1035), "teacher action\nlatent z_{t+1}", F_MONO_B)
    arrow(d, (3450, 560), (3450, 655), color=GRAY, width=4, dashed=True)
    arrow(d, (3450, 820), (3450, 915), color=GRAY, width=4, dashed=True)

    round_box(d, (1140, 305, 2700, 405), WHITE, GRAY_LIGHT, width=2, radius=18)
    text_box(d, (1140, 305, 2700, 405), "Teacher probe uses shifted policy contexts,\nnot in-window token matching", F_TEXT, MUTED)
    d.text((1320, 575), "LR-NODE student transition", font=F_SECTION, fill=GREEN_DARK)

    module(d, (1070, 730, 1450, 855), "visual/proprio delta\nfrom C_t to C_{t+1}", GREEN, WHITE, F_TEXT, width=3)
    module(d, (1560, 700, 1980, 855), "Fast Delta\nEncoder", GREEN, GREEN_FILL, F_LABEL, width=4)
    module(d, (1560, 950, 1980, 1110), "Gated Latent\nNODE", GREEN, GREEN_FILL, F_LABEL, width=4)
    latent(d, (2245, 950, 2765, 1080), "predicted latent\nz_hat_{t+1}", F_MONO_B)
    arrow(d, (625, 980), (1555, 1030), color=GREEN, width=6)
    d.text((870, 935), "z_t", font=F_MONO_B, fill=GREEN_DARK)
    arrow(d, (1450, 792), (1555, 792), color=GREEN, width=6)
    d.text((1470, 740), "u_delta", font=F_MONO_B, fill=GREEN_DARK)
    arrow(d, (1770, 855), (1770, 945), color=GREEN, width=6)
    arrow(d, (1980, 1030), (2240, 1030), color=GREEN, width=6)
    module(d, (1210, 1230, 2630, 1315), "z_hat_{t+1} = z_t + gate * dt * dz", GREEN, WHITE, F_MONO_B, width=2)

    action = (1380, 1500, 2460, 1635)
    module(d, action, "Existing Seer Action Head H(·)", BLUE, BLUE_FILL, F_LABEL, width=4)
    round_box(d, (680, 1505, 1320, 1630), WHITE, GRAY_LIGHT, width=2, radius=20)
    text_box(d, (680, 1505, 1320, 1630), "Action head reused\n/ not replaced", F_TEXT, MUTED)
    elbow(d, [(2480, 1080), (2820, 1140), (2820, 1425), (1760, 1495)], color=GREEN, width=6)
    elbow(d, [(3450, 1035), (3180, 1120), (3180, 1420), (2050, 1495)], color=GRAY, width=4, dashed=True)

    loss = (750, 1770, 3090, 1985)
    round_box(d, loss, RED_FILL, RED, width=3, radius=22)
    d.text((790, 1805), "Training losses", font=F_LABEL, fill=RED)
    d.text((795, 1860), "Latent distillation: ||z_hat_{t+1} - stopgrad(z_{t+1})||^2", font=F_MONO, fill=BLACK)
    d.text((795, 1915), "Action distillation: ||H(z_hat_{t+1}) - stopgrad(H(z_{t+1}))||_1", font=F_MONO, fill=BLACK)
    arrow(d, (1920, 1635), (1920, 1765), color=RED, width=4)
    elbow(d, [(2765, 1015), (2950, 1015), (2950, 1705), (2200, 1765)], color=RED, width=3)
    elbow(d, [(3450, 1035), (3450, 1705), (2450, 1765)], color=RED, width=3, dashed=True)

    round_box(d, (520, 2050, 3320, 2125), WHITE, GRAY_LIGHT, width=2, radius=20)
    text_box(d, (520, 2050, 3320, 2125), "Current code: LR-NODE predicts the next policy-context action latent, then reuses the original action decoder.", F_TEXT, BLACK)

    out = OUT_DIR / "02_main_idea.png"
    img.save(out, "PNG", optimize=True)


def section(draw, box, header, fnt):
    round_box(draw, box, WHITE, GRAY_LIGHT, width=2, radius=22)
    x0, y0, x1, _ = box
    draw.text((x0 + 24, y0 + 20), header, font=fnt, fill=BLACK)
    draw.line((x0 + 22, y0 + 70, x1 - 22, y0 + 70), fill=GRAY_LIGHT, width=2)


def teacher_stream(draw, y, title_text, shifted, fonts):
    F_LABEL, F_TEXT, F_SMALL, F_MONO_B = fonts
    items = [
        "RGB primary + wrist history" + (" shifted by one step" if shifted else ""),
        "proprio history q_{t+1}" if shifted else "proprio history q_t",
        "same language tokens" if shifted else "language tokens",
    ]
    obs(draw, (125, y + 70, 570, y + 285), title_text, items, F_LABEL, F_SMALL)
    boxes = {
        "visual": (650, y + 45, 1025, y + 125),
        "perceiver": (1100, y + 45, 1450, y + 125),
        "state": (650, y + 165, 1025, y + 245),
        "text": (650, y + 285, 1025, y + 365),
        "transformer": (1100, y + 185, 1450, y + 365),
        "latent": (1510, y + 210, 1765, y + 340),
    }
    module(draw, boxes["visual"], "Frozen/normal\nvisual encoder", BLUE, BLUE_FILL, F_TEXT)
    module(draw, boxes["perceiver"], "Perceiver\nresampler", BLUE, BLUE_FILL, F_TEXT)
    module(draw, boxes["state"], "State encoder", BLUE, BLUE_FILL, F_TEXT)
    module(draw, boxes["text"], "Text encoder /\nprojector", BLUE, BLUE_FILL, F_TEXT)
    module(draw, boxes["transformer"], "Causal\nTransformer", BLUE, BLUE_FILL, F_TEXT)
    latent(draw, boxes["latent"], "z_{t+1} teacher\nstopgrad" if shifted else "z_t", F_MONO_B)
    arrow(draw, (570, y + 135), (645, y + 85), color=GRAY, width=3, dashed=True)
    arrow(draw, (570, y + 190), (645, y + 205), color=GRAY, width=3, dashed=True)
    arrow(draw, (570, y + 245), (645, y + 325), color=GRAY, width=3, dashed=True)
    arrow(draw, (1025, y + 85), (1095, y + 85), color=GRAY, width=3, dashed=True)
    arrow(draw, (1450, y + 85), (1275, y + 180), color=GRAY, width=3, dashed=True)
    arrow(draw, (1025, y + 205), (1095, y + 235), color=GRAY, width=3, dashed=True)
    arrow(draw, (1025, y + 325), (1095, y + 315), color=GRAY, width=3, dashed=True)
    arrow(draw, (1450, y + 275), (1505, y + 275), color=GRAY, width=3, dashed=True)
    module(draw, (1185, y + 390, 1765, y + 455), "extract action_latent_full[:, selected_step]", GRAY_LIGHT, WHITE, font(18, mono=True), txt=MUTED, width=2, radius=14)
    arrow(draw, (1638, y + 340), (1638, y + 385), color=GRAY, width=3, dashed=True)


def figure_detailed():
    W, H = 5120, 2880
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    F_TITLE = font(66, bold=True)
    F_SECTION = font(33, bold=True)
    F_SUB = font(27, bold=True)
    F_LABEL = font(21, bold=True)
    F_TEXT = font(19)
    F_SMALL = font(16)
    F_TINY = font(14)
    F_MONO = font(18, mono=True)
    F_MONO_B = font(20, bold=True, mono=True)

    title(d, W, "Detailed LR-NODE Architecture in Seer", F_TITLE)
    legend(
        d,
        (1390, 135, 3730, 225),
        [
            (BLUE_FILL, BLUE, "original Seer modules", False),
            (GREEN_FILL, GREEN, "new LR-NODE modules", False),
            (ORANGE_FILL, ORANGE, "latent tensors", False),
            (RED_FILL, RED, "losses", False),
            (WHITE, GRAY, "gray dashed = teacher-only / stop-gradient", True),
        ],
        F_TINY,
    )

    section(d, (85, 275, 1830, 2665), "(A) Full Seer Teacher Probe", F_SECTION)
    section(d, (1870, 275, 3445, 2665), "(B) LR-NODE Latent Update", F_SECTION)
    section(d, (3490, 275, 5035, 2665), "(C) Reused Action Decoder and Training Losses", F_SECTION)
    teacher_stream(d, 405, "Context C_t", False, (F_LABEL, F_TEXT, F_SMALL, F_MONO_B))
    teacher_stream(d, 1435, "Shifted Context C_{t+1}", True, (F_LABEL, F_TEXT, F_SMALL, F_MONO_B))
    module(d, (375, 2475, 1555, 2578), "Teacher probe uses shifted policy contexts, not in-window token matching", GRAY_LIGHT, WHITE, F_TEXT, txt=MUTED, width=2)

    # FastVisualDeltaEncoder.
    round_box(d, (1905, 410, 3400, 1330), WHITE, GREEN, width=3, radius=24)
    d.text((1935, 438), "FastVisualDeltaEncoder", font=F_SUB, fill=GREEN_DARK)
    for txt, yy in [
        ("key_rgb from C_t selected step", 530),
        ("cur_rgb from C_{t+1} selected step", 615),
        ("q_key", 865),
        ("q_cur", 950),
    ]:
        module(d, (1945, yy, 2240, yy + 58), txt, GREEN, WHITE, F_SMALL, width=2, radius=14)
    steps = [
        ("1", "resize to 64x64", 520),
        ("2", "concat [key_rgb, cur_rgb,\ncur_rgb - key_rgb]", 635),
        ("3", "3-layer small CNN:\nConv 9→32→64→128", 775),
        ("4", "camera features averaged:\nprimary + wrist", 905),
        ("5", "proprio MLP on\n[q_key, q_cur, q_cur - q_key]", 1060),
    ]
    for num, txt, yy in steps:
        d.ellipse((2310, yy + 14, 2354, yy + 58), fill=GREEN, outline=GREEN)
        text_box(d, (2310, yy + 14, 2354, yy + 58), num, F_SMALL, WHITE)
        module(d, (2370, yy, 2965, yy + 86), txt, GREEN, GREEN_FILL, F_TEXT, width=2, radius=16)
    module(d, (3105, 845, 3345, 955), "u_delta", GREEN, GREEN_FILL, F_MONO_B, width=4)
    arrow(d, (2240, 559), (2365, 561), color=GREEN, width=4)
    arrow(d, (2240, 644), (2365, 675), color=GREEN, width=4)
    arrow(d, (2665, 606), (2665, 630), color=GREEN, width=4)
    arrow(d, (2665, 720), (2665, 770), color=GREEN, width=4)
    arrow(d, (2665, 861), (2665, 900), color=GREEN, width=4)
    arrow(d, (2965, 948), (3100, 890), color=GREEN, width=4)
    arrow(d, (2240, 894), (2365, 1105), color=GREEN, width=4)
    arrow(d, (2240, 979), (2365, 1105), color=GREEN, width=4)
    arrow(d, (2965, 1105), (3100, 930), color=GREEN, width=4)

    # ControlledLatentNODE.
    round_box(d, (1905, 1450, 3400, 2385), WHITE, GREEN, width=3, radius=24)
    d.text((1935, 1478), "ControlledLatentNODE", font=F_SUB, fill=GREEN_DARK)
    input_boxes = [
        ((1945, 1570, 2220, 1630), "previous latent z_t", ORANGE, ORANGE_FILL),
        ((1945, 1660, 2220, 1720), "u_delta", GREEN, WHITE),
        ((1945, 1750, 2220, 1810), "time embedding dt", GREEN, WHITE),
        ((1945, 1840, 2220, 1900), "cache age embedding", GREEN, WHITE),
    ]
    for box, txt, stroke, fill in input_boxes:
        module(d, box, txt, stroke, fill, F_SMALL, width=2, radius=14)
    module(d, (2310, 1560, 2995, 1665), "dynamics MLP:\ndz = f(z_t, u_delta, dt, age)", GREEN, GREEN_FILL, F_TEXT)
    module(d, (2310, 1715, 2995, 1835), "gate MLP:\ngate = sigmoid(g(u_delta, age) + gate_bias)\ncode default gate_bias = -4.0", GREEN, GREEN_FILL, F_TEXT)
    module(d, (2310, 1900, 2995, 2000), "fixed Euler update", GREEN, GREEN_FILL, F_LABEL)
    latent(d, (3090, 1878, 3355, 2018), "z_hat_{t+1}", F_MONO_B)
    module(d, (2120, 2130, 3195, 2240), "z_hat_{t+1} = z_t + gate * dt * dz", GREEN, WHITE, F_MONO_B, width=3)
    d.text((2290, 2265), "post LayerNorm optional; current default off", font=F_SMALL, fill=MUTED)
    for yy, dest in [(1600, 1615), (1690, 1615), (1780, 1615), (1870, 1775)]:
        arrow(d, (2220, yy), (2305, dest), color=GREEN, width=4)
    arrow(d, (2995, 1612), (3065, 1915), color=GREEN, width=4)
    arrow(d, (2995, 1775), (3065, 1945), color=GREEN, width=4)
    arrow(d, (2995, 1950), (3085, 1950), color=GREEN, width=4)
    arrow(d, (3220, 2018), (2680, 2125), color=GREEN, width=4)

    # Section C.
    d.text((3530, 420), "Same original action decoder", font=F_SUB, fill=BLUE)
    latent(d, (3565, 530, 3885, 655), "z_hat_{t+1}\nstudent", F_MONO_B)
    latent(d, (4620, 530, 4940, 655), "z_{t+1}\nteacher", F_MONO_B)
    module(d, (4705, 670, 4940, 730), "from shifted probe / stopgrad", GRAY_LIGHT, WHITE, F_SMALL, txt=GRAY, width=2, radius=12)
    module(d, (3910, 810, 4595, 950), "Existing Seer Action Head H(·)", BLUE, BLUE_FILL, F_LABEL, width=4, radius=22)
    module(d, (3645, 1110, 4145, 1210), "predicted action sequence", BLUE, WHITE, F_TEXT, width=2, radius=16)
    module(d, (4360, 1110, 4860, 1210), "teacher action", BLUE, WHITE, F_TEXT, width=2, radius=16)
    arrow(d, (3725, 655), (4050, 805), color=GREEN, width=4)
    arrow(d, (4780, 655), (4460, 805), color=GRAY, width=4, dashed=True)
    arrow(d, (4165, 950), (3895, 1105), color=BLUE, width=4)
    arrow(d, (4350, 950), (4610, 1105), color=GRAY, width=4, dashed=True)
    loss = (3560, 1360, 4965, 1800)
    round_box(d, loss, RED_FILL, RED, width=3, radius=22)
    d.text((3595, 1394), "Training losses", font=F_SUB, fill=RED)
    losses = [
        "L_latent = MSE(z_hat_{t+1}, stopgrad(z_{t+1}))",
        "L_action = L1(H(z_hat_{t+1}), stopgrad(H(z_{t+1})))",
        "L_smooth = MSE(z_hat_{t+1} - z_t, 0)",
        "Total LR-NODE loss = λ_z L_latent + λ_a L_action + λ_s L_smooth",
    ]
    yy = 1455
    for line in losses:
        d.text((3605, yy), line, font=F_MONO, fill=BLACK)
        yy += 72
    arrow(d, (3895, 1210), (3895, 1355), color=RED, width=3)
    arrow(d, (4610, 1210), (4610, 1355), color=RED, width=3, dashed=True)
    protocol = (3560, 1950, 4965, 2425)
    round_box(d, protocol, WHITE, GRAY_LIGHT, width=2, radius=22)
    d.text((3595, 1984), "Protocol", font=F_SUB, fill=BLACK)
    for line, yy in [
        ("Detached teacher-student mode: LR-NODE loss updates only LR-NODE modules", 2068),
        ("Coupled joint mode: z_t and action head can receive LR-NODE gradients", 2164),
        ("Eval: refresh full Seer every K steps; skip steps use cached z and LR-NODE update", 2260),
    ]:
        round_box(d, (3600, yy - 10, 4925, yy + 58), GRAY_FILL, GRAY_LIGHT, width=1, radius=12)
        d.text((3625, yy + 6), line, font=F_SMALL, fill=MUTED)
    module(d, (3735, 2510, 4785, 2605), "Action head reused / not replaced", GRAY_LIGHT, WHITE, F_LABEL, txt=BLUE, width=2)

    # Clean high-level dependencies.
    elbow(d, [(1765, 680), (1845, 680), (1845, 1600), (1940, 1600)], color=GREEN, width=4)
    d.text((1795, 632), "z_t seeds latent update", font=F_TINY, fill=GREEN_DARK)
    elbow(d, [(3355, 1948), (3480, 1948), (3480, 585), (3560, 585)], color=GREEN, width=4)
    d.text((3470, 520), "student prediction", font=F_TINY, fill=GREEN_DARK)

    out = OUT_DIR / "03_detailed_architecture.png"
    img.save(out, "PNG", optimize=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_pipeline()
    figure_main_idea()
    figure_detailed()
    for path in sorted(OUT_DIR.glob("*.png")):
        with Image.open(path) as im:
            print(f"{path} {im.size[0]}x{im.size[1]} {im.mode}")


if __name__ == "__main__":
    main()
