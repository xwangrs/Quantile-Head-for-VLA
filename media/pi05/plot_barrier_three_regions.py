#!/usr/bin/env python3
"""Draw the three operating regions of the PI0.5 quantile gap barrier."""

from __future__ import annotations

import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SVG_PATH = ROOT / "quantile-gap-barrier-three-regions.svg"
PNG_PATH = ROOT / "quantile-gap-barrier-three-regions.png"

WIDTH, HEIGHT = 1600, 900
PNG_SCALE = 2

PLOT_LEFT, PLOT_TOP = 115, 170
PLOT_RIGHT, PLOT_BOTTOM = 1135, 785
X_MIN, X_MAX = 0.05, 8.0
Y_MIN, Y_MAX = -1.55, 8.35

COLORS = {
    "background": "#FAFBFD",
    "plot_background": "#FFFFFF",
    "text": "#172033",
    "muted": "#5C667A",
    "grid": "#DCE2EB",
    "axis": "#748096",
    "pinball": "#E07A3F",
    "barrier": "#2979C6",
    "total": "#6A4BBC",
    "collapse_fill": "#FDE8E3",
    "balance_fill": "#E5F5EA",
    "large_fill": "#E7F0FB",
    "collapse_accent": "#C94C3A",
    "balance_accent": "#218A52",
    "large_accent": "#326FA8",
    "card": "#FFFFFF",
}


def pinball(t: float) -> float:
    """L_pinball / (a_i g_0)."""
    return t


def barrier(t: float) -> float:
    """B_i / (a_i g_0)."""
    return 2.0 * math.log((1.0 + t) / (2.0 * t))


def total(t: float) -> float:
    return pinball(t) + barrier(t)


def x_px(t: float) -> float:
    fraction = (math.log(t) - math.log(X_MIN)) / (math.log(X_MAX) - math.log(X_MIN))
    return PLOT_LEFT + fraction * (PLOT_RIGHT - PLOT_LEFT)


def y_px(value: float) -> float:
    fraction = (value - Y_MIN) / (Y_MAX - Y_MIN)
    return PLOT_BOTTOM - fraction * (PLOT_BOTTOM - PLOT_TOP)


def samples(count: int = 420) -> list[float]:
    return [
        math.exp(math.log(X_MIN) + i * (math.log(X_MAX) - math.log(X_MIN)) / (count - 1))
        for i in range(count)
    ]


def svg_path(function) -> str:
    points = [(x_px(t), y_px(function(t))) for t in samples()]
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f},{y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def svg_text(x: float, y: float, text: str, **attrs: str | int | float) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" {attributes}>{html.escape(text)}</text>'


def write_svg() -> None:
    x_ticks = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    y_ticks = [-1, 0, 1, 2, 4, 6, 8]
    balance_x = x_px(1.0)
    balance_y = y_px(total(1.0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#23304A" flood-opacity="0.13"/>',
        "</filter>",
        '<marker id="arrow-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">',
        f'<path d="M0,0 L0,6 L7,3 z" fill="{COLORS["balance_accent"]}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["background"]}"/>',
        svg_text(80, 65, "Gradient-balanced Quantile Gap Barrier", fill=COLORS["text"], font_size=34, font_weight=700),
        svg_text(80, 105, "三个区域：防塌缩、梯度平衡与大间隔饱和", fill=COLORS["muted"], font_size=22),
        f'<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{PLOT_RIGHT-PLOT_LEFT}" height="{PLOT_BOTTOM-PLOT_TOP}" rx="14" fill="{COLORS["plot_background"]}" stroke="#E1E6EE"/>',
        f'<rect x="{PLOT_LEFT}" y="{PLOT_TOP}" width="{x_px(0.45)-PLOT_LEFT:.2f}" height="{PLOT_BOTTOM-PLOT_TOP}" fill="{COLORS["collapse_fill"]}" opacity="0.88"/>',
        f'<rect x="{x_px(0.45):.2f}" y="{PLOT_TOP}" width="{x_px(2.0)-x_px(0.45):.2f}" height="{PLOT_BOTTOM-PLOT_TOP}" fill="{COLORS["balance_fill"]}" opacity="0.88"/>',
        f'<rect x="{x_px(2.0):.2f}" y="{PLOT_TOP}" width="{PLOT_RIGHT-x_px(2.0):.2f}" height="{PLOT_BOTTOM-PLOT_TOP}" fill="{COLORS["large_fill"]}" opacity="0.88"/>',
    ]

    for tick in y_ticks:
        y = y_px(tick)
        parts.append(f'<line x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{PLOT_RIGHT}" y2="{y:.2f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(svg_text(PLOT_LEFT - 16, y + 7, str(tick), fill=COLORS["muted"], font_size=17, text_anchor="end"))

    for tick in x_ticks:
        x = x_px(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{PLOT_BOTTOM}" stroke="{COLORS["grid"]}" stroke-width="1" stroke-dasharray="3 7"/>')
        label = f"{tick:g}"
        parts.append(svg_text(x, PLOT_BOTTOM + 32, label, fill=COLORS["muted"], font_size=17, text_anchor="middle"))

    parts.extend(
        [
            f'<line x1="{PLOT_LEFT}" y1="{y_px(0):.2f}" x2="{PLOT_RIGHT}" y2="{y_px(0):.2f}" stroke="{COLORS["axis"]}" stroke-width="1.4"/>',
            f'<line x1="{balance_x:.2f}" y1="{PLOT_TOP}" x2="{balance_x:.2f}" y2="{PLOT_BOTTOM}" stroke="{COLORS["balance_accent"]}" stroke-width="2.2" stroke-dasharray="7 7"/>',
            f'<path d="{svg_path(pinball)}" fill="none" stroke="{COLORS["pinball"]}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<path d="{svg_path(barrier)}" fill="none" stroke="{COLORS["barrier"]}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<path d="{svg_path(total)}" fill="none" stroke="{COLORS["total"]}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<circle cx="{balance_x:.2f}" cy="{balance_y:.2f}" r="8" fill="#FFFFFF" stroke="{COLORS["total"]}" stroke-width="4"/>',
            svg_text((PLOT_LEFT + x_px(0.45)) / 2, PLOT_TOP + 42, "① 塌缩区  g << g0", fill=COLORS["collapse_accent"], font_size=20, font_weight=700, text_anchor="middle"),
            svg_text((x_px(0.45) + x_px(2.0)) / 2, PLOT_TOP + 42, "② 平衡区  g ≈ g0", fill=COLORS["balance_accent"], font_size=20, font_weight=700, text_anchor="middle"),
            svg_text((x_px(2.0) + PLOT_RIGHT) / 2, PLOT_TOP + 42, "③ 大间隔区  g >> g0", fill=COLORS["large_accent"], font_size=20, font_weight=700, text_anchor="middle"),
            svg_text((PLOT_LEFT + PLOT_RIGHT) / 2, 855, "归一化相邻间隔  g / g0  （对数坐标）", fill=COLORS["text"], font_size=20, font_weight=600, text_anchor="middle"),
            f'<text x="36" y="{(PLOT_TOP+PLOT_BOTTOM)/2}" fill="{COLORS["text"]}" font-size="20" font-weight="600" text-anchor="middle" transform="rotate(-90 36 {(PLOT_TOP+PLOT_BOTTOM)/2})">归一化损失  / (a_i g0)</text>',
        ]
    )

    card_x, card_y, card_w, card_h = 1170, 170, 375, 615
    parts.extend(
        [
            f'<rect x="{card_x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="18" fill="{COLORS["card"]}" filter="url(#shadow)"/>',
            svg_text(card_x + 28, card_y + 45, "曲线", fill=COLORS["text"], font_size=21, font_weight=700),
            f'<line x1="{card_x+30}" y1="{card_y+82}" x2="{card_x+82}" y2="{card_y+82}" stroke="{COLORS["pinball"]}" stroke-width="5" stroke-linecap="round"/>',
            svg_text(card_x + 98, card_y + 89, "Pinball:  t", fill=COLORS["text"], font_size=19),
            f'<line x1="{card_x+30}" y1="{card_y+124}" x2="{card_x+82}" y2="{card_y+124}" stroke="{COLORS["barrier"]}" stroke-width="5" stroke-linecap="round"/>',
            svg_text(card_x + 98, card_y + 131, "Barrier:  2 log[(1+t)/(2t)]", fill=COLORS["text"], font_size=18),
            f'<line x1="{card_x+30}" y1="{card_y+166}" x2="{card_x+82}" y2="{card_y+166}" stroke="{COLORS["total"]}" stroke-width="6" stroke-linecap="round"/>',
            svg_text(card_x + 98, card_y + 173, "Total:  Pinball + Barrier", fill=COLORS["text"], font_size=18),
            f'<line x1="{card_x+28}" y1="{card_y+205}" x2="{card_x+card_w-28}" y2="{card_y+205}" stroke="#E5E9F0"/>',
            svg_text(card_x + 28, card_y + 247, "在平衡点  t = g/g0 = 1", fill=COLORS["balance_accent"], font_size=20, font_weight=700),
            svg_text(card_x + 42, card_y + 291, "Pinball 梯度", fill=COLORS["muted"], font_size=18),
            svg_text(card_x + card_w - 35, card_y + 291, "+a_i", fill=COLORS["pinball"], font_size=22, font_weight=700, text_anchor="end"),
            svg_text(card_x + 42, card_y + 334, "Barrier 梯度", fill=COLORS["muted"], font_size=18),
            svg_text(card_x + card_w - 35, card_y + 334, "−a_i", fill=COLORS["barrier"], font_size=22, font_weight=700, text_anchor="end"),
            f'<line x1="{card_x+42}" y1="{card_y+355}" x2="{card_x+card_w-35}" y2="{card_y+355}" stroke="{COLORS["axis"]}"/>',
            svg_text(card_x + 42, card_y + 393, "Total 梯度", fill=COLORS["text"], font_size=19, font_weight=700),
            svg_text(card_x + card_w - 35, card_y + 393, "0", fill=COLORS["balance_accent"], font_size=24, font_weight=800, text_anchor="end"),
            f'<line x1="{card_x+28}" y1="{card_y+428}" x2="{card_x+card_w-28}" y2="{card_y+428}" stroke="#E5E9F0"/>',
            svg_text(card_x + 28, card_y + 471, "三个区域", fill=COLORS["text"], font_size=21, font_weight=700),
            svg_text(card_x + 32, card_y + 512, "①  g → 0：Barrier → +∞", fill=COLORS["collapse_accent"], font_size=18),
            svg_text(card_x + 32, card_y + 550, "②  g = g0：梯度恰好抵消", fill=COLORS["balance_accent"], font_size=18),
            svg_text(card_x + 32, card_y + 588, "③  g → ∞：Barrier → 常数", fill=COLORS["large_accent"], font_size=18),
        ]
    )

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc" if bold else "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size * PNG_SCALE)
        except OSError:
            continue
    return ImageFont.load_default(size=size * PNG_SCALE)


def write_png() -> None:
    scale = PNG_SCALE
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), COLORS["background"])
    draw = ImageDraw.Draw(image)

    def box(coords, **kwargs):
        draw.rounded_rectangle(tuple(int(v * scale) for v in coords), radius=int(kwargs.pop("radius", 0) * scale), **kwargs)

    def line(points, **kwargs):
        draw.line([(int(x * scale), int(y * scale)) for x, y in points], **kwargs)

    def text_at(x, y, text, size, color=COLORS["text"], bold=False, anchor="la"):
        draw.text((int(x * scale), int(y * scale)), text, font=load_font(size, bold), fill=color, anchor=anchor)

    text_at(80, 55, "Gradient-balanced Quantile Gap Barrier", 34, bold=True)
    text_at(80, 96, "三个区域：防塌缩、梯度平衡与大间隔饱和", 22, color=COLORS["muted"])

    box((PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM), fill=COLORS["plot_background"], outline="#E1E6EE", width=2 * scale, radius=14)
    draw.rectangle((int(PLOT_LEFT*scale), int(PLOT_TOP*scale), int(x_px(0.45)*scale), int(PLOT_BOTTOM*scale)), fill=COLORS["collapse_fill"])
    draw.rectangle((int(x_px(0.45)*scale), int(PLOT_TOP*scale), int(x_px(2.0)*scale), int(PLOT_BOTTOM*scale)), fill=COLORS["balance_fill"])
    draw.rectangle((int(x_px(2.0)*scale), int(PLOT_TOP*scale), int(PLOT_RIGHT*scale), int(PLOT_BOTTOM*scale)), fill=COLORS["large_fill"])

    for tick in [-1, 0, 1, 2, 4, 6, 8]:
        y = y_px(tick)
        line([(PLOT_LEFT, y), (PLOT_RIGHT, y)], fill=COLORS["grid"], width=1 * scale)
        text_at(PLOT_LEFT - 16, y, str(tick), 17, color=COLORS["muted"], anchor="rm")

    for tick in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
        x = x_px(tick)
        for y in range(PLOT_TOP, PLOT_BOTTOM, 10):
            line([(x, y), (x, min(y + 3, PLOT_BOTTOM))], fill=COLORS["grid"], width=1 * scale)
        text_at(x, PLOT_BOTTOM + 25, f"{tick:g}", 17, color=COLORS["muted"], anchor="ma")

    line([(PLOT_LEFT, y_px(0)), (PLOT_RIGHT, y_px(0))], fill=COLORS["axis"], width=2 * scale)
    balance_x = x_px(1.0)
    for y in range(PLOT_TOP, PLOT_BOTTOM, 14):
        line([(balance_x, y), (balance_x, min(y + 7, PLOT_BOTTOM))], fill=COLORS["balance_accent"], width=2 * scale)

    for function, color, width in [
        (pinball, COLORS["pinball"], 4),
        (barrier, COLORS["barrier"], 4),
        (total, COLORS["total"], 5),
    ]:
        points = [(x_px(t), y_px(function(t))) for t in samples()]
        line(points, fill=color, width=width * scale, joint="curve")

    balance_y = y_px(total(1.0))
    draw.ellipse(
        ((balance_x-8)*scale, (balance_y-8)*scale, (balance_x+8)*scale, (balance_y+8)*scale),
        fill="#FFFFFF",
        outline=COLORS["total"],
        width=4*scale,
    )

    text_at((PLOT_LEFT+x_px(0.45))/2, PLOT_TOP+31, "① 塌缩区  g << g0", 20, COLORS["collapse_accent"], True, "ma")
    text_at((x_px(0.45)+x_px(2.0))/2, PLOT_TOP+31, "② 平衡区  g ≈ g0", 20, COLORS["balance_accent"], True, "ma")
    text_at((x_px(2.0)+PLOT_RIGHT)/2, PLOT_TOP+31, "③ 大间隔区  g >> g0", 20, COLORS["large_accent"], True, "ma")
    text_at((PLOT_LEFT+PLOT_RIGHT)/2, 842, "归一化相邻间隔  g / g0  （对数坐标）", 20, bold=True, anchor="ma")

    # Render the vertical axis label on a small transparent layer, then rotate it.
    axis_layer = Image.new("RGBA", (520*scale, 50*scale), (255, 255, 255, 0))
    axis_draw = ImageDraw.Draw(axis_layer)
    axis_draw.text((260*scale, 25*scale), "归一化损失  / (a_i g0)", font=load_font(20, True), fill=COLORS["text"], anchor="mm")
    axis_layer = axis_layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(axis_layer, (8*scale, int(((PLOT_TOP+PLOT_BOTTOM)/2)*scale-axis_layer.height/2)), axis_layer)
    draw = ImageDraw.Draw(image)

    card_x, card_y, card_w, card_h = 1170, 170, 375, 615
    # A simple shadow plus card.
    box((card_x+4, card_y+7, card_x+card_w+4, card_y+card_h+7), fill="#E4E8EF", radius=18)
    box((card_x, card_y, card_x+card_w, card_y+card_h), fill=COLORS["card"], radius=18)
    text_at(card_x+28, card_y+31, "曲线", 21, bold=True)
    line([(card_x+30, card_y+82), (card_x+82, card_y+82)], fill=COLORS["pinball"], width=5*scale)
    text_at(card_x+98, card_y+82, "Pinball:  t", 19, anchor="lm")
    line([(card_x+30, card_y+124), (card_x+82, card_y+124)], fill=COLORS["barrier"], width=5*scale)
    text_at(card_x+98, card_y+124, "Barrier:  2 log[(1+t)/(2t)]", 17, anchor="lm")
    line([(card_x+30, card_y+166), (card_x+82, card_y+166)], fill=COLORS["total"], width=6*scale)
    text_at(card_x+98, card_y+166, "Total:  Pinball + Barrier", 17, anchor="lm")
    line([(card_x+28, card_y+205), (card_x+card_w-28, card_y+205)], fill="#E5E9F0", width=1*scale)
    text_at(card_x+28, card_y+229, "在平衡点  t = g/g0 = 1", 20, COLORS["balance_accent"], True)
    text_at(card_x+42, card_y+291, "Pinball 梯度", 18, COLORS["muted"], anchor="lm")
    text_at(card_x+card_w-35, card_y+291, "+a_i", 22, COLORS["pinball"], True, "rm")
    text_at(card_x+42, card_y+334, "Barrier 梯度", 18, COLORS["muted"], anchor="lm")
    text_at(card_x+card_w-35, card_y+334, "−a_i", 22, COLORS["barrier"], True, "rm")
    line([(card_x+42, card_y+355), (card_x+card_w-35, card_y+355)], fill=COLORS["axis"], width=1*scale)
    text_at(card_x+42, card_y+393, "Total 梯度", 19, bold=True, anchor="lm")
    text_at(card_x+card_w-35, card_y+393, "0", 24, COLORS["balance_accent"], True, "rm")
    line([(card_x+28, card_y+428), (card_x+card_w-28, card_y+428)], fill="#E5E9F0", width=1*scale)
    text_at(card_x+28, card_y+453, "三个区域", 21, bold=True)
    text_at(card_x+32, card_y+512, "①  g → 0：Barrier → +∞", 18, COLORS["collapse_accent"], anchor="lm")
    text_at(card_x+32, card_y+550, "②  g = g0：梯度恰好抵消", 18, COLORS["balance_accent"], anchor="lm")
    text_at(card_x+32, card_y+588, "③  g → ∞：Barrier → 常数", 18, COLORS["large_accent"], anchor="lm")

    image.save(PNG_PATH, optimize=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    write_svg()
    write_png()
    print(SVG_PATH)
    print(PNG_PATH)
