"""
Tiny dependency-free SVG chart helpers, themed to match the dark site.
Each returns an SVG string sized with a viewBox so it scales to its container.
"""
INK = "#e6edf3"; MUT = "#8aa0b2"; GRID = "#2a3543"; ACC = "#4cc2ff"
POS = "#39d98a"; WARN = "#f0a35e"; BG = "#0f141c"


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def hbars(items, color=ACC, value_fmt="{:.1f}", width=560, label_w=150,
          bar_h=20, gap=8, max_value=None, unit=""):
    """items: list of (label, value). Horizontal bars."""
    items = list(items)
    n = len(items)
    h = n * (bar_h + gap) + gap
    mx = max_value if max_value is not None else max((v for _, v in items), default=1) or 1
    plot_w = width - label_w - 70
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    y = gap
    for label, v in items:
        bw = max(0, (v / mx) * plot_w)
        out.append(f'<text x="{label_w-8}" y="{y+bar_h*0.7}" text-anchor="end" '
                   f'fill="{MUT}" font-size="12.5">{_esc(label)}</text>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                   f'rx="4" fill="{color}" opacity="0.85"/>')
        out.append(f'<text x="{label_w+bw+6:.1f}" y="{y+bar_h*0.7}" fill="{INK}" '
                   f'font-size="12">{_esc(value_fmt.format(v))}{unit}</text>')
        y += bar_h + gap
    out.append("</svg>")
    return "".join(out)


def line_chart(series, x_range, y_range, width=560, height=340, diagonal=False,
               x_label="", y_label="", x_ticks=None, y_ticks=None, dots=True):
    """series: list of dicts {name, color, points:[(x,y)], dash?}. Optional y=x diagonal."""
    ml, mr, mt, mb = 52, 16, 14, 40
    pw, ph = width - ml - mr, height - mt - mb
    x0, x1 = x_range
    y0, y1 = y_range
    sx = lambda x: ml + (x - x0) / (x1 - x0 + 1e-9) * pw
    sy = lambda y: mt + (1 - (y - y0) / (y1 - y0 + 1e-9)) * ph
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    # grid + ticks
    xt = x_ticks or _nice_ticks(x0, x1)
    yt = y_ticks or _nice_ticks(y0, y1)
    for xv in xt:
        out.append(f'<line x1="{sx(xv):.1f}" y1="{mt}" x2="{sx(xv):.1f}" y2="{mt+ph}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{sx(xv):.1f}" y="{mt+ph+16}" text-anchor="middle" fill="{MUT}" font-size="11">{_fmt(xv)}</text>')
    for yv in yt:
        out.append(f'<line x1="{ml}" y1="{sy(yv):.1f}" x2="{ml+pw}" y2="{sy(yv):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ml-6}" y="{sy(yv)+4:.1f}" text-anchor="end" fill="{MUT}" font-size="11">{_fmt(yv)}</text>')
    if diagonal:
        out.append(f'<line x1="{sx(max(x0,y0)):.1f}" y1="{sy(max(x0,y0)):.1f}" '
                   f'x2="{sx(min(x1,y1)):.1f}" y2="{sy(min(x1,y1)):.1f}" '
                   f'stroke="{MUT}" stroke-width="1.2" stroke-dasharray="4 4"/>')
    for s in series:
        pts = s["points"]
        if pts:
            d = "M" + " L".join(f"{sx(x):.1f} {sy(y):.1f}" for x, y in pts)
            dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
            out.append(f'<path d="{d}" fill="none" stroke="{s.get("color", ACC)}" stroke-width="2"{dash}/>')
            if dots:
                for x, y in pts:
                    out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{s.get("color", ACC)}"/>')
    if x_label:
        out.append(f'<text x="{ml+pw/2:.1f}" y="{height-4}" text-anchor="middle" fill="{MUT}" font-size="11.5">{_esc(x_label)}</text>')
    if y_label:
        out.append(f'<text x="13" y="{mt+ph/2:.1f}" text-anchor="middle" fill="{MUT}" font-size="11.5" transform="rotate(-90 13 {mt+ph/2:.1f})">{_esc(y_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def histogram(edges, counts, color=ACC, width=560, height=240, x_label="", mean=None):
    ml, mr, mt, mb = 40, 12, 12, 34
    pw, ph = width - ml - mr, height - mt - mb
    x0, x1 = edges[0], edges[-1]
    mx = max(counts) or 1
    sx = lambda x: ml + (x - x0) / (x1 - x0 + 1e-9) * pw
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i, c in enumerate(counts):
        bx, bw = sx(edges[i]), sx(edges[i + 1]) - sx(edges[i])
        bh = (c / mx) * ph
        out.append(f'<rect x="{bx:.1f}" y="{mt+ph-bh:.1f}" width="{max(bw-1,1):.1f}" '
                   f'height="{bh:.1f}" fill="{color}" opacity="0.8"/>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{GRID}"/>')
    if mean is not None:
        out.append(f'<line x1="{sx(mean):.1f}" y1="{mt}" x2="{sx(mean):.1f}" y2="{mt+ph}" '
                   f'stroke="{WARN}" stroke-width="1.5" stroke-dasharray="3 3"/>')
    for xv in _nice_ticks(x0, x1):
        out.append(f'<text x="{sx(xv):.1f}" y="{mt+ph+15}" text-anchor="middle" fill="{MUT}" font-size="11">{_fmt(xv)}</text>')
    if x_label:
        out.append(f'<text x="{ml+pw/2:.1f}" y="{height-2}" text-anchor="middle" fill="{MUT}" font-size="11">{_esc(x_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def _nice_ticks(a, b, n=5):
    if a == b:
        return [a]
    step = _nice(( b - a) / n)
    import math
    start = math.ceil(a / step) * step
    ticks, v = [], start
    while v <= b + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks or [a, b]


def _nice(x):
    import math
    if x <= 0:
        return 1
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    nf = 1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10
    return nf * 10 ** e


def _fmt(v):
    if abs(v) >= 1000:
        return f"{v/1000:.1f}k"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")
