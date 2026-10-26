"""Redraw the log-scale speed-regret figure on linear axes.

Reads the log SVG from sections/_appendix.qmd, adds the fixed-horizon level
band when needed, inverts pixel coordinates to (S, R), and writes the linear
SVG into the "Where a shipped test sits" slide of sections/_p1.qmd.
Stdlib only; the model itself lives in p1-frontier.py.
"""

import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE.parent / "sections" / "_appendix.qmd"
TARGET = HERE.parent / "sections" / "_p1.qmd"

# Frame and log scales of p1-frontier.py.
X0, X1 = 70, 624
Y0, Y1 = 10, 322
LX0, LX1 = math.log10(0.05), math.log10(50)
LY0, LY1 = math.log10(0.4), math.log10(4e-4)

S_MAX, R_MAX = 10, 0.35
S_TICKS = (0, 2, 4, 6, 8, 10)
R_TICKS = (0, 0.1, 0.2, 0.3)
VOID = (78, 312)
FRAGMENTS = (2, 1, 3)  # envelope, the three tests, Φ
FIXED = 'stroke="#5b626e" stroke-width="2.6"'


def unlog(x, y):
    s = 10 ** (LX0 + (x - X0) / (X1 - X0) * (LX1 - LX0))
    r = 10 ** (LY0 - (y - Y0) / (Y1 - Y0) * (LY0 - LY1))
    return s, r


def px(s, r):
    return X0 + s / S_MAX * (X1 - X0), Y1 - r / R_MAX * (Y1 - Y0)


def parse_points(points):
    return [tuple(map(float, p.split(","))) for p in points.split()]


def crop(xy):
    """Crop an increasing-x series to the linear panel's right edge."""
    edge = X0 + (math.log10(S_MAX) - LX0) / (LX1 - LX0) * (X1 - X0)
    kept = [p for p in xy if p[0] <= edge]
    if len(kept) < len(xy):
        (xa, ya), (xb, yb) = kept[-1], xy[len(kept)]
        kept.append((edge, ya + (yb - ya) * (edge - xa) / (xb - xa)))
    return kept


def relinear(points):
    kept = crop(parse_points(points))
    return " ".join("%.1f,%.1f" % px(*unlog(x, y)) for x, y in kept)


def relinear_polygon(points):
    xy = parse_points(points)
    split = len(xy) // 2
    upper = crop(xy[:split])
    lower = list(reversed(crop(list(reversed(xy[split:])))))
    return " ".join("%.1f,%.1f" % px(*unlog(x, y)) for x, y in upper + lower)


def fixed_risk(s, k=1.645):
    """Direct quadrature of E[|theta| 1{wrong sign}], theta ~ N(0, 1)."""
    n, limit = 4000, 8.0
    h = limit / n
    total = 0.0
    for i in range(n + 1):
        theta = i * h
        z = math.sqrt(s) * theta
        cdf = lambda a: 0.5 * (1 + math.erf(a / math.sqrt(2)))
        value = theta * math.exp(-theta * theta / 2) / math.sqrt(2 * math.pi)
        value *= cdf(k - z) + cdf(-k - z)
        total += value * (1 if i in (0, n) else 4 if i % 2 else 2)
    return total * h / 3


def decorate(svg):
    """Add the computed fixed-level band and truthful family labels."""
    svg = svg.replace(
        'aria-label="Regret against speed on log scales for four named tests, their envelope, and the best of all procedures"',
        'aria-label="Regret against speed on log scales: a fixed-horizon region, family lower edges, their envelope, and the best of all procedures"',
    )
    if 'id="fr-clip"' not in svg:
        defs = (f'<defs><clipPath id="fr-clip"><rect x="{X0}" y="{Y0}" '
                f'width="{X1 - X0}" height="{Y1 - Y0}"/></clipPath></defs>\n')
        svg = svg.replace('>\n', '>\n' + defs, 1)
    fixed = re.search(r'<polyline points="([^"]+)" fill="none" ' + re.escape(FIXED) + r'/>', svg)
    if '<polygon points=' not in svg:
        lower = parse_points(fixed.group(1))
        upper = []
        for x, _ in lower:
            s, _ = unlog(x, Y0)
            risk = fixed_risk(s)
            y = Y0 + (LY0 - math.log10(risk)) / (LY0 - LY1) * (Y1 - Y0)
            upper.append((x, y))
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + list(reversed(lower)))
        band = (f'<polygon points="{points}" fill="#5b626e" fill-opacity="0.16" '
                'clip-path="url(#fr-clip)"/>\n')
        svg = svg[:fixed.start()] + band + svg[fixed.start():]
    labels = {
        "fixed horizon": "fixed: size + level",
        "fixed: 2 knobs (band)": "fixed: size + level",
        "SPRT": "SPRT: 1 knob",
        "mixture SPRT": "mix. SPRT: edge",
        "mix. SPRT: lower edge": "mix. SPRT: edge",
        "expected loss": "expected loss: edge",
        "expected loss: lower edge": "expected loss: edge",
    }
    for old, new in labels.items():
        svg = svg.replace(f'>{old}</text>', f'>{new}</text>')
    return svg


def axes():
    lines = []
    for s in S_TICKS:
        x, _ = px(s, 0)
        if 0 < s < S_MAX:
            lines.append(f'<line class="fr-grid" x1="{x:.1f}" y1="{Y0:.1f}" x2="{x:.1f}" y2="{Y1:.1f}"/>')
        lines.append(f'<text class="fr-tick" x="{x:.1f}" y="{Y1 + 20:.1f}" text-anchor="middle">{s:g}</text>')
    for r in R_TICKS:
        _, y = px(0, r)
        if 0 < r < R_MAX:
            lines.append(f'<line class="fr-grid" x1="{X0:.1f}" y1="{y:.1f}" x2="{X1:.1f}" y2="{y:.1f}"/>')
        lines.append(f'<text class="fr-tick" x="{X0 - 8:.1f}" y="{y + 5:.1f}" text-anchor="end">{r:g}</text>')
    return "\n".join(lines) + "\n"


def linear(svg):
    svg = re.sub(r'<(line class="fr-grid"|text class="fr-tick")[^\n]*\n', "", svg)
    svg = svg.replace('<rect class="fr-frame"', axes() + '<rect class="fr-frame"')
    svg = re.sub(r'(<polyline points=")([^"]*)', lambda m: m[1] + relinear(m[2]), svg)
    svg = re.sub(r'(<polygon points=")([^"]*)', lambda m: m[1] + relinear_polygon(m[2]), svg)
    svg = svg.replace("on log scales", "on linear scales")
    svg = re.sub(r'class="fr-void" x="[^"]*" y="[^"]*"', 'class="fr-void" x="%d" y="%d"' % VOID, svg)
    groups = iter(FRAGMENTS)
    return re.sub(r"<g>", lambda m: f'<g class="fragment" data-fragment-index="{next(groups)}">', svg)


SVG = re.compile(r'<svg class="p1-frontier".*?</svg>', re.S)
log_svg = SVG.search(SOURCE.read_text()).group(0)
log_svg = decorate(log_svg)
SOURCE.write_text(SVG.sub(lambda _: log_svg, SOURCE.read_text(), count=1))
text = TARGET.read_text()
head, sep, tail = text.partition("## Where a shipped test sits")
TARGET.write_text(head + sep + SVG.sub(lambda m: linear(log_svg), tail, count=1))
