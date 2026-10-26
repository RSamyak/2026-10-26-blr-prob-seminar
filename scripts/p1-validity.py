#!/usr/bin/env python3
"""Power and regret of a fixed-sample test, with and without level control.

phi_k(X) = A iff theta_hat >= k se.  The horizontal coordinate is x = theta/se,
so P_theta{phi_k = A} = Phi(x - k) and the regret |theta| P{wrong sign}, in
units of se, is |x| Phi(-|x| + k sign(x)).  Closed form; stdlib only.

Usage, from the repo root:
  .venv/bin/python scripts/p1-validity.py sections/_p1.qmd  # replace in place
  .venv/bin/python scripts/p1-validity.py > figure.svg      # or print it
"""
import math
import re
import sys

RULES = {
    "sign rule, k = 0": (0.0, "#0b6e99"),
    "5% level, k = 1.645": (1.645, "#c1452b"),
}
XS = [i / 50 for i in range(-200, 201)]
POWER_Y, REGRET_Y = 1.0, 1.0


def cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def power(x, c):
    return cdf(x - c)


def regret(x, c):
    return abs(x) * (1 - power(x, c) if x > 0 else power(x, c))


PANELS = [
    ("P\u03b8(\u03c6(X) = A)", power, POWER_Y, 70.0, 360.0),
    ("regret |\u03b8| P\u03b8(wrong sign)", regret, REGRET_Y, 440.0, 730.0),
]
Y0, Y1 = 34.0, 236.0


def panel(title, f, ymax, x0, x1):
    def px(x, y):
        return x0 + (x + 4) / 8 * (x1 - x0), Y1 - y / ymax * (Y1 - Y0)

    out = []
    for x in (-4, -2, 0, 2, 4):
        xp, _ = px(x, 0)
        out += [f'<line class="fr-grid" x1="{xp:.1f}" y1="{Y0}" x2="{xp:.1f}" y2="{Y1}"/>',
                f'<text class="fr-tick" x="{xp:.1f}" y="{Y1 + 19}" text-anchor="middle">{x}</text>']
    for k in range(5):
        y = ymax * k / 4
        _, yp = px(0, y)
        out += [f'<line class="fr-grid" x1="{x0}" y1="{yp:.1f}" x2="{x1}" y2="{yp:.1f}"/>',
                f'<text class="fr-tick" x="{x0 - 8}" y="{yp + 4.5:.1f}" text-anchor="end">{y:g}</text>']
    xz, _ = px(0, 0)
    out += [f'<line class="va-level" x1="{xz:.1f}" y1="{Y0}" x2="{xz:.1f}" y2="{Y1}"/>',
            f'<rect class="fr-frame" x="{x0}" y="{Y0}" width="{x1 - x0}" height="{Y1 - Y0}"/>',
            f'<text class="fr-axis" x="{(x0 + x1) / 2}" y="{Y0 - 12}" text-anchor="middle">{title}</text>',
            f'<text class="fr-axis" x="{(x0 + x1) / 2}" y="{Y1 + 44}" text-anchor="middle">'
            'true effect \u03b8 / se</text>']
    for name, (c, colour) in RULES.items():
        pts = " ".join("%.1f,%.1f" % px(x, f(x, c)) for x in XS)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="3"/>')
    return out


def svg():
    out = ['<svg class="p1-validity" viewBox="0 0 760 290" role="img" aria-label='
           '"Power and regret against the true effect for the sign rule and the 5% level test">']
    for p in PANELS:
        out += panel(*p)
    for i, (name, (_, colour)) in enumerate(RULES.items()):
        y = 52 + 18 * i
        out += [f'<line x1="452" y1="{y}" x2="476" y2="{y}" stroke="{colour}" stroke-width="3"/>',
                f'<text class="fr-key" x="483" y="{y + 4.5}">{name}</text>']
    out.append("</svg>")
    return "\n".join(out)


def main():
    for name, (c, _) in RULES.items():
        peak = max(XS, key=lambda x: regret(x, c))
        print(f"{name}: size {power(0, c):.3f}, peak regret {regret(peak, c):.3f} at {peak:+.2f}",
              file=sys.stderr)
    figure = svg()
    if len(sys.argv) < 2:
        print(figure)
        return
    with open(sys.argv[1]) as f:
        text = f.read()
    text, count = re.subn(r'<svg class="p1-validity".*?</svg>', lambda _: figure, text, flags=re.S)
    if count != 1:
        raise RuntimeError(f"expected one p1-validity SVG, found {count}")
    with open(sys.argv[1], "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
