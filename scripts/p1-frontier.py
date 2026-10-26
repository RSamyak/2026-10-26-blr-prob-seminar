#!/usr/bin/env python3
"""Speed-regret curves for Problem 1, computed on a grid rather than simulated.

Effect mu ~ N(0, 1).  Observe Y(u) = mu u + B(u) at looks u_k, so u is
information (n gamma^2 / sigma^2) and regret is in units of gamma, the prior
standard deviation of the effect.  The posterior mean m = Y / (1 + u) is a
standard Brownian motion in q = u / (1 + u), so each procedure is one forward
convolution pass and the Bayes procedure one backward induction, on the same
grid.  Every procedure acts on the sign of m when it stops.  Phi is the lower
convex hull of everything computed: the Bayes procedures for 40 costs c lie on
it, and the families' points, being attainable, can only tighten its chords.

Usage, from the repo root, with numpy and scipy in .venv (about ten minutes;
a summary goes to stderr).  The log figure lives in the appendix; the linear
slide is redrawn from it by p1-frontier-linear.py:
  .venv/bin/python scripts/p1-frontier.py sections/_appendix.qmd   # replace in place
  .venv/bin/python scripts/p1-frontier-linear.py
  .venv/bin/python scripts/p1-frontier.py > figure.svg             # or print it
"""
import re
import sys

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm

U_MAX = 2000.0
U = np.concatenate([[0.0], np.geomspace(1e-3, U_MAX, 1800)])
Q = U / (1 + U)
K = len(U) - 1
DM = 0.001
M = np.arange(-5.0, 5.0 + DM / 2, DM)
ABS_M = np.abs(M)


def psi(z):
    return norm.pdf(z) - z * norm.sf(z)


RISK = [np.sqrt(1 - q) * psi(ABS_M / np.sqrt(1 - q)) for q in Q]


def kernel(var):
    h = int(np.ceil(6 * np.sqrt(var) / DM))
    w = np.exp(-((np.arange(-h, h + 1) * DM) ** 2) / (2 * var))
    return w / w.sum()


KERNELS = [kernel(v) for v in np.diff(Q)]


def run(stop):
    """Forward pass.  Per look: mass and risk stopping, and continuing."""
    f = np.zeros_like(M)
    f[len(M) // 2] = 1.0
    out = np.zeros((4, K + 1))
    for k in range(1, K + 1):
        f = np.maximum(fftconvolve(f, KERNELS[k - 1], mode="same"), 0.0)
        d = stop(k) if k < K else np.ones_like(M, bool)
        fr = f * RISK[k]
        out[:, k] = f[d].sum(), fr[d].sum(), f[~d].sum(), fr[~d].sum()
        f[d] = 0.0
    return out


def curve(out, truncate=True):
    """(S, R) for the procedure, and for it truncated at every look."""
    P, RS, A, B = out
    S = np.cumsum(U * P) + U * A
    R = np.cumsum(RS) + B
    return (S, R) if truncate else (S[-1:], R[-1:])


def bayes(c):
    V = RISK[K].copy()
    stops = [None] * (K + 1)
    for k in range(K - 1, 0, -1):
        cont = c * (U[k + 1] - U[k]) + fftconvolve(V, KERNELS[k], mode="same")
        stops[k] = RISK[k] <= cont
        V = np.where(stops[k], RISK[k], cont)
    return lambda k: stops[k]


def sprt(b):
    return lambda k: ABS_M * (1 + U[k]) >= b


def msprt(alpha):
    def stop(k):
        y2 = (ABS_M * (1 + U[k])) ** 2
        return y2 >= (1 + U[k]) * (2 * np.log(1 / alpha) + np.log1p(U[k]))
    return stop


def expected_loss(eps, u_min):
    """Flat-prior expected loss below eps, after a minimum sample u_min."""
    def stop(k):
        z = ABS_M * (1 + U[k]) / np.sqrt(U[k])
        return (psi(z) <= eps * np.sqrt(U[k])) & (U[k] >= u_min)
    return stop


def never(k):
    return np.zeros_like(M, bool)


def frontier(S, R):
    """Lower convex hull of a point cloud, sorted by S.

    Randomising between two procedures at time zero averages their (S, R), so
    a family's frontier, like Phi, is the hull of what its settings attain.
    """
    S, R = np.concatenate(S), np.concatenate(R)
    keep = (S > 0) & np.isfinite(R)
    order = np.lexsort((R[keep], S[keep]))
    hull = []
    for p in zip(S[keep][order], R[keep][order]):
        while len(hull) > 1 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                 - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) <= 0:
            hull.pop()
        hull.append(p)
    S, R = map(np.array, zip(*hull))
    last = np.argmin(R) + 1
    return S[:last], R[:last]


def family(stops, truncate):
    pts = [curve(run(s), truncate) for s in stops]
    return frontier([p[0] for p in pts], [p[1] for p in pts])


X0, X1, LX0, LX1 = 70.0, 624.0, np.log10(0.05), np.log10(50.0)
Y0, Y1, LY0, LY1 = 10.0, 322.0, np.log10(0.4), np.log10(4e-4)
S_PLOT = np.geomspace(0.05, 50.0, 140)
LOOK = {
    "fixed": ("fixed: size + level", 'stroke="#5b626e" stroke-width="2.6"'),
    "sprt": ("SPRT: 1 knob", 'stroke="#0b6e99" stroke-width="2.6"'),
    "msprt": ("mix. SPRT: edge", 'stroke="#8e3b78" stroke-width="2.6"'),
    "eloss": ("expected loss: edge", 'stroke="#c1452b" stroke-width="2.6"'),
    "menu": ("their envelope", 'stroke="#b8860b" stroke-opacity="0.45" '
             'stroke-width="11" stroke-linecap="round" stroke-linejoin="round"'),
    "bayes": ("\u03a6: all procedures", 'stroke="#16213e" stroke-width="2.2" '
              'stroke-dasharray="7 5"'),
}
ORDER = ["fixed", "sprt", "msprt", "eloss", "menu", "bayes"]


def px(S, R):
    x = X0 + (np.log10(S) - LX0) / (LX1 - LX0) * (X1 - X0)
    y = Y0 + (LY0 - np.log10(R)) / (LY0 - LY1) * (Y1 - Y0)
    return x, y


def on_plot(S, R):
    return np.interp(S_PLOT, S, R, left=np.nan, right=np.nan)


def fixed_level_risk(S, k=1.645):
    """Bayes regret of a fixed-horizon threshold rule, by direct quadrature."""
    theta = np.linspace(0, 8, 8001)
    z = np.sqrt(np.asarray(S))[:, None] * theta
    wrong = norm.cdf(k - z) + norm.cdf(-k - z)
    return np.trapezoid(theta * norm.pdf(theta) * wrong, theta, axis=1)


def svg(fams):
    """The figure: fixed, then three sequential tests, their envelope, then Phi."""
    curves = {name: on_plot(*fams[name]) for name in fams}
    curves["menu"] = np.min([curves[n] for n in ORDER[:4]], axis=0)

    def draw(name):
        x, y = px(S_PLOT, curves[name])
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y) if np.isfinite(b))
        label, attrs = LOOK[name]
        k = 24 + 20 * ORDER.index(name)
        return [f'<polyline points="{pts}" fill="none" {attrs}/>',
                f'<line x1="476" y1="{k}" x2="498" y2="{k}" {attrs}/>',
                f'<text class="fr-key" x="505" y="{k + 4.5}">{label}</text>']

    def fixed_band():
        upper = fixed_level_risk(S_PLOT)
        xu, yu = px(S_PLOT, upper)
        xl, yl = px(S_PLOT, curves["fixed"])
        pts = [*zip(xu, yu), *zip(xl[::-1], yl[::-1])]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts if np.isfinite(y))
        return [f'<polygon points="{points}" fill="#5b626e" fill-opacity="0.16" '
                'clip-path="url(#fr-clip)"/>', *draw("fixed")]

    def fragment(index, body):
        return [f'<g class="fragment" data-fragment-index="{index}">', *body, "</g>"]

    out = ['<svg class="p1-frontier" viewBox="0 0 640 372" role="img" aria-label='
           '"Regret against speed on log scales: a fixed-horizon region, family lower '
           'edges, their envelope, and the best of all procedures">',
           f'<defs><clipPath id="fr-clip"><rect x="{X0}" y="{Y0}" '
           f'width="{X1 - X0}" height="{Y1 - Y0}"/></clipPath></defs>']
    for t in (0.1, 1, 10):
        x, _ = px(t, 1)
        out += [f'<line class="fr-grid" x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y1}"/>',
                f'<text class="fr-tick" x="{x:.1f}" y="{Y1 + 20}" text-anchor="middle">{t:g}</text>']
    for t in (0.001, 0.01, 0.1):
        _, y = px(1, t)
        out += [f'<line class="fr-grid" x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}"/>',
                f'<text class="fr-tick" x="{X0 - 8}" y="{y + 5:.1f}" text-anchor="end">{t:g}</text>']
    out += [f'<rect class="fr-frame" x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}"/>',
            f'<text class="fr-axis" x="{(X0 + X1) / 2}" y="{Y1 + 46}" text-anchor="middle">'
            'speed S: expected sample size</text>',
            f'<text class="fr-axis" transform="translate(16 {(Y0 + Y1) / 2}) rotate(-90)" '
            'text-anchor="middle">regret R</text>']
    out += fragment(2, draw("menu"))
    out += fixed_band()
    out += fragment(1, draw("msprt") + draw("eloss") + draw("sprt"))
    out += fragment(3, draw("bayes") + ['<text class="fr-void" x="100" y="292">unattainable</text>'])
    out.append("</svg>")
    return "\n".join(out)


def compute():
    """Each family's frontier over its knobs, and Phi over the cost c."""
    fams = {
        "fixed": frontier(*[[x] for x in curve(run(never))]),
        "sprt": family([sprt(b) for b in np.geomspace(0.05, 40, 48)], False),
        "msprt": family([msprt(a) for a in np.geomspace(1e-4, 0.7, 16)], True),
        "eloss": family([expected_loss(e, u0) for e in np.geomspace(2e-4, 0.3, 40)
                         for u0 in [0, *np.geomspace(0.01, 50, 15)]], False),
    }
    bay = [curve(run(bayes(c)), False) for c in np.geomspace(1e-6, 1.0, 40)]
    pts = bay + list(fams.values())
    fams["bayes"] = frontier([p[0] for p in pts], [p[1] for p in pts])
    return fams


def report(fams):
    """Regret of each frontier relative to Phi, at matched speed."""
    phi = on_plot(*fams["bayes"])
    print("S      " + " ".join(f"{s:6.2f}" for s in S_PLOT[::10]), file=sys.stderr)
    for name, front in fams.items():
        ratio = on_plot(*front) / phi
        print(f"{name:6} " + " ".join(f"{r:6.2f}" for r in ratio[::10]), file=sys.stderr)


def main():
    """Print the figure, or replace it inside the file named on the command line."""
    fams = compute()
    report(fams)
    figure = svg(fams)
    if len(sys.argv) < 2:
        print(figure)
        return
    with open(sys.argv[1]) as f:
        text = f.read()
    text = re.sub(r'<svg class="p1-frontier".*?</svg>', lambda _: figure, text, flags=re.S)
    with open(sys.argv[1], "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
