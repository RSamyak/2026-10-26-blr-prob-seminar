"""Problem 2, 'How many ideas?': total value against ideas tested, at Bing's fitted prior.

Model and parameters from Eduardo Azevedo et al. (2020), 'A/B testing with fat tails'.
Standard library only.  Checks against the paper go to stderr; the slide's SVG goes to stdout.
"""
import math
import sys

# Success-rate prior M + s * t_alpha (Table 2); standard error 0.022 at 20M users (Figure 1).
M, SCALE, SE20, ALPHA = -0.000946, 0.00296, 0.022, 1.31
m, r20 = M / SCALE, SCALE / SE20        # units of the prior scale; r = prior scale / standard error

def phi(z): return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
def sf(z): return math.erfc(z / math.sqrt(2)) / 2

def v_normal(r):                        # normal prior: E[(posterior mean)^+], closed form
    w = r / math.sqrt(1 + r * r)
    return w * phi(m / w) + m * (1 - sf(m / w))

C = math.exp(math.lgamma((ALPHA + 1) / 2) - math.lgamma(ALPHA / 2)) / math.sqrt(ALPHA * math.pi)
def g(u): return C * (1 + u * u / ALPHA) ** (-(ALPHA + 1) / 2)

X, n = math.asinh(1e4), 6000            # u = sinh(x), Simpson's rule on |u| < 1e4, closed-form upper tail
nodes = [(math.sinh(x), (1 if i in (0, n) else 4 if i % 2 else 2) * 2 * X / n / 3 * math.cosh(x))
         for i, x in ((i, -X + 2 * X * i / n) for i in range(n + 1))]
A = math.sinh(X)
TAIL = (ALPHA + A * A) / (ALPHA - 1) * g(A) + m * A * g(A) / ALPHA

def shipped(c, r):                      # E[theta 1{theta_hat > c}], theta_hat ~ N(theta, 1/r^2)
    return sum(w * (m + u) * g(u) * sf((c - u) * r) for u, w in nodes) + TAIL

def v_t(r):                             # t prior: best threshold by golden section
    lo, hi, k = -3 / r, 12 / r, (math.sqrt(5) - 1) / 2
    for _ in range(50):
        a, b = hi - k * (hi - lo), lo + k * (hi - lo)
        if shipped(a, r) < shipped(b, r): lo = a
        else: hi = b
    return shipped((lo + hi) / 2, r)

def log(line): print(line, file=sys.stderr)

v20 = v_t(r20)                          # checks against the paper
log("value of a 20M test: %.4f scale units (paper 2.39e-3 / 0.00296 = %.4f)" % (v20, 2.39e-3 / SCALE))
for p, paper in ((1.1, 8.59), (1.2, 17.05)):
    log("%.0f%% more ideas, same users: +%.2f%% (paper +%.2f%%)" % (100 * p - 100, 100 * (p * v_t(r20 / math.sqrt(p)) / v20 - 1), paper))

K = 21                                  # 21 ideas; traffic for 21 tests of 20M users, split equally over k of them
thin = [k * v_normal(r20 * math.sqrt(K / k)) for k in range(1, K + 1)]
fat = [k * v_t(r20 * math.sqrt(K / k)) for k in range(1, K + 1)]
thin, fat = [y / max(thin) for y in thin], [y / max(fat) for y in fat]
best_thin, best_fat = 1 + thin.index(1.0), 1 + fat.index(1.0)
log("best k: thin %d, fat %d" % (best_thin, best_fat))

def x(k): return 80 + 32.5 * (k - 1)
def y(v): return 290 - 246 * v
def points(ys): return " ".join("%.1f,%.1f" % (x(k), y(v)) for k, v in enumerate(ys, 1))

SVG = """<svg viewBox="0 0 760 345" role="img" aria-label="Total value against the number of ideas tested, for a thin-tailed and a fat-tailed prior">
<path class="ax" d="M80 30V290H740M80 290v6M210 290v6M372.5 290v6M535 290v6M730 290v6"/>
<g class="tk"><text x="80" y="314">1</text><text x="210" y="314">5</text><text x="372.5" y="314">10</text><text x="535" y="314">15</text><text x="730" y="314">21</text></g>
<g class="tky"><text x="70" y="295">0</text><text x="70" y="49">max</text></g>
<text class="lab" x="405" y="339" text-anchor="middle">ideas tested, k</text>
<text class="lab" x="0" y="0" text-anchor="middle" transform="translate(30 167) rotate(-90)">total value</text>
<g class="thin">
<polyline points="{thin}"/>
<circle cx="{thin_x:g}" cy="44" r="6"/>
<text x="128" y="30">thin tail (normal): best k = {best_thin}</text>
</g>
<g class="fat fragment">
<polyline points="{fat}"/>
<circle cx="{fat_x:g}" cy="44" r="6"/>
<text x="725" y="128" text-anchor="end">fat tail, α = {alpha:g}</text>
<text x="725" y="151" text-anchor="end">Bing's fit: {fat_note}</text>
</g>
</svg>"""

print(SVG.format(thin=points(thin), fat=points(fat), thin_x=x(best_thin), fat_x=x(best_fat), best_thin=best_thin,
                 alpha=ALPHA, fat_note="test all %d" % K if best_fat == K else "best k = %d" % best_fat))
