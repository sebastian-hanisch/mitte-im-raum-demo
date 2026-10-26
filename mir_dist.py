"""Die Mitte von Verteilungen: zwei Arten, Verteilungen zu „mitteln“, als Einheiten (DEMO-PLAYBOOK §1b), numpy-only (kein scipy zur Laufzeit).

* **Mischung** (Wahrscheinlichkeiten mitteln): F(x) = Σ w_j F_j(x). Das ist die richtige Beschreibung von „jeder Auftrag geht zufällig an Quelle j“.
* **Quantilmittel** (Quantile mitteln, Vincent-Mittel): Q(p) = Σ w_j Q_j(p). Auf der Zahlengeraden die Mitte der Verteilungen im Sinn des Abstands
  d(F, G)² = ∫ (Q_F − Q_G)² dp (eindimensionales Wasserstein-Baryzentrum); dies ist hier nur die Definition, keine Wasserstein-Theorie.
Beide haben dasselbe Mittel Σ w_j μ_j; sie unterscheiden sich in der Streuung.

Die Normalverteilungs-Funktionen (Φ, Φ⁻¹) sind selbst geschrieben, damit die App ohne scipy läuft; tests/test_oracle_verteilungen.py prüft sie gegen `math.erfc` und scipy."""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)
SERIES_LIMIT = 2.0       # bis hierhin die Reihe für erf, darüber der Kettenbruch für erfc
SERIES_TERMS = 45
FRACTION_TERMS = 60
BRACKET_SD = 9.0         # Suchbereich der Quantile: Lage ± 9 Streuungen
QUANTILE_ITERS = 100


# ---------------------------------------------------------------- Normalverteilung

def erfc(x):
    """Komplementäre Fehlerfunktion (Feld oder Zahl). |x| < 2: Reihe 2/√π·e^(−x²)·Σ 2^k x^(2k+1)/(2k+1)!! (nur positive Summanden), sonst Kettenbruch."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    small = ax < SERIES_LIMIT
    xs = np.where(small, ax, 0.0)
    term = xs.copy()
    total = xs.copy()
    for k in range(1, SERIES_TERMS):
        term = term * (2.0 * xs * xs / (2 * k + 1))
        total = total + term
    erf_small = 2.0 / SQRT_PI * np.exp(-xs * xs) * total
    xl = np.where(small, SERIES_LIMIT, ax)
    t = np.zeros_like(xl)
    for k in range(FRACTION_TERMS, 0, -1):
        t = (k / 2.0) / (xl + t)
    erfc_large = np.exp(-xl * xl) / SQRT_PI / (xl + t)
    positive = np.where(small, 1.0 - erf_small, erfc_large)
    return np.where(x >= 0, positive, 2.0 - positive)


def norm_cdf(x):
    """Verteilungsfunktion Φ der Standardnormalverteilung."""
    return 0.5 * erfc(-np.asarray(x, dtype=float) / SQRT2)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_ppf(p):
    """Φ⁻¹ per Bisektion auf [−9, 9] (60 Halbierungen); p muss in (0, 1) liegen."""
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("p muss zwischen 0 und 1 liegen")
    lo = np.full(p.shape, -BRACKET_SD)
    hi = np.full(p.shape, BRACKET_SD)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = norm_cdf(mid) < p
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- Mischung von Normalverteilungen

def _components(locs, scales):
    locs = np.atleast_2d(np.asarray(locs, dtype=float))
    scales = np.broadcast_to(np.atleast_2d(np.asarray(scales, dtype=float)), locs.shape)
    return locs, scales


def mixture_cdf(x, locs, scales, weights):
    """Verteilungsfunktion der Mischung Σ w_j N(loc_j, scale_j²) an den Stellen x. locs, scales: (T, J) oder (J,); x: (T, K) oder passend zu broadcasten."""
    locs, scales = _components(locs, scales)
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    return sum(w[j] * norm_cdf((x - locs[:, j:j + 1]) / scales[:, j:j + 1]) for j in range(locs.shape[1]))


def mixture_pdf(x, locs, scales, weights):
    locs, scales = _components(locs, scales)
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    return sum(w[j] * norm_pdf((x - locs[:, j:j + 1]) / scales[:, j:j + 1]) / scales[:, j:j + 1] for j in range(locs.shape[1]))


def mixture_quantile_batch(p, locs, scales, weights):
    """Quantile der Mischung zu den Stufen p (K,) für T Fälle: locs, scales (T, J), Ergebnis (T, K).

    Eingeklammertes Newton-Verfahren: Newton-Schritt, wo er im Klammerintervall bleibt, sonst Halbierung; die Klammer wird mit jedem Wert enger."""
    locs, scales = _components(locs, scales)
    p = np.asarray(p, dtype=float)
    w = np.asarray(weights, dtype=float)
    T, K = locs.shape[0], p.shape[0]
    lo = np.broadcast_to((locs - BRACKET_SD * scales).min(axis=1)[:, None], (T, K)).copy()
    hi = np.broadcast_to((locs + BRACKET_SD * scales).max(axis=1)[:, None], (T, K)).copy()
    x = np.broadcast_to((locs * w[None, :]).sum(axis=1)[:, None], (T, K)).copy()
    for _ in range(QUANTILE_ITERS):
        gap = mixture_cdf(x, locs, scales, w) - p[None, :]
        lo = np.where(gap < 0, x, lo)
        hi = np.where(gap < 0, hi, x)
        dens = mixture_pdf(x, locs, scales, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - gap / dens
        use = (dens > 0) & (newton > lo) & (newton < hi)
        new = np.where(use, newton, 0.5 * (lo + hi))
        done = np.all(np.abs(new - x) <= 1e-13 * (1.0 + np.abs(new)))
        x = new
        if done:
            break
    return x


def mixture_quantile(p, locs, scales, weights):
    """Quantile einer einzelnen Mischung zu den Stufen p (K,): Ergebnis (K,)."""
    return mixture_quantile_batch(p, np.asarray(locs, dtype=float).reshape(1, -1), np.asarray(scales, dtype=float).reshape(1, -1), weights)[0]


def vincent_quantile_batch(p, locs, scales, weights):
    """Quantilmittel Q(p) = Σ w_j (loc_j + scale_j·Φ⁻¹(p)) für T Fälle: Ergebnis (T, K). (Mittel der Quantile von Normalverteilungen ist wieder eine Normalverteilung.)"""
    locs, scales = _components(locs, scales)
    z = norm_ppf(np.asarray(p, dtype=float))
    w = np.asarray(weights, dtype=float)
    return (locs * w).sum(axis=1)[:, None] + (scales * w).sum(axis=1)[:, None] * z[None, :]


def mixture_moments(locs, scales, weights):
    """Mittel und Standardabweichung der Mischung von Normalverteilungen (Gesetz der totalen Varianz)."""
    locs = np.asarray(locs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float((w * locs).sum())
    var = float((w * (scales ** 2 + locs ** 2)).sum() - mean ** 2)
    return mean, math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------- Lognormal-Lieferanten (alle mit derselben Streuung s im Logarithmus)

def lognormal_mean(median, s):
    """Mittel einer Lognormalverteilung mit Median `median` und Log-Streuung s: median·exp(s²/2)."""
    return np.asarray(median, dtype=float) * np.exp(0.5 * s * s)


def lognormal_mixture_cdf(x, medians, s, weights):
    """P(Lieferzeit ≤ x), wenn jeder Auftrag mit Anteil w_j an den Lieferanten j mit Median m_j und Log-Streuung s geht."""
    x = np.asarray(x, dtype=float)
    medians = np.asarray(medians, dtype=float)
    w = np.asarray(weights, dtype=float)
    with np.errstate(divide="ignore"):
        lx = np.log(np.maximum(x, 1e-300))
    return sum(w[j] * norm_cdf((lx - math.log(medians[j])) / s) for j in range(len(medians))) * (x > 0)


def lognormal_mixture_pdf(x, medians, s, weights):
    x = np.asarray(x, dtype=float)
    medians = np.asarray(medians, dtype=float)
    w = np.asarray(weights, dtype=float)
    xs = np.maximum(x, 1e-300)
    return sum(w[j] * norm_pdf((np.log(xs) - math.log(medians[j])) / s) / (xs * s) for j in range(len(medians))) * (x > 0)


def lognormal_mixture_quantile(p, medians, s, weights):
    """Quantile der Mischung von Lognormalverteilungen (Bisektion/Newton im Logarithmus, dann exp)."""
    locs = np.log(np.asarray(medians, dtype=float))
    return np.exp(mixture_quantile(p, locs, np.full(len(locs), s), weights))


def lognormal_mixture_moments(medians, s, weights):
    """Mittel und Standardabweichung der Mischung von Lognormalverteilungen: E[X] = Σ w m e^(s²/2), E[X²] = Σ w m² e^(2s²)."""
    medians = np.asarray(medians, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float((w * medians).sum() * math.exp(0.5 * s * s))
    second = float((w * medians ** 2).sum() * math.exp(2.0 * s * s))
    return mean, math.sqrt(max(second - mean * mean, 0.0))


def lognormal_vincent_median(medians, weights):
    """Das Quantilmittel Σ w_j m_j·exp(s·Φ⁻¹(p)) gleicher Log-Streuung ist selbst lognormal: Median = Σ w_j m_j, dieselbe Log-Streuung s."""
    return float((np.asarray(weights, dtype=float) * np.asarray(medians, dtype=float)).sum())


def lognormal_vincent_moments(medians, s, weights):
    """Mittel und Standardabweichung des Quantilmittels gleich gestreuter Lognormalverteilungen (lognormal mit Median Σ w m)."""
    m = lognormal_vincent_median(medians, weights)
    mean = m * math.exp(0.5 * s * s)
    return mean, mean * math.sqrt(math.expm1(s * s))


def lognormal_cdf(x, median, s):
    return lognormal_mixture_cdf(x, [median], s, [1.0])


# ---------------------------------------------------------------- Prognosegüte

def crps_from_quantiles(Q, y, p):
    """CRPS aus einem Quantilraster: 2·Mittel der Pinball-Verluste über die Stufen p. Q: (T, K), y: (T,), p: (K,). Kleiner ist besser."""
    d = np.asarray(y, dtype=float)[:, None] - np.asarray(Q, dtype=float)
    p = np.asarray(p, dtype=float)[None, :]
    pinball = np.where(d >= 0, p * d, (p - 1.0) * d)
    return 2.0 * pinball.mean(axis=1)


def interval_covered(y, lower, upper):
    """Liegt y im geschlossenen Intervall [lower, upper]? (Feld von Wahrheitswerten)"""
    y = np.asarray(y, dtype=float)
    return (y >= np.asarray(lower, dtype=float)) & (y <= np.asarray(upper, dtype=float))


def quantile_grid(k):
    """Gleich verteilte Quantilstufen (i + 0,5)/k, i = 0 … k − 1 (Mittelpunktsregel)."""
    return (np.arange(k) + 0.5) / k


# ---------------------------------------------------------------- diskrete Mini-Verteilungen (Erklärbeispiel)

def quantile_mean_discrete(samples, weights):
    """Quantilmittel gleich großer diskreter Verteilungen: jede Stichprobe sortieren, dann gewichtet Platz für Platz mitteln."""
    arr = np.sort(np.asarray(samples, dtype=float), axis=1)
    return np.asarray(weights, dtype=float) @ arr


def mixture_discrete(samples, weights):
    """Mischung gleich großer diskreter Verteilungen: alle Werte zusammen, jeder mit Gewicht w_j/N. Rückgabe (sortierte Werte, Gewichte)."""
    arr = np.asarray(samples, dtype=float)
    w = np.repeat(np.asarray(weights, dtype=float) / arr.shape[1], arr.shape[1])
    values = arr.reshape(-1)
    order = np.argsort(values, kind="stable")
    return values[order], w[order]
