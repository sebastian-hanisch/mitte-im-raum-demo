"""Die Mitte von Zeitpunkten auf dem 24-Stunden-Kreis: Einheiten (DEMO-PLAYBOOK §1b), alle numpy-only und über die letzte Achse vektorisiert.

Winkel im Bogenmaß, Stunden h gehören zum Winkel 2π·h/24. Eine Mitte ist der Minimierer von Σ d(θ_i, c)^p mit dem Bogenabstand d auf dem Kreis
(Fréchet-Mittel, in dieser Demo nur als Name für diese Definition benutzt):
p = 2 intrinsisches Fréchet-Mittel (`frechet_mean_batch`), p = 1 zirkulärer Median (`circular_median_batch`), p = ∞ Mitte des kleinsten Bogens,
der alle Punkte enthält (`cover_arc_batch`). Dazu das extrinsische Kreismittel (`mean_direction`, Richtung der Resultierenden) und das arithmetische Mittel
der Uhrzeiten mit einem Schnittpunkt des Tages (`arithmetic_mean_with_cut`).

Das Fréchet-Mittel auf dem Kreis wird **exakt aufgezählt**, nicht auf einem Gitter gesucht: Zwischen je zwei aufeinanderfolgenden Gegenpunkten θ_i + π ist
F(c) = mean((c − a_i)²) mit festen entfalteten Winkeln a_i, also quadratisch mit Minimum bei mean(a_i); ein lokales Minimum von F ist genau ein solches Minimum, das im
eigenen Bogen liegt (an den Gegenpunkten selbst hat F einen nach oben zeigenden Knick, dort liegt nie ein Minimum)."""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
NO_DIRECTION = 1e-12     # unter dieser resultierenden Länge hat der Winkel keine Bedeutung (Rundungsrauschen)
MIN_ARC = 1e-12          # kürzere Bögen zwischen zwei Gegenpunkten (doppelte Punkte) zählen nicht
ARC_TOL = 1e-12          # Toleranz beim Test „Minimum liegt im eigenen Bogen“
TIE_TOL = 1e-9           # Werte der Zielfunktion, die sich höchstens so unterscheiden, gelten als gleich (Gleichstand)
CHUNK = 2_000_000        # höchstens so viele Zahlen je Zwischenfeld (Speicher)


# ---------------------------------------------------------------- Winkel, Stunden, Abstände

def h2a(hours):
    """Uhrzeit in Stunden -> Winkel im Bogenmaß (24 h = 2π)."""
    return np.asarray(hours, dtype=float) * (TWO_PI / 24.0)


def a2h(angle):
    """Winkel im Bogenmaß -> Uhrzeit in Stunden aus [0, 24)."""
    return (np.asarray(angle, dtype=float) % TWO_PI) * (24.0 / TWO_PI)


def wrap_pi(angle):
    """Winkel auf (−π, π] abbilden."""
    return math.pi - ((math.pi - np.asarray(angle, dtype=float)) % TWO_PI)


def arc_distance(a, b):
    """Bogenabstand auf dem Kreis, aus [0, π]."""
    return np.abs(wrap_pi(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def hours_distance(a, b):
    """Abstand zweier Uhrzeiten in Stunden um die Mitternacht herum, aus [0, 12]."""
    return np.abs(((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 12.0) % 24.0) - 12.0)


# ---------------------------------------------------------------- extrinsisch: Richtung und Länge der Resultierenden

def resultant_components(theta):
    """Mittelwerte von cos und sin entlang der letzten Achse."""
    theta = np.asarray(theta, dtype=float)
    return np.cos(theta).mean(axis=-1), np.sin(theta).mean(axis=-1)


def resultant_length(theta):
    """Resultierende Länge R aus [0, 1]: 1, wenn alle Punkte zusammenfallen, 0 bei vollkommenem Ausgleich."""
    c, s = resultant_components(theta)
    return np.hypot(c, s)


def mean_direction(theta):
    """Extrinsisches Kreismittel: Richtung der Resultierenden, in [0, 2π). Unter R < 1e-12 ist die Richtung nicht definiert; geliefert wird dann 0."""
    c, s = resultant_components(theta)
    angle = np.arctan2(s, c) % TWO_PI
    return np.where(np.hypot(c, s) < NO_DIRECTION, 0.0, angle)


def rayleigh_p(R, n):
    """Einfacher Rayleigh-Test (asymptotisch): P(n·R² ≥ z) = exp(−z) unter Gleichverteilung; nach oben auf 1 begrenzt."""
    return np.minimum(1.0, np.exp(-np.asarray(n, dtype=float) * np.asarray(R, dtype=float) ** 2))


def rayleigh_threshold(n, level=0.05):
    """Schwelle für R, über der der einfache Rayleigh-Test zum Niveau `level` verwirft: R = √(−ln(level)/n); bei level 0,05 ist das √(2,996/n) ≈ √(3/n)."""
    return np.sqrt(-math.log(level) / np.asarray(n, dtype=float))


# ---------------------------------------------------------------- Fréchet p = 2, exakt aufgezählt

def frechet_value(c, theta, p=2):
    """Zielfunktion F(c) = mean_i d(θ_i, c)^p für eine Mitte c (Zahl oder Feld) und die Punkte θ (eine Achse)."""
    theta = np.asarray(theta, dtype=float)
    c = np.atleast_1d(np.asarray(c, dtype=float))
    return np.mean(arc_distance(theta[:, None], c[None, :]) ** p, axis=0)


def _frechet_arcs(theta):
    """Für jeden Bogen zwischen zwei Gegenpunkten: Stelle des quadratischen Minimums (entfaltet), Wert dort, Grenzen, Gültigkeit.

    theta: Form (T, n). Rückgabe (cstar, F, lo, hi, valid), alle (T, n)."""
    T, n = theta.shape
    th = theta % TWO_PI
    antipodes = np.sort((th + math.pi) % TWO_PI, axis=1)
    lo = antipodes
    hi = np.concatenate([antipodes[:, 1:], antipodes[:, :1] + TWO_PI], axis=1)
    mid = 0.5 * (lo + hi)
    unfolded = mid[:, :, None] - wrap_pi(mid[:, :, None] - th[:, None, :])    # (T, Bögen, n): a_i, so dass |mid − a_i| ≤ π
    cstar = unfolded.mean(axis=2)
    value = ((cstar[:, :, None] - unfolded) ** 2).mean(axis=2)
    valid = (hi - lo > MIN_ARC) & (cstar >= lo - ARC_TOL) & (cstar <= hi + ARC_TOL)
    return cstar, value, lo, hi, valid


def frechet_mean_batch(theta):
    """Globales intrinsisches Fréchet-Mittel (p = 2) je Datensatz; theta: (T, n).

    Rückgabe (c, F, n_minima, n_tied): Mitte in [0, 2π), Zielwert, Zahl der lokalen Minima, Zahl der Minima mit (fast) gleichem Zielwert.
    Bei Gleichstand gilt das Minimum mit dem kleinsten Winkel."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    T, n = theta.shape
    c = np.empty(T)
    best = np.empty(T)
    n_min = np.empty(T, dtype=int)
    n_tie = np.empty(T, dtype=int)
    step = max(1, CHUNK // (n * n))
    for s in range(0, T, step):
        cstar, value, _, _, valid = _frechet_arcs(theta[s:s + step])
        vmin = np.where(valid, value, np.inf).min(axis=1)
        tied = valid & (value <= vmin[:, None] + TIE_TOL)
        angle = np.where(tied, cstar % TWO_PI, np.inf)
        c[s:s + step] = angle.min(axis=1)
        best[s:s + step] = vmin
        n_min[s:s + step] = valid.sum(axis=1)
        n_tie[s:s + step] = tied.sum(axis=1)
    return c, best, n_min, n_tie


def frechet_minima(theta):
    """Alle lokalen Minima der Zielfunktion p = 2 für einen Datensatz: (Mitten in [0, 2π), Zielwerte), nach Zielwert, dann Winkel sortiert; doppelte entfernt."""
    theta = np.asarray(theta, dtype=float).reshape(1, -1)
    cstar, value, _, _, valid = _frechet_arcs(theta)
    centres = (cstar[valid] % TWO_PI)
    values = value[valid]
    order = np.lexsort((centres, np.round(values, 9)))
    centres, values = centres[order], values[order]
    keep = []
    for i, ci in enumerate(centres):
        if not any(arc_distance(ci, centres[j]) < 1e-9 for j in keep):
            keep.append(i)
    return centres[keep], values[keep]


def frechet_mean(theta):
    """Intrinsisches Fréchet-Mittel eines Datensatzes: (Mitte, Zielwert, Zahl lokaler Minima, Zahl gleich guter Minima)."""
    c, f, n_min, n_tie = frechet_mean_batch(np.asarray(theta, dtype=float).reshape(1, -1))
    return float(c[0]), float(f[0]), int(n_min[0]), int(n_tie[0])


# ---------------------------------------------------------------- p = 1 (zirkulärer Median) und p = ∞ (kleinster Bogen)

def circular_median_batch(theta):
    """Zirkulärer Median (p = 1) je Datensatz: der Datenpunkt mit der kleinsten mittleren Bogenentfernung zu allen Punkten.

    Die Zielfunktion ist stückweise linear; ihre Minima liegen auf Datenpunkten. Rückgabe (c, F1, n_tied) mit der Zahl der Datenpunkte mit (fast) gleichem Zielwert;
    bei Gleichstand gilt der Punkt mit dem kleinsten Winkel (die Punkte werden vorher sortiert, das Ergebnis hängt nicht an der Eingabereihenfolge)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    T, n = theta.shape
    c = np.empty(T)
    value = np.empty(T)
    ties = np.empty(T, dtype=int)
    step = max(1, CHUNK // (n * n))
    for s in range(0, T, step):
        th = np.sort(theta[s:s + step] % TWO_PI, axis=1)
        mean_dist = arc_distance(th[:, :, None], th[:, None, :]).mean(axis=2)     # (t, n): mittlere Entfernung von Punkt i zu allen
        tied = mean_dist <= mean_dist.min(axis=1)[:, None] + TIE_TOL
        k = np.argmax(tied, axis=1)                                              # erster gleich guter Punkt in aufsteigender Winkelreihenfolge
        rows = np.arange(len(th))
        c[s:s + step] = th[rows, k]
        value[s:s + step] = mean_dist[rows, k]
        ties[s:s + step] = tied.sum(axis=1)
    return c, value, ties


def circular_median(theta):
    """Zirkulärer Median eines Datensatzes: (Mitte, mittlere Bogenentfernung, Zahl gleich guter Datenpunkte)."""
    c, f, t = circular_median_batch(np.asarray(theta, dtype=float).reshape(1, -1))
    return float(c[0]), float(f[0]), int(t[0])


def cover_arc_batch(theta):
    """Mitte des kleinsten Bogens, der alle Punkte enthält (p = ∞), und seine Länge; theta: (T, n).

    Der Bogen ist das Komplement der größten Lücke zwischen zwei benachbarten Punkten. Bei gleich großen Lücken gilt die erste (nach Winkel sortiert)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    T, n = theta.shape
    s = np.sort(theta % TWO_PI, axis=1)
    gaps = np.diff(np.concatenate([s, s[:, :1] + TWO_PI], axis=1), axis=1)       # Lücke nach Punkt k, die letzte schließt den Kreis
    k = np.argmax(gaps, axis=1)
    rows = np.arange(T)
    start = s[rows, (k + 1) % n]
    length = TWO_PI - gaps[rows, k]
    return (start + 0.5 * length) % TWO_PI, length


def cover_arc_center(theta):
    """Mitte und Länge des kleinsten Bogens, der alle Punkte eines Datensatzes enthält."""
    c, length = cover_arc_batch(np.asarray(theta, dtype=float).reshape(1, -1))
    return float(c[0]), float(length[0])


# ---------------------------------------------------------------- arithmetisches Mittel der Uhrzeiten

def arithmetic_mean_with_cut(hours, cut=0.0):
    """Arithmetisches Mittel der Uhrzeiten, wenn der Tag bei `cut` Uhr beginnt (Werte aus [cut, cut + 24)), als Uhrzeit in [0, 24). Letzte Achse."""
    h = np.asarray(hours, dtype=float)
    shifted = (h - cut) % 24.0 + cut
    return shifted.mean(axis=-1) % 24.0


# ---------------------------------------------------------------- die fünf Mitten zusammen

CENTRE_NAMES = ("Arithmetisch (Tag ab 0 Uhr)", "Kreismittel (arctan2)", "Fréchet p = 2", "Zirkulärer Median p = 1", "Kleinster Bogen p = ∞")


def five_centres(hours, cut=0.0):
    """Die fünf Mitten (in Stunden, [0, 24)) für einen Stapel von Nächten, hours: (T, n). Dazu die resultierende Länge R.

    Rückgabe (dict Name -> (T,)-Feld, R). Das arithmetische Mittel nutzt den Tagesbeginn `cut` (Name bleibt, der Schnittpunkt ist Teil der Frage)."""
    hours = np.atleast_2d(np.asarray(hours, dtype=float))
    theta = h2a(hours)
    centres = {
        CENTRE_NAMES[0]: arithmetic_mean_with_cut(hours, cut),
        CENTRE_NAMES[1]: a2h(mean_direction(theta)),
        CENTRE_NAMES[2]: a2h(frechet_mean_batch(theta)[0]),
        CENTRE_NAMES[3]: a2h(circular_median_batch(theta)[0]),
        CENTRE_NAMES[4]: a2h(cover_arc_batch(theta)[0]),
    }
    return centres, resultant_length(theta)
