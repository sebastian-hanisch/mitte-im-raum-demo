"""Szenario: Ankunftszeiten am Tor (Kreis) und Prognosen mit bekannter Wahrheit (Verteilungen), erzeugt aus dem eigenen SplitMix64-Strom (mir_rng.py).

`Draws` zieht Zahlen blockweise (numpy, uint64 rechnet modulo 2^64) und liefert exakt den Strom der wörtlich kopierten Klasse `SplitMix64`;
tests/test_scenario.py vergleicht beide. Damit sind alle Instanzen auf jeder Plattform gleich und Messreihen über viele Instanzen bleiben schnell.

Erzeuger Kreis: **wrapped normal** (eine Normalverteilung auf der Zahlengeraden, dann modulo 24 Stunden bzw. 2π gefaltet). Ihre resultierende Länge ist bekannt:
ρ = exp(−σ²/2) für die Streuung σ im Bogenmaß. Eine Von-Mises-Verteilung wird nicht verwendet.
Erzeuger Verteilungen: Lieferanten sind lognormal (geschlossene Formeln, kein Zufall); Prognosen ziehen die wahre Lage m, die Nachfrage y und den Fehler je Prognose."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from mir_rng import MASK

GAMMA = 0x9E3779B97F4A7C15
TWO_PI = 2.0 * math.pi


class Draws:
    """SplitMix64-Strom, blockweise gezogen: gleicher Seed, gleiche Zahlen wie `mir_rng.SplitMix64`, aber ohne Python-Schleife."""

    def __init__(self, seed: int):
        self.state = int(seed) & MASK

    def raw(self, count: int) -> np.ndarray:
        """Die nächsten `count` Ausgaben des Stroms als uint64-Feld."""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GAMMA) + np.uint64(self.state)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + count * GAMMA) & MASK
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        size = int(np.prod(shape))
        return ((self.raw(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormal nach Box-Muller: zwei gleichverteilte Zahlen ergeben ein Paar (Kosinus-Wert zuerst)."""
        size = int(np.prod(shape))
        pairs = (size + 1) // 2
        u = self.uniform((2, pairs))
        radius = np.sqrt(-2.0 * np.log(1.0 - u[0]))        # 1 − u liegt in (0, 1]: kein log(0)
        angle = TWO_PI * u[1]
        z = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return z[:size].reshape(shape)


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Messzelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte); ein Seed gehört zu einer Zelle, nie zu einer Instanz allein."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


# ---------------------------------------------------------------- Kreis

def sigma_from_rho(rho: float) -> float:
    """Streuung σ (Bogenmaß) der wrapped normal mit resultierender Länge ρ: σ = √(−2 ln ρ)."""
    return math.sqrt(-2.0 * math.log(rho))


def rho_from_sigma(sigma: float) -> float:
    """Resultierende Länge der wrapped normal mit Streuung σ (Bogenmaß): ρ = exp(−σ²/2)."""
    return math.exp(-0.5 * sigma * sigma)


def uniform_angles(draws: Draws, runs: int, n: int) -> np.ndarray:
    """`runs` Datensätze mit je n gleichverteilten Winkeln aus [0, 2π)."""
    return draws.uniform((runs, n)) * TWO_PI


def wrapped_angles(draws: Draws, runs: int, n: int, rho: float, centre: float = 0.0) -> np.ndarray:
    """`runs` Datensätze mit je n Winkeln aus einer wrapped normal um `centre` mit wahrer resultierender Länge ρ."""
    return (centre + sigma_from_rho(rho) * draws.normal((runs, n))) % TWO_PI


def two_cluster_angles(draws: Draws, runs: int, n: int, delta_deg: float, n_first, sigma_deg: float) -> np.ndarray:
    """`runs` Datensätze mit zwei Häufungen: `n_first` Winkel um 0° (Zahl oder Feld der Länge runs), die übrigen n − n_first um `delta_deg`, beide wrapped normal mit Streuung `sigma_deg` (Grad).
    Zieht immer dieselbe Zahl von Normalwerten, damit zwei Zählungen (n_first, n_first ± 1) mit gleichem Strom gepaart werden können."""
    z = draws.normal((runs, n)) * math.radians(sigma_deg)
    first = np.asarray(n_first).reshape(-1, 1) if np.ndim(n_first) else n_first
    centre = np.where(np.arange(n)[None, :] < first, 0.0, math.radians(delta_deg))
    return (centre + z) % TWO_PI


def binomial_counts(draws: Draws, runs: int, n: int, p: float) -> np.ndarray:
    """Je Datensatz die Zahl der Erfolge von n Münzwürfen mit Wahrscheinlichkeit p (Binomial(n, p)), aus n Gleichverteilten je Datensatz."""
    return (draws.uniform((runs, n)) < p).sum(axis=1)


def gate_hours(draws: Draws, runs: int, n: int, mu_h, sigma_h: float) -> np.ndarray:
    """`runs` Nächte mit je n Ankünften (Uhrzeit in Stunden aus [0, 24)): wrapped normal um die Mitte mu_h (Zahl oder Feld der Länge runs) mit Streuung sigma_h Stunden."""
    mu = np.asarray(mu_h, dtype=float)
    if mu.ndim == 1:
        mu = mu[:, None]
    return (mu + sigma_h * draws.normal((runs, n))) % 24.0


def two_peak_hours(draws: Draws, runs: int, n: int, mu_h: float, gap_h: float, n_second, sigma_h: float) -> np.ndarray:
    """Nächte mit zwei Gipfeln: die ersten n − n_second Ankünfte um mu_h, die übrigen n_second (Zahl oder Feld der Länge runs) um mu_h + gap_h; Streuung sigma_h.
    Die Reihenfolge in der Nacht ist unerheblich."""
    z = draws.normal((runs, n)) * sigma_h
    second = np.asarray(n_second).reshape(-1, 1) if np.ndim(n_second) else n_second
    centre = np.where(np.arange(n)[None, :] < n - second, mu_h, mu_h + gap_h)
    return (centre + z) % 24.0


def second_peak_count(n: int, share_percent: float) -> int:
    """Zahl der Ankünfte im zweiten Gipfel: runden(n · Anteil/100), höchstens n."""
    return int(min(n, max(0, round(n * share_percent / 100.0))))


@dataclass(frozen=True)
class Night:
    hours: np.ndarray          # Uhrzeiten der Ankünfte (n,)
    n_second: int              # davon im zweiten Gipfel
    truth: float | None        # wahre Mitte in Stunden bei einem Gipfel, sonst None (zwei Gipfel haben keine wahre Mitte)


def make_night(n: int, mu_h: float, sigma_h: float, share_percent: float, gap_h: float, seed: int) -> Night:
    """Eine Nacht für die Hauptansicht, deterministisch aus den Einstellungen."""
    k = second_peak_count(n, share_percent)
    hours = two_peak_hours(Draws(cell_seed(seed, "nacht", n, mu_h, sigma_h, k, gap_h)), 1, n, mu_h, gap_h, k, sigma_h)[0]
    return Night(hours, k, float(mu_h) % 24.0 if k == 0 else None)


# ---------------------------------------------------------------- Prognosen mit bekannter Wahrheit

def forecast_cases(draws: Draws, cases: int, s_e: float, spread_y: float = 1.0, spread_m: float = 1.0):
    """Wahre Lage m ~ N(0, spread_m²), Nachfrage y = m + spread_y·ε und die Lagefehler zweier Prognosen e_i ~ N(0, s_e²).

    Rückgabe (m, y, mu) mit mu = m + e_i, Form (cases, 2). Die Ziehung ist von der gemeldeten Streuung unabhängig, so dass überzuversichtliche und ehrliche
    Meldung auf denselben Daten verglichen werden (gepaarter Vergleich)."""
    z = draws.normal((cases, 4))
    m = spread_m * z[:, 0]
    y = m + spread_y * z[:, 1]
    mu = m[:, None] + s_e * z[:, 2:4]
    return m, y, mu


def reported_sigma(calibration: str, s_e: float, spread_y: float = 1.0) -> float:
    """Streuung, die jede Prognose meldet: „überzuversichtlich“ = nur die Streuung der Nachfrage (ohne den eigenen Lagefehler), „ehrlich“ = √(Nachfrage² + Lagefehler²)."""
    if calibration == "ehrlich":
        return math.sqrt(spread_y ** 2 + s_e ** 2)
    if calibration == "überzuversichtlich":
        return spread_y
    raise ValueError(f"unbekannte Meldung: {calibration}")
