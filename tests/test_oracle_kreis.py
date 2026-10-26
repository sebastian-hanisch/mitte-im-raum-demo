"""Orakel für die Kreis-Einheiten (mir_circle.py): ein anderer Rechenweg je Einheit.

* Resultierende Länge und Mittelrichtung gegen `scipy.stats.circvar` / `circmean` (R = 1 − circvar).
* Fréchet-Mittel (p = 2) gegen die Aufzählung aller Entfaltungen: jedes lokale Minimum ist Mittelwert von θ_i + 2π·m_i mit m_i ∈ {0, 1};
  für n ≤ 6 werden alle 2^n Kandidaten mit der echten Bogenabstands-Zielfunktion bewertet und per Nachbarschaftstest auf lokales Minimum geprüft.
  Für größere n: globaler Zielwert gegen ein Feingitter.
* Zirkulärer Median (p = 1) gegen ein Feingitter, kleinster Bogen gegen die Aufzählung „Bogen beginnt bei Punkt i“.
Die Handinstanzen stehen in tests/test_circle.py; hier die Zufallsdatensätze (fester Seed, unter 10 Sekunden)."""

import itertools
import math

import numpy as np
import pytest

import mir_circle as K

TWO_PI = 2 * math.pi


def _objective(c, th, p=2):
    """Unabhängige Zielfunktion: Bogenabstand über min(|Δ| mod 2π, 2π − …), nicht über wrap_pi."""
    d = np.abs(th[:, None] - np.atleast_1d(c)[None, :]) % TWO_PI
    d = np.minimum(d, TWO_PI - d)
    return (d ** p).mean(axis=0)


def _datasets(seed, count, n_lo, n_hi):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        n = int(rng.integers(n_lo, n_hi + 1))
        kind = i % 3
        if kind == 0:
            th = rng.uniform(0, TWO_PI, n)
        elif kind == 1:
            th = rng.normal(rng.uniform(0, TWO_PI), 0.6, n) % TWO_PI
        else:
            th = np.where(rng.random(n) < 0.5, 0.0, math.pi) + rng.normal(0, 0.3, n)
        out.append(th % TWO_PI)
    return out


def test_resultant_and_direction_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    for th in _datasets(1, 200, 2, 40):
        assert float(K.resultant_length(th)) == pytest.approx(1.0 - stats.circvar(th), abs=1e-12)
        if K.resultant_length(th) > 1e-6:
            diff = float(K.arc_distance(K.mean_direction(th), stats.circmean(th)))
            assert diff < 1e-12


def test_frechet_minima_equal_the_enumeration_of_all_unfoldings():
    delta = 1e-6
    for th in _datasets(2, 60, 2, 6):
        n = len(th)
        found = []
        for m in itertools.product((0, 1), repeat=n):
            c = float(np.mean(th + TWO_PI * np.array(m)) % TWO_PI)
            f0 = _objective(c, th)[0]
            if f0 <= _objective(c + delta, th)[0] + 1e-13 and f0 <= _objective(c - delta, th)[0] + 1e-13:
                if not any(float(K.arc_distance(c, d)) < 1e-6 for d in found):
                    found.append(c)
        centres, values = K.frechet_minima(th)
        assert len(centres) == len(found), (th, centres, found)
        for c in centres:
            assert any(float(K.arc_distance(c, d)) < 1e-6 for d in found)
            assert _objective(c, th)[0] == pytest.approx(values[list(centres).index(c)], abs=1e-12)


def test_global_frechet_value_against_a_fine_grid():
    grid = np.linspace(0, TWO_PI, 40000, endpoint=False)
    for th in _datasets(3, 45, 2, 30):
        c, f, n_min, n_tied = K.frechet_mean(th)
        grid_min = _objective(grid, th).min()
        assert f <= grid_min + 1e-12
        assert grid_min - f < 1e-6
        assert _objective(c, th)[0] == pytest.approx(f, abs=1e-12)


def test_every_exact_minimum_is_a_local_minimum_of_the_true_objective():
    for th in _datasets(4, 60, 3, 25):
        centres, values = K.frechet_minima(th)
        assert len(centres) >= 1
        for c, v in zip(centres, values):
            for eps in (1e-5, -1e-5):
                assert _objective(c + eps, th)[0] >= v - 1e-13


def test_circular_median_against_a_fine_grid():
    grid = np.linspace(0, TWO_PI, 40000, endpoint=False)
    for th in _datasets(5, 45, 2, 30):
        c, f, ties = K.circular_median(th)
        grid_min = _objective(grid, th, p=1).min()
        assert f <= grid_min + 1e-12 and grid_min - f < 3e-4
        assert _objective(c, th, p=1)[0] == pytest.approx(f, abs=1e-12)


def test_cover_arc_against_the_enumeration_of_start_points():
    for th in _datasets(6, 60, 2, 25):
        lengths = []
        for start in th:
            ccw = (th - start) % TWO_PI                      # wie weit man von `start` gegen den Uhrzeigersinn bis zu jedem Punkt geht
            lengths.append(ccw.max())
        best = int(np.argmin(lengths))
        length_ref = lengths[best]
        centre, length = K.cover_arc_center(th)
        assert length == pytest.approx(length_ref, abs=1e-12)
        ties = [i for i, l in enumerate(lengths) if abs(l - length_ref) < 1e-9]
        if len(ties) == 1:
            assert float(K.arc_distance(centre, (th[best] + length_ref / 2) % TWO_PI)) < 1e-9
        assert float(K.arc_distance(centre, th).max()) <= length / 2 + 1e-9       # alle Punkte liegen im Bogen um die Mitte


def test_grid_search_would_count_minima_wrongly_but_the_enumeration_does_not():
    """Orakel-Fund der Vorarbeit: auf einem groben Gitter fallen schmale lokale Minima durch. Die Aufzählung zählt für 60 Zufallsdatensätze (n = 3 bis 12) genau so viele Minima
    wie ein sehr feines Gitter (400 000 Punkte)."""
    grid = np.linspace(0, TWO_PI, 400000, endpoint=False)
    mismatches = 0
    for th in _datasets(7, 60, 3, 12):
        f = _objective(grid, th)
        grid_minima = int(((f < np.roll(f, 1)) & (f <= np.roll(f, -1))).sum())
        exact = len(K.frechet_minima(th)[0])
        mismatches += grid_minima != exact
    assert mismatches == 0        # bei diesem feinen Gitter zählt auch das Gitter richtig (bei 720 Punkten nicht, siehe README); die Aufzählung ist die Referenz
