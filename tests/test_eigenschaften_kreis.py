"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Mitten gelten müssen, ohne dass man das richtige Ergebnis kennt.

Kreis: Drehung der Daten dreht jede Mitte mit (Rotationsäquivarianz), Spiegelung spiegelt sie, die Reihenfolge der Punkte ist gleichgültig, Mehrfachkopie der Daten ändert nichts.
Verteilungen: Verschiebung und Skalierung der Quellen wirken auf beide Mitten gleich; die Mischung ist von monotonen Umrechnungen unabhängig, das Quantilmittel nur von linearen."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_dist as D

RNG = np.random.default_rng(11)
DATA = [RNG.uniform(0, 2 * math.pi, int(RNG.integers(3, 15))) for _ in range(30)]
CENTRE_FUNCS = {
    "ext": lambda th: float(K.mean_direction(th)),
    "int": lambda th: K.frechet_mean(th)[0],
    "med": lambda th: K.circular_median(th)[0],
    "cov": lambda th: K.cover_arc_center(th)[0],
}


@pytest.mark.parametrize("name", list(CENTRE_FUNCS))
def test_rotation_moves_every_center_with_the_data(name):
    f = CENTRE_FUNCS[name]
    for th in DATA:
        turn = 1.234
        ref, moved = f(th), f((th + turn) % (2 * math.pi))
        if name == "med":                                      # Gleichstände wählen den kleineren Winkel; die Drehung ändert die Reihenfolge: nur eindeutige Fälle
            if K.circular_median(th)[2] > 1:
                continue
        if name == "int" and K.frechet_mean(th)[3] > 1:
            continue
        if name == "cov" and len(set(np.round(np.diff(np.sort(np.concatenate([th % (2 * math.pi), [th.min() % (2 * math.pi) + 2 * math.pi]])))[:], 9))) < len(th):
            continue
        assert float(K.arc_distance(moved, ref + turn)) < 1e-9


@pytest.mark.parametrize("name", ["ext", "int", "cov"])
def test_reflection_mirrors_every_center(name):
    f = CENTRE_FUNCS[name]
    for th in DATA:
        if name == "int" and K.frechet_mean(th)[3] > 1:
            continue
        gaps = np.diff(np.sort(np.concatenate([th % (2 * math.pi), [th.min() % (2 * math.pi) + 2 * math.pi]])))
        if name == "cov" and np.sum(gaps >= gaps.max() - 1e-9) > 1:
            continue
        assert float(K.arc_distance(f((-th) % (2 * math.pi)), -f(th))) < 1e-9


@pytest.mark.parametrize("name", list(CENTRE_FUNCS))
def test_the_order_of_the_points_does_not_matter(name):
    f = CENTRE_FUNCS[name]
    rng = np.random.default_rng(5)
    for th in DATA:
        assert float(K.arc_distance(f(th), f(rng.permutation(th)))) < 1e-9


def test_duplicating_the_data_does_not_change_the_intrinsic_center_or_the_resultant():
    for th in DATA[:15]:
        twice = np.concatenate([th, th])
        assert float(K.resultant_length(twice)) == pytest.approx(float(K.resultant_length(th)), abs=1e-12)
        c1, f1, _, _ = K.frechet_mean(th)
        c2, f2, _, _ = K.frechet_mean(twice)
        assert f2 == pytest.approx(f1, abs=1e-9)


def test_frechet_value_is_scale_free_in_the_unit_of_the_angle():
    """Stunden und Winkel liefern dieselbe Mitte (h2a/a2h vertauschen mit der Mitte)."""
    rng = np.random.default_rng(6)
    for _ in range(15):
        hours = rng.uniform(0, 24, 8)
        a = K.a2h(K.frechet_mean(K.h2a(hours))[0])
        assert float(K.hours_distance(a, K.a2h(K.frechet_mean(K.h2a((hours + 7.0) % 24.0))[0]) - 7.0)) < 1e-9


def test_resultant_length_is_between_zero_and_one_and_extreme_cases():
    for th in DATA:
        assert 0.0 <= float(K.resultant_length(th)) <= 1.0 + 1e-12
    assert float(K.resultant_length(np.full(7, 2.0))) == pytest.approx(1.0)


def test_mixture_and_quantile_mean_shift_with_the_sources_and_scale_linearly():
    p = np.array([0.1, 0.5, 0.9])
    locs, scales, w = np.array([[0.0, 3.0]]), np.array([[1.0, 2.0]]), [0.4, 0.6]
    mix, vin = D.mixture_quantile_batch(p, locs, scales, w), D.vincent_quantile_batch(p, locs, scales, w)
    assert D.mixture_quantile_batch(p, locs + 5.0, scales, w) == pytest.approx(mix + 5.0, abs=1e-9)
    assert D.vincent_quantile_batch(p, locs + 5.0, scales, w) == pytest.approx(vin + 5.0)
    assert D.mixture_quantile_batch(p, 2.0 * locs, 2.0 * scales, w) == pytest.approx(2.0 * mix, abs=1e-8)
    assert D.vincent_quantile_batch(p, 2.0 * locs, 2.0 * scales, w) == pytest.approx(2.0 * vin)


def test_quantile_functions_are_monotone():
    p = D.quantile_grid(50)
    rng = np.random.default_rng(7)
    for _ in range(10):
        locs, scales = rng.normal(0, 3, (1, 2)), rng.uniform(0.3, 2, (1, 2))
        assert np.all(np.diff(D.mixture_quantile_batch(p, locs, scales, [0.5, 0.5])[0]) > 0)
        assert np.all(np.diff(D.vincent_quantile_batch(p, locs, scales, [0.5, 0.5])[0]) > 0)


def test_mixture_commutes_with_monotone_maps_but_the_quantile_mean_does_not():
    """Mischung in log x und dann exp = Mischung in x der Lognormalverteilungen; das Quantilmittel in x ist größer als das in log x (Ungleichung zwischen arithmetischem und geometrischem Mittel)."""
    p = np.array([0.2, 0.5, 0.8, 0.95])
    medians, s = [10.0, 30.0], 0.4
    in_log = np.exp(D.mixture_quantile(p, np.log(medians), [s, s], [0.5, 0.5]))
    assert D.lognormal_mixture_cdf(in_log, medians, s, [0.5, 0.5]) == pytest.approx(p, abs=1e-12)
    arithmetic = 0.5 * (10.0 + 30.0) * np.exp(s * D.norm_ppf(p))
    geometric = np.sqrt(10.0 * 30.0) * np.exp(s * D.norm_ppf(p))
    assert np.all(arithmetic > geometric)


def test_crps_is_smallest_for_the_true_distribution_on_average():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 4000)
    p = D.quantile_grid(100)
    z = D.norm_ppf(p)
    def mean_crps(mu, sigma):
        return float(D.crps_from_quantiles(np.tile(mu + sigma * z, (len(y), 1)), y, p).mean())
    best = mean_crps(0.0, 1.0)
    assert best < mean_crps(0.5, 1.0) and best < mean_crps(0.0, 0.5) and best < mean_crps(0.0, 2.0)
