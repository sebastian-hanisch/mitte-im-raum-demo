"""Orakel für die Verteilungs-Einheiten (mir_dist.py): ein anderer Rechenweg je Einheit.

* Φ gegen `math.erfc`, Φ⁻¹ gegen `scipy.stats.norm.ppf`, Mischungsquantil gegen `scipy.optimize.brentq` auf der Mischungs-Verteilungsfunktion von scipy.
* Mischungsmomente und Lognormal-Momente gegen eine Ziehung aus der Definition (jeder Auftrag geht zufällig an eine Quelle) mit Toleranz aus dem Standardfehler.
* CRPS aus dem Quantilraster gegen die geschlossene Formel der Normalverteilung: σ·(z(2Φ(z) − 1) + 2φ(z) − 1/√π), z = (y − μ)/σ."""

import math

import numpy as np
import pytest

import mir_dist as D


def test_norm_cdf_against_math_erfc_over_the_whole_range():
    xs = np.linspace(-9.0, 9.0, 721)
    ref = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in xs])
    assert np.abs(D.norm_cdf(xs) - ref).max() < 1e-14
    xs2 = np.linspace(-1.0, 1.0, 41)                                                 # Übergang Reihe/Kettenbruch liegt bei |x/√2| = 2, also |x| = 2,83
    assert np.abs(D.norm_cdf(xs2 * 4.0) - np.array([0.5 * math.erfc(-x * 4.0 / math.sqrt(2)) for x in xs2])).max() < 1e-14
    xe = np.array([1.999999, 2.0, 2.000001])
    assert np.abs(D.erfc(xe) - np.array([math.erfc(x) for x in xe])).max() < 1e-15


def test_norm_ppf_against_scipy():
    stats = pytest.importorskip("scipy.stats")
    ps = np.concatenate([np.linspace(0.001, 0.999, 200), [1e-8, 1e-5, 1 - 1e-5, 1 - 1e-8]])
    assert np.abs(D.norm_ppf(ps) - stats.norm.ppf(ps)).max() < 1e-8


def test_mixture_quantile_against_brentq_on_scipy_cdf():
    stats = pytest.importorskip("scipy.stats")
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    for _ in range(40):
        k = int(rng.integers(2, 4))
        locs = rng.normal(0, 5, k)
        scales = rng.uniform(0.3, 3, k)
        w = rng.dirichlet(np.ones(k))
        p = rng.uniform(0.01, 0.99, 6)
        mine = D.mixture_quantile(p, locs, scales, w)

        def cdf(x):
            return float(sum(wj * stats.norm.cdf(x, lj, sj) for wj, lj, sj in zip(w, locs, scales)))

        for pi, qi in zip(p, mine):
            ref = optimize.brentq(lambda x: cdf(x) - pi, locs.min() - 10 * scales.max(), locs.max() + 10 * scales.max(), xtol=1e-13)
            assert qi == pytest.approx(ref, abs=1e-9)


def test_batch_quantiles_equal_single_cases():
    rng = np.random.default_rng(2)
    locs = rng.normal(0, 2, (5, 2))
    scales = rng.uniform(0.5, 2, (5, 2))
    p = np.array([0.05, 0.5, 0.95])
    batch = D.mixture_quantile_batch(p, locs, scales, [0.3, 0.7])
    for i in range(5):
        assert batch[i] == pytest.approx(D.mixture_quantile(p, locs[i], scales[i], [0.3, 0.7]), abs=1e-10)


def test_mixture_moments_and_lognormal_moments_against_a_draw_from_the_definition():
    rng = np.random.default_rng(3)
    n = 100000
    pick = rng.random(n) < 0.35                                                      # 35 % Quelle B
    s = 0.3
    x = np.where(pick, rng.lognormal(math.log(30.0), s, n), rng.lognormal(math.log(10.0), s, n))
    mean, sd = D.lognormal_mixture_moments([10.0, 30.0], s, [0.65, 0.35])
    assert x.mean() == pytest.approx(mean, rel=0.015) and x.std() == pytest.approx(sd, rel=0.03)
    share = ((x > 15) & (x < 25)).mean()
    assert share == pytest.approx(float(D.lognormal_mixture_cdf(25.0, [10.0, 30.0], s, [0.65, 0.35]) - D.lognormal_mixture_cdf(15.0, [10.0, 30.0], s, [0.65, 0.35])), abs=0.005)
    # Quantilmittel: gemeinsames z für beide Quellen (Σ w m·exp(s z) mit einem einzigen z)
    z = rng.normal(0.0, 1.0, n)
    y = 0.65 * 10.0 * np.exp(s * z) + 0.35 * 30.0 * np.exp(s * z)
    vmean, vsd = D.lognormal_vincent_moments([10.0, 30.0], s, [0.65, 0.35])
    assert y.mean() == pytest.approx(vmean, rel=0.015) and y.std() == pytest.approx(vsd, rel=0.03)
    # Normalmischung
    xn = np.where(rng.random(n) < 0.4, rng.normal(3.0, 2.0, n), rng.normal(-1.0, 0.5, n))
    nm, nsd = D.mixture_moments([3.0, -1.0], [2.0, 0.5], [0.4, 0.6])
    assert xn.mean() == pytest.approx(nm, abs=0.02) and xn.std() == pytest.approx(nsd, rel=0.01)


def test_crps_from_quantile_grid_against_the_closed_form_for_normals():
    def phi(z):
        return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

    def closed(mu, sigma, y):
        z = (y - mu) / sigma
        return sigma * (z * (2 * float(D.norm_cdf(z)) - 1) + 2 * phi(z) - 1 / math.sqrt(math.pi))

    p = D.quantile_grid(400)
    z = D.norm_ppf(p)
    rng = np.random.default_rng(4)
    for _ in range(30):
        mu, sigma, y = rng.normal(0, 2), rng.uniform(0.5, 3), rng.normal(0, 3)
        mine = D.crps_from_quantiles((mu + sigma * z)[None, :], np.array([y]), p)[0]
        assert mine == pytest.approx(closed(mu, sigma, y), abs=1e-3 * sigma)  # Rasterfehler wächst, wenn y weit außerhalb liegt (Quantile jenseits 0,125 % fehlen)
    assert closed(0.0, 1.0, 0.0) == pytest.approx(math.sqrt(2) / 2 * 0 + (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)))    # Handwert der Formel bei y = μ
