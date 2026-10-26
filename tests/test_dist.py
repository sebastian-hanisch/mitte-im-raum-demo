"""Die Verteilungs-Einheiten gegen Handrechnung (Mini-Instanzen in tests/conftest.py)."""

import math

import numpy as np
import pytest

import mir_dist as D


def test_normal_cdf_and_ppf_by_hand():
    assert float(D.norm_cdf(0.0)) == pytest.approx(0.5, abs=1e-15)
    assert float(D.norm_cdf(1.0)) + float(D.norm_cdf(-1.0)) == pytest.approx(1.0, abs=1e-15)
    assert float(D.norm_cdf(1.959964)) == pytest.approx(0.975, abs=1e-6)             # Tabellenwert des 97,5-%-Quantils
    assert float(D.norm_ppf(0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(D.norm_ppf(0.975)) == pytest.approx(1.959964, abs=1e-6)
    assert float(D.norm_pdf(0.0)) == pytest.approx(1 / math.sqrt(2 * math.pi))
    with pytest.raises(ValueError):
        D.norm_ppf(1.0)
    with pytest.raises(ValueError):
        D.norm_ppf(np.array([0.2, 0.0]))


def test_mixture_of_two_far_normals_by_hand():
    """N(0, 1) und N(10, 1) zu gleichen Teilen: Mittel 5, Varianz 1 + 25 = 26; Median 5 (Symmetrie); Quantilmittel N(5, 1)."""
    locs, scales, w = [0.0, 10.0], [1.0, 1.0], [0.5, 0.5]
    mean, sd = D.mixture_moments(locs, scales, w)
    assert mean == pytest.approx(5.0) and sd == pytest.approx(math.sqrt(26.0))
    p = np.array([0.025, 0.5, 0.975])
    assert D.mixture_quantile(p, locs, scales, w)[1] == pytest.approx(5.0, abs=1e-9)
    assert float(D.mixture_cdf(np.array([[5.0]]), locs, scales, w)[0, 0]) == pytest.approx(0.5, abs=1e-12)
    vin = D.vincent_quantile_batch(p, [locs], scales, w)[0]
    assert vin == pytest.approx([5.0 - 1.959964, 5.0, 5.0 + 1.959964], abs=1e-6)
    mix = D.mixture_quantile(p, locs, scales, w)
    assert mix[0] < 2.0 and mix[2] > 8.0                                            # die Mischung hat ihre Masse an beiden Enden, nicht in der Mitte


def test_quantile_mean_and_mixture_of_discrete_mini_distributions():
    samples = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    assert D.quantile_mean_discrete(samples, [0.5, 0.5]) == pytest.approx([5.5, 11.0, 16.5])
    assert D.quantile_mean_discrete([[3.0, 1.0, 2.0], [30.0, 10.0, 20.0]], [0.5, 0.5]) == pytest.approx([5.5, 11.0, 16.5])    # Reihenfolge der Eingabe egal
    values, weights = D.mixture_discrete(samples, [0.5, 0.5])
    assert values.tolist() == [1.0, 2.0, 3.0, 10.0, 20.0, 30.0] and weights == pytest.approx([1 / 6] * 6) and weights.sum() == pytest.approx(1.0)
    assert (values * weights).sum() == pytest.approx(11.0)                          # gleiches Mittel wie das Quantilmittel (5,5 + 11 + 16,5)/3
    assert D.quantile_mean_discrete(samples, [0.5, 0.5]).mean() == pytest.approx(11.0)


def test_lognormal_suppliers_closed_forms_by_hand():
    """Lieferanten mit Median 10 und 30 Tage, Log-Streuung 0,15, je 50 %: Mischungsmittel 20·e^(0,15²/2), Quantilmittel lognormal mit Median 20."""
    s = 0.15
    mean, sd = D.lognormal_mixture_moments([10.0, 30.0], s, [0.5, 0.5])
    assert mean == pytest.approx(20 * math.exp(0.01125))
    assert sd == pytest.approx(math.sqrt(500 * math.exp(0.045) - (20 * math.exp(0.01125)) ** 2))
    vmean, vsd = D.lognormal_vincent_moments([10.0, 30.0], s, [0.5, 0.5])
    assert D.lognormal_vincent_median([10.0, 30.0], [0.5, 0.5]) == 20.0
    assert vmean == pytest.approx(mean) and vsd == pytest.approx(vmean * math.sqrt(math.exp(s * s) - 1))
    assert vsd < sd / 3                                                              # gleiches Mittel, ganz andere Streuung
    # Anteil der Masse in (15, 25): Quantilmittel Φ(ln(25/20)/s) − Φ(ln(15/20)/s)
    share_vin = float(D.lognormal_cdf(25.0, 20.0, s) - D.lognormal_cdf(15.0, 20.0, s))
    assert share_vin == pytest.approx(float(D.norm_cdf(math.log(25 / 20) / s) - D.norm_cdf(math.log(15 / 20) / s)))
    share_mix = float(D.lognormal_mixture_cdf(25.0, [10.0, 30.0], s, [0.5, 0.5]) - D.lognormal_mixture_cdf(15.0, [10.0, 30.0], s, [0.5, 0.5]))
    assert share_mix < 0.1 < 0.8 < share_vin


def test_lognormal_quantile_inverts_the_cdf_and_is_scale_equivariant():
    q = D.lognormal_mixture_quantile(np.array([0.1, 0.5, 0.95]), [10.0, 30.0], 0.15, [0.5, 0.5])
    assert D.lognormal_mixture_cdf(q, [10.0, 30.0], 0.15, [0.5, 0.5]) == pytest.approx([0.1, 0.5, 0.95], abs=1e-12)
    q2 = D.lognormal_mixture_quantile(np.array([0.1, 0.5, 0.95]), [20.0, 60.0], 0.15, [0.5, 0.5])
    assert q2 == pytest.approx(2 * q)
    assert float(D.lognormal_mixture_cdf(0.0, [10.0], 0.2, [1.0])) == 0.0
    assert float(D.lognormal_mixture_pdf(-1.0, [10.0], 0.2, [1.0])) == 0.0


def test_crps_by_hand():
    """Konstante Quantile c: CRPS = |y − c|. Gleichverteilung U(0, 1) auf dem Raster mit vier Stufen und y = 0,5: 0,09375 (Rasterwert); feines Raster nähert sich 1/12."""
    assert D.crps_from_quantiles(np.array([[3.0, 3.0, 3.0, 3.0]]), np.array([5.0]), D.quantile_grid(4)) == pytest.approx([2.0])
    assert D.crps_from_quantiles(np.array([[3.0, 3.0, 3.0, 3.0]]), np.array([1.0]), D.quantile_grid(4)) == pytest.approx([2.0])
    p4 = D.quantile_grid(4)
    assert D.crps_from_quantiles(p4[None, :], np.array([0.5]), p4) == pytest.approx([0.09375])
    p400 = D.quantile_grid(400)
    assert D.crps_from_quantiles(p400[None, :], np.array([0.5]), p400)[0] == pytest.approx(1 / 12, abs=1e-4)


def test_quantile_grid_and_interval_coverage():
    assert D.quantile_grid(4).tolist() == [0.125, 0.375, 0.625, 0.875]
    covered = D.interval_covered(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0] * 4), np.array([2.0] * 4))
    assert covered.tolist() == [False, True, True, False]                            # beide Ränder gehören zum Intervall


def test_lognormal_mean_and_the_density_and_cdf_below_one_day():
    """Mittel = Median·exp(s²/2); Verteilungsfunktion und Dichte sind auch zwischen 0 und 1 positiv (Median 2, s = 0,5: Φ(ln(0,5)/0,5) = Φ(−1,386))."""
    assert float(D.lognormal_mean(10.0, 0.15)) == pytest.approx(10.0 * math.exp(0.01125))
    assert D.lognormal_mean(np.array([10.0, 30.0]), 0.0).tolist() == [10.0, 30.0]
    cdf = float(D.lognormal_mixture_cdf(1.0, [2.0], 0.5, [1.0]))
    assert cdf == pytest.approx(0.5 * math.erfc(1.3862943611198906 / math.sqrt(2)), abs=1e-14) and cdf > 0.08
    pdf = float(D.lognormal_mixture_pdf(1.0, [2.0], 0.5, [1.0]))
    assert pdf == pytest.approx(math.exp(-0.5 * (math.log(0.5) / 0.5) ** 2) / (math.sqrt(2 * math.pi) * 0.5)) and pdf > 0.1
    assert float(D.lognormal_mixture_cdf(0.5, [2.0], 0.5, [1.0])) > 0 and float(D.lognormal_mixture_pdf(0.5, [2.0], 0.5, [1.0])) > 0
