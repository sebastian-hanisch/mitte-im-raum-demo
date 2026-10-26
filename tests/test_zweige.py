"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): Jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab. Sonst wäre sie ein toter Zweig,
dessen Spalte in den Messreihen lauter Nullen oder lauter Gleiches zeigt."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_dist as D
import mir_experiments as X
import mir_scenario as S


def test_arithmetic_and_circle_mean_differ_on_the_night_shift(uhr4):
    centres, _ = K.five_centres(uhr4[None, :])
    assert float(K.hours_distance(centres[K.CENTRE_NAMES[0]][0], centres[K.CENTRE_NAMES[1]][0])) == pytest.approx(12.0, abs=1e-9)


def test_arithmetic_mean_depends_on_the_cut_and_the_circle_mean_does_not(uhr4):
    means = [float(K.arithmetic_mean_with_cut(uhr4, c)) for c in X.CUTS]
    assert max(means) - min(means) > 1.0                                              # 12 h (Schnitt 0) gegen 0 h (Schnitt 12)
    assert float(K.hours_distance(K.arithmetic_mean_with_cut(uhr4, 12.0), 0.0)) == pytest.approx(0.0, abs=1e-9)
    shifted = (uhr4 + 5.0) % 24.0
    assert float(K.hours_distance(K.a2h(K.mean_direction(K.h2a(shifted))), 5.0)) == pytest.approx(0.0, abs=1e-9)


def test_extrinsic_and_intrinsic_differ_on_skewed_data():
    """Drei Punkte bei 0°, einer bei 100°: intrinsisch 25° (Mittel der entfalteten Winkel), extrinsisch atan2((sin 100°)/4, (3 + cos 100°)/4) = 19,2°."""
    th = np.radians([0.0, 0.0, 0.0, 100.0])
    ext = math.degrees(float(K.mean_direction(th)))
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert math.degrees(c) == pytest.approx(25.0)
    assert ext == pytest.approx(math.degrees(math.atan2(math.sin(math.radians(100)) / 4, (3 + math.cos(math.radians(100))) / 4))) and ext == pytest.approx(19.2, abs=0.05)


def test_median_differs_from_both_means_at_sixty_forty_antipodes():
    """6 Punkte bei 0°, 4 bei 180°: Median und Kreismittel liegen bei 0°, das Fréchet-Mittel bei ±72° (Minimum von 0,6θ² + 0,4(π − θ)² bei θ = 0,4π), F = 0,24π²."""
    th = np.radians([0.0] * 6 + [180.0] * 4)
    assert float(K.circular_median(th)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(K.mean_direction(th)) == pytest.approx(0.0, abs=1e-12)
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert math.degrees(c) == pytest.approx(72.0) and f == pytest.approx(0.24 * math.pi ** 2) and n_min >= 2
    assert sorted(np.round(np.degrees(K.frechet_minima(th)[0][:2]))) == [72.0, 288.0]


def test_cover_arc_differs_from_the_median_with_one_outlier():
    th = np.radians([0.0] * 9 + [170.0])
    assert float(K.circular_median(th)[0]) == pytest.approx(0.0, abs=1e-12)
    centre, length = K.cover_arc_center(th)
    assert math.degrees(centre) == pytest.approx(85.0) and math.degrees(length) == pytest.approx(170.0)


def test_mixture_and_quantile_mean_differ_as_soon_as_the_sources_differ():
    mu = np.array([[0.0, 10.0]])
    p = np.array([0.05, 0.5, 0.95])
    mix = D.mixture_quantile_batch(p, mu, 1.0, [0.5, 0.5])[0]
    vin = D.vincent_quantile_batch(p, mu, 1.0, [0.5, 0.5])[0]
    assert np.abs(mix - vin).max() > 2.0
    same = np.array([[3.0, 3.0]])
    assert D.mixture_quantile_batch(p, same, 1.0, [0.5, 0.5])[0] == pytest.approx(D.vincent_quantile_batch(p, same, 1.0, [0.5, 0.5])[0], abs=1e-9)    # gleiche Quellen: kein Unterschied


def test_calibrated_and_overconfident_reports_differ():
    over = X.forecast_cell(2.0, "überzuversichtlich", 100, seeds=(8101,))
    honest = X.forecast_cell(2.0, "ehrlich", 100, seeds=(8101,))
    assert over["methods"]["Mischung"]["covered"]["mean"] != honest["methods"]["Mischung"]["covered"]["mean"]
    assert over["methods"]["Quantilmittel"]["width"]["mean"] != honest["methods"]["Quantilmittel"]["width"]["mean"]


def test_quantile_mean_in_x_and_in_log_x_differ():
    cell = X.scale_cell(2.0, "überzuversichtlich", 100, seeds=(8101,))
    assert cell["methods"]["Quantilmittel in x"]["crps"]["mean"] != cell["methods"]["Quantilmittel in log x"]["crps"]["mean"]
    assert cell["methods"]["Mischung"]["crps"]["mean"] not in (cell["methods"]["Quantilmittel in x"]["crps"]["mean"], cell["methods"]["Quantilmittel in log x"]["crps"]["mean"])


def test_one_peak_and_two_peak_nights_differ():
    one = S.make_night(60, 6.0, 1.0, 0, 12.0, 35)
    two = S.make_night(60, 6.0, 1.0, 50, 12.0, 35)
    assert one.truth == 6.0 and two.truth is None
    assert float(K.resultant_length(K.h2a(one.hours))) > 0.9 and float(K.resultant_length(K.h2a(two.hours))) < 0.2


def test_every_stored_column_has_nonzero_entries_in_the_results():
    """Nullspalten-Regel an den vorgerechneten Zahlen: keine Spalte der Experimente besteht nur aus Nullen."""
    import mir_results as R
    assert any(r["power"]["mean"] > 0 for r in R.power_rows()) and any(r["err_gt_45"]["mean"] > 0 for r in R.power_rows())
    for key in ("ext_shift", "int_shift", "med_shift", "cov_shift", "ext_gt_60", "med_gt_60"):
        assert any(r[key]["mean"] > 0 for r in R.lorry_rows()), key
    for key in ("ext_empty", "int_empty", "med_a", "cov_empty"):
        assert any(r[key]["mean"] > 0 for r in R.cluster_rows()), key
    assert R.grid_miscount()["wrong_share"]["mean"] > 0
