"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstand, inklusive Grenze, Entartungen der Regler."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_constants as C
import mir_dist as D
import mir_evaluation as V
import mir_experiments as X
import mir_scenario as S


def test_exactly_two_points_not_opposite():
    """n = 2 bei 0° und 90°: extrinsisch 45°, intrinsisch 45° (ein zweites, schlechteres Minimum am Gegenpunkt 225°), Median Gleichstand → kleinerer Winkel, Bogen Mitte 45°."""
    th = np.radians([0.0, 90.0])
    assert math.degrees(float(K.mean_direction(th))) == pytest.approx(45.0)
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert math.degrees(c) == pytest.approx(45.0) and n_min == 2 and n_tied == 1 and f == pytest.approx(math.radians(45.0) ** 2)
    m, fm, ties = K.circular_median(th)
    assert math.degrees(m) == pytest.approx(0.0, abs=1e-9) and ties == 2 and fm == pytest.approx(math.radians(45.0))
    centre, length = K.cover_arc_center(th)
    assert math.degrees(centre) == pytest.approx(45.0) and math.degrees(length) == pytest.approx(90.0)


def test_two_points_across_the_cut():
    """350° und 10°: der kleine Bogen liegt über 0°; alle vier Mitten bei 0° (Median: der kleinere Winkel, also 10°)."""
    th = np.radians([350.0, 10.0])
    assert float(K.arc_distance(K.mean_direction(th), 0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(K.arc_distance(K.frechet_mean(th)[0], 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(K.circular_median(th)[0]) == pytest.approx(10.0)
    assert float(K.arc_distance(K.cover_arc_center(th)[0], 0.0)) == pytest.approx(0.0, abs=1e-9) and K.cover_arc_center(th)[1] == pytest.approx(math.radians(20.0))


def test_points_exactly_at_the_cut_and_at_two_pi():
    th = np.array([0.0, 2 * math.pi, 4 * math.pi])
    assert float(K.cover_arc_center(th)[1]) == pytest.approx(0.0, abs=1e-9)
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert float(K.arc_distance(c, 0.0)) < 1e-9 and f == pytest.approx(0.0, abs=1e-12)


def test_two_identical_clusters_of_equal_points_still_enumerate():
    th = np.radians([0.0, 0.0, 180.0, 180.0])
    centres, values = K.frechet_minima(th)
    assert sorted(np.round(np.degrees(centres))) == [90.0, 270.0] and values == pytest.approx([math.pi ** 2 / 4] * 2)


def test_n_equals_one_in_the_batch_functions():
    th = K.h2a(np.array([[7.0]]))
    assert K.frechet_mean_batch(th)[2].tolist() == [1] and K.circular_median_batch(th)[2].tolist() == [1] and K.cover_arc_batch(th)[1].tolist() == [0.0]
    centres, R = K.five_centres(np.array([[7.0]]))
    assert all(float(K.hours_distance(c[0], 7.0)) < 1e-9 for c in centres.values()) and R[0] == pytest.approx(1.0)


def test_window_boundary_is_inclusive_for_exact_values():
    assert X.window_coverage(np.array([[1.0, 3.0]]), np.array([2.0]), 1.0)[0] == 1.0
    assert X.window_coverage(np.array([[1.0, 3.0]]), np.array([2.0]), 0.99)[0] == 0.0


def test_rayleigh_boundary_n_one_and_two():
    assert float(K.rayleigh_p(1.0, 1)) == pytest.approx(math.exp(-1.0)) and float(K.rayleigh_threshold(1)) == pytest.approx(math.sqrt(-math.log(0.05)))
    assert float(K.rayleigh_threshold(1)) > 1.0                       # bei n = 1 gibt es keine erkennbare Häufung (R ≤ 1)


def test_second_peak_share_extremes():
    for share in (0, 100):
        night = S.make_night(10, 6.0, 1.0, share, 12.0, 1)
        assert len(night.hours) == 10 and night.n_second == (0 if share == 0 else 10)
    full = V.analyse_gate(V.Settings(n=5, share=100, gap=12.0, mu=6.0))
    assert full.night.truth is None and full.n_minima >= 1


def test_gate_analysis_at_the_slider_corners():
    for kwargs in ({"n": C.N_MIN}, {"n": C.N_MAX}, {"sigma": C.SIGMA_MIN}, {"sigma": C.SIGMA_MAX}, {"mu": C.MU_MAX}, {"gap": C.GAP_MIN, "share": 50}, {"cut": C.CUT_MAX}, {"window": C.WINDOW_MAX}):
        a = V.analyse_gate(V.Settings(**kwargs))
        assert 0.0 <= a.R <= 1.0 and 0.0 <= a.p_value <= 1.0 and all(0.0 <= v <= 1.0 for v in a.coverage.values()), kwargs
        assert all(0.0 <= h < 24.0 for h in a.centres.values()), kwargs


def test_supplier_summary_at_the_slider_corners():
    for ratio in (C.RATIO_MIN, C.RATIO_MAX):
        for s in (C.S_MIN, C.S_MAX):
            for wb in (C.WB_MIN, C.WB_MAX):
                for service in (C.SERVICE_MIN, C.SERVICE_MAX):
                    r = X.supplier_summary(ratio, s, float(wb), float(service))
                    assert r["service_at_plan"]["Mischung, Quantil (richtig)"] == pytest.approx(service / 100.0, abs=1e-9)
                    assert 0.0 < r["service_at_plan"]["Quantilmittel, Quantil"] <= 1.0 and math.isfinite(r["sd_ratio"])


def test_forecast_cell_at_the_slider_corners():
    for se in (C.SE_MIN, C.SE_MAX):
        for calib in C.CALIBRATIONS:
            cell = X.forecast_cell(se, calib, 40, seeds=(8101,))
            assert all(math.isfinite(m["crps"]["mean"]) for m in cell["methods"].values())


def test_quantile_extremes_and_degenerate_mixture():
    p = np.array([1e-9, 0.5, 1 - 1e-9])
    q = D.mixture_quantile(p, [0.0, 0.0], [1.0, 1.0], [0.3, 0.7])                  # zwei gleiche Quellen = eine Normalverteilung
    assert q == pytest.approx(D.norm_ppf(p), abs=1e-8)
    assert D.mixture_quantile(np.array([0.5]), [4.0], [2.0], [1.0])[0] == pytest.approx(4.0)
    assert D.mixture_quantile(np.array([0.5]), [0.0, 0.0], [1.0, 1.0], [1.0, 0.0])[0] == pytest.approx(0.0, abs=1e-9)      # Gewicht 0 für eine Quelle


def test_weights_of_zero_and_one_collapse_the_mixture_to_one_supplier():
    r = X.supplier_summary(3.0, 0.3, share_b=0.0)
    assert r["mix_mean"] == pytest.approx(10.0 * math.exp(0.045)) and r["sd_ratio"] == pytest.approx(1.0) and r["plans"]["Mischung, Quantil (richtig)"] == pytest.approx(r["plans"]["Quantilmittel, Quantil"], rel=1e-9)
    r1 = X.supplier_summary(3.0, 0.3, share_b=100.0)
    assert r1["mix_mean"] == pytest.approx(30.0 * math.exp(0.045)) and r1["sd_ratio"] == pytest.approx(1.0)
