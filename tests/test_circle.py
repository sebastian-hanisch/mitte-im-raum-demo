"""Die Kreis-Einheiten gegen Handrechnung auf den Mini-Instanzen (tests/conftest.py) und gegen ihre Randfälle."""

import math

import numpy as np
import pytest

import mir_circle as K


def test_hour_angle_conversion_and_distances():
    assert K.h2a(6.0) == pytest.approx(math.pi / 2) and K.h2a(24.0) == pytest.approx(2 * math.pi)
    assert K.a2h(math.pi) == pytest.approx(12.0) and K.a2h(-math.pi / 2) == pytest.approx(18.0) and K.a2h(2 * math.pi) == pytest.approx(0.0)
    assert float(K.hours_distance(23.0, 1.0)) == pytest.approx(2.0) and float(K.hours_distance(1.0, 23.0)) == pytest.approx(2.0)
    assert float(K.hours_distance(0.0, 12.0)) == pytest.approx(12.0) and float(K.hours_distance(5.0, 5.0)) == 0.0
    assert float(K.arc_distance(math.radians(350), math.radians(10))) == pytest.approx(math.radians(20))
    assert float(K.arc_distance(0.0, math.pi)) == pytest.approx(math.pi)
    assert float(K.wrap_pi(math.pi)) == pytest.approx(math.pi) and float(K.wrap_pi(3 * math.pi / 2)) == pytest.approx(-math.pi / 2)


def test_clock_example_arithmetic_versus_circle_mean(uhr4):
    """23, 23,5, 0,5, 1 Uhr: arithmetisch 12,00 h, Kreismittel 0,00 h, R = 0,97869."""
    assert float(K.arithmetic_mean_with_cut(uhr4, 0.0)) == pytest.approx((23 + 23.5 + 0.5 + 1) / 4)
    assert float(K.hours_distance(K.arithmetic_mean_with_cut(uhr4, 12.0), 0.0)) == pytest.approx(0.0, abs=1e-12)
    th = K.h2a(uhr4)
    assert float(K.hours_distance(K.a2h(K.mean_direction(th)), 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(K.resultant_length(th)) == pytest.approx((math.cos(math.radians(15)) + math.cos(math.radians(7.5))) / 2)


def test_clock_example_frechet_has_two_local_minima(uhr4):
    th = K.h2a(uhr4)
    centres, values = K.frechet_minima(th)
    assert len(centres) == 2
    assert float(K.hours_distance(K.a2h(centres[0]), 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert values[0] == pytest.approx((2 * math.radians(15) ** 2 + 2 * math.radians(7.5) ** 2) / 4)
    assert float(K.a2h(centres[1])) == pytest.approx(12.0)
    assert values[1] > 100 * values[0]
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert n_min == 2 and n_tied == 1 and f == pytest.approx(values[0])


def test_clock_example_median_and_cover_arc(uhr4):
    th = K.h2a(uhr4)
    c, f, ties = K.circular_median(th)
    assert float(K.a2h(c)) == pytest.approx(0.5) and ties == 2                    # Gleichstand 23,5 h / 0,5 h: der kleinere Winkel
    assert f == pytest.approx(math.radians(11.25))
    centre, length = K.cover_arc_center(th)
    assert float(K.hours_distance(K.a2h(centre), 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert length == pytest.approx(math.radians(30))


def test_two_opposite_points_have_two_equal_minima_at_plus_minus_90_degrees(gegenpunkte):
    th = K.h2a(gegenpunkte)
    assert float(K.resultant_length(th)) == pytest.approx(0.0, abs=1e-12)
    assert float(K.mean_direction(th)) == 0.0                                      # Richtung nicht definiert: 0 geliefert
    centres, values = K.frechet_minima(th)
    assert sorted(np.round(K.a2h(centres), 9)) == [6.0, 18.0]
    assert values == pytest.approx([math.pi ** 2 / 4] * 2)
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert (n_min, n_tied) == (2, 2) and float(K.a2h(c)) == pytest.approx(6.0)    # Gleichstand: der kleinere Winkel
    centre, length = K.cover_arc_center(th)
    assert length == pytest.approx(math.pi) and float(K.a2h(centre)) == pytest.approx(18.0)     # gleich große Lücken: die erste Lücke zählt


def test_three_even_points_have_three_minima_on_the_points(drei_punkte):
    th = K.h2a(drei_punkte)
    assert float(K.resultant_length(th)) == pytest.approx(0.0, abs=1e-12)
    centres, values = K.frechet_minima(th)
    assert sorted(np.round(K.a2h(centres), 9)) == [0.0, 8.0, 16.0]
    assert values == pytest.approx([8 * math.pi ** 2 / 27] * 3)


def test_four_even_points_have_four_minima_between_the_points(vier_punkte):
    centres, values = K.frechet_minima(K.h2a(vier_punkte))
    assert sorted(np.round(K.a2h(centres), 9)) == [3.0, 9.0, 15.0, 21.0]
    assert values == pytest.approx([10125 * math.radians(1) ** 2] * 4)


def test_single_point_is_its_own_center():
    th = K.h2a(np.array([5.0]))
    assert float(K.a2h(K.mean_direction(th))) == pytest.approx(5.0)
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert float(K.a2h(c)) == pytest.approx(5.0) and f == pytest.approx(0.0, abs=1e-12) and n_min == 1
    assert float(K.a2h(K.circular_median(th)[0])) == pytest.approx(5.0)
    centre, length = K.cover_arc_center(th)
    assert float(K.a2h(centre)) == pytest.approx(5.0) and length == pytest.approx(0.0)


def test_identical_points_do_not_break_the_enumeration():
    th = K.h2a(np.array([3.0, 3.0, 3.0]))
    c, f, n_min, n_tied = K.frechet_mean(th)
    assert float(K.a2h(c)) == pytest.approx(3.0) and f == pytest.approx(0.0, abs=1e-12) and n_min == 1
    assert K.cover_arc_center(th)[1] == pytest.approx(0.0)


def test_rayleigh_threshold_and_p_value():
    assert float(K.rayleigh_threshold(30)) == pytest.approx(math.sqrt(-math.log(0.05) / 30)) and float(K.rayleigh_threshold(30)) == pytest.approx(0.316, abs=1e-3)
    assert float(K.rayleigh_p(K.rayleigh_threshold(50), 50)) == pytest.approx(0.05)
    assert float(K.rayleigh_p(0.0, 10)) == 1.0 and float(K.rayleigh_p(1.0, 10)) == pytest.approx(math.exp(-10))
    assert float(K.rayleigh_threshold(20, level=0.01)) == pytest.approx(math.sqrt(-math.log(0.01) / 20))


def test_five_centres_on_the_clock_example(uhr4):
    centres, R = K.five_centres(uhr4)
    assert set(centres) == set(K.CENTRE_NAMES)
    assert centres[K.CENTRE_NAMES[0]][0] == pytest.approx(12.0)
    assert centres[K.CENTRE_NAMES[3]][0] == pytest.approx(0.5)
    for name in (K.CENTRE_NAMES[1], K.CENTRE_NAMES[2], K.CENTRE_NAMES[4]):
        assert float(K.hours_distance(centres[name][0], 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert R[0] == pytest.approx(0.97869, abs=1e-5)
    centres12, _ = K.five_centres(uhr4, cut=12.0)
    assert float(K.hours_distance(centres12[K.CENTRE_NAMES[0]][0], 0.0)) == pytest.approx(0.0, abs=1e-9)


def test_batch_results_equal_single_calls():
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 2 * math.pi, (12, 7))
    c, f, n_min, n_tied = K.frechet_mean_batch(data)
    for i in range(12):
        ci, fi, mi, ti = K.frechet_mean(data[i])
        assert (ci, fi, mi, ti) == pytest.approx((c[i], f[i], n_min[i], n_tied[i]))
    mc, mv, mt = K.circular_median_batch(data)
    for i in range(12):
        assert K.circular_median(data[i]) == pytest.approx((mc[i], mv[i], mt[i]))


def test_results_do_not_depend_on_the_input_order():
    rng = np.random.default_rng(4)
    th = rng.uniform(0, 2 * math.pi, 9)
    perm = rng.permutation(9)
    assert K.frechet_mean(th) == pytest.approx(K.frechet_mean(th[perm]))
    assert K.circular_median(th) == pytest.approx(K.circular_median(th[perm]))
    assert K.cover_arc_center(th) == pytest.approx(K.cover_arc_center(th[perm]))
