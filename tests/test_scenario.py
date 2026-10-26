"""Szenario: der blockweise Strom ist der Strom der kopierten Klasse; die Erzeuger haben ihre dokumentierten Eigenschaften (R der wrapped normal, Anteile, Reihenfolge)."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_scenario as S
from mir_rng import SplitMix64


@pytest.mark.parametrize("seed", [0, 1, 7, 2**63 + 5, 999999])
def test_block_stream_equals_the_copied_splitmix64(seed):
    rng = SplitMix64(seed)
    expected = [rng.next() for _ in range(50)]
    assert S.Draws(seed).raw(50).tolist() == expected


def test_blocks_continue_the_stream_where_the_last_block_ended():
    d = S.Draws(5)
    first, second = d.raw(7), d.raw(9)
    assert np.concatenate([first, second]).tolist() == S.Draws(5).raw(16).tolist()
    assert S.Draws(5).raw(0).size == 0


def test_uniform_is_the_top_53_bits_of_the_stream():
    rng = SplitMix64(3)
    expected = [(rng.next() >> 11) / 2**53 for _ in range(10)]
    assert S.Draws(3).uniform(10).tolist() == expected
    u = S.Draws(4).uniform((1000, 3))
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.03


def test_normal_by_hand_from_two_uniforms():
    """Box-Muller: z0 = √(−2 ln(1 − u1))·cos(2π u2), z1 = … sin(2π u2); bei Größe 2 steht z0 zuerst."""
    u = S.Draws(9).uniform(2)
    r = math.sqrt(-2 * math.log(1 - u[0]))
    z = S.Draws(9).normal(2)
    assert z.tolist() == pytest.approx([r * math.cos(2 * math.pi * u[1]), r * math.sin(2 * math.pi * u[1])])


def test_normal_has_mean_zero_and_variance_one():
    z = S.Draws(10).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert S.Draws(10).normal(3).shape == (3,)
    assert S.Draws(10).normal((4, 2, 2)).shape == (4, 2, 2)


def test_cell_seed_is_stable_and_separates_cells():
    assert S.cell_seed(5, "a", 1.0) == S.cell_seed(5, "a", 1.0)
    assert len({S.cell_seed(5, "a", 1.0), S.cell_seed(5, "b", 1.0), S.cell_seed(5, "a", 2.0), S.cell_seed(6, "a", 1.0)}) == 4
    assert S.cell_seed(5) == 5


def test_rho_and_sigma_are_inverse_to_each_other():
    assert S.rho_from_sigma(S.sigma_from_rho(0.45)) == pytest.approx(0.45)
    assert S.sigma_from_rho(math.exp(-0.5)) == pytest.approx(1.0)
    assert S.rho_from_sigma(0.0) == 1.0


@pytest.mark.parametrize("rho", [0.125, 0.45, 0.7, 0.95])
def test_wrapped_normal_has_the_stated_resultant_length(rho):
    """R einer großen Stichprobe liegt beim wahren ρ (Standardfehler ≈ √((1 − ρ²)/(2n)), hier n = 200 000: ≤ 0,0016)."""
    th = S.wrapped_angles(S.Draws(11), 1, 200000, rho, centre=1.0)
    assert float(K.resultant_length(th)[0]) == pytest.approx(rho, abs=0.008)
    assert float(K.arc_distance(K.mean_direction(th)[0], 1.0)) < 0.02


def test_uniform_angles_have_small_resultant_length_and_cover_the_circle():
    th = S.uniform_angles(S.Draws(12), 1, 100000)
    assert th.min() >= 0 and th.max() < 2 * math.pi
    assert float(K.resultant_length(th)[0]) < 0.02


def test_two_clusters_have_the_requested_counts_and_centres():
    th = S.two_cluster_angles(S.Draws(13), 50, 41, 120.0, 20, 5.0)
    first, second = th[:, :20], th[:, 20:]
    assert np.abs(K.wrap_pi(first)).max() < math.radians(5 * 6)
    assert np.abs(K.wrap_pi(second - math.radians(120))).max() < math.radians(5 * 6)
    paired = S.two_cluster_angles(S.Draws(13), 50, 41, 120.0, 21, 5.0)             # gleicher Strom, ein Lkw mehr in der ersten Häufung
    changed = np.flatnonzero((np.abs(K.wrap_pi(paired - th)) > 1e-12).any(axis=0))
    assert changed.tolist() == [20]


def test_gate_hours_and_two_peak_hours():
    h = S.gate_hours(S.Draws(14), 3, 5000, 23.5, 1.0)
    assert h.min() >= 0 and h.max() < 24
    assert float(K.hours_distance(K.a2h(K.mean_direction(K.h2a(h))), 23.5).max()) < 0.2
    h2 = S.gate_hours(S.Draws(14), 2, 7, np.array([3.0, 15.0]), 0.0)
    assert h2.tolist() == [[3.0] * 7, [15.0] * 7]
    tp = S.two_peak_hours(S.Draws(15), 1, 10, 6.0, 12.0, 4, 0.0)
    assert sorted(tp[0].tolist()) == [6.0] * 6 + [18.0] * 4


def test_second_peak_count_rounds_and_is_bounded():
    assert S.second_peak_count(41, 51) == 21 and S.second_peak_count(41, 49) == 20 and S.second_peak_count(41, 50) == 20
    assert S.second_peak_count(60, 0) == 0 and S.second_peak_count(60, 100) == 60 and S.second_peak_count(5, 100) == 5


def test_make_night_is_deterministic_and_reports_the_truth():
    a = S.make_night(60, 23.5, 1.0, 0, 12.0, 35)
    b = S.make_night(60, 23.5, 1.0, 0, 12.0, 35)
    assert a.hours.tolist() == b.hours.tolist() and a.truth == 23.5 and a.n_second == 0 and len(a.hours) == 60
    two = S.make_night(60, 6.0, 1.0, 50, 12.0, 35)
    assert two.truth is None and two.n_second == 30
    assert S.make_night(60, 23.5, 1.0, 0, 12.0, 36).hours.tolist() != a.hours.tolist()


def test_forecast_cases_have_the_stated_structure_and_are_paired_across_calibrations():
    m, y, mu = S.forecast_cases(S.Draws(16), 100000, 2.0)
    assert m.std() == pytest.approx(1.0, abs=0.02) and (y - m).std() == pytest.approx(1.0, abs=0.02)
    assert (mu - m[:, None]).std() == pytest.approx(2.0, abs=0.03)
    assert abs(np.corrcoef(mu[:, 0] - m, mu[:, 1] - m)[0, 1]) < 0.02
    m2, y2, mu2 = S.forecast_cases(S.Draws(16), 100000, 2.0)
    assert (m == m2).all() and (mu == mu2).all()


def test_reported_sigma_by_hand():
    assert S.reported_sigma("überzuversichtlich", 2.0) == 1.0
    assert S.reported_sigma("ehrlich", 2.0) == pytest.approx(math.sqrt(5.0))
    assert S.reported_sigma("ehrlich", 0.0, spread_y=0.4) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        S.reported_sigma("egal", 1.0)
