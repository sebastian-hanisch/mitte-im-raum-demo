"""Messreihen-Funktionen (mir_experiments.py) mit kleinen Läufen: Aufbau, Gepaartheit, Handwerte, geschlossene Formeln gegen Ziehung. Die vorgerechneten Zahlen prüft test_claims.py."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_constants as C
import mir_dist as D
import mir_experiments as X


def test_agg_gives_mean_and_std_over_the_seeds():
    a = X.agg([1.0, 2.0, 3.0, 4.0])
    assert a["mean"] == 2.5 and a["std"] == pytest.approx(math.sqrt(1.25)) and a["per_seed"] == [1.0, 2.0, 3.0, 4.0]


def test_source_hash_is_stable_and_covers_every_source():
    assert X.source_hash() == X.source_hash() and len(X.source_hash()) == 16
    for name in X.SOURCES:
        assert (X.ROOT / name).exists(), name


def test_uniform_cell_matches_the_rayleigh_laws_and_is_deterministic():
    cell = X.uniform_cell(30, 4000)
    assert cell == X.uniform_cell(30, 4000)
    assert cell["mean"]["mean"] == pytest.approx(cell["approx_mean"], abs=0.01)
    assert cell["r95"]["mean"] == pytest.approx(cell["threshold"], abs=0.02)
    assert cell["size"]["mean"] == pytest.approx(0.05, abs=0.01)
    assert cell["below_01"]["mean"] == pytest.approx(1 - math.exp(-30 * 0.01), abs=0.03)          # P(R < 0,1) = 1 − exp(−n·0,01)
    assert len(cell["mean"]["per_seed"]) == len(X.SEEDS)


def test_power_cell_grows_with_n_and_with_rho():
    low, high = X.power_cell(0.25, 10, 500), X.power_cell(0.25, 100, 500)
    assert high["power"]["mean"] > low["power"]["mean"] + 0.4 and high["err_mean_deg"]["mean"] < low["err_mean_deg"]["mean"]
    assert X.power_cell(0.7, 10, 500)["power"]["mean"] > low["power"]["mean"]


def test_one_peak_cell_has_consistent_columns():
    cell = X.one_peak_cell(0.7, 20, 20, 200)
    assert cell["rmse_cov"]["mean"] > cell["rmse_ext"]["mean"]                                    # der kleinste Bogen folgt den Störern
    assert 0.0 <= cell["multi_minima"]["mean"] <= 1.0 and cell["dist_mean_deg"]["mean"] >= 0.0
    clean = X.one_peak_cell(0.7, 20, 0, 200)
    assert clean["rmse_cov"]["mean"] < cell["rmse_cov"]["mean"]


def test_two_cluster_cell_shares_add_up_to_one():
    cell = X.two_cluster_cell(120, "binomial 50/50", 100)
    for key in ("ext", "int", "med", "cov"):
        total = sum(cell[f"{key}_{w}"]["mean"] for w in ("a", "b", "empty"))
        assert total == pytest.approx(1.0)
    assert cell["ext_empty"]["mean"] > 0.97 and cell["int_empty"]["mean"] > 0.97                   # bei Δ = 120° liegen beide im Leeren (bis auf extreme Zufallsaufteilungen)
    with pytest.raises(KeyError):
        X.two_cluster_cell(120, "binomial 50/50", 10)["nothing"]


def test_one_lorry_cell_by_the_antipodal_hand_picture():
    """Bei Gegenhäufungen springt das extrinsische Mittel und der Median kippt auf die andere Häufung, das intrinsische bleibt nah, der kleinste Bogen ruhig."""
    cell = X.one_lorry_cell(180, 150)
    assert cell["med_shift"]["mean"] > 120 and cell["ext_shift"]["mean"] > 60
    assert cell["int_shift"]["mean"] < cell["ext_shift"]["mean"] / 2 and cell["cov_shift"]["mean"] < 15
    near = X.one_lorry_cell(60, 150)
    assert near["med_shift"]["mean"] < 30 and near["ext_shift"]["mean"] < 5


def test_grid_miscount_shrinks_with_a_finer_grid():
    coarse = X.grid_miscount(60, seeds=(8101, 8102), grid_points=720)["wrong_share"]["mean"]
    fine = X.grid_miscount(60, seeds=(8101, 8102), grid_points=20000)["wrong_share"]["mean"]
    assert coarse > 0.0 and fine < coarse


def test_window_coverage_and_best_window_by_hand(uhr4):
    hours = uhr4[None, :]
    assert X.window_coverage(hours, np.array([0.0]), 1.0)[0] == 1.0                                # Abstände 1, 0,5, 0,5, 1: der Rand zählt mit
    assert X.window_coverage(hours, np.array([0.0]), 0.5)[0] == 0.5
    assert X.window_coverage(hours, np.array([12.0]), 1.0)[0] == 0.0
    assert X.best_window_coverage(np.array([[0.0, 12.0]]), 1.0)[0] == 0.5
    assert X.best_window_coverage(hours, 1.0)[0] == 0.75                                            # um 23 Uhr: 23, 23,5; um 23,5 Uhr: 23, 23,5, 0,5; nie alle vier


def test_evaluate_nights_on_the_clock_example(uhr4):
    out = X.evaluate_nights(uhr4[None, :], np.array([0.0]), window=1.01)       # 1,01 statt 1: der Rand 1,0 wäre an der Rundung des Kreismittels knapp daneben
    arith = out["Arithmetisch (Tag ab 0 Uhr)"]
    assert arith["err_min"] == pytest.approx(720.0) and arith["err_gt_1h"] == 1.0 and arith["coverage"] == 0.0
    circle = out["Kreismittel (arctan2)"]
    assert circle["err_min"] == pytest.approx(0.0, abs=1e-6) and circle["err_gt_1h"] == 0.0 and circle["coverage"] == 1.0
    assert out["Arithmetisch (Tag ab 12 Uhr)"]["err_min"] == pytest.approx(0.0, abs=1e-6)
    assert out["_R"] == pytest.approx(0.97869, abs=1e-5) and out["_oracle"] == 1.0
    two = X.evaluate_nights(np.array([[6.0, 6.0, 18.0, 18.0]]), None, peaks=(6.0, 18.0))
    assert two["Arithmetisch (Tag ab 0 Uhr)"]["empty"] == 1.0 and "err_min" not in two["Fréchet p = 2"] and "_oracle" not in two


def test_torzeit_and_nightshift_cells_show_the_wraparound_error():
    cell = X.torzeit_cell(1.0, 300)
    rows = cell["rows"]
    assert rows["Arithmetisch (Tag ab 0 Uhr)"]["err_min"]["mean"] > 5 * rows["Kreismittel (arctan2)"]["err_min"]["mean"]
    night = X.nightshift_cell(200)["rows"]
    assert night["Arithmetisch (Tag ab 0 Uhr)"]["median_estimate_h"]["mean"] == pytest.approx(16.1, abs=0.3)
    assert night["Arithmetisch (Tag ab 0 Uhr)"]["coverage"]["mean"] < 0.001 and night["Kreismittel (arctan2)"]["coverage"]["mean"] > 0.6


def test_cut_and_two_peak_cells():
    cut = X.cut_cell(300)
    assert 0.4 < cut["range_gt_1h"]["mean"] < 0.75 and cut["cuts"] == list(X.CUTS)
    peaks = X.two_peak_cell(200)
    assert peaks["rows"]["Fréchet p = 2"]["empty"]["mean"] == 1.0 and peaks["rows"]["Zirkulärer Median p = 1"]["empty"]["mean"] < 0.3
    assert peaks["best_window"]["mean"] > peaks["rows"]["Zirkulärer Median p = 1"]["coverage"]["mean"]


def test_supplier_summary_against_hand_numbers():
    """Median 10 und 30 Tage, Log-Streuung 0,15, je 50 %: Plan auf dem Quantilmittel = 20·exp(0,15·1,645) Tage; Servicegrad Σ 0,5·Φ((ln x − ln m)/0,15)."""
    r = X.supplier_summary(3.0, 0.15)
    plan = 20.0 * math.exp(0.15 * float(D.norm_ppf(0.95)))
    assert r["plans"]["Quantilmittel, Quantil"] == pytest.approx(plan)
    phi = lambda z: 0.5 * math.erfc(-z / math.sqrt(2))
    expected = 0.5 * phi((math.log(plan) - math.log(10)) / 0.15) + 0.5 * phi((math.log(plan) - math.log(30)) / 0.15)
    assert r["service_at_plan"]["Quantilmittel, Quantil"] == pytest.approx(expected, abs=1e-12)
    assert r["service_at_plan"]["Mischung, Quantil (richtig)"] == pytest.approx(0.95, abs=1e-12)
    assert r["mix_mean"] == pytest.approx(r["vin_mean"]) and r["sd_ratio"] == pytest.approx(r["mix_sd"] / r["vin_sd"])
    assert r["service_at_plan"]["Typische Lieferzeit (Mittel)"] < r["service_at_plan"]["Quantilmittel, Quantil"] < 0.95


def test_equal_suppliers_make_both_means_coincide():
    r = X.supplier_summary(1.0, 0.3)
    assert r["sd_ratio"] == pytest.approx(1.0) and r["band_mix"] == pytest.approx(r["band_vin"])
    assert r["service_at_plan"]["Quantilmittel, Quantil"] == pytest.approx(0.95, abs=1e-12)
    assert r["plans"]["Quantilmittel, Quantil"] == pytest.approx(r["plans"]["Mischung, Quantil (richtig)"], rel=1e-9)


def test_supplier_weights_and_levels_are_used():
    heavy_b = X.supplier_summary(3.0, 0.15, share_b=80.0, service=90.0)
    light_b = X.supplier_summary(3.0, 0.15, share_b=20.0, service=90.0)
    assert heavy_b["mix_mean"] > light_b["mix_mean"] and heavy_b["service"] == 90.0
    assert heavy_b["service_at_plan"]["Mischung, Quantil (richtig)"] == pytest.approx(0.90, abs=1e-12)


def test_supplier_simulation_agrees_with_the_closed_form():
    sim = X.supplier_simulation(3.0, 0.15, 40000)
    closed = X.supplier_summary(3.0, 0.15)
    assert sim["mix_mean"]["mean"] == pytest.approx(closed["mix_mean"], rel=0.01) and sim["mix_sd"]["mean"] == pytest.approx(closed["mix_sd"], rel=0.02)
    assert sim["vin_sd"]["mean"] == pytest.approx(closed["vin_sd"], rel=0.02)
    assert sim["mix_range"]["mean"] == pytest.approx(closed["range_mix"], abs=0.01) and sim["vin_range"]["mean"] == pytest.approx(closed["range_vin"], abs=0.01)


def test_forecast_cell_without_forecast_error_collapses_to_one_forecast():
    """Ohne Lagefehler (s_e ≈ 0) sind beide Prognosen gleich: Mischung, Quantilmittel und die einzelne Prognose haben denselben CRPS."""
    cell = X.forecast_cell(1e-9, "ehrlich", 150, seeds=(8101, 8102))
    crps = {k: v["crps"]["mean"] for k, v in cell["methods"].items()}
    assert crps["Mischung"] == pytest.approx(crps["Quantilmittel"], abs=1e-6) and crps["Mischung"] == pytest.approx(crps["Eine Prognose allein"], abs=1e-6)
    assert cell["diffs"]["mix_minus_vin"]["crps_diff"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_forecast_cell_pairs_the_data_across_calibrations():
    over = X.forecast_cell(1.0, "überzuversichtlich", 200, seeds=(8101,))
    honest = X.forecast_cell(1.0, "ehrlich", 200, seeds=(8101,))
    assert over["methods"]["Wahrheit (bekannte Lage)"] == honest["methods"]["Wahrheit (bekannte Lage)"]        # gleiche Fälle, die Wahrheit hängt nicht an der Meldung
    assert over["sigma_rep"] == 1.0 and honest["sigma_rep"] == pytest.approx(math.sqrt(2.0))
    assert over["methods"]["Mischung"]["covered"]["mean"] != honest["methods"]["Mischung"]["covered"]["mean"]
    assert over["methods"]["Wahrheit (bekannte Lage)"]["covered"]["mean"] == pytest.approx(0.8, abs=0.08)


def test_forecast_cell_on_the_oracle_is_the_best():
    cell = X.forecast_cell(2.0, "überzuversichtlich", 300, seeds=(8101, 8102))
    crps = {k: v["crps"]["mean"] for k, v in cell["methods"].items()}
    assert crps["Wahrheit (bekannte Lage)"] < min(crps["Mischung"], crps["Quantilmittel"], crps["Eine Prognose allein"])
    assert crps["Eine Prognose allein"] > crps["Quantilmittel"]                                      # zwei Prognosen mitteln hilft gegen den Lagefehler


def test_forecast_methods_on_a_hand_case():
    """Zwei Prognosen N(0, 1) und N(10, 1): das Quantilmittel hat den Median 5 und die Breite von N(0, 1), die Mischung hat die Ränder weit außen."""
    mu = np.array([[0.0, 10.0]])
    q = X.forecast_methods(mu, 1.0, np.array([5.0]))
    k = len(X.P_GRID)
    lo, hi = k, k + 1
    assert q["Quantilmittel"][0, lo] == pytest.approx(5.0 - 1.2815516, abs=1e-6) and q["Quantilmittel"][0, hi] == pytest.approx(5.0 + 1.2815516, abs=1e-6)
    assert q["Mischung"][0, lo] < 0.0 and q["Mischung"][0, hi] > 10.0
    assert q["Eine Prognose allein"][0, lo] == pytest.approx(0.0 - 1.2815516, abs=1e-6)


def test_log_methods_differ_by_the_scale_of_averaging():
    """Lognormalprognosen mit Lage 0 und 1: Quantilmittel in x ist das arithmetische, in log x das geometrische Mittel der Quantile."""
    mu = np.array([[0.0, 1.0]])
    q = X.log_methods(mu, 0.5, np.array([0.5]))
    k = len(X.P_GRID)
    z = float(D.norm_ppf(0.9))
    q1, q2 = math.exp(0.0 + 0.5 * z), math.exp(1.0 + 0.5 * z)
    assert q["Quantilmittel in x"][0, k + 1] == pytest.approx(0.5 * (q1 + q2)) and q["Quantilmittel in log x"][0, k + 1] == pytest.approx(math.sqrt(q1 * q2))
    assert q["Quantilmittel in x"][0, k + 1] > q["Quantilmittel in log x"][0, k + 1]               # arithmetisches Mittel ≥ geometrisches (AM-GM)


def test_scale_cell_has_the_documented_structure():
    cell = X.scale_cell(2.0, "überzuversichtlich", 150, seeds=(8101, 8102))
    assert set(cell["methods"]) == {"Mischung", "Quantilmittel in x", "Quantilmittel in log x", "Wahrheit (bekannte Lage)"}
    assert set(cell["diffs"]) == {"x_minus_logx", "mix_minus_x", "mix_minus_logx"}
    assert cell["s_e"] == pytest.approx(0.8) and cell["sigma_rep"] == X.SCALE_SPREAD_Y
    assert X.scale_cell(2.0, "ehrlich", 50, seeds=(8101,))["sigma_rep"] == pytest.approx(math.sqrt(0.16 + 0.64))


def test_gate_study_runs_for_one_and_two_peaks():
    one = X.gate_study(30, 23.5, 1.0, 0, 12.0, 0, 1.0, 40, seeds=(8101, 8102))
    assert one["n_second"] == 0 and one["rows"]["Kreismittel (arctan2)"]["err_min"]["mean"] < 20 and "_oracle" in one["rows"]
    two = X.gate_study(30, 6.0, 1.0, 50, 12.0, 0, 1.0, 40, seeds=(8101,))
    assert two["n_second"] == 15 and "err_min" not in two["rows"]["Kreismittel (arctan2)"] and two["rows"]["Fréchet p = 2"]["empty"]["mean"] == 1.0


def test_parts_cover_the_documented_sections():
    assert set(X.PARTS) == {"m1", "m2", "m3", "m4", "m5", "m6"}
    assert C.RESULTS_FILE == "mir_results.json"
