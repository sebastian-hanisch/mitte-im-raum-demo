"""Jede Zahl der README steht hier und wird aus den vorgerechneten Messreihen (data/mir_results.json) bzw. aus einer Handrechnung nachgerechnet.

Gerechnet wird mit denselben Zugriffsfunktionen wie in der App (mir_results), verglichen mit der Schreibweise der README (deutsches Dezimalkomma, `V.de`/`V.pct`).
Bei Zahlen aus Zufallsläufen steht die Messreihe mit vier festen Zufalls-Seeds dahinter (Mittel über die Seeds); die README nennt Mittel und, wo es zählt, die Streuung."""

import math

import numpy as np
import pytest

import mir_circle as K
import mir_dist as D
import mir_evaluation as V
import mir_experiments as X
import mir_results as R


def de(x, d=1):
    return V.de(x, d)


def pc(x, d=0):
    return V.pct(x, d)


# ---------------------------------------------------------------- Handrechnung (H1)

def test_h1_clock_example_by_hand():
    hours = np.array([23.0, 23.5, 0.5, 1.0])
    assert float(K.arithmetic_mean_with_cut(hours, 0.0)) == 12.0
    assert float(K.hours_distance(K.a2h(K.mean_direction(K.h2a(hours))), 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert de(float(K.resultant_length(K.h2a(hours))), 3) == "0,979"
    assert len(K.frechet_minima(K.h2a(hours))[0]) == 2


# ---------------------------------------------------------------- E1 (H2, H3)

def test_e1_random_resultant_length():
    assert [de(R.uniform_row(n)["mean"]["mean"], 3) for n in (5, 10, 30, 100, 1000)] == ["0,402", "0,282", "0,162", "0,089", "0,028"]
    assert de(R.uniform_row(30)["r95"]["mean"], 3) == "0,315" and de(R.uniform_row(30)["threshold"], 3) == "0,316"
    assert de(R.uniform_row(8)["r95"]["mean"], 3) == "0,603" and de(R.uniform_row(8)["threshold"], 3) == "0,612"
    assert abs(R.uniform_row(8)["r95"]["mean"] - R.uniform_row(8)["threshold"]) < 0.01 and abs(R.uniform_row(3)["r95"]["mean"] - R.uniform_row(3)["threshold"]) > 0.05
    assert de(R.uniform_row(3)["r95"]["mean"], 3) == "0,940" and de(R.uniform_row(3)["threshold"], 3) == "0,999"


def test_e1_size_of_the_simple_rayleigh_test():
    sizes = {n: pc(R.uniform_row(n)["size"]["mean"], 1) for n in X.N_RASTER}
    assert sizes == {3: "0,1", 5: "4,3", 8: "4,5", 10: "4,6", 30: "4,9", 100: "4,8", 1000: "5,2"}
    assert max(abs(R.uniform_row(n)["size"]["mean"] - 0.05) for n in (30, 100, 1000)) < 0.004


def test_e1_share_of_random_nights_with_r_below_a_tenth():
    assert {n: pc(R.uniform_row(n)["below_01"]["mean"], 1) for n in (10, 30, 100)} == {10: "9,1", 30: "25,7", 100: "62,9"}
    for n in (10, 30, 100):                                    # geschlossene Form: P(R < 0,1) = 1 − exp(−n·0,01) unter der Näherung
        assert R.uniform_row(n)["below_01"]["mean"] == pytest.approx(1 - math.exp(-n * 0.01), abs=0.03)


def test_e1_power_and_direction_error():
    power = {(rho, n): pc(R.power_row(rho, n)["power"]["mean"], 0) for rho, n in ((0.125, 100), (0.125, 200), (0.25, 30), (0.25, 100), (0.45, 10), (0.45, 30), (0.7, 10))}
    assert power == {(0.125, 100): "34", (0.125, 200): "60", (0.25, 30): "39", (0.25, 100): "91", (0.45, 10): "41", (0.45, 30): "93", (0.7, 10): "94"}
    assert pc(R.power_row(0.125, 30)["err_gt_45"]["mean"], 0) == "43" and de(R.power_row(0.125, 30)["err_mean_deg"]["mean"], 0) == "50"
    assert pc(R.power_row(0.25, 100)["err_gt_45"]["mean"], 1) == "1,1" and de(R.power_row(0.25, 100)["err_mean_deg"]["mean"], 0) == "13"
    assert max(R.power_rows(), key=lambda r: r["err_gt_45"]["mean"])["rho"] == 0.125


# ---------------------------------------------------------------- E2 (H4 bis H6, Gitter)

def test_e2_one_peak_distance_and_accuracy():
    r20, r100 = R.one_peak_row(0.45, 20, 0), R.one_peak_row(0.45, 100, 0)
    assert de(r20["dist_mean_deg"]["mean"]) == "7,4" and de(r100["dist_mean_deg"]["mean"]) == "3,5" and pc(r20["dist_gt_10"]["mean"]) == "26"
    assert [de(r20["rmse_" + k]["mean"]) for k in ("ext", "int", "med", "cov")] == ["20,9", "20,9", "23,9", "38,8"]
    assert [de(r100["rmse_" + k]["mean"]) for k in ("ext", "int", "med", "cov")] == ["8,8", "8,7", "10,0", "28,8"]
    assert abs(r20["rmse_ext"]["mean"] - r20["rmse_int"]["mean"]) < 0.3                              # gleich genau: das intrinsische Mittel ist nicht besser
    assert pc(r20["multi_minima"]["mean"]) == "72" and pc(r100["multi_minima"]["mean"]) == "67" and pc(R.one_peak_row(0.7, 20, 0)["multi_minima"]["mean"]) == "44"


def test_e2_contamination_hurts_the_intrinsic_mean_more():
    r20, r100 = R.one_peak_row(0.9, 20, 20), R.one_peak_row(0.9, 100, 20)
    assert [de(r20["rmse_" + k]["mean"]) for k in ("ext", "int", "med", "cov")] == ["8,7", "13,9", "9,3", "53,0"]
    assert [de(r100["rmse_" + k]["mean"]) for k in ("ext", "int", "med", "cov")] == ["3,9", "6,4", "4,1", "58,5"]
    assert r20["rmse_int"]["mean"] > 1.5 * r20["rmse_ext"]["mean"] and r20["rmse_cov"]["mean"] > 5 * r20["rmse_ext"]["mean"]
    assert R.one_peak_row(0.7, 20, 0)["rmse_int"]["mean"] < R.one_peak_row(0.7, 20, 0)["rmse_ext"]["mean"] and de(R.one_peak_row(0.7, 20, 0)["rmse_ext"]["mean"]) == "11,3"


def test_e2_skewed_data_have_no_common_center():
    assert de(R.skew_row(100)["dist_mean_deg"]["mean"]) == "5,5" and pc(R.skew_row(100)["dist_gt_2"]["mean"]) == "100"
    assert de(R.skew_row(20)["dist_mean_deg"]["mean"]) == "5,7" and pc(R.skew_row(20)["dist_gt_2"]["mean"]) == "91"


def test_e2_two_clusters_at_120_degrees():
    for split in X.SPLITS:
        row = R.cluster_row(120, split)
        assert pc(row["ext_empty"]["mean"]) == "100" and pc(row["int_empty"]["mean"]) == "100" and pc(row["cov_empty"]["mean"]) == "100"
    assert de(R.cluster_row(120, "binomial 50/50")["R"]["mean"], 2) == "0,51"
    assert pc(R.cluster_row(120, "binomial 50/50")["med_a"]["mean"] + R.cluster_row(120, "binomial 50/50")["med_b"]["mean"]) == "71"
    assert pc(R.cluster_row(120, "fest 50/50")["med_a"]["mean"]) == "28" and pc(R.cluster_row(120, "fest 60/40")["med_a"]["mean"]) == "99"


def test_e2_antipodal_clusters():
    binom, sixty = R.cluster_row(180, "binomial 50/50"), R.cluster_row(180, "fest 60/40")
    assert de(binom["R"]["mean"], 2) == "0,13" and [pc(binom[k]["mean"]) for k in ("ext_a", "ext_b", "ext_empty")] == ["32", "33", "35"]
    assert pc(binom["int_empty"]["mean"]) == "100" and pc(binom["cov_empty"]["mean"]) == "100" and [pc(binom[k]["mean"]) for k in ("med_a", "med_b", "med_empty")] == ["42", "42", "16"]
    assert de(sixty["R"]["mean"], 2) == "0,20" and pc(sixty["ext_a"]["mean"]) == "96" and pc(sixty["med_a"]["mean"]) == "98" and pc(sixty["int_empty"]["mean"]) == "100"


def test_e2_one_lorry():
    shifts = lambda d: [de(R.lorry_row(d)[k + "_shift"]["mean"]) for k in ("ext", "int", "med", "cov")]
    assert shifts(120) == ["4,9", "2,9", "71,0", "0,2"] and shifts(180) == ["97,8", "20,8", "143,5", "6,6"]
    over60 = [pc(R.lorry_row(180)[k + "_gt_60"]["mean"]) for k in ("ext", "int", "med", "cov")]
    assert over60 == ["79", "9", "100", "4"] and pc(R.lorry_row(120)["med_gt_60"]["mean"]) == "88"
    assert de(R.lorry_row(180)["R"]["mean"], 2) == "0,04" and de(R.lorry_row(120)["R"]["mean"], 2) == "0,49"
    assert R.lorry_row(180)["n"] == 41


def test_e2_grid_miscount():
    g = R.grid_miscount()["wrong_share"]
    assert pc(g["mean"], 1) == "6,0" and pc(g["std"], 1) == "1,5" and R.grid_miscount()["grid_points"] == 720 and R.grid_miscount()["runs_per_seed"] == 500


# ---------------------------------------------------------------- E3 (H7, H8)

def test_e3_gate_errors_and_window_coverage():
    err = {s: (de(R.torzeit_row(s)["rows"]["Arithmetisch (Tag ab 0 Uhr)"]["err_min"]["mean"], 0), de(R.torzeit_row(s)["rows"]["Kreismittel (arctan2)"]["err_min"]["mean"], 0)) for s in X.SIGMAS_H}
    assert err == {0.5: ("26", "3"), 1.0: ("52", "6"), 2.0: ("99", "13"), 3.0: ("147", "19")}
    one = R.torzeit_row(1.0)["rows"]
    assert pc(one["Arithmetisch (Tag ab 0 Uhr)"]["err_gt_1h"]["mean"]) == "14" and pc(one["Kreismittel (arctan2)"]["err_gt_1h"]["mean"]) == "0" and pc(R.torzeit_row(3.0)["rows"]["Arithmetisch (Tag ab 0 Uhr)"]["err_gt_1h"]["mean"]) == "43"
    assert [pc(one[n]["coverage"]["mean"]) for n in ("Arithmetisch (Tag ab 0 Uhr)", "Kreismittel (arctan2)", "Zirkulärer Median p = 1", "Kleinster Bogen p = ∞")] == ["60", "69", "69", "66"]
    assert pc(one["_oracle"]["mean"]) == "68" and de(one["Zirkulärer Median p = 1"]["err_min"]["mean"], 0) == "8" and de(one["Kleinster Bogen p = ∞"]["err_min"]["mean"], 0) == "16"
    assert de(one["Kreismittel (arctan2)"]["err_min"]["mean"], 1) == de(one["Fréchet p = 2"]["err_min"]["mean"], 1) == "6,2"


def test_e3_night_shift():
    rows = R.nightshift()["rows"]
    a, c = rows["Arithmetisch (Tag ab 0 Uhr)"], rows["Kreismittel (arctan2)"]
    assert de(a["median_estimate_h"]["mean"], 2) == "16,15" and de(a["err_min"]["mean"], 0) == "444" and pc(a["coverage"]["mean"]) == "0"
    assert de(c["median_estimate_h"]["mean"], 2) == "23,50" and de(c["err_min"]["mean"], 1) == "6,3" and pc(c["coverage"]["mean"]) == "69"
    assert de(rows["Zirkulärer Median p = 1"]["err_min"]["mean"], 1) == "7,9" and de(rows["Kleinster Bogen p = ∞"]["err_min"]["mean"], 1) == "15,3"
    assert de(rows["Arithmetisch (Tag ab 12 Uhr)"]["err_min"]["mean"], 1) == "6,3"                      # ein Schnitt weit von der Häufung: so gut wie das Kreismittel


def test_e3_cut_dependence():
    cut = R.cut_cell()
    assert pc(cut["range_gt_1h"]["mean"]) == "58" and de(cut["range_mean_min"]["mean"], 0) == "191"


def test_e3_two_peaks():
    rows = R.two_peak()["rows"]
    assert de(rows["_R"]["mean"], 3) == "0,107" and pc(R.two_peak()["best_window"]["mean"]) == "41"
    cov = [pc(rows[n]["coverage"]["mean"]) for n in ("Arithmetisch (Tag ab 0 Uhr)", "Kreismittel (arctan2)", "Fréchet p = 2", "Zirkulärer Median p = 1", "Kleinster Bogen p = ∞")]
    assert cov == ["0", "24", "0", "30", "0"]
    assert [pc(rows[n]["empty"]["mean"]) for n in ("Arithmetisch (Tag ab 0 Uhr)", "Fréchet p = 2", "Kleinster Bogen p = ∞")] == ["100", "100", "100"]
    assert pc(rows["Kreismittel (arctan2)"]["near_peak"]["mean"]) == "73" and pc(rows["Zirkulärer Median p = 1"]["near_peak"]["mean"]) == "93"


# ---------------------------------------------------------------- E4 (H9, H10)

def test_e4_the_two_suppliers_by_hand():
    """Mittel 20·e^(0,15²/2); E[X²] = 500·e^(0,15²·2); Quantilmittel lognormal mit Median 20."""
    e = R.supplier_example()
    assert e["mix_mean"] == pytest.approx(20 * math.exp(0.01125)) and de(e["mix_mean"], 2) == "20,23"
    assert e["mix_sd"] == pytest.approx(math.sqrt(500 * math.exp(0.045) - e["mix_mean"] ** 2)) and de(e["mix_sd"], 2) == "10,67" and de(e["vin_sd"], 2) == "3,05"
    assert de(e["sd_ratio"], 1) == "3,5" and pc(e["range_mix"], 1) == "5,8" and pc(e["range_vin"], 1) == "90,4"


def test_e4_simulation_confirms_the_formulas():
    sim, e = R.supplier_simulation(), R.supplier_example()
    assert de(sim["mix_sd"]["mean"], 2) == "10,67" and de(sim["vin_sd"]["mean"], 2) == "3,05" and de(sim["mix_mean"]["mean"], 1) == "20,2"
    assert pc(sim["mix_range"]["mean"], 1) == "5,8" and pc(sim["vin_range"]["mean"], 1) == "90,4"
    assert abs(sim["mix_sd"]["mean"] - e["mix_sd"]) < 0.02 and abs(sim["vin_sd"]["mean"] - e["vin_sd"]) < 0.02


def test_e4_order_point_and_actual_service_level():
    e, e3 = R.supplier_example(), R.supplier_example_s03()
    assert {k: de(v, 1) for k, v in e["plans"].items()} == {"Mischung, Quantil (richtig)": "36,4", "Quantilmittel, Quantil": "25,6", "Quantilmittel: Mittel + z · Std": "25,2", "Typische Lieferzeit (Mittel)": "20,2"}
    assert {k: pc(v, 1) for k, v in e["service_at_plan"].items()} == {"Mischung, Quantil (richtig)": "95,0", "Quantilmittel, Quantil": "57,2", "Quantilmittel: Mittel + z · Std": "56,2", "Typische Lieferzeit (Mittel)": "50,2"}
    assert {k: de(v, 1) for k, v in e3["plans"].items()}["Mischung, Quantil (richtig)"] == "44,1" and de(e3["plans"]["Quantilmittel, Quantil"], 1) == "32,8"
    assert {k: pc(v, 1) for k, v in e3["service_at_plan"].items()} == {"Mischung, Quantil (richtig)": "95,0", "Quantilmittel, Quantil": "80,8", "Quantilmittel: Mittel + z · Std": "78,2", "Typische Lieferzeit (Mittel)": "55,4"}


def test_e4_grid():
    grid = lambda s: [pc(R.supplier_cell(r, s)["service_at_plan"]["Quantilmittel, Quantil"], 1) for r in X.RATIOS]
    assert grid(0.15) == ["95,0", "92,3", "83,3", "69,6", "57,2"] and grid(0.3) == ["95,0", "94,4", "92,1", "87,6", "80,8"] and grid(0.5) == ["95,0", "94,8", "94,1", "92,5", "89,8"]
    ratio = lambda s: [de(R.supplier_cell(r, s)["sd_ratio"], 2) for r in X.RATIOS]
    assert ratio(0.15) == ["1,00", "1,17", "1,67", "2,45", "3,50"] and ratio(0.5) == ["1,00", "1,02", "1,09", "1,23", "1,46"]
    assert pc(R.supplier_cell(3.0, 0.15)["band_mix"], 1) == "6,5" and pc(R.supplier_cell(3.0, 0.15)["band_vin"], 1) == "90,8"


# ---------------------------------------------------------------- E5, E6 (H11, H12)

def test_e5_overconfident_reports():
    over = lambda s: R.forecast_cell(s, "überzuversichtlich")
    diff = {s: (de(over(s)["diffs"]["mix_minus_vin"]["crps_diff"]["mean"], 4), de(over(s)["diffs"]["mix_minus_vin"]["crps_diff"]["std"], 4)) for s in (1.0, 2.0, 4.0)}
    assert diff == {1.0: ("0,0039", "0,0029"), 2.0: ("0,0080", "0,0094"), 4.0: ("-0,0167", "0,0228")}
    for s in X.SE_VALUES:                                                                          # nirgends mehr als eine Streuung zwischen den Seeds vom Nullpunkt entfernt, außer s_e = 0,5
        d = over(s)["diffs"]["mix_minus_vin"]["crps_diff"]
        assert abs(d["mean"]) < 0.01 or abs(d["mean"]) <= d["std"]
    cov = {s: (pc(over(s)["methods"]["Mischung"]["covered"]["mean"], 1), pc(over(s)["methods"]["Quantilmittel"]["covered"]["mean"], 1)) for s in (1.0, 2.0, 4.0)}
    assert cov == {1.0: ("77,4", "70,3"), 2.0: ("72,2", "53,9"), 4.0: ("64,2", "33,3")}
    assert de(over(2.0)["methods"]["Mischung"]["width"]["mean"], 2) == "4,11" and de(over(2.0)["methods"]["Quantilmittel"]["width"]["mean"], 2) == "2,56"
    assert pc(over(2.0)["methods"]["Eine Prognose allein"]["covered"]["mean"], 1) == "44,0" and de(over(2.0)["methods"]["Wahrheit (bekannte Lage)"]["crps"]["mean"], 3) == "0,568"
    assert pc(over(4.0)["diffs"]["mix_minus_vin"]["first_better_share"]["mean"]) == "61" and pc(over(1.0)["diffs"]["mix_minus_vin"]["first_better_share"]["mean"]) == "47"


def test_e5_honest_reports():
    honest = lambda s: R.forecast_cell(s, "ehrlich")
    diff = {s: (de(honest(s)["diffs"]["mix_minus_vin"]["crps_diff"]["mean"], 4), de(honest(s)["diffs"]["mix_minus_vin"]["crps_diff"]["std"], 4)) for s in (1.0, 2.0, 4.0)}
    assert diff == {1.0: ("0,0135", "0,0018"), 2.0: ("0,0509", "0,0036"), 4.0: ("0,1261", "0,0052")}
    assert {s: pc(honest(s)["diffs"]["mix_minus_vin"]["first_better_share"]["mean"]) for s in (1.0, 2.0, 4.0)} == {1.0: "32", 2.0: "26", 4.0: "22"}
    cov = {s: (pc(honest(s)["methods"]["Mischung"]["covered"]["mean"], 1), pc(honest(s)["methods"]["Quantilmittel"]["covered"]["mean"], 1)) for s in (1.0, 2.0, 4.0)}
    assert cov == {1.0: ("89,3", "85,7"), 2.0: ("93,7", "90,5"), 4.0: ("95,0", "91,8")}
    assert [de(honest(2.0)["methods"][m]["crps"]["mean"], 3) for m in ("Mischung", "Quantilmittel")] == ["1,046", "0,995"] and honest(2.0)["sigma_rep"] == pytest.approx(math.sqrt(5.0))


def test_e5_unequal_reported_spreads():
    u = R.forecast_unequal()["diffs"]["mix_minus_vin"]
    assert de(u["crps_diff"]["mean"], 4) == "0,0128" and de(u["crps_diff"]["std"], 4) == "0,0024" and pc(u["first_better_share"]["mean"], 1) == "49,2"


def test_e5_small_errors_make_both_equal():
    assert abs(R.forecast_cell(0.25, "überzuversichtlich")["diffs"]["mix_minus_vin"]["crps_diff"]["mean"]) < 0.0003
    assert de(R.forecast_cell(0.25, "überzuversichtlich")["methods"]["Mischung"]["crps"]["mean"], 3) == "0,572"


def test_e6_scale():
    cell = R.scale_cell(2.0, "überzuversichtlich")
    assert [de(cell["methods"][m]["crps"]["mean"], 3) for m in ("Quantilmittel in x", "Quantilmittel in log x", "Mischung")] == ["0,612", "0,525", "0,557"]
    assert [de(cell["diffs"][k]["crps_diff"]["mean"], 4) for k in ("x_minus_logx", "mix_minus_x", "mix_minus_logx")] == ["0,0866", "-0,0552", "0,0314"] and de(cell["diffs"]["x_minus_logx"]["crps_diff"]["std"], 4) == "0,0031"
    honest = R.scale_cell(2.0, "ehrlich")
    assert [de(honest["diffs"][k]["crps_diff"]["mean"], 4) for k in ("x_minus_logx", "mix_minus_x", "mix_minus_logx")] == ["0,0902", "-0,0342", "0,0560"]
    assert [de(R.scale_cell(r, "überzuversichtlich")["diffs"]["x_minus_logx"]["crps_diff"]["mean"], 4) for r in (0.5, 1.0)] == ["0,0007", "0,0055"]
    assert cell["s_e"] == pytest.approx(0.8)


# ---------------------------------------------------------------- Rasterfehler des CRPS und weitere genannte Zahlen

def test_crps_raster_error_for_100_levels():
    """Rasterfehler des CRPS aus 100 Quantilstufen gegen die geschlossene Formel (Wahrheit N(0, 1), 2000 Beobachtungen aus N(0, 1)): im Mittel unter 0,0001, höchstens 0,003."""
    def phi(z):
        return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

    p = D.quantile_grid(100)
    z = D.norm_ppf(p)
    errors = []
    for y in np.random.default_rng(1).normal(0, 1, 2000):
        closed = y * (2 * float(D.norm_cdf(y)) - 1) + 2 * phi(y) - 1 / math.sqrt(math.pi)
        errors.append(D.crps_from_quantiles(z[None, :], np.array([y]), p)[0] - closed)
    assert abs(np.mean(errors)) < 1e-4 and np.abs(errors).max() < 3e-3


def test_numbers_named_in_the_text_of_the_readme():
    assert math.sqrt(-math.log(0.05)) == pytest.approx(1.7308, abs=1e-4) and math.sqrt(3) == pytest.approx(1.7321, abs=1e-4)           # √(−ln 0,05) ≈ √3
    assert de(float(K.rayleigh_threshold(30)), 3) == "0,316" and de(float(K.rayleigh_threshold(100)), 3) == "0,173"
    assert len(X.SEEDS) == 4 and X.CASES == 1500 and X.CLUSTER_N == 40 and X.LORRY_N == 41 and X.TORZEIT_N == 60 and X.CLUSTER_SIGMA_DEG == 13.0
    assert V.de(math.degrees(math.sqrt(-2 * math.log(0.45))), 0) == "72" and V.de(math.sqrt(-2 * math.log(0.45)), 2) == "1,26"             # ρ = 0,45 entspricht σ = 1,26 rad = 72°
    assert V.de(X.RHO_RASTER[2], 2) == "0,45" and V.de(math.degrees(math.sqrt(-2 * math.log(0.9))), 0) == "26"                          # ρ = 0,9 entspricht σ = 26°


def test_mutation_state_matches_the_readme():
    import json
    from pathlib import Path
    state = json.loads((Path(__file__).resolve().parent / "mutation_state.json").read_text(encoding="utf-8"))
    kinds = [v for v in state.values()]
    assert len(kinds) == 133 and kinds.count("killed") == 103 and kinds.count("equivalent") == 30
    assert {k.split("|")[0] for k in state} == {"mir_circle.py", "mir_dist.py"}
