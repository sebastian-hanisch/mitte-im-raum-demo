"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, keine Seeds. Damit prüft der Test die Grenzen (<, <=) und jede genannte Zahl direkt."""

import math

import pytest

import mir_evaluation as V


def ag(mean, std=0.0):
    return {"mean": mean, "std": std, "per_seed": [mean]}


# ---------------------------------------------------------------- Formatierung

def test_german_number_formatting():
    assert V.de(1234.5678, 2) == "1234,57" and V.de(0.25, 1) == "0,2" and V.de(-3.0, 0) == "-3"
    assert V.de(2.0, 1) == "2,0" and V.pct(0.5724857, 1) == "57,2" and V.pct(0.9999, 0) == "100"
    assert V.hhmm(23.5) == "23:30" and V.hhmm(0.0) == "00:00" and V.hhmm(24.0) == "00:00" and V.hhmm(23.9999) == "00:00" and V.hhmm(6.25) == "06:15"


# ---------------------------------------------------------------- Teil A

def test_gate_verdict_random_when_r_is_below_the_threshold():
    n = 60
    thr = math.sqrt(-math.log(0.05) / n)
    kind, text = V.verdict_gate(R=thr - 0.001, n=n, circle_spread_h=3.0, arith_gap_h=0.0, cut=0, cut_range_min=0.0, n_second=0)
    assert kind == "warning" and "unter der Schwelle 0,223" in text and "fast Zufall" in text and "zwei Häufungen" not in text
    assert f"R = {V.de(thr - 0.001, 3)}" in text and "n = 60" in text


def test_gate_verdict_threshold_is_inclusive_for_a_real_cluster():
    n = 60
    thr = math.sqrt(-math.log(0.05) / n)
    kind, text = V.verdict_gate(R=thr, n=n, circle_spread_h=0.0, arith_gap_h=0.0, cut=0, cut_range_min=0.0, n_second=0)
    assert kind == "success" and "Ankünfte häufen sich" in text


def test_gate_verdict_names_two_clusters_from_one_fifth_on():
    kind, text = V.verdict_gate(R=0.05, n=40, circle_spread_h=5.5, arith_gap_h=0.0, cut=0, cut_range_min=0.0, n_second=8)        # 8 von 40 = genau ein Fünftel
    assert kind == "warning" and "zwei Häufungen (32 und 8 Ankünfte)" in text and "bis zu 5,5 Stunden auseinander" in text
    kind2, text2 = V.verdict_gate(R=0.05, n=40, circle_spread_h=5.5, arith_gap_h=0.0, cut=0, cut_range_min=0.0, n_second=7)       # knapp darunter
    assert "zwei Häufungen" not in text2


def test_gate_verdict_blames_the_cut_when_the_arithmetic_mean_is_far_off():
    kind, text = V.verdict_gate(R=0.9, n=60, circle_spread_h=0.1, arith_gap_h=7.2, cut=0, cut_range_min=190.0, n_second=0)
    assert kind == "warning" and "Tag ab 0 Uhr" in text and "7,2 Stunden" in text and "190 Minuten" in text
    assert V.verdict_gate(R=0.9, n=60, circle_spread_h=0.1, arith_gap_h=0.25, cut=6, cut_range_min=3.0, n_second=0)[0] == "success"          # genau 15 Minuten: noch kein Befund
    kind, text = V.verdict_gate(R=0.9, n=60, circle_spread_h=0.2, arith_gap_h=0.1, cut=6, cut_range_min=3.0, n_second=0)
    assert kind == "success" and "6 Minuten vom Kreismittel" in text and "12 Minuten auseinander" in text and "R = 0,900" in text


# ---------------------------------------------------------------- Teil B

def _summary(sd_ratio, real, mix_sd=10.7, vin_sd=3.1):
    return {"sd_ratio": sd_ratio, "service": 95.0, "mix_mean": 20.2, "mix_sd": mix_sd, "vin_sd": vin_sd,
            "plans": {V.PLAN_QUANTILE: 25.6, V.PLAN_MIXTURE: 36.4}, "service_at_plan": {V.PLAN_QUANTILE: real}}


def test_supplier_verdict_warns_when_the_spread_differs():
    kind, text = V.verdict_suppliers(_summary(3.5, 0.572))
    assert kind == "warning"
    assert "dasselbe Mittel (20,2 Tage)" in text and "3,5-mal so stark" in text and "(10,7 gegen 3,1 Tage)" in text
    assert "95-%-Quantil des Quantilmittels (25,6 Tage)" in text and "tatsächlich 57,2 % Servicegrad statt 95 %" in text and "36,4 Tagen" in text


def test_supplier_verdict_threshold_is_five_percent():
    assert V.verdict_suppliers(_summary(1.05, 0.94))[0] == "warning"
    kind, text = V.verdict_suppliers(_summary(1.049, 0.949))
    assert kind == "success" and "nur 4,9 % stärker" in text and "erreicht 94,9 % Servicegrad" in text and "(25,6 Tage)" in text


# ---------------------------------------------------------------- Prognosen

def _forecast(diff, std, cov_mix, cov_vin, calibration="überzuversichtlich"):
    return {"calibration": calibration, "s_e": 2.0,
            "diffs": {"mix_minus_vin": {"crps_diff": ag(diff, std)}},
            "methods": {"Mischung": {"crps": ag(1.041), "covered": ag(cov_mix)}, "Quantilmittel": {"crps": ag(1.033), "covered": ag(cov_vin)}}}


def test_forecast_verdict_three_cases_for_the_crps_difference():
    kind, text = V.verdict_forecast(_forecast(0.008, 0.009, 0.722, 0.539))
    assert kind == "info" and "CRPS nicht zu unterscheiden (Mischung 1,041, Quantilmittel 1,033, Differenz 0,008 ± 0,009" in text
    assert "bei der Mischung 72,2 %, beim Quantilmittel 53,9 %" in text and "die Mischung liegt näher am Soll" in text and "überzuversichtlich, Streuung des Lagefehlers 2,00" in text
    assert "Kein allgemeiner Sieger" in text
    assert "nicht zu unterscheiden" in V.verdict_forecast(_forecast(0.009, 0.009, 0.7, 0.7))[1]                     # Differenz genau eine Streuung: noch nicht erkennbar
    assert "Das Quantilmittel hat den besseren CRPS (1,033 gegen 1,041, Differenz 0,051 ± 0,004)" in V.verdict_forecast(_forecast(0.051, 0.004, 0.93, 0.90))[1]
    assert "Die Mischung hat den besseren CRPS (1,041 gegen 1,033, Differenz -0,017 ± 0,010)" in V.verdict_forecast(_forecast(-0.017, 0.010, 0.64, 0.33))[1]


def test_forecast_verdict_picks_the_coverage_closer_to_eighty_percent():
    assert "das Quantilmittel liegt näher am Soll" in V.verdict_forecast(_forecast(0.05, 0.004, 0.937, 0.905))[1].replace("Das Quantilmittel hat", "")
    assert "die Mischung liegt näher am Soll" in V.verdict_forecast(_forecast(0.05, 0.004, 0.81, 0.90))[1]


# ---------------------------------------------------------------- Experimente

def _uniform():
    return {"n": 30, "mean": ag(0.162), "r95": ag(0.315), "threshold": 0.3162, "below_01": ag(0.257)}


def _power(power, rho=0.25, n=100):
    return {"rho": rho, "n": n, "power": ag(power), "err_mean_deg": ag(13.2), "err_gt_45": ag(0.011)}


def test_e1_verdict_names_chance_level_and_power():
    kind, text = V.verdict_e1(_uniform(), _power(0.91))
    assert kind == "success"
    assert "n = 30) ist R im Mittel 0,162, in 5 % der Nächte über 0,315 (Schwelle √(−ln 0,05 / n) = 0,316) und in 26 % der Nächte unter 0,1" in text
    assert "ρ = 0,250 und n = 100 erkennt der Test die Häufung in 91 % der Nächte" in text and "13° daneben, in 1 % der Nächte um mehr als 45°" in text
    assert V.verdict_e1(_uniform(), _power(0.80))[0] == "success" and V.verdict_e1(_uniform(), _power(0.7999))[0] == "warning"


def _peak(contamination=0):
    return {"rho": 0.45, "n": 20, "contamination": contamination, "dist_mean_deg": ag(7.4), "dist_gt_10": ag(0.26), "rmse_ext": ag(20.9), "rmse_int": ag(20.9), "rmse_med": ag(23.9),
            "rmse_cov": ag(38.8), "multi_minima": ag(0.72)}


def test_e2_peak_verdict_texts():
    kind, text = V.verdict_e2_peak(_peak())
    assert kind == "info" and "Ein Gipfel (ρ = 0,45, n = 20):" in text and "7,4° auseinander" in text and "(in 26 % der Nächte um mehr als 10°)" in text
    assert "Kreismittel 20,9°, Fréchet 20,9°, Median 23,9°, kleinster Bogen 38,8°" in text and "In 72 % der Datensätze" in text and "Störankünften" not in text
    assert "n = 20) mit 20 % gleichverteilten Störankünften:" in V.verdict_e2_peak(_peak(20))[1]


def _clusters(ext_empty):
    row = {"delta": 180, "split": "binomial 50/50", "R": ag(0.13), "cov_empty": ag(1.0), "med_a": ag(0.42), "med_b": ag(0.42), "ext_empty": ag(ext_empty), "int_empty": ag(1.0)}
    return row


def test_e2_clusters_verdict_threshold_and_numbers():
    kind, text = V.verdict_e2_clusters(_clusters(0.35))
    assert kind == "info" and "Abstand 180° (binomial 50/50, R = 0,13)" in text and "Kreismittel liegt in 35 % der Datensätze im Leeren" in text
    assert "Fréchet-Mittel in 100 %, der kleinste Bogen in 100 %" in text and "Median sitzt in 84 % auf einer der Häufungen (Häufung A: 42 %, B: 42 %)" in text
    assert V.verdict_e2_clusters(_clusters(0.5))[0] == "info" and V.verdict_e2_clusters(_clusters(0.51))[0] == "warning"


def _lorry(ext=97.8, int_=20.8, med=143.5, cov=6.6):
    return {"delta": 180, "n": 41, "R": ag(0.04), "ext_shift": ag(ext), "int_shift": ag(int_), "med_shift": ag(med), "cov_shift": ag(cov)}


def test_e2_lorry_verdict_threshold_and_numbers():
    kind, text = V.verdict_e2_lorry(_lorry())
    assert kind == "warning" and "(n = 41, Abstand 180°, R = 0,04)" in text
    assert "Kreismittel verschiebt sich im Mittel um 97,8°, das Fréchet-Mittel um 20,8°, der Median um 143,5°, der kleinste Bogen um 6,6°" in text
    assert V.verdict_e2_lorry(_lorry(ext=60.0, int_=20.0, med=59.9, cov=1.0))[0] == "success"                      # genau 60° ist noch kein Sprung
    assert V.verdict_e2_lorry(_lorry(ext=10.0, int_=20.0, med=60.1, cov=1.0))[0] == "warning"
    assert V.verdict_e2_lorry(_lorry(ext=10.0, int_=20.0, med=10.0, cov=60.5))[0] == "warning" and V.verdict_e2_lorry(_lorry(ext=10.0, int_=60.5, med=10.0, cov=1.0))[0] == "warning"


def test_e2_grid_verdict():
    kind, text = V.verdict_e2_grid({"grid_points": 720, "wrong_share": ag(0.06, 0.015)})
    assert kind == "info" and "Gitter mit 720 Punkten" in text and "in 6,0 % der Datensätze falsch" in text and "±1,5 Prozentpunkte" in text and "zählt deshalb exakt" in text


def _gate_rows(arith_err=51.6, circle_err=6.2):
    return {"Arithmetisch (Tag ab 0 Uhr)": {"err_min": ag(arith_err), "coverage": ag(0.60), "median_estimate_h": ag(16.15)},
            "Kreismittel (arctan2)": {"err_min": ag(circle_err), "coverage": ag(0.69), "median_estimate_h": ag(23.5)}, "_oracle": ag(0.683)}


def test_e3_verdicts():
    kind, text = V.verdict_e3({"sigma": 1.0, "rows": _gate_rows()})
    assert kind == "warning" and "Streuung von 1,0 Stunden" in text and "im Mittel 52 Minuten neben der wahren Mitte, das Kreismittel 6 Minuten" in text
    assert "fängt 60 % gegen 69 % der Ankünfte (ein Fenster um die wahre Mitte: 68 %)" in text
    kind, text = V.verdict_e3_night({"mu": 23.5, "sigma": 1.0, "n": 60, "rows": {**_gate_rows(444.0, 6.3)}}, {"range_gt_1h": ag(0.58)})
    assert kind == "warning" and "Mitte 23:30 Uhr (Streuung 1,0 h, n = 60)" in text and "im Median bei 16,1 Uhr, im Mittel 444 Minuten neben der Mitte, Fensterabdeckung 60 %" in text
    assert "Kreismittel liegt bei 23,50 Uhr, 6 Minuten daneben, Fensterabdeckung 69 %" in text and "in 58 % der Nächte um mehr als eine Stunde" in text
    rows = {name: {"coverage": ag(c)} for name, c in (("Arithmetisch (Tag ab 0 Uhr)", 0.0), ("Kreismittel (arctan2)", 0.24), ("Fréchet p = 2", 0.0), ("Zirkulärer Median p = 1", 0.30), ("Kleinster Bogen p = ∞", 0.0))}
    rows["_R"] = ag(0.107)
    kind, text = V.verdict_e3_peaks({"peaks": [6.0, 18.0], "rows": rows, "best_window": ag(0.41)})
    assert kind == "warning" and "(06:00 und 18:00 Uhr, je etwa die Hälfte, R = 0,11)" in text
    assert "arithmetisch 0 %, Kreismittel 24 %, Fréchet 0 %, Median 30 %, kleinster Bogen 0 %. Das beste Einzelfenster fängt 41 %." in text


def _scale(d, std):
    return {"rel": 2.0, "calibration": "überzuversichtlich", "diffs": {"x_minus_logx": {"crps_diff": ag(d, std)}},
            "methods": {"Quantilmittel in x": {"crps": ag(0.612)}, "Quantilmittel in log x": {"crps": ag(0.525)}, "Mischung": {"crps": ag(0.557)}}}


def test_e6_verdict_threshold_and_numbers():
    kind, text = V.verdict_e6(_scale(0.0866, 0.0031))
    assert kind == "warning" and "Verhältnis 2,0 zur Streuung der Nachfrage (überzuversichtlich)" in text and "in x gemittelt 0,612, in log x gemittelt 0,525" in text
    assert "Differenz 0,0866 ± 0,0031" in text and "Die Mischung (0,557) ändert sich" in text
    assert V.verdict_e6(_scale(0.0031, 0.0031))[0] == "info" and V.verdict_e6(_scale(-0.0032, 0.0031))[0] == "warning"
