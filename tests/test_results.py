"""Die vorgerechneten Messreihen (data/mir_results.json): vollständig, aktuell gegenüber den Quelldateien und in sich stimmig."""

import json

import pytest

import mir_constants as C
import mir_experiments as X
import mir_results as R


def test_results_file_is_valid_utf8_json_with_all_parts():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta", "m1", "m2", "m3", "m4", "m5", "m6"}
    assert "Arithmetisch (Tag ab 0 Uhr)" in text and "überzuversichtlich" in text            # Umlaute stehen im Klartext (kein ASCII-Escape)
    assert "\r" not in text


def test_e1_tables_are_complete():
    assert [r["n"] for r in R.uniform_rows()] == list(X.N_RASTER)
    for r in R.uniform_rows():
        assert len(r["mean"]["per_seed"]) == len(X.SEEDS) and r["runs_per_seed"] == X.runs_uniform(r["n"])
    assert len(R.power_rows()) == len(X.RHO_RASTER) * len(X.N_POWER)
    assert R.power_row(0.45, 30)["n"] == 30 and R.uniform_row(30)["n"] == 30


def test_e2_tables_are_complete_and_shares_add_up():
    assert [(r["rho"], r["n"], r["contamination"]) for r in R.one_peak_rows()] == list(X.ONE_PEAK_SETTINGS)
    assert [r["n"] for r in R.skew_rows()] == list(X.SKEW_SETTINGS)
    assert len(R.cluster_rows()) == len(X.DELTAS) * len(X.SPLITS)
    for r in R.cluster_rows():
        for key in ("ext", "int", "med", "cov"):
            assert sum(r[f"{key}_{w}"]["mean"] for w in ("a", "b", "empty")) == pytest.approx(1.0)
    assert [r["delta"] for r in R.lorry_rows()] == list(X.DELTAS) and R.lorry_row(180)["n"] == X.LORRY_N
    assert R.cluster_row(120, "fest 60/40")["split"] == "fest 60/40" and R.grid_miscount()["grid_points"] == 720


def test_e3_to_e6_tables_have_their_rows():
    assert [r["sigma"] for r in R.torzeit_rows()] == list(X.SIGMAS_H) and R.torzeit_row(1.0)["n"] == X.TORZEIT_N
    assert set(R.torzeit_row(1.0)["rows"]) == set(X.TABLE_NAMES) | {"_R", "_oracle"}
    assert R.nightshift()["mu"] == 23.5 and R.cut_cell()["cuts"] == list(X.CUTS) and R.two_peak()["peaks"] == [6.0, 18.0]
    assert len(R.supplier_grid()) == len(X.S_VALUES) * len(X.RATIOS) and R.supplier_cell(3.0, 0.15)["ratio"] == 3.0
    assert R.supplier_example()["s"] == 0.15 and R.supplier_example_s03()["s"] == 0.3 and R.supplier_simulation()["draws_per_seed"] == 200000
    assert len(R.forecast_cells()) == len(C.CALIBRATIONS) * len(X.SE_VALUES) and R.forecast_cell(2.0, "ehrlich")["cases"] == X.CASES
    assert R.forecast_unequal()["sigma_rep"] == [0.5, 2.0]
    assert len(R.scale_cells()) == len(C.CALIBRATIONS) * len(X.SCALE_REL) and R.scale_cell(2.0, "ehrlich")["rel"] == 2.0


def assert_same_cell(live, stored, rel=1e-9):
    """Gleiche Struktur, Zeichenketten und ganze Zahlen exakt, Gleitkommazahlen bis auf die letzte Binärstelle: `log`, `cos` und die Summation können je Plattform
    (Windows, Linux) in der letzten Stelle abweichen; die Messreihe wird auf einem anderen Rechner nachgerechnet."""
    if isinstance(stored, dict):
        assert isinstance(live, dict) and set(live) == set(stored)
        for key in stored:
            assert_same_cell(live[key], stored[key], rel)
    elif isinstance(stored, (list, tuple)):
        assert isinstance(live, (list, tuple)) and len(live) == len(stored)
        for a, b in zip(live, stored):
            assert_same_cell(a, b, rel)
    elif isinstance(stored, float):
        assert live == pytest.approx(stored, rel=rel, abs=1e-15)
    else:
        assert live == stored


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds, gleiche Läufe)."""
    row = R.uniform_row(10)
    assert_same_cell(X.uniform_cell(10, row["runs_per_seed"]), row)
    assert_same_cell(X.power_cell(0.45, 30, 4000), R.power_row(0.45, 30))
    assert_same_cell(X.supplier_summary(3.0, 0.15), R.supplier_example())
    live = X.forecast_cell(1.0, "ehrlich", X.CASES, seeds=X.SEEDS[:1])
    stored = R.forecast_cell(1.0, "ehrlich")
    assert live["methods"]["Mischung"]["crps"]["per_seed"][0] == pytest.approx(stored["methods"]["Mischung"]["crps"]["per_seed"][0], rel=1e-12)
    assert X.one_lorry_cell(120, 600, seeds=X.SEEDS[:1])["ext_shift"]["per_seed"][0] == pytest.approx(R.lorry_row(120)["ext_shift"]["per_seed"][0], rel=1e-12)
