"""Vorgerechnete Messreihen (data/mir_results.json, erzeugt von tools/make_results.py aus mir_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py)."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import mir_constants as C

PATH = Path(os.environ.get("MIR_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import mir_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def _one(rows: list[dict], **match) -> dict:
    (row,) = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return row


# E1
def uniform_rows() -> list[dict]:
    return load()["m1"]["uniform"]


def uniform_row(n: int) -> dict:
    return _one(uniform_rows(), n=n)


def power_rows() -> list[dict]:
    return load()["m1"]["power"]


def power_row(rho: float, n: int) -> dict:
    return _one(power_rows(), rho=rho, n=n)


# E2
def one_peak_rows() -> list[dict]:
    return load()["m2"]["one_peak"]


def one_peak_row(rho: float, n: int, contamination: int) -> dict:
    return _one(one_peak_rows(), rho=rho, n=n, contamination=contamination)


def skew_rows() -> list[dict]:
    return load()["m2"]["skew"]


def skew_row(n: int) -> dict:
    return _one(skew_rows(), n=n)


def cluster_rows() -> list[dict]:
    return load()["m2"]["clusters"]


def cluster_row(delta: float, split: str) -> dict:
    return _one(cluster_rows(), delta=delta, split=split)


def lorry_rows() -> list[dict]:
    return load()["m2"]["lorry"]


def lorry_row(delta: float) -> dict:
    return _one(lorry_rows(), delta=delta)


def grid_miscount() -> dict:
    return load()["m2"]["grid"]


# E3
def torzeit_rows() -> list[dict]:
    return load()["m3"]["torzeit"]


def torzeit_row(sigma: float) -> dict:
    return _one(torzeit_rows(), sigma=sigma)


def nightshift() -> dict:
    return load()["m3"]["nightshift"]


def cut_cell() -> dict:
    return load()["m3"]["cut"]


def two_peak() -> dict:
    return load()["m3"]["two_peak"]


# E4
def supplier_example() -> dict:
    return load()["m4"]["example"]


def supplier_example_s03() -> dict:
    return load()["m4"]["example_s03"]


def supplier_simulation() -> dict:
    return load()["m4"]["simulation"]


def supplier_grid() -> list[dict]:
    return load()["m4"]["grid"]


def supplier_cell(ratio: float, s: float) -> dict:
    return _one(supplier_grid(), ratio=ratio, s=s)


# E5, E6
def forecast_cells() -> list[dict]:
    return load()["m5"]["cells"]


def forecast_cell(s_e: float, calibration: str) -> dict:
    return _one(forecast_cells(), s_e=s_e, calibration=calibration)


def forecast_unequal() -> dict:
    return load()["m5"]["unequal"]


def scale_cells() -> list[dict]:
    return load()["m6"]["cells"]


def scale_cell(rel: float, calibration: str) -> dict:
    return _one(scale_cells(), rel=rel, calibration=calibration)
