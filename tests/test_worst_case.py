"""Worst-Case am Slider-Maximum (DEMO-PLAYBOOK §1c): Hänger und Aufblähen im eigenen Prozess mit Zeitgrenze fangen. Keine Zeit-Assertion, nur ein Timeout als Netz."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = """
import sys
sys.path.insert(0, {root!r})
import mir_constants as C, mir_evaluation as V, mir_experiments as X
a = V.analyse_gate(V.Settings(n=C.N_MAX, share=50, gap=12.0, sigma=C.SIGMA_MAX))
assert 0.0 <= a.R <= 1.0 and a.n_minima >= 1
nights = int(max(20, min(C.LIVE_NIGHTS, C.LIVE_BUDGET // (C.N_MAX * C.N_MAX))))
g = X.gate_study(C.N_MAX, 12.0, 1.0, 0, 12.0, 0, 1.0, nights, seeds=X.SEEDS[:1])
assert g['rows']['_R']['mean'] > 0.5
f = X.forecast_cell(C.SE_MAX, 'ehrlich', C.LIVE_CASES, seeds=X.SEEDS[:1])
assert f['cases'] == C.LIVE_CASES
s = X.supplier_summary(C.RATIO_MAX, C.S_MAX, 90.0, 99.0)
assert s['service_at_plan']['Mischung, Quantil (richtig)'] > 0.98
print('ok')
"""


def test_slider_maximum_finishes_in_its_own_process_within_the_time_limit():
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", CODE.format(root=str(ROOT))], capture_output=True, text=True, timeout=180)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-500:]


def test_quantile_search_terminates_on_a_difficult_mixture():
    code = ("import sys; sys.path.insert(0, %r); import numpy as np, mir_dist as D\n"
            "q = D.mixture_quantile(np.array([1e-12, 0.5, 1 - 1e-12]), [0.0, 1000.0], [0.001, 500.0], [0.5, 0.5]); assert np.isfinite(q).all(); print('ok')" % str(ROOT))
    done = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-500:]
