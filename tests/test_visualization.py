"""Plotly-Abbildungen: baubar für jedes Preset und jede Tabelle, Achsen fest, Beschriftungen und Spuren wie dokumentiert."""

import numpy as np
import plotly.graph_objects as go
import pytest

import mir_constants as C
import mir_evaluation as V
import mir_experiments as X
import mir_results as R
import mir_visualization as W


def _fixed(fig):
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.fixedrange is True
    assert fig.layout.dragmode is False


def test_clock_has_the_arrivals_the_resultant_and_five_centres():
    a = V.analyse_gate(V.Settings())
    fig = W.build_clock(a)
    names = [t.name for t in fig.data]
    assert names[0] == "Ankünfte" and len(fig.data[0].theta) == 60 and names[1].startswith("Resultierende")
    assert sum(1 for n in names if n.startswith("Fenster")) == 1                               # ein Gipfel: Fenster um die wahre Mitte
    centres = [t for t in fig.data if t.marker.symbol == "diamond"]
    assert len(centres) == 5 and [t.theta[0] for t in centres] == pytest.approx([h * 15.0 for h in a.centres.values()])
    assert fig.layout.polar.angularaxis.direction == "clockwise" and fig.layout.polar.angularaxis.rotation == 90 and fig.layout.dragmode is False


def test_clock_without_a_window_for_two_peaks():
    a = V.analyse_gate(V.Settings(share=50))
    assert not any(t.name and t.name.startswith("Fenster") for t in W.build_clock(a).data)


def test_resultant_arrow_has_the_length_r():
    a = V.analyse_gate(V.Settings(mu=14.0))
    arrow = W.build_clock(a).data[1]
    assert arrow.r == pytest.approx((0.0, a.R))


def test_supplier_figures_for_the_extremes():
    for ratio, s, wb in ((1.0, 0.05, 10), (4.0, 0.8, 90), (3.0, 0.15, 50)):
        summary = X.supplier_summary(ratio, s, float(wb))
        for fig in (W.build_supplier_densities(summary), W.build_quantile_functions(summary)):
            assert isinstance(fig, go.Figure) and len(fig.data) == 4
            _fixed(fig)
        dens = W.build_supplier_densities(summary)
        assert all(np.isfinite(np.asarray(t.y, dtype=float)).all() for t in dens.data)


def test_density_traces_integrate_to_the_weights():
    summary = X.supplier_summary(3.0, 0.15, 50.0)
    fig = W.build_supplier_densities(summary)
    x = np.asarray(fig.data[2].x)
    mix_area = np.trapezoid(np.asarray(fig.data[2].y), x)
    vin_area = np.trapezoid(np.asarray(fig.data[3].y), x)
    assert mix_area == pytest.approx(1.0, abs=0.01) and vin_area == pytest.approx(1.0, abs=0.01)
    assert np.trapezoid(np.asarray(fig.data[0].y), x) == pytest.approx(0.5, abs=0.01)


def test_experiment_charts_are_built_from_the_stored_tables():
    figs = [W.build_uniform_chart(R.uniform_rows()), W.build_power_chart(R.power_rows()), W.build_lorry_chart(R.lorry_rows()), W.build_torzeit_chart(R.torzeit_rows()),
            W.build_service_chart(R.supplier_grid()), W.build_forecast_chart(R.forecast_cells()), W.build_crps_chart(R.forecast_cells()), W.build_scale_chart(R.scale_cells())]
    for split in X.SPLITS:
        figs.append(W.build_cluster_chart(R.cluster_rows(), split))
    for fig in figs:
        assert isinstance(fig, go.Figure) and len(fig.data) >= 2
        _fixed(fig)


def test_chart_values_come_from_the_tables():
    fig = W.build_uniform_chart(R.uniform_rows())
    assert list(fig.data[0].y) == [r["mean"]["mean"] for r in R.uniform_rows()] and fig.layout.xaxis.type == "log"
    service = W.build_service_chart(R.supplier_grid())
    assert len(service.data) == len(X.S_VALUES) and list(service.data[0].y)[-1] == pytest.approx(100 * R.supplier_cell(3.0, 0.15)["service_at_plan"]["Quantilmittel, Quantil"])
    clusters = W.build_cluster_chart(R.cluster_rows(), "fest 60/40")
    assert [t.name for t in clusters.data] == ["Kreismittel", "Fréchet p = 2", "zirk. Median", "kleinster Bogen"] and len(clusters.data[0].y) == len(X.DELTAS)
    cov = W.build_forecast_chart(R.forecast_cells())
    assert len(cov.data) == 4 and cov.data[0].y[0] == pytest.approx(100 * R.forecast_cell(0.25, "überzuversichtlich")["methods"]["Mischung"]["covered"]["mean"])


def test_every_preset_clock_builds():
    for name, preset in C.PRESETS.items():
        a = V.analyse_gate(V.Settings(**preset))
        assert len(W.build_clock(a).data) >= 7, name
