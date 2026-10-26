"""Plotly-Abbildungen: alle Achsen fest (`fixedrange`), Beschriftungen als Annotationen, Farben aus mir_constants.py."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import mir_circle as K
import mir_constants as C
import mir_dist as D
import mir_experiments as X

CENTRE_SHORT = {
    "Arithmetisch (Tag ab 0 Uhr)": "arithmetisch",
    "Kreismittel (arctan2)": "Kreismittel",
    "Fréchet p = 2": "Fréchet p = 2",
    "Zirkulärer Median p = 1": "zirk. Median",
    "Kleinster Bogen p = ∞": "kleinster Bogen",
    "Arithmetisch (Tag ab 12 Uhr)": "arithmetisch (ab 12 Uhr)",
}


def _layout(fig: go.Figure, xtitle: str = "", ytitle: str = "", height: int = 380, legend_below: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=55, r=20, t=30, b=95 if legend_below else 40), xaxis_title=xtitle, yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else None, dragmode=False,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# ---------------------------------------------------------------- Teil A: der Uhr-Kreis

def build_clock(a) -> go.Figure:
    """Die Ankünfte einer Nacht auf dem 24-Stunden-Kreis (0 Uhr oben, im Uhrzeigersinn), die fünf Mitten, die Resultierende und das Fenster der wahren Mitte."""
    hours = a.night.hours
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1.0] * len(hours), theta=hours * 15.0, mode="markers", name="Ankünfte",
                                  marker=dict(size=8, color=C.COLOR_DATA, line=dict(width=0)), hovertemplate="%{customdata}<extra>Ankunft</extra>",
                                  customdata=[f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d} Uhr" for h in hours]))
    c_mean = K.mean_direction(K.h2a(hours))
    fig.add_trace(go.Scatterpolar(r=[0.0, a.R], theta=[0.0, math.degrees(float(c_mean))], mode="lines", name="Resultierende (Länge R)",
                                  line=dict(color=C.COLOR_TRUE, width=3), hoverinfo="skip"))
    if a.night.truth is not None:
        w = a.settings.window
        arc = np.linspace(a.night.truth - w, a.night.truth + w, 30) * 15.0
        fig.add_trace(go.Scatterpolar(r=[1.12] * 30, theta=arc, mode="lines", name=f"Fenster um die wahre Mitte (± {a.settings.window:g} h)",
                                      line=dict(color=C.COLOR_TRUE, width=5), hoverinfo="skip"))
    for k, (name, h) in enumerate(a.centres.items()):
        r = 0.80 - 0.07 * k
        fig.add_trace(go.Scatterpolar(r=[r], theta=[h * 15.0], mode="markers", name=f"{CENTRE_SHORT[name]}: {int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}",
                                      marker=dict(size=15, color=C.CENTRE_COLORS[name], symbol="diamond", line=dict(color="white", width=1)),
                                      hovertemplate=f"{name}<br>%{{theta:.1f}}°<extra></extra>"))
    fig.update_layout(
        height=520, margin=dict(l=30, r=30, t=20, b=110), dragmode=False,
        polar=dict(radialaxis=dict(range=[0, 1.2], showticklabels=False, showgrid=False, ticks=""),
                   angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=[h * 15 for h in range(0, 24, 3)],
                                    ticktext=[f"{h} Uhr" for h in range(0, 24, 3)])),
        legend=dict(orientation="h", yanchor="top", y=-0.05, x=0),
    )
    return fig


# ---------------------------------------------------------------- Teil B: Lieferanten

def build_supplier_densities(summary: dict) -> go.Figure:
    """Dichten der beiden Lieferanten, der Mischung und des Quantilmittels (Lognormal mit Median Σ w m), dazu die beiden Pläne für den Zielservicegrad."""
    medians = [C.MEDIAN_A, C.MEDIAN_A * summary["ratio"]]
    w = [1.0 - summary["share_b"] / 100.0, summary["share_b"] / 100.0]
    s = summary["s"]
    top = float(D.lognormal_mixture_quantile(np.array([0.995]), medians, s, w)[0])
    x = np.linspace(0.01, top * 1.1, 500)
    vin_median = D.lognormal_vincent_median(medians, w)
    fig = go.Figure()
    for j, (name, color) in enumerate((("Lieferant A (Anteil)", C.COLOR_A), ("Lieferant B (Anteil)", C.COLOR_B))):
        fig.add_trace(go.Scatter(x=x, y=w[j] * D.lognormal_mixture_pdf(x, [medians[j]], s, [1.0]), mode="lines", name=name, line=dict(color=color, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=D.lognormal_mixture_pdf(x, medians, s, w), mode="lines", name="Mischung (zufällige Zuordnung)", line=dict(color=C.COLOR_MIX, width=3)))
    fig.add_trace(go.Scatter(x=x, y=D.lognormal_mixture_pdf(x, [vin_median], s, [1.0]), mode="lines", name="Quantilmittel", line=dict(color=C.COLOR_QUANTILE, width=3)))
    for name, color, dash in (("Mischung, Quantil (richtig)", C.COLOR_MIX, "solid"), ("Quantilmittel, Quantil", C.COLOR_QUANTILE, "solid")):
        fig.add_vline(x=summary["plans"][name], line=dict(color=color, width=1.5, dash="dash"))
    fig.add_annotation(x=summary["plans"]["Mischung, Quantil (richtig)"], y=1, yref="paper", text="Plan: Mischung", showarrow=False, xanchor="left", font=dict(color=C.COLOR_MIX))
    fig.add_annotation(x=summary["plans"]["Quantilmittel, Quantil"], y=1, yref="paper", text="Plan: Quantilmittel", showarrow=False, xanchor="right", font=dict(color=C.COLOR_QUANTILE))
    return _layout(fig, "Lieferzeit (Tage)", "Dichte")


def build_quantile_functions(summary: dict) -> go.Figure:
    """Quantilfunktionen: Mischung und Quantilmittel liegen beim Mittel gleich, aber die Mischung hat die flachere Mitte und die längeren Ränder."""
    medians = [C.MEDIAN_A, C.MEDIAN_A * summary["ratio"]]
    w = [1.0 - summary["share_b"] / 100.0, summary["share_b"] / 100.0]
    s = summary["s"]
    p = np.linspace(0.01, 0.99, 99)
    z = D.norm_ppf(p)
    fig = go.Figure()
    for j, (name, color) in enumerate((("Lieferant A", C.COLOR_A), ("Lieferant B", C.COLOR_B))):
        fig.add_trace(go.Scatter(x=p, y=medians[j] * np.exp(s * z), mode="lines", name=name, line=dict(color=color, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=p, y=D.lognormal_mixture_quantile(p, medians, s, w), mode="lines", name="Mischung", line=dict(color=C.COLOR_MIX, width=3)))
    fig.add_trace(go.Scatter(x=p, y=D.lognormal_vincent_median(medians, w) * np.exp(s * z), mode="lines", name="Quantilmittel", line=dict(color=C.COLOR_QUANTILE, width=3)))
    fig.add_vline(x=summary["service"] / 100.0, line=dict(color="gray", width=1, dash="dash"))
    return _layout(fig, "Wahrscheinlichkeit p", "Quantil (Tage)")


# ---------------------------------------------------------------- Experimente

def build_uniform_chart(rows: list[dict]) -> go.Figure:
    """E1: R unter Gleichverteilung gegen die Zahl der Ankünfte: Mittel, 95-%-Quantil und die Schwelle √(−ln 0,05/n)."""
    n = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=[r["mean"]["mean"] for r in rows], mode="lines+markers", name="Mittel von R", line=dict(color=C.COLOR_MIX)))
    fig.add_trace(go.Scatter(x=n, y=[r["r95"]["mean"] for r in rows], mode="lines+markers", name="95-%-Quantil von R", line=dict(color=C.COLOR_QUANTILE)))
    grid = np.logspace(math.log10(min(n)), math.log10(max(n)), 60)
    fig.add_trace(go.Scatter(x=grid, y=K.rayleigh_threshold(grid), mode="lines", name="Schwelle √(−ln 0,05 / n)", line=dict(color="gray", dash="dash")))
    fig.update_xaxes(type="log")
    return _layout(fig, "Ankünfte n (logarithmisch)", "resultierende Länge R")


def build_power_chart(rows: list[dict]) -> go.Figure:
    """E1: Anteil der Nächte, in denen R über der Schwelle liegt (die Häufung erkannt wird), gegen n, je wahre Länge ρ."""
    fig = go.Figure()
    for rho in sorted({r["rho"] for r in rows}):
        sel = [r for r in rows if r["rho"] == rho]
        fig.add_trace(go.Scatter(x=[r["n"] for r in sel], y=[100 * r["power"]["mean"] for r in sel], mode="lines+markers", name=f"wahre Länge ρ = {rho:g}".replace(".", ",")))
    fig.add_hline(y=80, line=dict(color="gray", dash="dash"))
    return _layout(fig, "Ankünfte n", "Häufung erkannt (%)")


def build_cluster_chart(rows: list[dict], split: str) -> go.Figure:
    """E2: Anteil der Datensätze, in denen die Mitte im Leeren liegt (weiter als 20° von beiden Häufungen), gegen den Abstand Δ."""
    sel = [r for r in rows if r["split"] == split]
    fig = go.Figure()
    for key, name in (("ext", "Kreismittel"), ("int", "Fréchet p = 2"), ("med", "zirk. Median"), ("cov", "kleinster Bogen")):
        fig.add_trace(go.Bar(x=[f"{int(r['delta'])}°" for r in sel], y=[100 * r[f"{key}_empty"]["mean"] for r in sel], name=name,
                             marker_color={"ext": C.CENTRE_COLORS["Kreismittel (arctan2)"], "int": C.CENTRE_COLORS["Fréchet p = 2"],
                                           "med": C.CENTRE_COLORS["Zirkulärer Median p = 1"], "cov": C.CENTRE_COLORS["Kleinster Bogen p = ∞"]}[key]))
    fig.update_layout(barmode="group")
    return _layout(fig, "Abstand der beiden Häufungen", "Mitte im Leeren (%)")


def build_lorry_chart(rows: list[dict]) -> go.Figure:
    """E2: mittlere Verschiebung jeder Mitte, wenn ein einziger Lkw die Häufung wechselt (20 gegen 21 → 21 gegen 20), gegen den Abstand Δ."""
    fig = go.Figure()
    for key, name in (("ext", "Kreismittel"), ("int", "Fréchet p = 2"), ("med", "zirk. Median"), ("cov", "kleinster Bogen")):
        color = {"ext": "Kreismittel (arctan2)", "int": "Fréchet p = 2", "med": "Zirkulärer Median p = 1", "cov": "Kleinster Bogen p = ∞"}[key]
        fig.add_trace(go.Scatter(x=[r["delta"] for r in rows], y=[r[f"{key}_shift"]["mean"] for r in rows], mode="lines+markers", name=name, line=dict(color=C.CENTRE_COLORS[color])))
    return _layout(fig, "Abstand der beiden Häufungen (Grad)", "Verschiebung durch einen Lkw (Grad)")


def build_torzeit_chart(rows: list[dict]) -> go.Figure:
    """E3: mittlerer Fehler gegen die wahre Mitte in Minuten, je Streuung der Ankünfte."""
    fig = go.Figure()
    for name in X.TABLE_NAMES[:5]:
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rows], y=[r["rows"][name]["err_min"]["mean"] for r in rows], mode="lines+markers", name=CENTRE_SHORT[name],
                                 line=dict(color=C.CENTRE_COLORS[name])))
    return _layout(fig, "Streuung der Ankünfte (Stunden)", "mittlerer Fehler (Minuten)")


def build_service_chart(grid: list[dict]) -> go.Figure:
    """E4: tatsächlicher Servicegrad, wenn mit dem 95-%-Quantil des Quantilmittels geplant wird, gegen das Medianverhältnis, je Streuung."""
    fig = go.Figure()
    for s in sorted({g["s"] for g in grid}):
        sel = [g for g in grid if g["s"] == s]
        fig.add_trace(go.Scatter(x=[g["ratio"] for g in sel], y=[100 * g["service_at_plan"]["Quantilmittel, Quantil"] for g in sel], mode="lines+markers",
                                 name=f"Streuung s = {s:g}".replace(".", ",")))
    fig.add_hline(y=95, line=dict(color="gray", dash="dash"))
    return _layout(fig, "Medianverhältnis B / A", "tatsächlicher Servicegrad (%), Ziel 95 %")


def build_forecast_chart(cells: list[dict]) -> go.Figure:
    """E5: Abdeckung des 80-%-Intervalls von Mischung und Quantilmittel gegen den Lagefehler s_e, getrennt nach Meldeart."""
    fig = go.Figure()
    for cal, dash in (("überzuversichtlich", "solid"), ("ehrlich", "dot")):
        sel = sorted([c for c in cells if c["calibration"] == cal], key=lambda c: c["s_e"])
        for method, color in (("Mischung", C.COLOR_MIX), ("Quantilmittel", C.COLOR_QUANTILE)):
            fig.add_trace(go.Scatter(x=[c["s_e"] for c in sel], y=[100 * c["methods"][method]["covered"]["mean"] for c in sel], mode="lines+markers",
                                     name=f"{method}, {cal}", line=dict(color=color, dash=dash)))
    fig.add_hline(y=80, line=dict(color="gray", dash="dash"))
    return _layout(fig, "Streuung des Lagefehlers s_e der Prognosen", "Abdeckung des 80-%-Intervalls (%)", height=420)


def build_crps_chart(cells: list[dict]) -> go.Figure:
    """E5: CRPS-Differenz Mischung minus Quantilmittel (negativ = Mischung besser), mit der Streuung über die Seeds."""
    fig = go.Figure()
    for cal, color in (("überzuversichtlich", C.COLOR_MIX), ("ehrlich", C.COLOR_B)):
        sel = sorted([c for c in cells if c["calibration"] == cal], key=lambda c: c["s_e"])
        fig.add_trace(go.Bar(x=[str(c["s_e"]).replace(".", ",") for c in sel], y=[c["diffs"]["mix_minus_vin"]["crps_diff"]["mean"] for c in sel], name=cal, marker_color=color,
                             error_y=dict(type="data", array=[c["diffs"]["mix_minus_vin"]["crps_diff"]["std"] for c in sel])))
    fig.update_layout(barmode="group")
    return _layout(fig, "Streuung des Lagefehlers s_e", "CRPS Mischung − Quantilmittel (negativ: Mischung besser)", height=400)


def build_scale_chart(cells: list[dict]) -> go.Figure:
    """E6: CRPS-Differenz zwischen Quantilmittel in x und in log x, je Verhältnis s_e zur Streuung der Nachfrage."""
    fig = go.Figure()
    for cal, color in (("überzuversichtlich", C.COLOR_MIX), ("ehrlich", C.COLOR_B)):
        sel = sorted([c for c in cells if c["calibration"] == cal], key=lambda c: c["rel"])
        fig.add_trace(go.Bar(x=[str(c["rel"]).replace(".", ",") for c in sel], y=[c["diffs"]["x_minus_logx"]["crps_diff"]["mean"] for c in sel], name=cal, marker_color=color,
                             error_y=dict(type="data", array=[c["diffs"]["x_minus_logx"]["crps_diff"]["std"] for c in sel])))
    fig.update_layout(barmode="group")
    return _layout(fig, "Lagefehler s_e im Verhältnis zur Streuung der Nachfrage", "CRPS Quantilmittel in x − in log x", height=380)
