"""Mitte in anderen Räumen: Kreis und Verteilungen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 der Reihe „Mittelwerte und Lagemaße“ (optional). Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mir_constants as C
import mir_evaluation as V
import mir_experiments as X
import mir_results as R
from mir_layout import dunkles_design, kopf_css
from mir_pdf_export import build_pdf
from mir_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mir_visualization import (
    build_clock,
    build_cluster_chart,
    build_crps_chart,
    build_forecast_chart,
    build_lorry_chart,
    build_power_chart,
    build_quantile_functions,
    build_scale_chart,
    build_service_chart,
    build_supplier_densities,
    build_torzeit_chart,
    build_uniform_chart,
)

st.set_page_config(page_title="Mitte in anderen Räumen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _gate(settings):
    return V.analyse_gate(settings)


@st.cache_data(show_spinner=False)
def _suppliers(ratio, s, wb, service):
    return X.supplier_summary(ratio, s, float(wb), float(service))


@st.cache_data(show_spinner=False)
def _forecast_live(se, calib, cases):
    return X.forecast_cell(se, calib, cases)


@st.cache_data(show_spinner=False)
def _gate_live(n, mu, sigma, share, gap, cut, window, nights):
    return X.gate_study(n, mu, sigma, share, gap, cut, window, nights)


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _fmt(x, d=2):
    return V.de(x, d)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mir_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🕛 Mitte in anderen Räumen: Kreis und Verteilungen")
st.markdown(
    """
**Stück 8 der Reihe „Mittelwerte und Lagemaße“ (optional).** Die Reihe fragt, welche Mitte zu welcher Frage passt. Hier verlässt sie die Zahlengerade: **Teil A** mittelt Uhrzeiten (Lkw-Ankünfte am Tor, ein Kreis von 24 Stunden),
**Teil B** mittelt ganze Verteilungen (Lieferzeiten zweier Lieferanten, zwei Prognosen). Leitbegriff ist das **Fréchet-Mittel**: die Mitte als Minimierer einer Summe von Abständen im jeweiligen Raum.
Gemessen wird, **wann die Mitte gut bestimmt ist, wann sie bestimmt, aber sinnlos ist** (zwei Häufungen), und **was es im Betrieb kostet**, wenn man trotzdem das gewohnte Mittel nimmt.
Das Torzeit- und das Lieferzeit-Szenario sind **gewählt, keine Messung an echten Daten**.
Abgrenzung: [shift-demo](https://github.com/sebastian-hanisch/shift-demo) (Schichten über Mitternacht als Überdeckung, nicht als Mitte), [forecast-combination-demo](https://github.com/sebastian-hanisch/forecast-combination-demo) (Punktprognosen kombinieren, Quantile nicht),
[bunching-demo](https://github.com/sebastian-hanisch/bunching-demo) (Takt und Wartezeit). Rangfolgen als Mitte: `rangaggregation-demo`; Graphen: [centrality-demo](https://github.com/sebastian-hanisch/centrality-demo).

Weiter unten: sechs Experimente (**🔬 E1 bis E6**), **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was „Mitte“ in einem anderen Raum heißt", expanded=True):
    st.markdown(
        r"""
1. **Eine Mitte ist ein Minimierer.** Zu Punkten $x_1, \dots, x_n$ und einem Abstand $d$ ist die Mitte $c$ das, was $\sum_i d(x_i, c)^p$ klein macht: $p = 2$ gibt das Mittel, $p = 1$ den Median, $p \to \infty$ die Mitte der Spannweite.
2. **Auf dem Kreis ändert sich der Abstand.** Zwischen 23 Uhr und 1 Uhr liegen 2 Stunden, nicht 22. Das arithmetische Mittel von 23, 23,5, 0,5 und 1 Uhr ist 12 Uhr; das **Kreismittel** ist 0 Uhr.
3. **Bei Verteilungen gibt es zwei „Mitten“.** Die **Mischung** mittelt die Wahrscheinlichkeiten (jeder Auftrag geht zufällig an einen der Lieferanten), das **Quantilmittel** mittelt die Quantile (der schnellste Fall beider Lieferanten wird mit dem schnellsten gemittelt, der langsamste mit dem langsamsten).
   Beide haben dasselbe Mittel, aber nicht dieselbe Streuung.

| Mitte | Raum | Was sie minimiert oder mittelt | Wer setzt an |
|---|---|---|---|
| Arithmetisch (Tag ab $c$ Uhr) | Zahlengerade, Schnitt bei $c$ Uhr | $\sum (h_i - c)^2$ mit Uhrzeiten in $[c, c + 24)$ | hängt am Schnittpunkt des Tages |
| Kreismittel (arctan2) | Kreis | Richtung der Resultierenden (extrinsisch: Sehne statt Bogen) | bei zwei Häufungen im Leeren; bei $R \approx 0$ Zufall |
| Fréchet $p = 2$ | Kreis | $\sum d_{\text{Bogen}}^2$ (intrinsisch), **exakt aufgezählt** | kann mehrere lokale Minima haben |
| Zirkulärer Median $p = 1$ | Kreis | $\sum d_{\text{Bogen}}$, liegt auf einem Datenpunkt | springt bei Gegenpunkten auf die andere Häufung |
| Kleinster Bogen $p = \infty$ | Kreis | Mitte des kürzesten Bogens, der alle Punkte enthält | folgt Störern |
| Mischung | Verteilungen | $F = \sum w_j F_j$ | richtig für zufällige Zuordnung |
| Quantilmittel | Verteilungen | $Q = \sum w_j Q_j$ | schärfer, hängt an der Skala |

Rangfolgen und Graphen haben ebenfalls Mitten; sie stehen in `rangaggregation-demo` und [centrality-demo](https://github.com/sebastian-hanisch/centrality-demo).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🕛 Teil A: Torzeit")
    n = st.slider("Ankünfte n pro Nacht", *bounds("n_slider"), key="n_slider", help="Wie viele Lkw in der Nacht ankommen.")
    mu = st.slider("Wahre Mitte (Uhr)", *bounds("mu_slider"), key="mu_slider", step=0.25, help="Um diese Uhrzeit häufen sich die Ankünfte (bei zwei Gipfeln: der erste Gipfel).")
    sigma = st.slider("Streuung σ (Stunden)", *bounds("sigma_slider"), key="sigma_slider", step=0.25, help="Standardabweichung der Ankünfte um die Mitte (wrapped normal, über Mitternacht gefaltet).")
    share = st.slider("Anteil des zweiten Gipfels (%)", *bounds("share_slider"), key="share_slider", help="Wie viele der Ankünfte zu einem zweiten Gipfel gehören. 0: ein Gipfel; um 50: zwei gleich große Häufungen.")
    gap = st.slider("Abstand des zweiten Gipfels (Stunden)", *bounds("gap_slider"), key="gap_slider", step=0.5, disabled=share == 0,
                    help="Nur bei einem zweiten Gipfel: wie weit er vom ersten entfernt liegt. 12 Stunden liegen sich auf dem Kreis gegenüber.")
    cut = st.slider("Tagesbeginn des arithmetischen Mittels (Uhr)", *bounds("cut_slider"), key="cut_slider",
                    help="Das arithmetische Mittel rechnet mit Uhrzeiten von dieser Stunde an (0 = übliche Tageszählung). Das Kreismittel kennt keinen Schnittpunkt.")
    window = st.slider("Fenster um die Mitte (± Stunden)", *bounds("window_slider"), key="window_slider", step=0.25, help="Torzeit-Fenster um die gewählte Mitte; gezählt wird der Anteil der Ankünfte darin.")
    st.number_input("Zufalls-Seed der Nacht", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Nacht generieren", width="stretch", on_click=randomize_seed)
    st.header("📦 Teil B: Lieferanten")
    ratio = st.slider("Median von B im Verhältnis zu A", *bounds("ratio_slider"), key="ratio_slider", step=0.1, help=f"Lieferant A liefert im Median in {C.MEDIAN_A:g} Tagen, Lieferant B in so vielen Vielfachen davon.")
    s_log = st.slider("Streuung s der Lieferzeit (Logarithmus)", *bounds("s_slider"), key="s_slider", step=0.05, help="Beide Lieferanten haben dieselbe Lognormal-Streuung s; groß heißt, dass die Lücke zwischen den Medianen verschwimmt.")
    wb = st.slider("Anteil der Aufträge an Lieferant B (%)", *bounds("wb_slider"), key="wb_slider", help="Jeder Auftrag geht zufällig mit dieser Wahrscheinlichkeit an B (Mischung).")
    service = st.slider("Zielservicegrad (%)", *bounds("service_slider"), key="service_slider", help="Der Bestellpunkt soll diesen Anteil der Aufträge rechtzeitig abdecken.")
    st.header("🔮 Teil B: Prognosen zu zweit")
    se = st.slider("Lagefehler s_e der Prognosen", *bounds("se_slider"), key="se_slider", step=0.25, help="Jede der zwei Prognosen liegt um diese Streuung neben dem wahren Erwartungswert; die Nachfrage selbst hat die Streuung 1.")
    calib = st.selectbox("Gemeldete Streuung", C.CALIBRATIONS, key="calib_select",
                         help="Überzuversichtlich: jede Prognose meldet nur die Streuung der Nachfrage (1) und lässt ihren eigenen Lagefehler weg. Ehrlich: sie meldet √(1 + s_e²).")

settings = V.Settings(n=int(n), mu=float(mu), sigma=float(sigma), share=int(share), gap=float(gap), cut=int(cut), window=float(window), seed=int(st.session_state["seed_input"]),
                      ratio=float(ratio), s=float(s_log), wb=int(wb), service=int(service), se=float(se), calib=str(calib))
sync_query_params({k: st.session_state[k] for k in ("n_slider", "mu_slider", "sigma_slider", "share_slider", "gap_slider", "cut_slider", "window_slider", "seed_input",
                                                      "ratio_slider", "s_slider", "wb_slider", "service_slider", "se_slider", "calib_select")})

a = _gate(settings)
summary = _suppliers(settings.ratio, settings.s, settings.wb, settings.service)

# --- Teil A -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🕛 Teil A: Eine Nacht am Tor")
names = list(a.centres)
kreis, arith = names[1], names[0]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Resultierende Länge R", _fmt(a.R, 3), delta=f"Schwelle {_fmt(a.threshold, 3)}", delta_color="off")
m2.metric("Rayleigh-p (einfach)", _fmt(a.p_value, 3) if a.p_value >= 0.001 else "< 0,001", delta="Gleichverteilung als Nullhypothese", delta_color="off")
m3.metric("Kreismittel", f"{V.hhmm(a.centres[kreis])} Uhr", delta="Referenz", delta_color="off")
m4.metric(f"Arithmetisch (Tag ab {settings.cut} Uhr)", f"{V.hhmm(a.centres[arith])} Uhr", delta=f"{_fmt(a.arith_gap_h, 1)} h neben dem Kreismittel", delta_color="off")
st.plotly_chart(build_clock(a), width="stretch", key="clock_chart")
st.caption("Uhr-Kreis: 0 Uhr oben, im Uhrzeigersinn. Graue Punkte: die Ankünfte (außen). Schwarzer Pfeil: die Resultierende, ihre Länge ist R (kurz heißt: die Ankünfte gleichen sich aus). "
           "Rauten: die fünf Mitten. Schwarzer Bogen außen: das Fenster um die wahre Mitte (nur bei einem Gipfel).")

rows = []
for name, h in a.centres.items():
    row = {"Mitte": name, "Uhrzeit": V.hhmm(h), "Fensterabdeckung (%)": 100.0 * a.coverage[name]}
    if a.error_min is not None:
        row["Abstand zur wahren Mitte (min)"] = a.error_min[name]
    rows.append(row)
cfg = {"Fensterabdeckung (%)": st.column_config.NumberColumn(format="%.1f %%"), "Abstand zur wahren Mitte (min)": st.column_config.NumberColumn(format="%.1f")}
st.dataframe(rows, hide_index=True, width="stretch", column_config=cfg)
cut_text = ", ".join(f"ab {c} Uhr: {V.hhmm(h)}" for c, h in a.cut_means.items())
st.caption(f"Arithmetisches Mittel derselben Nacht je Tagesbeginn: {cut_text} (Spannweite {_fmt(a.cut_range_min, 0)} Minuten). "
           f"Die Fréchet-Funktion hat {a.n_minima} lokale{'s Minimum' if a.n_minima == 1 else ' Minima'}"
           + (f"; {a.median_ties} Datenpunkte sind als zirkulärer Median gleich gut (Gleichstand: der kleinere Winkel gilt)." if a.median_ties > 1 else "."))
if a.oracle_coverage is not None:
    st.caption(f"Ein Fenster um die wahre Mitte ({V.hhmm(a.night.truth)} Uhr, ± {_fmt(settings.window, 2)} h) fängt {_fmt(100.0 * a.oracle_coverage, 1)} % der Ankünfte. Eine aus der Stichprobe berechnete Mitte kann mehr fangen, weil sie sich der Stichprobe anpasst.")

# --- Teil B -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📦 Teil B: Zwei Lieferanten, ein Bestellpunkt")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Mittel (beide gleich)", f"{_fmt(summary['mix_mean'], 1)} Tage", delta="Mischung = Quantilmittel", delta_color="off")
b2.metric("Streuung der Mischung", f"{_fmt(summary['mix_sd'], 1)} Tage", delta=f"{_fmt(summary['sd_ratio'], 2)}-mal das Quantilmittel", delta_color="off")
b3.metric("Streuung des Quantilmittels", f"{_fmt(summary['vin_sd'], 1)} Tage", delta="Referenz", delta_color="off")
real = summary["service_at_plan"][V.PLAN_QUANTILE]
b4.metric("Servicegrad (Plan: Quantilmittel)", f"{V.pct(real, 1)} %", delta=f"{_fmt(100.0 * real - summary['service'], 1)} Prozentpunkte gegen das Ziel")      # mein Wert minus Ziel; höher ist besser, also normale Färbung
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_supplier_densities(summary), width="stretch", key="supplier_density")
col_b.plotly_chart(build_quantile_functions(summary), width="stretch", key="supplier_quantiles")
st.caption("Links: Dichten der Lieferzeit. Blau die Mischung (jeder Auftrag geht zufällig an A oder B), orange das Quantilmittel (die Quantile beider gemittelt); gestrichelt die Bestellpunkte der beiden Pläne. "
           "Rechts: Quantilfunktionen; gestrichelt der Zielservicegrad.")
plan_rows = [{"Plan": name, "Bestellpunkt (Tage)": value, "Tatsächlicher Servicegrad (%)": 100.0 * summary["service_at_plan"][name]} for name, value in summary["plans"].items()]
st.dataframe(plan_rows, hide_index=True, width="stretch",
             column_config={"Bestellpunkt (Tage)": st.column_config.NumberColumn(format="%.1f"), "Tatsächlicher Servicegrad (%)": st.column_config.NumberColumn(format="%.1f %%")})
st.caption("Wahrheit ist die Mischung: Jeder Auftrag geht zufällig an A oder B. Der tatsächliche Servicegrad ist der Anteil der Aufträge, deren Lieferzeit höchstens der Bestellpunkt ist.")

st.markdown("### 🔮 Zwei Prognosen mitteln")
st.markdown("Zwei Prognosen derselben Größe, die Nachfrage ist bekannt normalverteilt um eine wahre Lage. Beide Prognosen liegen mit dem Lagefehler $s_e$ daneben und melden eine Streuung. Gemessen wird, wie gut die **Mischung** und das **Quantilmittel** der zwei gemeldeten Verteilungen sind.")
live_key = (settings.se, settings.calib, C.LIVE_CASES)
if st.button(f"Prognosen nachmessen ({C.LIVE_CASES} Fälle je Seed, vier Seeds, wenige Sekunden)", key="forecast_live"):
    with st.spinner("Rechne vier Seeds …"):
        st.session_state["forecast_result"] = (live_key, _forecast_live(*live_key))
res = st.session_state.get("forecast_result")
if res is not None and res[0] == live_key:
    cell = res[1]
    st.dataframe([{"Verfahren": name, "CRPS (kleiner ist besser)": m["crps"]["mean"], "Abdeckung 80-%-Intervall (%)": 100.0 * m["covered"]["mean"], "Mittlere Breite": m["width"]["mean"]}
                  for name, m in cell["methods"].items()], hide_index=True, width="stretch",
                 column_config={"CRPS (kleiner ist besser)": st.column_config.NumberColumn(format="%.3f"), "Abdeckung 80-%-Intervall (%)": st.column_config.NumberColumn(format="%.1f %%"),
                                "Mittlere Breite": st.column_config.NumberColumn(format="%.2f")})
    _verdict(V.verdict_forecast(cell))
    st.caption(f"Die Live-Messung nimmt nur {C.LIVE_CASES} Fälle je Seed; ihre Differenzen schwanken stärker als in der vorgerechneten Reihe von Experiment E5 ({X.CASES} Fälle je Seed) und können dort ein anderes Vorzeichen haben.")
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# PDF erst auf Klick
pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    verdicts = [V.verdict_gate(a.R, settings.n, a.circle_spread_h, a.arith_gap_h, settings.cut, a.cut_range_min, a.night.n_second), V.verdict_suppliers(summary)]
    st.session_state["pdf"] = (pdf_key, build_pdf(a, summary, verdicts))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="mitte-im-raum.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann ist die Mitte gut bestimmt, und was kostet die falsche?")
_verdict(V.verdict_gate(a.R, settings.n, a.circle_spread_h, a.arith_gap_h, settings.cut, a.cut_range_min, a.night.n_second))
_verdict(V.verdict_suppliers(summary))

with st.expander("🧰 Die fünf Kreis-Mitten und die zwei Verteilungs-Mitten erklärt"):
    st.markdown(
        r"""
| Verfahren | Was es tut | Was schiefgehen kann |
|---|---|---|
| **Arithmetisch** | Mittelt die Uhrzeiten als Zahlen von 0 bis 24 (oder ab dem gewählten Tagesbeginn) | Ankünfte vor und nach dem Schnittpunkt landen an den entgegengesetzten Enden: 23 Uhr und 1 Uhr ergeben 12 Uhr. |
| **Kreismittel (extrinsisch)** | Richtung der Summe der Einheitsvektoren (Sehnenabstand); Länge der Summe ist $R$ | Bei $R \approx 0$ ist die Richtung Rauschen, bei zwei gleich großen Häufungen liegt sie dazwischen. |
| **Fréchet $p = 2$ (intrinsisch)** | Minimiert die Summe der quadrierten Bogenabstände; hier exakt über alle Bögen zwischen Gegenpunkten aufgezählt | Kann mehrere lokale Minima haben (auch bei einem Gipfel ein zweites am Gegenpunkt); bei Gleichstand gilt der kleinere Winkel. |
| **Zirkulärer Median $p = 1$** | Der Datenpunkt mit der kleinsten Summe der Bogenabstände | Bei geradem $n$ ein flaches Stück statt eines Punktes; bei Gegenhäufungen springt er auf die andere Häufung. |
| **Kleinster Bogen $p = \infty$** | Mitte des kürzesten Bogens, der alle Punkte enthält (Komplement der größten Lücke) | Ein einziger Störer verschiebt ihn stark. |
| **Mischung** | Verteilungsfunktionen mit Gewichten mitteln: $F = \sum w_j F_j$ | Verbreitert sich mit dem Abstand der Quellen; ist die richtige Beschreibung zufälliger Zuordnung. |
| **Quantilmittel (Vincent-Mittel)** | Quantilfunktionen mit Gewichten mitteln: $Q = \sum w_j Q_j$ | Dasselbe Mittel wie die Mischung, aber viel schmaler, wenn die Quellen weit auseinanderliegen; hängt an der Skala ($x$ oder $\log x$). |
        """
    )

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------

st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); die Knöpfe „Nachmessen“ rechnen mit Ihren Einstellungen neu.")

# E1 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E1: Wann ist die Mitte Zufall?")
st.markdown("Gleichverteilte Ankünfte haben eine Resultierende der Länge $R > 0$ allein durch Zufall. Gemessen: wie groß $R$ dort wird, wie viele Ankünfte nötig sind, um eine echte Häufung zu erkennen, und wie weit die Richtung dabei daneben liegt.")
c1, c2 = st.columns(2)
c1.plotly_chart(build_uniform_chart(R.uniform_rows()), width="stretch", key="e1_uniform")
c2.plotly_chart(build_power_chart(R.power_rows()), width="stretch", key="e1_power")
sel1, sel2, sel3 = st.columns(3)
n_uni = sel1.selectbox("Ankünfte n (Zufallsstreuung)", X.N_RASTER, index=X.N_RASTER.index(30), key="e1_n_uni")
rho_pow = sel2.selectbox("Wahre Länge ρ des Erzeugers", X.RHO_RASTER, index=1, key="e1_rho")
n_pow = sel3.selectbox("Ankünfte n (Güte)", X.N_POWER, index=2, key="e1_n_pow")
_verdict(V.verdict_e1(R.uniform_row(n_uni), R.power_row(rho_pow, n_pow)))

# E2 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E2: Zwei Häufungen, ein Lkw und der Unterschied zwischen Sehne und Bogen")
st.markdown("Teil 1 vergleicht die extrinsische Mitte (Kreismittel) mit der intrinsischen (Fréchet) bei **einem** Gipfel, Teil 2 setzt **zwei** Häufungen in den Kreis und zählt, wo die Mitten landen, Teil 3 lässt **einen einzigen Lkw** die Häufung wechseln.")
setting_label = {s_: f"ρ = {s_[0]:g}, n = {s_[1]}" + (f", {s_[2]} % Störer" if s_[2] else "") for s_ in X.ONE_PEAK_SETTINGS}
choice = st.selectbox("Ein Gipfel", X.ONE_PEAK_SETTINGS, format_func=lambda s_: setting_label[s_].replace(".", ","), key="e2_peak")
_verdict(V.verdict_e2_peak(R.one_peak_row(*choice)))
skew = R.skew_row(100)
st.caption(f"Schiefe Daten (70 % um 0°, 30 % um 100°, n = 100): Kreismittel und Fréchet-Mittel liegen im Mittel {_fmt(skew['dist_mean_deg']['mean'], 1)}° auseinander, "
           f"in {V.pct(skew['dist_gt_2']['mean'], 0)} % der Datensätze um mehr als 2°. Eine wahre Mitte gibt es dort nicht.")
d1, d2 = st.columns(2)
split = d1.selectbox("Aufteilung der zwei Häufungen", X.SPLITS, index=1, key="e2_split")
delta = d2.select_slider("Abstand Δ der Häufungen (Grad)", options=list(X.DELTAS), value=180, key="e2_delta")
st.plotly_chart(build_cluster_chart(R.cluster_rows(), split), width="stretch", key="e2_clusters")
_verdict(V.verdict_e2_clusters(R.cluster_row(delta, split)))
st.plotly_chart(build_lorry_chart(R.lorry_rows()), width="stretch", key="e2_lorry")
_verdict(V.verdict_e2_lorry(R.lorry_row(delta)))
_verdict(V.verdict_e2_grid(R.grid_miscount()))
st.caption("Beim zirkulären Median heißt „gerades n, feste Hälften“: Die Zielfunktion ist dort zwischen zwei Datenpunkten flach, und die Demo nimmt den Punkt mit dem kleineren Winkel. Die Zeile „fest 50/50“ zeigt deshalb mehr als den Median.")

# E3 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E3: Der Torzeit-Fall")
st.markdown(f"{X.TORZEIT_N} Ankünfte je Nacht, Mitte irgendwo am Tag, Fenster ± {X.WINDOW:g} Stunde um die gewählte Mitte. Gemessen: Fehler gegen die wahre Mitte und der Anteil der Ankünfte im Fenster.")
st.plotly_chart(build_torzeit_chart(R.torzeit_rows()), width="stretch", key="e3_chart")
sigma_sel = st.select_slider("Streuung σ der Ankünfte (Stunden)", options=list(X.SIGMAS_H), value=1.0, key="e3_sigma")
_verdict(V.verdict_e3(R.torzeit_row(sigma_sel)))
_verdict(V.verdict_e3_night(R.nightshift(), R.cut_cell()))
_verdict(V.verdict_e3_peaks(R.two_peak()))
nights_live = int(max(20, min(C.LIVE_NIGHTS, C.LIVE_BUDGET // (settings.n * settings.n))))
gate_key = (settings.n, settings.mu, settings.sigma, settings.share, settings.gap, settings.cut, settings.window, nights_live)
if st.button(f"Nächte nachmessen (Seitenleiste, {nights_live} Nächte je Seed, vier Seeds, wenige Sekunden)", key="gate_live"):
    with st.spinner("Rechne vier Seeds …"):
        st.session_state["gate_result"] = (gate_key, _gate_live(*gate_key))
res = st.session_state.get("gate_result")
if res is not None and res[0] == gate_key:
    live = res[1]
    live_rows = []
    for name in X.TABLE_NAMES:
        r_ = live["rows"][name]
        row = {"Mitte": name, "Fensterabdeckung (%)": 100.0 * r_["coverage"]["mean"]}
        if "err_min" in r_:
            row["mittlerer Fehler (min)"] = r_["err_min"]["mean"]
        if "empty" in r_:
            row["Mitte im Leeren (%)"] = 100.0 * r_["empty"]["mean"]
        live_rows.append(row)
    st.dataframe(live_rows, hide_index=True, width="stretch", column_config={"Fensterabdeckung (%)": st.column_config.NumberColumn(format="%.1f %%"),
                                                                            "mittlerer Fehler (min)": st.column_config.NumberColumn(format="%.1f"),
                                                                            "Mitte im Leeren (%)": st.column_config.NumberColumn(format="%.1f %%")})
    st.info(f"Nachgemessen: {live['nights_per_seed']} Nächte je Seed mit n = {live['n']}, Mitte {V.hhmm(live['mu'])} Uhr, σ = {_fmt(live['sigma'], 2)} h, "
            f"{live['n_second']} Ankünften im zweiten Gipfel; mittleres R = {_fmt(live['rows']['_R']['mean'], 3)}.")
elif res is not None:
    st.caption("Die Einstellungen haben sich geändert: Knopf erneut drücken.")

# E4 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E4: Lieferantenmix und Bestellpunkt")
ex = R.supplier_example()
st.markdown(f"Lieferant A liefert im Median in {C.MEDIAN_A:g} Tagen, B in {C.MEDIAN_A * ex['ratio']:g} Tagen, beide mit Log-Streuung {_fmt(ex['s'], 2)}, je die Hälfte der Aufträge. "
            f"Mischung und Quantilmittel haben das Mittel {_fmt(ex['mix_mean'], 2)} Tage. Anteil der Masse im Bereich von {C.SHARE_LO:g} bis {C.SHARE_HI:g} Tagen: Mischung {V.pct(ex['range_mix'], 1)} %, Quantilmittel {V.pct(ex['range_vin'], 1)} %.")
st.plotly_chart(build_service_chart(R.supplier_grid()), width="stretch", key="e4_chart")
e4a, e4b = st.columns(2)
ratio_sel = e4a.select_slider("Medianverhältnis B / A", options=list(X.RATIOS), value=3.0, key="e4_ratio")
s_sel = e4b.select_slider("Log-Streuung s", options=list(X.S_VALUES), value=0.15, key="e4_s")
_verdict(V.verdict_suppliers(R.supplier_cell(ratio_sel, s_sel)))
st.caption(f"Die Formeln stimmen mit einer Ziehung aus der Definition überein: Streuung der Mischung {_fmt(R.supplier_simulation()['mix_sd']['mean'], 2)} (Ziehung) gegen {_fmt(ex['mix_sd'], 2)} (Formel), "
           f"des Quantilmittels {_fmt(R.supplier_simulation()['vin_sd']['mean'], 2)} gegen {_fmt(ex['vin_sd'], 2)}.")

# E5 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E5: Zwei Prognosen mitteln")
st.markdown("Bekannte Wahrheit: Die Nachfrage ist $y = m + \\varepsilon$; jede Prognose liegt um $e_i$ neben $m$ und meldet eine Normalverteilung. 1 500 Fälle je Seed, gleiche Fälle für alle Verfahren. "
            "CRPS aus 100 Quantilstufen (kleiner ist besser); die Abdeckung des 80-%-Intervalls sollte bei 80 % liegen.")
f1, f2 = st.columns(2)
f1.plotly_chart(build_forecast_chart(R.forecast_cells()), width="stretch", key="e5_cov")
f2.plotly_chart(build_crps_chart(R.forecast_cells()), width="stretch", key="e5_crps")
g1, g2 = st.columns(2)
se_sel = g1.select_slider("Lagefehler s_e", options=list(X.SE_VALUES), value=2.0, key="e5_se")
cal_sel = g2.selectbox("Gemeldete Streuung", C.CALIBRATIONS, key="e5_cal")
_verdict(V.verdict_forecast(R.forecast_cell(se_sel, cal_sel)))
un = R.forecast_unequal()
st.caption(f"Ungleich gemeldete Streuungen (0,5 und 2, s_e = 0,5): CRPS-Differenz Mischung minus Quantilmittel {_fmt(un['diffs']['mix_minus_vin']['crps_diff']['mean'], 3)} ± {_fmt(un['diffs']['mix_minus_vin']['crps_diff']['std'], 3)}; "
           f"die Mischung ist in {V.pct(un['diffs']['mix_minus_vin']['first_better_share']['mean'], 0)} % der Fälle besser.")

# E6 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E6: Das Quantilmittel hängt an der Skala")
st.markdown(f"Lognormale Nachfrage ($\\ln y = m + \\varepsilon$, Streuung {X.SCALE_SPREAD_Y:g}). Dieselben Prognosen, einmal die Quantile in $x$ gemittelt, einmal in $\\log x$: Die Mischung ist von dieser Wahl unabhängig, das Quantilmittel nicht.")
st.plotly_chart(build_scale_chart(R.scale_cells()), width="stretch", key="e6_chart")
h1, h2 = st.columns(2)
rel_sel = h1.select_slider("Lagefehler im Verhältnis zur Streuung der Nachfrage", options=list(X.SCALE_REL), value=2.0, key="e6_rel")
cal6 = h2.selectbox("Gemeldete Streuung", C.CALIBRATIONS, key="e6_cal")
_verdict(V.verdict_e6(R.scale_cell(rel_sel, cal6)))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ankünfte aus einer wrapped normal** (auf den Kreis gefaltete Normalverteilung) | Echte Torzeiten haben Spitzen (Schichtbeginn), Pausen und Wochentage; alle Anteile hier gelten für den Erzeuger, nicht für Ihr Tor. Eine Von-Mises-Verteilung wurde nicht verwendet. | – |
| **Eine Mitte reicht** | Bei zwei Häufungen ist die Mitte definiert, aber sinnlos: Kreismittel und Fréchet liegen im Leeren, der Median auf einer Häufung, und ein einziger Lkw wirft sie um. | Clustern: [kmeans-demo](https://github.com/sebastian-hanisch/kmeans-demo) |
| **Fester Tagesbeginn** | Das arithmetische Mittel hängt am Schnittpunkt; das Kreismittel kennt keinen, dafür bleibt die Frage, welche Streuung zur Planung passt. | – |
| **Lieferzeiten lognormal, gleiche Streuung** | Echte Lieferzeiten sind selten lognormal; die Befunde zu Mischung und Quantilmittel gelten für diesen Erzeuger. | Prognose und Bestand: [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) |
| **Zufällige Zuordnung der Aufträge** | Wird nach Region, Menge oder Dringlichkeit zugeordnet, ist die Mischung nicht die Wahrheit. | – |
| **Zwei Prognosen, gleiche Gewichte, bekannte Wahrheit** | Mit mehr Prognosen, ungleichen Gewichten oder echten Daten kann das Ergebnis anders ausfallen. Es gibt hier keinen allgemeinen Sieger. | Punktprognosen kombinieren: [forecast-combination-demo](https://github.com/sebastian-hanisch/forecast-combination-demo) |
| **Simulierte Daten, vier Seeds je Zelle** | Die angegebene Streuung ist die zwischen vier festen Zufalls-Seeds, kein Konfidenzintervall. Es gibt keinen Datensatz aus einem echten Tor. | – |
    """
)
st.markdown(
    """
**Bekannte Probleme und Unsicherheiten**

- Die Zahlen der Experimente sind **Messungen an erzeugten Daten**. Das Torzeit- und das Lieferzeit-Szenario sind gewählt; dass sie einem echten Betrieb ähneln, ist nicht gezeigt.
- Die Einfachform des Rayleigh-Tests ($P(nR^2 \\ge z) \\approx e^{-z}$) ist eine Näherung; ihre Größe wurde nachgemessen (E1), die Formelquelle nicht eingesehen. Literatur zum Mitteln von Quantilen und Wahrscheinlichkeiten wurde nicht gelesen; alle Aussagen sind eigene Messungen.
- Der CRPS wird aus einem Raster von 100 Quantilstufen gerechnet; das Raster ist für beide Verfahren gleich, der Rasterfehler wurde gegen die Formel geprüft (siehe README).
- Mausklicks auf die Diagramme (Auswahl) sind nicht Teil der Demo.
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* Zeitpunkte $\theta_1, \dots, \theta_n$ auf dem Kreis (Uhrzeit $h$ gehört zum Winkel $2\pi h / 24$) bzw. Verteilungen $F_1, \dots, F_J$ mit Gewichten $w_j$. *Gesucht:* eine Mitte.

| Symbol | Bedeutung |
|---|---|
| $\theta_i$, $c$ | Zeitpunkte und gesuchte Mitte, Winkel in $[0, 2\pi)$ |
| $d(\theta, c)$ | Bogenabstand auf dem Kreis |
| $R$ | resultierende Länge der Einheitsvektoren, aus $[0, 1]$ |
| $F_j$, $Q_j$, $w_j$ | Verteilungsfunktion, Quantilfunktion und Gewicht der Quelle $j$ |
| $s$, $m_j$ | Log-Streuung und Median der Lieferzeit |

**Bogenabstand.**

$$d(\theta, c) = \min\big(|\theta - c|,\ 2\pi - |\theta - c|\big).$$

**Die Familie der Mitten.** Für $p \ge 1$ minimiert die Mitte

$$F_p(c) = \frac{1}{n} \sum_{i=1}^{n} d(\theta_i, c)^p;$$

$p = 2$ ist das (intrinsische) Fréchet-Mittel, $p = 1$ der zirkuläre Median, $p \to \infty$ die Mitte des kleinsten Bogens, der alle Punkte enthält.

**Extrinsisches Kreismittel und Länge.** Mit $\bar C = \frac{1}{n}\sum \cos\theta_i$ und $\bar S = \frac{1}{n}\sum \sin\theta_i$ ist

$$R = \sqrt{\bar C^{2} + \bar S^{2}}, \qquad \hat\mu = \operatorname{atan2}(\bar S, \bar C).$$

**Einfacher Rayleigh-Test.** Unter Gleichverteilung gilt näherungsweise $P(nR^2 \ge z) \approx e^{-z}$; die Schwelle zum Niveau $0{,}05$ ist

$$R_{0{,}05} = \sqrt{\frac{-\ln 0{,}05}{n}} \approx \sqrt{\frac{3}{n}}.$$

**Exakte Aufzählung des Fréchet-Mittels ($p = 2$).** Zwischen zwei aufeinanderfolgenden Gegenpunkten $\theta_i + \pi$ ist mit festen entfalteten Winkeln $a_i$

$$F_2(c) = \frac{1}{n} \sum_{i=1}^{n} (c - a_i)^2,$$

also quadratisch mit dem Minimum bei $c^{*} = \frac{1}{n}\sum a_i$. Ein lokales Minimum von $F_2$ ist genau ein $c^{*}$, das in seinem eigenen Bogen liegt; an den Gegenpunkten knickt $F_2$ nach oben, dort liegt nie ein Minimum.

**Erzeuger.** Die auf den Kreis gefaltete Normalverteilung mit Streuung $\sigma$ (Bogenmaß) hat die resultierende Länge

$$\rho = e^{-\sigma^{2}/2}.$$

**Mischung und Quantilmittel.**

$$F_{\mathrm{mix}}(x) = \sum_{j} w_j F_j(x), \qquad Q_{\mathrm{vin}}(p) = \sum_{j} w_j Q_j(p).$$

Beide haben das Mittel $\bar\mu = \sum_j w_j \mu_j$. Die Varianz der Mischung ist

$$\operatorname{Var}_{\mathrm{mix}} = \sum_{j} w_j \big(\sigma_j^{2} + \mu_j^{2}\big) - \bar\mu^{2}.$$

**Lieferanten mit gleicher Log-Streuung.** Das Quantilmittel lognormaler Verteilungen mit Medianen $m_j$ und gleichem $s$ ist wieder lognormal mit dem Median $\sum_j w_j m_j$ und derselben Streuung $s$. Der tatsächliche Servicegrad eines Bestellpunkts $x$ ist die Mischungsverteilungsfunktion

$$\mathrm{SG}(x) = \sum_{j} w_j\, \Phi\!\Big(\frac{\ln x - \ln m_j}{s}\Big).$$

**Prognosegüte.** Der CRPS einer Verteilung mit Quantilfunktion $Q$ an der Beobachtung $y$ ist

$$\mathrm{CRPS}(Q, y) = 2 \int_0^1 \rho_p\big(y - Q(p)\big)\, dp, \qquad \rho_p(u) = u\,\big(p - \mathbf{1}[u < 0]\big),$$

hier als Mittel über 100 Stufen $p = (i + 0{,}5)/100$.

**Literatur.** Die Begriffe Fréchet-Mittel, Vincent-Mittel (Quantilmittel), lineare Mischung und Rayleigh-Test sind in der Literatur eingeführt; die Quellen wurden für diese Demo **nicht eingesehen**.
Alle Aussagen der Demo sind eigene Messungen an erzeugten Daten und Umformungen, die in `tests/` gegen Handrechnung, `math.erfc` und `scipy` geprüft sind.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
