"""Kennzahlen und Verdikte. Jedes Verdikt der App ist eine reine Funktion `verdict_*(...) -> (Art, Text)` (DEMO-PLAYBOOK §12, tests/test_verdicts.py):
Sie bekommen Zahlen, keine Seeds, und jede genannte Zahl steht im Text aus ihren Argumenten."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import mir_circle as K
import mir_constants as C
import mir_experiments as X
import mir_scenario as S


@dataclass(frozen=True)
class Settings:
    """Alle Regler der Seitenleiste (nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld verschiebt sonst still Werte)."""
    n: int = C.DEFAULT_N
    mu: float = C.DEFAULT_MU
    sigma: float = C.DEFAULT_SIGMA
    share: int = C.DEFAULT_SHARE
    gap: float = C.DEFAULT_GAP
    cut: int = C.DEFAULT_CUT
    window: float = C.DEFAULT_WINDOW
    seed: int = C.DEFAULT_SEED
    ratio: float = C.DEFAULT_RATIO
    s: float = C.DEFAULT_S
    wb: int = C.DEFAULT_WB
    service: int = C.DEFAULT_SERVICE
    se: float = C.DEFAULT_SE
    calib: str = C.DEFAULT_CALIBRATION


# ---------------------------------------------------------------- Darstellung deutscher Zahlen

def de(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Prozentzeichen."""
    return de(100.0 * x, digits)


def hhmm(hours: float) -> str:
    """Uhrzeit in Stunden als „23:30“."""
    minutes = int(round((hours % 24.0) * 60.0)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ---------------------------------------------------------------- Teil A: eine Nacht am Tor

@dataclass(frozen=True)
class GateAnalysis:
    night: S.Night
    centres: dict                   # Name -> Uhrzeit (Stunden) der fünf Mitten
    R: float
    threshold: float                # Schwelle des einfachen Rayleigh-Tests zum Niveau 5 %
    p_value: float
    coverage: dict                  # Name -> Anteil der Ankünfte im Fenster Mitte ± window
    error_min: dict | None          # Name -> Abstand zur wahren Mitte in Minuten (nur bei einem Gipfel)
    oracle_coverage: float | None   # Fensterabdeckung um die wahre Mitte
    cut_means: dict                 # Tagesbeginn -> arithmetisches Mittel
    cut_range_min: float            # größter Abstand zweier dieser Mittel in Minuten
    spread_h: float                 # größter Abstand zweier der fünf Mitten (Stunden)
    circle_spread_h: float          # dasselbe ohne das arithmetische Mittel
    arith_gap_h: float              # Abstand arithmetisches Mittel – Kreismittel
    n_minima: int                   # lokale Minima der Fréchet-Funktion
    median_ties: int                # gleich gute Datenpunkte des zirkulären Medians
    settings: Settings = field(default_factory=Settings)


def pairwise_spread(values) -> float:
    """Größter Abstand zweier Uhrzeiten (Stunden) aus einer Liste, um Mitternacht herum gemessen."""
    values = list(values)
    return max((float(K.hours_distance(a, b)) for a in values for b in values), default=0.0)


def analyse_gate(settings: Settings) -> GateAnalysis:
    night = S.make_night(settings.n, settings.mu, settings.sigma, settings.share, settings.gap, settings.seed)
    hours = night.hours[None, :]
    centres, R = K.five_centres(hours, cut=settings.cut)
    centres = {name: float(c[0]) for name, c in centres.items()}
    Rv = float(R[0])
    theta = K.h2a(hours)
    coverage = {name: float(X.window_coverage(hours, np.array([c]), settings.window)[0]) for name, c in centres.items()}
    truth = night.truth
    error = None if truth is None else {name: 60.0 * float(K.hours_distance(c, truth)) for name, c in centres.items()}
    oracle = None if truth is None else float(X.window_coverage(hours, np.array([truth]), settings.window)[0])
    cut_means = {c: float(K.arithmetic_mean_with_cut(night.hours, c)) for c in X.CUTS}
    names = list(centres)
    circle = [centres[name] for name in names[1:]]
    return GateAnalysis(
        night=night, centres=centres, R=Rv, threshold=float(K.rayleigh_threshold(settings.n)), p_value=float(K.rayleigh_p(Rv, settings.n)),
        coverage=coverage, error_min=error, oracle_coverage=oracle, cut_means=cut_means, cut_range_min=60.0 * pairwise_spread(cut_means.values()),
        spread_h=pairwise_spread(centres.values()), circle_spread_h=pairwise_spread(circle), arith_gap_h=float(K.hours_distance(centres[names[0]], centres[names[1]])),
        n_minima=int(K.frechet_mean_batch(theta)[2][0]), median_ties=int(K.circular_median_batch(theta)[2][0]), settings=settings,
    )


def verdict_gate(R: float, n: int, circle_spread_h: float, arith_gap_h: float, cut: int, cut_range_min: float, n_second: int) -> tuple[str, str]:
    """Kernfrage von Teil A: Ist die Mitte der Ankünfte überhaupt bestimmt, und hängt sie am Rechenweg?

    1. R unter der Schwelle des Rayleigh-Tests (5 %): die Ankünfte sind von gleichverteilten nicht zu unterscheiden (bei zwei etwa gleich großen Häufungen: zwei Häufungen statt einer Mitte).
    2. Sonst: weicht das arithmetische Mittel um mehr als 15 Minuten vom Kreismittel ab, trägt der Schnittpunkt des Tages die Schuld.
    3. Sonst: alle Mitten stimmen innerhalb der Streuung überein."""
    threshold = float(K.rayleigh_threshold(n))
    if R < threshold:
        text = (f"Die resultierende Länge R = {de(R, 3)} liegt unter der Schwelle {de(threshold, 3)} (einfacher Rayleigh-Test, 5 %, etwa √(3/n) bei n = {n}): "
                "Diese Ankünfte sind von gleichmäßig über den Tag verteilten nicht zu unterscheiden. Eine einzelne Mitte ist hier fast Zufall.")
        if min(n_second, n - n_second) >= 0.2 * n:
            text += (f" Die Daten bestehen aus zwei Häufungen ({n - n_second} und {n_second} Ankünfte): Keine der fünf Mitten beschreibt sie, "
                     f"die Mitten liegen bis zu {de(circle_spread_h, 1)} Stunden auseinander.")
        return "warning", text
    if arith_gap_h > 0.25:
        return "warning", (f"Das arithmetische Mittel (Tag ab {cut} Uhr) liegt {de(arith_gap_h, 1)} Stunden neben dem Kreismittel: Die Ankünfte laufen über den Schnittpunkt des Tages. "
                           f"Mit einem anderen Tagesbeginn (0, 6, 12 oder 18 Uhr) streuen die arithmetischen Mittel dieser Nacht um {de(cut_range_min, 0)} Minuten, das Kreismittel nicht.")
    return "success", (f"Die Ankünfte häufen sich (R = {de(R, 3)}, Schwelle {de(threshold, 3)}), und kein Schnittpunkt stört: "
                       f"Das arithmetische Mittel liegt {de(60.0 * arith_gap_h, 0)} Minuten vom Kreismittel, die vier Kreis-Mitten höchstens {de(60.0 * circle_spread_h, 0)} Minuten auseinander.")


# ---------------------------------------------------------------- Teil B: Lieferanten

PLAN_QUANTILE = "Quantilmittel, Quantil"
PLAN_MIXTURE = "Mischung, Quantil (richtig)"


def verdict_suppliers(summary: dict) -> tuple[str, str]:
    """Kernfrage von Teil B (Lieferanten): Fallen Mischung und Quantilmittel zusammen, und was kostet es, wenn nicht?

    Sind die Streuungen beider höchstens 5 % verschieden, ja. Sonst haben beide dasselbe Mittel, die Mischung streut aber `sd_ratio`-mal so stark, und ein Plan auf
    dem Quantilmittel erreicht den Servicegrad nicht."""
    real = summary["service_at_plan"][PLAN_QUANTILE]
    target = f"{summary['service']:g}"
    if summary["sd_ratio"] < 1.05:
        return "success", (f"Die Mitten fallen praktisch zusammen: Die Mischung streut nur {de(100.0 * (summary['sd_ratio'] - 1.0), 1)} % stärker als das Quantilmittel. "
                           f"Ein Plan mit dem {target}-%-Quantil des Quantilmittels ({de(summary['plans'][PLAN_QUANTILE], 1)} Tage) erreicht {pct(real, 1)} % Servicegrad.")
    return "warning", (f"Mischung und Quantilmittel haben dasselbe Mittel ({de(summary['mix_mean'], 1)} Tage), aber die Mischung streut {de(summary['sd_ratio'], 1)}-mal so stark "
                       f"({de(summary['mix_sd'], 1)} gegen {de(summary['vin_sd'], 1)} Tage). Wer mit dem {target}-%-Quantil des Quantilmittels ({de(summary['plans'][PLAN_QUANTILE], 1)} Tage) plant, "
                       f"erreicht tatsächlich {pct(real, 1)} % Servicegrad statt {target} %; richtig wäre das Mischungsquantil von {de(summary['plans'][PLAN_MIXTURE], 1)} Tagen.")


# ---------------------------------------------------------------- Teil B: Prognosen zu zweit

def verdict_forecast(cell: dict) -> tuple[str, str]:
    """Kernfrage von Teil B (Prognosen): Welche Mitte zweier Prognosen ist besser? Es gibt keinen allgemeinen Sieger.

    Der CRPS-Unterschied (Mischung minus Quantilmittel, kleiner ist besser) gilt als erkennbar, wenn sein Mittel über die Seeds mehr als eine Standardabweichung über die Seeds von 0 entfernt ist."""
    diff = cell["diffs"]["mix_minus_vin"]["crps_diff"]
    mix, vin = cell["methods"]["Mischung"], cell["methods"]["Quantilmittel"]
    cov_mix, cov_vin = mix["covered"]["mean"], vin["covered"]["mean"]
    if abs(diff["mean"]) <= diff["std"]:
        crps_text = f"CRPS nicht zu unterscheiden (Mischung {de(mix['crps']['mean'], 3)}, Quantilmittel {de(vin['crps']['mean'], 3)}, Differenz {de(diff['mean'], 3)} ± {de(diff['std'], 3)} zwischen den Seeds)"
    elif diff["mean"] < 0:
        crps_text = f"Die Mischung hat den besseren CRPS ({de(mix['crps']['mean'], 3)} gegen {de(vin['crps']['mean'], 3)}, Differenz {de(diff['mean'], 3)} ± {de(diff['std'], 3)})"
    else:
        crps_text = f"Das Quantilmittel hat den besseren CRPS ({de(vin['crps']['mean'], 3)} gegen {de(mix['crps']['mean'], 3)}, Differenz {de(diff['mean'], 3)} ± {de(diff['std'], 3)})"
    better = "die Mischung" if abs(cov_mix - 0.8) < abs(cov_vin - 0.8) else "das Quantilmittel"
    return "info", (f"{crps_text}. Das 80-%-Intervall deckt bei der Mischung {pct(cov_mix, 1)} %, beim Quantilmittel {pct(cov_vin, 1)} % der Fälle (Soll 80 %): "
                    f"{better} liegt näher am Soll. Kein allgemeiner Sieger; das Ergebnis hängt an der Kalibrierung der Prognosen ({cell['calibration']}, Streuung des Lagefehlers {de(cell['s_e'], 2)}).")


# ---------------------------------------------------------------- Verdikte der Experimente (aus den vorgerechneten Zeilen)

def verdict_e1(uni: dict, power: dict) -> tuple[str, str]:
    """E1: R unter Zufallsstreuung (Zeile der Gleichverteilungs-Tabelle) und die Güte einer wahren Häufung (Zeile der Gütetabelle). Erkannt heißt: R über der Schwelle des Rayleigh-Tests (5 %).

    Die Art ist „success“, wenn die Häufung in mindestens 80 % der Nächte erkannt wird, sonst „warning“: dann ist die Mitte oft nicht von Zufall zu trennen."""
    text = (f"Unter reiner Zufallsstreuung (gleichverteilte Ankünfte, n = {uni['n']}) ist R im Mittel {de(uni['mean']['mean'], 3)}, in 5 % der Nächte über {de(uni['r95']['mean'], 3)} "
            f"(Schwelle √(−ln 0,05 / n) = {de(uni['threshold'], 3)}) und in {pct(uni['below_01']['mean'], 0)} % der Nächte unter 0,1. "
            f"Bei wahrer Länge ρ = {de(power['rho'], 3)} und n = {power['n']} erkennt der Test die Häufung in {pct(power['power']['mean'], 0)} % der Nächte; "
            f"die Richtung liegt im Mittel {de(power['err_mean_deg']['mean'], 0)}° daneben, in {pct(power['err_gt_45']['mean'], 0)} % der Nächte um mehr als 45°.")
    return ("success" if power["power"]["mean"] >= 0.8 else "warning"), text


def verdict_e2_peak(row: dict) -> tuple[str, str]:
    """E2 (ein Gipfel): Abstand und Genauigkeit der Mitten. Die Art ist immer „info“: das Ergebnis ist ein Vergleich, kein Alarm."""
    noise = f" mit {row['contamination']} % gleichverteilten Störankünften" if row["contamination"] else ""
    return "info", (f"Ein Gipfel (ρ = {de(row['rho'], 2)}, n = {row['n']}){noise}: Kreismittel und Fréchet-Mittel liegen im Mittel {de(row['dist_mean_deg']['mean'], 1)}° auseinander "
                    f"(in {pct(row['dist_gt_10']['mean'], 0)} % der Nächte um mehr als 10°). Abweichung von der wahren Mitte (Wurzel der mittleren quadratischen Abweichung): "
                    f"Kreismittel {de(row['rmse_ext']['mean'], 1)}°, Fréchet {de(row['rmse_int']['mean'], 1)}°, Median {de(row['rmse_med']['mean'], 1)}°, kleinster Bogen {de(row['rmse_cov']['mean'], 1)}°. "
                    f"In {pct(row['multi_minima']['mean'], 0)} % der Datensätze hat die Fréchet-Funktion mehr als ein lokales Minimum.")


def verdict_e2_clusters(row: dict) -> tuple[str, str]:
    """E2 (zwei Häufungen): Wo liegt die Mitte? „warning“, wenn das Kreismittel in mehr als der Hälfte der Datensätze im Leeren liegt (weiter als 20° von beiden Häufungen), sonst „info“."""
    ext_e, int_e = row["ext_empty"]["mean"], row["int_empty"]["mean"]
    med_on = row["med_a"]["mean"] + row["med_b"]["mean"]
    text = (f"Zwei Häufungen im Abstand {row['delta']:.0f}° ({row['split']}, R = {de(row['R']['mean'], 2)}): Das Kreismittel liegt in {pct(ext_e, 0)} % der Datensätze im Leeren, "
            f"das Fréchet-Mittel in {pct(int_e, 0)} %, der kleinste Bogen in {pct(row['cov_empty']['mean'], 0)} %. Der Median sitzt in {pct(med_on, 0)} % auf einer der Häufungen "
            f"(Häufung A: {pct(row['med_a']['mean'], 0)} %, B: {pct(row['med_b']['mean'], 0)} %).")
    return ("warning" if ext_e > 0.5 else "info"), text


def verdict_e2_lorry(row: dict) -> tuple[str, str]:
    """E2 (ein Lkw): Verschiebung jeder Mitte, wenn ein einziger Lkw die Häufung wechselt. „warning“, wenn irgendeine Mitte im Mittel um mehr als 60° springt."""
    shifts = {name: row[f"{key}_shift"]["mean"] for key, name in (("ext", "Kreismittel"), ("int", "Fréchet"), ("med", "Median"), ("cov", "kleinster Bogen"))}
    text = (f"Ein einziger Lkw wechselt die Häufung (n = {row['n']}, Abstand {row['delta']:.0f}°, R = {de(row['R']['mean'], 2)}): Das Kreismittel verschiebt sich im Mittel um {de(shifts['Kreismittel'], 1)}°, "
            f"das Fréchet-Mittel um {de(shifts['Fréchet'], 1)}°, der Median um {de(shifts['Median'], 1)}°, der kleinste Bogen um {de(shifts['kleinster Bogen'], 1)}°.")
    return ("warning" if max(shifts.values()) > 60 else "success"), text


def verdict_e2_grid(grid: dict) -> tuple[str, str]:
    """E2 (Orakel-Fund): Wie oft zählt ein Gitter die lokalen Minima falsch?"""
    return "info", (f"Ein Gitter mit {grid['grid_points']} Punkten zählt die lokalen Minima der Fréchet-Funktion in {pct(grid['wrong_share']['mean'], 1)} % der Datensätze falsch "
                    f"(Streuung zwischen den Seeds ±{pct(grid['wrong_share']['std'], 1)} Prozentpunkte). Die Demo zählt deshalb exakt.")


def verdict_e3(row: dict) -> tuple[str, str]:
    """E3: Torzeit-Fall bei einer Streuung. „warning“, weil das arithmetische Mittel schlechter ist, wenn die Mitte gleichverteilt über den Tag liegt."""
    a, c = row["rows"]["Arithmetisch (Tag ab 0 Uhr)"], row["rows"]["Kreismittel (arctan2)"]
    return "warning", (f"Bei einer Streuung von {de(row['sigma'], 1)} Stunden und Mitte irgendwo am Tag liegt das arithmetische Mittel im Mittel {de(a['err_min']['mean'], 0)} Minuten neben der wahren Mitte, "
                       f"das Kreismittel {de(c['err_min']['mean'], 0)} Minuten; ein Fenster von ± 1 Stunde um die Mitte fängt {pct(a['coverage']['mean'], 0)} % gegen {pct(c['coverage']['mean'], 0)} % der Ankünfte "
                       f"(ein Fenster um die wahre Mitte: {pct(row['rows']['_oracle']['mean'], 0)} %).")


def verdict_e3_night(night: dict, cut: dict) -> tuple[str, str]:
    """E3: Nachtschicht (feste Mitte) und Abhängigkeit vom Schnittpunkt."""
    a, c = night["rows"]["Arithmetisch (Tag ab 0 Uhr)"], night["rows"]["Kreismittel (arctan2)"]
    return "warning", (f"Nachtschicht mit der Mitte {hhmm(night['mu'])} Uhr (Streuung {de(night['sigma'], 1)} h, n = {night['n']}): Das arithmetische Mittel (Tag ab 0 Uhr) liegt im Median bei "
                       f"{de(a['median_estimate_h']['mean'], 1)} Uhr, im Mittel {de(a['err_min']['mean'], 0)} Minuten neben der Mitte, Fensterabdeckung {pct(a['coverage']['mean'], 0)} %; "
                       f"das Kreismittel liegt bei {de(c['median_estimate_h']['mean'], 2)} Uhr, {de(c['err_min']['mean'], 0)} Minuten daneben, Fensterabdeckung {pct(c['coverage']['mean'], 0)} %. "
                       f"Je nach Tagesbeginn (0, 6, 12 oder 18 Uhr) streuen die arithmetischen Mittel derselben Nacht bei beliebiger Mitte in {pct(cut['range_gt_1h']['mean'], 0)} % der Nächte um mehr als eine Stunde.")


def verdict_e3_peaks(peaks: dict) -> tuple[str, str]:
    """E3: zwei Gipfel (früh/spät). Keine Mitte beschreibt das; wer im Leeren sitzt, fängt kaum Ankünfte."""
    rows = peaks["rows"]
    names = {"Kreismittel (arctan2)": "Kreismittel", "Fréchet p = 2": "Fréchet", "Zirkulärer Median p = 1": "Median", "Kleinster Bogen p = ∞": "kleinster Bogen"}
    cov = ", ".join(f"{label} {pct(rows[name]['coverage']['mean'], 0)} %" for name, label in names.items())
    best = peaks["best_window"]["mean"]
    return "warning", (f"Zwei Gipfel ({hhmm(peaks['peaks'][0])} und {hhmm(peaks['peaks'][1])} Uhr, je etwa die Hälfte, R = {de(rows['_R']['mean'], 2)}): Fensterabdeckung (± 1 h) arithmetisch "
                       f"{pct(rows['Arithmetisch (Tag ab 0 Uhr)']['coverage']['mean'], 0)} %, {cov}. Das beste Einzelfenster fängt {pct(best, 0)} %.")


def verdict_e6(cell: dict) -> tuple[str, str]:
    """E6: Dasselbe Quantilmittel, in x oder in log x gemittelt. Die Differenz (x minus log x, positiv = log x besser) gilt als erkennbar, wenn ihr Mittel über die Seeds mehr als eine
    Standardabweichung über die Seeds von 0 entfernt ist („warning“: das Quantilmittel hängt an der Skala), sonst „info“. Die Mischung ist von der Skala unabhängig (monotone Umrechnung vertauscht mit dem Mitteln der Wahrscheinlichkeiten)."""
    d = cell["diffs"]["x_minus_logx"]["crps_diff"]
    m = cell["methods"]
    text = (f"Das Quantilmittel hängt an der Skala: Lagefehler im Verhältnis {de(cell['rel'], 1)} zur Streuung der Nachfrage ({cell['calibration']}), CRPS in x gemittelt {de(m['Quantilmittel in x']['crps']['mean'], 3)}, "
            f"in log x gemittelt {de(m['Quantilmittel in log x']['crps']['mean'], 3)} (Differenz {de(d['mean'], 4)} ± {de(d['std'], 4)} zwischen den Seeds). "
            f"Die Mischung ({de(m['Mischung']['crps']['mean'], 3)}) ändert sich mit der Skala nicht.")
    return ("warning" if abs(d["mean"]) > d["std"] else "info"), text
