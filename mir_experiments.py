"""Messreihen der Experimente E1 bis E6: reine Funktionen über feste Zufalls-Seeds, die sowohl die vorgerechneten Tabellen (tools/make_results.py → data/mir_results.json)
als auch die Live-Knöpfe der App erzeugen. Jede Zelle nutzt vier feste Seeds (SEEDS); angegeben wird der Mittelwert mit der Standardabweichung über die Seeds.

Zufall: `Draws(cell_seed(...))` aus mir_scenario.py; ein Seed gehört zu einer Zelle (Seed, Marken), nie zu einer Instanz allein. Alle verglichenen Verfahren sehen dieselben Daten
(gepaarter Vergleich: eine Mischung und ein Quantilmittel derselben Prognosefälle, vier Mitten desselben Datensatzes).

Die Raster und Seeds stehen hier und nicht in mir_constants.py, damit das Ändern eines Hilfetextes die Prüfsumme der Erzeuger (`source_hash`) nicht verändert."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import mir_circle as K
import mir_constants as C
import mir_dist as D
import mir_scenario as S

ROOT = Path(__file__).resolve().parent
SOURCES = ("mir_rng.py", "mir_scenario.py", "mir_circle.py", "mir_dist.py", "mir_experiments.py", "tools/make_results.py")

SEEDS = (8101, 8102, 8103, 8104)                        # vier feste Zufalls-Seeds je Zelle
N_RASTER = (3, 5, 8, 10, 30, 100, 1000)                 # E1: Ankünfte je Nacht unter Gleichverteilung
RHO_RASTER = (0.125, 0.25, 0.45, 0.7)                   # E1: wahre resultierende Länge des Erzeugers
N_POWER = (10, 30, 100, 200)                            # E1: Ankünfte je Nacht für die Güte
DELTAS = (60, 90, 120, 150, 170, 180)                   # E2: Abstand der beiden Häufungen in Grad
CLUSTER_SIGMA_DEG = 13.0                                # E2: Streuung innerhalb einer Häufung (Grad)
CLUSTER_N = 40                                          # E2: Punkte der Zwei-Häufungen-Datensätze
LORRY_N = 41                                            # E2: ungerade, damit der Median eindeutig ist (20 gegen 21)
NEAR_DEG = 20.0                                         # E2: „in der Häufung“ heißt höchstens so weit vom Häufungszentrum
SIGMAS_H = (0.5, 1.0, 2.0, 3.0)                         # E3: Streuung der Ankünfte in Stunden
TORZEIT_N = 60                                          # E3: Ankünfte je Nacht
WINDOW = 1.0                                            # E3: Fenster = Mitte ± WINDOW Stunden
CUTS = (0, 6, 12, 18)                                   # E3: Tagesbeginne, die verglichen werden
RATIOS = (1.0, 1.2, 1.5, 2.0, 3.0)                      # E4: Medianverhältnis B/A
S_VALUES = (0.15, 0.3, 0.5)                             # E4: Log-Streuung
SE_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)                  # E5: Streuung des Prognosefehlers (Nachfrage hat Streuung 1)
SCALE_SPREAD_Y = 0.4                                    # E6: Streuung der Nachfrage im Logarithmus
SCALE_SPREAD_M = 0.5                                    # E6: Streuung der wahren Lage im Logarithmus
SCALE_REL = (0.5, 1.0, 2.0)                             # E6: s_e im Verhältnis zur Nachfrage-Streuung
CASES = 1500                                            # E5/E6: Prognosefälle je Seed
P_GRID = D.quantile_grid(C.QUANTILE_LEVELS)             # K Stufen für den CRPS
LEVELS = np.concatenate([P_GRID, np.asarray(C.INTERVAL_LEVELS)])      # dazu die Ränder des 80-%-Intervalls


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def agg(values) -> dict:
    """Mittel und Standardabweichung über die Seeds einer Zelle (die Einzelwerte bleiben erhalten)."""
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "per_seed": [float(x) for x in v]}


def _deg(x) -> np.ndarray:
    return np.degrees(np.asarray(x, dtype=float))


# ================================================================== E1: wann ist die Mitte zufällig?

def uniform_cell(n: int, runs: int, seeds=SEEDS) -> dict:
    """Resultierende Länge R unter Gleichverteilung: Mittel, 95-%-Quantil, Größe des einfachen Rayleigh-Tests, Anteil R < 0,1; dazu die Näherungen √(π/4n) und √(−ln 0,05/n)."""
    thr = float(K.rayleigh_threshold(n))
    cols = {"mean": [], "r95": [], "size": [], "below_01": []}
    for seed in seeds:
        R = K.resultant_length(S.uniform_angles(S.Draws(S.cell_seed(seed, "gleich", n)), runs, n))
        cols["mean"].append(R.mean())
        cols["r95"].append(np.quantile(R, 0.95))
        cols["size"].append((R > thr).mean())
        cols["below_01"].append((R < 0.1).mean())
    return {"n": n, "runs_per_seed": runs, "approx_mean": math.sqrt(math.pi / (4 * n)), "threshold": thr, **{k: agg(v) for k, v in cols.items()}}


def power_cell(rho: float, n: int, runs: int, seeds=SEEDS) -> dict:
    """Güte und Richtungsfehler bei wahrer Richtung 0 und wahrer resultierender Länge ρ: P(R über der Rayleigh-Schwelle), P(Fehler > 45°), mittlerer Fehler in Grad."""
    thr = float(K.rayleigh_threshold(n))
    cols = {"power": [], "err_gt_45": [], "err_mean_deg": []}
    for seed in seeds:
        th = S.wrapped_angles(S.Draws(S.cell_seed(seed, "guete", rho, n)), runs, n, rho)
        err = K.arc_distance(K.mean_direction(th), 0.0)
        cols["power"].append((K.resultant_length(th) > thr).mean())
        cols["err_gt_45"].append((err > math.pi / 4).mean())
        cols["err_mean_deg"].append(_deg(err).mean())
    return {"rho": rho, "n": n, "runs_per_seed": runs, **{k: agg(v) for k, v in cols.items()}}


def runs_uniform(n: int) -> int:
    return 20000 if n <= 100 else 5000


def run_m1() -> dict:
    return {
        "uniform": [uniform_cell(n, runs_uniform(n)) for n in N_RASTER],
        "power": [power_cell(rho, n, 4000) for rho in RHO_RASTER for n in N_POWER],
    }


# ================================================================== E2: extrinsisch gegen intrinsisch, zwei Häufungen

ONE_PEAK_SETTINGS = ((0.45, 20, 0), (0.45, 100, 0), (0.7, 20, 0), (0.9, 20, 20), (0.9, 100, 20))      # (ρ, n, Prozent gleichverteilte Störankünfte)
SKEW_SETTINGS = (20, 100)                                                                              # n der schiefen Datensätze (70 % bei 0°, 30 % bei 100°)
SPLITS = ("fest 50/50", "binomial 50/50", "fest 60/40")


def four_centres(theta) -> dict:
    """Extrinsisch, intrinsisch, Median, kleinster Bogen für einen Stapel Winkel (T, n), mit Zahl lokaler Fréchet-Minima und Gleichstandszahl des Medians."""
    ext = K.mean_direction(theta)
    intr, _, n_min, n_tie = K.frechet_mean_batch(theta)
    med, _, med_ties = K.circular_median_batch(theta)
    cov, _ = K.cover_arc_batch(theta)
    return {"ext": ext, "int": intr, "med": med, "cov": cov, "n_minima": n_min, "median_ties": med_ties}


def one_peak_cell(rho: float, n: int, contamination: int, runs: int, seeds=SEEDS) -> dict:
    """Ein Gipfel bei 0 mit wahrer Länge ρ; `contamination` Prozent der Punkte (die letzten) sind gleichverteilte Störankünfte. Abstand extrinsisch–intrinsisch,
    Wurzel der mittleren quadratischen Abweichung (RMSE) der vier Mitten von der wahren Mitte 0, Anteil Datensätze mit mindestens zwei lokalen Fréchet-Minima."""
    k = int(round(n * contamination / 100))
    cols = {"dist_mean_deg": [], "dist_gt_10": [], "rmse_ext": [], "rmse_int": [], "rmse_med": [], "rmse_cov": [], "multi_minima": []}
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "ein-gipfel", rho, n, contamination))
        th = S.wrapped_angles(d, runs, n, rho)
        if k:
            th[:, n - k:] = S.uniform_angles(d, runs, k)
        c = four_centres(th)
        dist = _deg(K.arc_distance(c["ext"], c["int"]))
        cols["dist_mean_deg"].append(dist.mean())
        cols["dist_gt_10"].append((dist > 10).mean())
        for key in ("ext", "int", "med", "cov"):
            cols[f"rmse_{key}"].append(math.sqrt((_deg(K.arc_distance(c[key], 0.0)) ** 2).mean()))
        cols["multi_minima"].append((c["n_minima"] >= 2).mean())
    return {"rho": rho, "n": n, "contamination": contamination, "runs_per_seed": runs, **{k_: agg(v) for k_, v in cols.items()}}


def skew_cell(n: int, runs: int, seeds=SEEDS) -> dict:
    """Schiefe Daten: 70 % um 0°, 30 % um 100° (beide mit wahrer Länge 0,9). Abstand der beiden Mitten extrinsisch und intrinsisch."""
    k = int(round(0.3 * n))
    cols = {"dist_mean_deg": [], "dist_gt_2": []}
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "schief", n))
        th = S.wrapped_angles(d, runs, n, 0.9)
        th[:, n - k:] = S.wrapped_angles(d, runs, k, 0.9, centre=math.radians(100.0))
        c = four_centres(th)
        dist = _deg(K.arc_distance(c["ext"], c["int"]))
        cols["dist_mean_deg"].append(dist.mean())
        cols["dist_gt_2"].append((dist > 2).mean())
    return {"n": n, "runs_per_seed": runs, **{k_: agg(v) for k_, v in cols.items()}}


def two_cluster_cell(delta: float, split: str, runs: int, seeds=SEEDS, n: int = CLUSTER_N) -> dict:
    """Zwei Häufungen bei 0 und Δ (Streuung CLUSTER_SIGMA_DEG): Anteil der Datensätze, in denen die Mitte in Häufung A, in Häufung B oder im Leeren (weiter als NEAR_DEG von beiden) liegt."""
    cols = {f"{key}_{where}": [] for key in ("ext", "int", "med", "cov") for where in ("a", "b", "empty")}
    cols.update({"R": [], "median_ties": []})
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "zwei-haeufungen", delta, split))
        if split == "binomial 50/50":
            first = S.binomial_counts(d, runs, n, 0.5)
        else:
            first = round(n * (0.5 if split == "fest 50/50" else 0.6))
        th = S.two_cluster_angles(d, runs, n, delta, first, CLUSTER_SIGMA_DEG)
        c = four_centres(th)
        for key in ("ext", "int", "med", "cov"):
            in_a = _deg(K.arc_distance(c[key], 0.0)) < NEAR_DEG
            in_b = _deg(K.arc_distance(c[key], math.radians(delta))) < NEAR_DEG
            cols[f"{key}_a"].append(in_a.mean())
            cols[f"{key}_b"].append((in_b & ~in_a).mean())
            cols[f"{key}_empty"].append((~in_a & ~in_b).mean())
        cols["R"].append(K.resultant_length(th).mean())
        cols["median_ties"].append((c["median_ties"] > 1).mean())
    return {"delta": delta, "split": split, "n": n, "runs_per_seed": runs, **{k_: agg(v) for k_, v in cols.items()}}


def one_lorry_cell(delta: float, runs: int, seeds=SEEDS, n: int = LORRY_N) -> dict:
    """Ein einziger Lkw wechselt die Häufung (n = 41: 20 gegen 21, dann 21 gegen 20; gleicher Zufallsstrom, nur der Punkt 21 sitzt in B oder in A).
    Mittlere Verschiebung jeder Mitte in Grad und Anteil der Paare mit Verschiebung über 60°."""
    cols = {f"{key}_{what}": [] for key in ("ext", "int", "med", "cov") for what in ("shift", "gt_60")}
    cols["R"] = []
    for seed in seeds:
        s = S.cell_seed(seed, "ein-lkw", delta)
        before = S.two_cluster_angles(S.Draws(s), runs, n, delta, n // 2, CLUSTER_SIGMA_DEG)
        after = S.two_cluster_angles(S.Draws(s), runs, n, delta, n // 2 + 1, CLUSTER_SIGMA_DEG)
        c1, c2 = four_centres(before), four_centres(after)
        for key in ("ext", "int", "med", "cov"):
            shift = _deg(K.arc_distance(c1[key], c2[key]))
            cols[f"{key}_shift"].append(shift.mean())
            cols[f"{key}_gt_60"].append((shift > 60).mean())
        cols["R"].append(K.resultant_length(before).mean())
    return {"delta": delta, "n": n, "runs_per_seed": runs, **{k_: agg(v) for k_, v in cols.items()}}


def grid_miscount(runs: int, seeds=SEEDS, grid_points: int = 720) -> dict:
    """Wie oft zählt ein Gitter die lokalen Minima der Fréchet-Funktion falsch? Datensätze mit n = 3 bis 12 Punkten (je ein Drittel gleichverteilt, ein Gipfel mit Länge 0,6, zwei Gegenhäufungen)."""
    grid = np.arange(grid_points) * (2 * math.pi / grid_points)
    wrong, total_min_exact, total_min_grid = [], [], []
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "gitter", grid_points))
        sizes = 3 + (d.uniform(runs) * 10).astype(int)
        bad = 0
        for i in range(runs):
            n = int(sizes[i])
            if i % 3 == 0:
                th = S.uniform_angles(d, 1, n)[0]
            elif i % 3 == 1:
                th = S.wrapped_angles(d, 1, n, 0.6)[0]
            else:
                th = S.two_cluster_angles(d, 1, n, 180.0, n // 2, 17.0)[0]
            f = K.frechet_value(grid, th)
            grid_minima = int(((f < np.roll(f, 1)) & (f <= np.roll(f, -1))).sum())
            exact_minima = len(K.frechet_minima(th)[0])
            bad += grid_minima != exact_minima
        wrong.append(bad / runs)
    return {"grid_points": grid_points, "runs_per_seed": runs, "wrong_share": agg(wrong)}


def run_m2() -> dict:
    return {
        "one_peak": [one_peak_cell(rho, n, cont, 1500) for rho, n, cont in ONE_PEAK_SETTINGS],
        "skew": [skew_cell(n, 1500) for n in SKEW_SETTINGS],
        "clusters": [two_cluster_cell(delta, split, 800) for delta in DELTAS for split in SPLITS],
        "lorry": [one_lorry_cell(delta, 600) for delta in DELTAS],
        "grid": grid_miscount(500),
    }


# ================================================================== E3: der Torzeit-Fall

EXTRA_CUT12 = "Arithmetisch (Tag ab 12 Uhr)"
TABLE_NAMES = (*K.CENTRE_NAMES, EXTRA_CUT12)


def centre_table(hours, cut: float = 0.0):
    """Die fünf Mitten plus das arithmetische Mittel mit Tagesbeginn 12 Uhr, in Stunden; dazu R."""
    centres, R = K.five_centres(hours, cut=cut)
    centres[EXTRA_CUT12] = K.arithmetic_mean_with_cut(hours, 12.0)
    return {name: centres[name] for name in TABLE_NAMES}, R


def window_coverage(hours, centre, window: float = WINDOW) -> np.ndarray:
    """Anteil der Ankünfte einer Nacht im Fenster Mitte ± window Stunden (hours: (T, n), centre: (T,))."""
    return (K.hours_distance(hours, np.asarray(centre)[:, None]) <= window).mean(axis=1)


def best_window_coverage(hours, window: float = WINDOW) -> np.ndarray:
    """Das beste Einzelfenster Ankunft ± window Stunden (Fenster um jede Ankunft der Nacht probieren), Anteil der Ankünfte darin."""
    best = np.zeros(hours.shape[0])
    for j in range(hours.shape[1]):
        best = np.maximum(best, window_coverage(hours, hours[:, j], window))
    return best


def evaluate_nights(hours, truth, cut: float = 0.0, window: float = WINDOW, peaks=None) -> dict:
    """Je Mitte: mittlerer Fehler in Minuten und P(Fehler > 1 h) gegen die wahre Mitte (falls truth gegeben), Fensterabdeckung; bei `peaks` (Liste von Uhrzeiten) auch der
    Anteil „nahe einem Gipfel“ (< 2 h) und „im Leeren“. Rückgabe {Mitte: {...}, "_R": mittleres R, "_oracle": Abdeckung eines Fensters um die wahre Mitte}."""
    centres, R = centre_table(hours, cut)
    out = {}
    for name, c in centres.items():
        row = {"coverage": float(window_coverage(hours, c, window).mean())}
        if truth is not None:
            err = K.hours_distance(c, truth)
            row["err_min"] = float(60.0 * err.mean())
            row["err_gt_1h"] = float((err > 1.0).mean())
        if peaks is not None:
            dist = np.min([K.hours_distance(c, p) for p in peaks], axis=0)
            row["near_peak"] = float((dist < 2.0).mean())
            row["empty"] = float((dist >= 2.0).mean())
        out[name] = row
    out["_R"] = float(R.mean())
    if truth is not None:
        out["_oracle"] = float(window_coverage(hours, truth, window).mean())
    return out


def _aggregate_rows(per_seed: list[dict]) -> dict:
    out = {}
    for key, value in per_seed[0].items():
        if isinstance(value, dict):
            out[key] = {k: agg([ps[key][k] for ps in per_seed]) for k in value}
        else:
            out[key] = agg([ps[key] for ps in per_seed])
    return out


def torzeit_cell(sigma: float, runs: int, seeds=SEEDS, n: int = TORZEIT_N) -> dict:
    """Eine Nacht mit n Ankünften um eine über den Tag gleichverteilte Mitte, Streuung sigma Stunden."""
    per_seed = []
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "torzeit", sigma))
        mu = d.uniform(runs) * 24.0
        per_seed.append(evaluate_nights(S.gate_hours(d, runs, n, mu, sigma), mu))
    return {"sigma": sigma, "n": n, "runs_per_seed": runs, "rows": _aggregate_rows(per_seed)}


def nightshift_cell(runs: int, seeds=SEEDS, n: int = TORZEIT_N, mu: float = 23.5, sigma: float = 1.0) -> dict:
    """Nachtschicht: feste Mitte 23,5 Uhr. Zusätzlich der Median der Schätzung über die Nächte (Stunden)."""
    per_seed, medians = [], []
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "nachtschicht", mu, sigma))
        hours = S.gate_hours(d, runs, n, mu, sigma)
        per_seed.append(evaluate_nights(hours, np.full(runs, mu)))
        centres, _ = centre_table(hours)
        medians.append({name: float(np.median(c)) for name, c in centres.items()})
    rows = _aggregate_rows(per_seed)
    for name in TABLE_NAMES:
        rows[name]["median_estimate_h"] = agg([m[name] for m in medians])
    return {"mu": mu, "sigma": sigma, "n": n, "runs_per_seed": runs, "rows": rows}


def cut_cell(runs: int, seeds=SEEDS, n: int = TORZEIT_N, sigma: float = 1.0) -> dict:
    """Schnittpunkt des Tages: dieselben Daten, arithmetische Mittel bei Tagesbeginn 0, 6, 12 und 18 Uhr. Spannweite = größter Abstand zweier dieser Mittel."""
    cols = {"range_gt_1h": [], "range_mean_min": []}
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "schnittpunkt", sigma))
        mu = d.uniform(runs) * 24.0
        hours = S.gate_hours(d, runs, n, mu, sigma)
        means = np.stack([K.arithmetic_mean_with_cut(hours, c) for c in CUTS], axis=1)
        spread = np.max([K.hours_distance(means[:, i], means[:, j]) for i in range(len(CUTS)) for j in range(len(CUTS))], axis=0)
        cols["range_gt_1h"].append((spread > 1.0).mean())
        cols["range_mean_min"].append(60.0 * spread.mean())
    return {"sigma": sigma, "n": n, "cuts": list(CUTS), "runs_per_seed": runs, **{k: agg(v) for k, v in cols.items()}}


def two_peak_cell(runs: int, seeds=SEEDS, n: int = TORZEIT_N, sigma: float = 1.0, early: float = 6.0, late: float = 18.0) -> dict:
    """Zwei Gipfel (früh 6 Uhr, spät 18 Uhr), jede Ankunft gehört mit Wahrscheinlichkeit 1/2 zu jedem. Fensterabdeckung, Anteil nahe einem Gipfel und im Leeren;
    dazu das beste Einzelfenster."""
    per_seed, best = [], []
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "zwei-gipfel", sigma))
        second = S.binomial_counts(d, runs, n, 0.5)
        hours = S.two_peak_hours(d, runs, n, early, late - early, second, sigma)
        per_seed.append(evaluate_nights(hours, None, peaks=(early, late)))
        best.append(best_window_coverage(hours).mean())
    return {"sigma": sigma, "n": n, "peaks": [early, late], "runs_per_seed": runs, "rows": _aggregate_rows(per_seed), "best_window": agg(best)}


def run_m3() -> dict:
    return {
        "torzeit": [torzeit_cell(sigma, 3000) for sigma in SIGMAS_H],
        "nightshift": nightshift_cell(3000),
        "cut": cut_cell(3000),
        "two_peak": two_peak_cell(3000),
    }


# ================================================================== E4: Lieferantenmix und Bestellpunkt (geschlossene Formeln)

def supplier_summary(ratio: float, s: float, share_b: float = 50.0, service: float = 95.0) -> dict:
    """Zwei Lieferanten mit Median 10 Tage (A) und 10·ratio Tage (B), gleiche Log-Streuung s; Anteil share_b Prozent der Aufträge geht an B.

    Mischung (jeder Auftrag geht zufällig an einen der beiden) gegen Quantilmittel (die Quantile gemittelt): Mittel, Streuung, Masse nahe dem gemeinsamen Mittel, Masse im Bereich
    (SHARE_LO, SHARE_HI) und der tatsächliche Servicegrad der Pläne, die auf dem Quantilmittel oder auf einer „typischen“ Lieferzeit beruhen (Wahrheit = die Mischung)."""
    medians = [C.MEDIAN_A, C.MEDIAN_A * ratio]
    w = [1.0 - share_b / 100.0, share_b / 100.0]
    level = service / 100.0
    z = float(D.norm_ppf(level))
    mix_mean, mix_sd = D.lognormal_mixture_moments(medians, s, w)
    vin_mean, vin_sd = D.lognormal_vincent_moments(medians, s, w)
    vin_median = D.lognormal_vincent_median(medians, w)

    def mix_cdf(x):
        return float(D.lognormal_mixture_cdf(x, medians, s, w))

    def vin_cdf(x):
        return float(D.lognormal_cdf(x, vin_median, s))

    plans = {
        "Mischung, Quantil (richtig)": float(D.lognormal_mixture_quantile(np.array([level]), medians, s, w)[0]),
        "Quantilmittel, Quantil": vin_median * math.exp(s * z),
        "Quantilmittel: Mittel + z · Std": vin_mean + z * vin_sd,
        "Typische Lieferzeit (Mittel)": mix_mean,
    }
    lo, hi = (1.0 - C.SHARE_BAND) * mix_mean, (1.0 + C.SHARE_BAND) * mix_mean
    return {
        "ratio": ratio, "s": s, "share_b": share_b, "service": service,
        "mix_mean": mix_mean, "mix_sd": mix_sd, "vin_mean": vin_mean, "vin_sd": vin_sd, "sd_ratio": mix_sd / vin_sd if vin_sd > 0 else float("nan"),
        "band_mix": mix_cdf(hi) - mix_cdf(lo), "band_vin": vin_cdf(hi) - vin_cdf(lo),
        "range_mix": mix_cdf(C.SHARE_HI) - mix_cdf(C.SHARE_LO), "range_vin": vin_cdf(C.SHARE_HI) - vin_cdf(C.SHARE_LO),
        "plans": plans, "service_at_plan": {name: mix_cdf(x) for name, x in plans.items()},
    }


def supplier_simulation(ratio: float, s: float, draws_per_seed: int = 200000, seeds=SEEDS, share_b: float = 50.0) -> dict:
    """Die Mischung aus der Definition gezogen (jeder Auftrag geht zufällig an A oder B) und das Quantilmittel mit gemeinsamem z: Mittel, Streuung, Masse in (SHARE_LO, SHARE_HI)."""
    medians = np.array([C.MEDIAN_A, C.MEDIAN_A * ratio])
    cols = {"mix_mean": [], "mix_sd": [], "mix_range": [], "vin_mean": [], "vin_sd": [], "vin_range": []}
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "lieferanten-sim", ratio, s))
        pick_b = d.uniform(draws_per_seed) < share_b / 100.0
        z = d.normal(draws_per_seed)
        x = medians[pick_b.astype(int)] * np.exp(s * d.normal(draws_per_seed))
        y = ((1.0 - share_b / 100.0) * medians[0] + share_b / 100.0 * medians[1]) * np.exp(s * z)
        for name, v in (("mix", x), ("vin", y)):
            cols[f"{name}_mean"].append(v.mean())
            cols[f"{name}_sd"].append(v.std())
            cols[f"{name}_range"].append(((v > C.SHARE_LO) & (v < C.SHARE_HI)).mean())
    return {"ratio": ratio, "s": s, "draws_per_seed": draws_per_seed, **{k: agg(v) for k, v in cols.items()}}


def run_m4() -> dict:
    return {
        "example": supplier_summary(3.0, 0.15),
        "example_s03": supplier_summary(3.0, 0.3),
        "simulation": supplier_simulation(3.0, 0.15),
        "grid": [supplier_summary(ratio, s) for s in S_VALUES for ratio in RATIOS],
    }


# ================================================================== E5/E6: Prognosen zu zweit mitteln (Mischung gegen Quantilmittel)

def _scores(Q, y) -> dict:
    """CRPS (Quantilraster), Abdeckung und Breite des 80-%-Intervalls je Fall für ein Feld Quantile Q (T, K + 2)."""
    k = len(P_GRID)
    return {
        "crps": D.crps_from_quantiles(Q[:, :k], y, P_GRID),
        "covered": D.interval_covered(y, Q[:, k], Q[:, k + 1]).astype(float),
        "width": Q[:, k + 1] - Q[:, k],
    }


def forecast_methods(mu, sigma_rep, m, spread_y: float = 1.0) -> dict:
    """Quantile (T, K + 2) der verglichenen Verfahren für zwei Normalprognosen (Lage mu (T, 2), gemeldete Streuung sigma_rep: Zahl oder Paar) bei gleichen Gewichten."""
    scales = np.broadcast_to(np.asarray(sigma_rep, dtype=float), mu.shape)
    z = D.norm_ppf(LEVELS)[None, :]
    return {
        "Mischung": D.mixture_quantile_batch(LEVELS, mu, scales, [0.5, 0.5]),
        "Quantilmittel": D.vincent_quantile_batch(LEVELS, mu, scales, [0.5, 0.5]),
        "Eine Prognose allein": mu[:, :1] + scales[:, :1] * z,
        "Wahrheit (bekannte Lage)": m[:, None] + spread_y * z,
    }


def forecast_cell(s_e: float, calibration: str, cases: int, seeds=SEEDS, sigma_rep=None) -> dict:
    """Zwei Prognosen derselben Größe, Nachfrage y = m + ε (ε ~ N(0, 1), m ~ N(0, 1)); Prognose i meldet N(m + e_i, σ_rep), e_i ~ N(0, s_e²).
    Gleiche Daten für beide Meldearten (die Ziehung hängt nur von s_e ab). `sigma_rep` überschreibt die gemeldete Streuung (Zahl oder Paar)."""
    sigma = S.reported_sigma(calibration, s_e) if sigma_rep is None else sigma_rep
    per_seed = []
    for seed in seeds:
        m, y, mu = S.forecast_cases(S.Draws(S.cell_seed(seed, "prognose", s_e)), cases, s_e)
        per_seed.append({name: _scores(Q, y) for name, Q in forecast_methods(mu, sigma, m).items()})
    return _forecast_result(per_seed, {"s_e": s_e, "calibration": calibration, "cases": cases, "sigma_rep": sigma if np.isscalar(sigma) else [float(x) for x in sigma]},
                            pairs={"mix_minus_vin": ("Mischung", "Quantilmittel")})


def log_methods(mu, sigma_rep, m) -> dict:
    """Quantile (T, K + 2) in x für Lognormalprognosen (ln-Lage mu): Mischung, Quantilmittel in x (arithmetisch), Quantilmittel in log x (geometrisch), Wahrheit."""
    scales = np.broadcast_to(np.asarray(sigma_rep, dtype=float), mu.shape)
    z = D.norm_ppf(LEVELS)[None, :]
    q1, q2 = np.exp(mu[:, :1] + scales[:, :1] * z), np.exp(mu[:, 1:] + scales[:, 1:] * z)
    return {
        "Mischung": np.exp(D.mixture_quantile_batch(LEVELS, mu, scales, [0.5, 0.5])),
        "Quantilmittel in x": 0.5 * (q1 + q2),
        "Quantilmittel in log x": np.sqrt(q1 * q2),
        "Wahrheit (bekannte Lage)": np.exp(m[:, None] + SCALE_SPREAD_Y * z),
    }


def scale_cell(rel: float, calibration: str, cases: int, seeds=SEEDS) -> dict:
    """Lognormale Nachfrage: ln y = m + 0,4·ε; die Prognosen melden lognormal(m + e_i, σ_rep) mit e_i ~ N(0, s_e²), s_e = rel · 0,4. Der CRPS wird in x gerechnet."""
    s_e = rel * SCALE_SPREAD_Y
    sigma = math.sqrt(SCALE_SPREAD_Y ** 2 + s_e ** 2) if calibration == "ehrlich" else SCALE_SPREAD_Y
    per_seed = []
    for seed in seeds:
        m, ly, mu = S.forecast_cases(S.Draws(S.cell_seed(seed, "skala", rel)), cases, s_e, spread_y=SCALE_SPREAD_Y, spread_m=SCALE_SPREAD_M)
        y = np.exp(ly)
        per_seed.append({name: _scores(Q, y) for name, Q in log_methods(mu, sigma, m).items()})
    return _forecast_result(per_seed, {"rel": rel, "s_e": s_e, "calibration": calibration, "cases": cases, "sigma_rep": sigma},
                            pairs={"x_minus_logx": ("Quantilmittel in x", "Quantilmittel in log x"), "mix_minus_x": ("Mischung", "Quantilmittel in x"),
                                   "mix_minus_logx": ("Mischung", "Quantilmittel in log x")})


def _forecast_result(per_seed: list[dict], head: dict, pairs: dict) -> dict:
    methods = {name: {key: agg([ps[name][key].mean() for ps in per_seed]) for key in ("crps", "covered", "width")} for name in per_seed[0]}
    diffs = {}
    for label, (a, b) in pairs.items():
        diffs[label] = {"crps_diff": agg([(ps[a]["crps"] - ps[b]["crps"]).mean() for ps in per_seed]),
                        "first_better_share": agg([(ps[a]["crps"] < ps[b]["crps"]).mean() for ps in per_seed])}
    return {**head, "methods": methods, "diffs": diffs}


def run_m5() -> dict:
    return {
        "cells": [forecast_cell(s_e, cal, CASES) for cal in C.CALIBRATIONS for s_e in SE_VALUES],
        "unequal": forecast_cell(0.5, "überzuversichtlich", CASES, sigma_rep=np.array([0.5, 2.0])),
    }


def run_m6() -> dict:
    return {"cells": [scale_cell(rel, cal, CASES) for cal in C.CALIBRATIONS for rel in SCALE_REL]}


# ================================================================== Live-Messung der App

def gate_study(n: int, mu: float, sigma: float, share_percent: float, gap: float, cut: float, window: float, nights: int, seeds=SEEDS) -> dict:
    """„Nächte nachmessen“: viele Nächte mit den Einstellungen der Seitenleiste (feste Mitte, bei zwei Gipfeln die Anteile aus der Seitenleiste), alle fünf Mitten."""
    k = S.second_peak_count(n, share_percent)
    per_seed = []
    for seed in seeds:
        d = S.Draws(S.cell_seed(seed, "live-nacht", n, mu, sigma, k, gap))
        hours = S.two_peak_hours(d, nights, n, mu, gap, k, sigma)
        per_seed.append(evaluate_nights(hours, np.full(nights, mu % 24.0) if k == 0 else None, cut=cut, window=window,
                                        peaks=None if k == 0 else (mu % 24.0, (mu + gap) % 24.0)))
    return {"n": n, "mu": mu, "sigma": sigma, "n_second": k, "gap": gap, "cut": cut, "window": window, "nights_per_seed": nights, "rows": _aggregate_rows(per_seed)}


PARTS = {"m1": run_m1, "m2": run_m2, "m3": run_m3, "m4": run_m4, "m5": run_m5, "m6": run_m6}
