"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Komplexität am Slider-Maximum (DEMO-PLAYBOOK §1c): Teil A rechnet das Fréchet-Mittel exakt und den zirkulären Median mit je O(n²) Zahlen (n ≤ 500: 250 000);
die Live-Messreihe „Nächte nachmessen“ nimmt LIVE_NIGHTS Nächte je Seed (Speicher/Rechnung LIVE_NIGHTS · n², begrenzt durch LIVE_BUDGET). Teil B ist geschlossen
(Mischungsquantil per Newton auf wenigen Stufen); die Live-Prognosemessung zieht LIVE_CASES Fälle mit K = QUANTILE_LEVELS Stufen."""

# --- Teil A: Ankunftszeiten am Tor ---------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 5, 500, 60                 # Ankünfte pro Nacht
MU_MIN, MU_MAX, DEFAULT_MU = 0.0, 23.75, 23.5         # wahre Mitte der Ankünfte (Uhr), Schritt 0,25
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.25, 6.0, 1.0  # Streuung der Ankünfte in Stunden (wrapped normal)
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 0, 100, 0       # Anteil des zweiten Gipfels in %
GAP_MIN, GAP_MAX, DEFAULT_GAP = 1.0, 12.0, 12.0       # Abstand des zweiten Gipfels in Stunden (12 h = gegenüber)
CUT_MIN, CUT_MAX, DEFAULT_CUT = 0, 23, 0              # Tagesbeginn des arithmetischen Mittels (Uhr)
WINDOW_MIN, WINDOW_MAX, DEFAULT_WINDOW = 0.5, 3.0, 1.0  # Fenster = Mitte ± so viele Stunden
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Teil B: Lieferzeiten und Prognosen ----------------------------------------------------------------------------------------------------
MEDIAN_A = 10.0                                       # Lieferant A: Median der Lieferzeit in Tagen (feste Annahme der Demo)
RATIO_MIN, RATIO_MAX, DEFAULT_RATIO = 1.0, 4.0, 1.5    # Median von B / Median von A
S_MIN, S_MAX, DEFAULT_S = 0.05, 0.8, 0.3               # Streuung der Lieferzeit im Logarithmus, für beide Lieferanten gleich
WB_MIN, WB_MAX, DEFAULT_WB = 10, 90, 50                # Anteil der Aufträge, die an B gehen, in %
SERVICE_MIN, SERVICE_MAX, DEFAULT_SERVICE = 80, 99, 95  # Zielservicegrad in %
SE_MIN, SE_MAX, DEFAULT_SE = 0.25, 4.0, 1.0            # Streuung des Prognosefehlers s_e (die wahre Streuung der Nachfrage ist 1)
CALIBRATIONS = ("überzuversichtlich", "ehrlich")
DEFAULT_CALIBRATION = "überzuversichtlich"
INTERVAL_LEVELS = (0.1, 0.9)                           # 80-%-Intervall
LIVE_CASES = 200                                       # Prognosefälle der Live-Messung (die vorgerechneten Reihen nehmen 1 500 je Seed)
QUANTILE_LEVELS = 100                                  # K Stufen des Quantilrasters für den CRPS
SHARE_BAND = 0.25                                      # „typischer“ Bereich: ± 25 % um das gemeinsame Mittel
SHARE_LO, SHARE_HI = 15.0, 25.0                        # Tage: Anteil der Masse in diesem Bereich (Beispiel der Einleitung)

# --- Messreihen (die Raster und Seeds der Messreihen selbst stehen in mir_experiments.py: sie gehören zur Prüfsumme der Erzeuger) ------------------
RESULTS_FILE = "mir_results.json"
LIVE_NIGHTS = 500                                      # Nächte je Seed bei „Nächte nachmessen“
LIVE_BUDGET = 400_000                                  # n² · Nächte je Seed höchstens (Rechenbudget der Live-Messung)

# --- Farben --------------------------------------------------------------------------------------------------------------------------------
COLOR_DATA = "rgba(120,120,120,0.55)"
COLOR_TRUE = "#222222"
CENTRE_COLORS = {
    "Arithmetisch (Tag ab 0 Uhr)": "#d62728",
    "Kreismittel (arctan2)": "#2F6B65",
    "Fréchet p = 2": "#4c78a8",
    "Zirkulärer Median p = 1": "#e08a1e",
    "Kleinster Bogen p = ∞": "#8c6bb1",
}
COLOR_MIX = "#4c78a8"
COLOR_QUANTILE = "#e08a1e"
COLOR_A = "#2F6B65"
COLOR_B = "#8c6bb1"

# --- Presets -------------------------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Nachtschicht am Tor (Standard)": {},
    "Tagschicht, kein Sprung": {"mu": 14.0, "sigma": 1.0},
    "Zwei Gipfel Früh/Spät": {"mu": 6.0, "share": 50, "gap": 12.0},
    "Ein Lkw mehr (Gegenpunkte)": {"mu": 0.0, "n": 41, "sigma": 0.5, "share": 51, "gap": 12.0},
    "Zwei Lieferanten": {"ratio": 3.0, "s": 0.15},
    "Prognose zu zweit": {"se": 2.0},
    "Prognose ehrlich gemeldet": {"se": 2.0, "calib": "ehrlich"},
}
PRESET_HELP = {
    "Nachtschicht am Tor (Standard)": "Teil A: Ankünfte um 23:30 Uhr mit einer Streuung von 1 Stunde, 60 Ankünfte. Die Uhrzeit läuft über Mitternacht: Das arithmetische Mittel (Tag ab 0 Uhr) landet am Nachmittag, das Kreismittel in der Nacht.",
    "Tagschicht, kein Sprung": "Teil A: Ankünfte um 14 Uhr, weit weg von Mitternacht. Hier fallen die Mitten praktisch zusammen: Zwischen arithmetischem Mittel und Kreismittel entscheidet nur der Schnittpunkt des Tages, und der liegt hier weit von den Daten.",
    "Zwei Gipfel Früh/Spät": "Teil A: eine Hälfte der Ankünfte um 6 Uhr, die andere um 18 Uhr. Die resultierende Länge R liegt weit unter der Schwelle; die meisten Mitten sitzen im Leeren zwischen den Häufungen, keine beschreibt die Daten.",
    "Ein Lkw mehr (Gegenpunkte)": "Teil A: 41 Ankünfte in zwei Häufungen gegenüber (0 und 12 Uhr), 20 gegen 21. Bei 49 statt 51 Prozent (ein Lkw wechselt die Seite, die Nacht ist aber eine andere Ziehung) springen in diesem Beispiel vier der fünf Mitten um mehr als 6 Stunden; der gepaarte Vergleich am selben Datensatz und wie weit die Mitten im Mittel springen, steht in E2.",
    "Zwei Lieferanten": "Teil B: Lieferant B liefert in der dreifachen Zeit von A (Median 30 gegen 10 Tage), beide streuen im Logarithmus mit 0,15. Mischung und Quantilmittel haben dasselbe Mittel, aber ganz andere Streuung.",
    "Prognose zu zweit": "Teil B: zwei Prognosen, jede liegt mit Streuung 2 neben dem wahren Erwartungswert, meldet aber nur die Streuung 1 der Nachfrage (überzuversichtlich). Messung mit 200 Fällen auf Knopfdruck.",
    "Prognose ehrlich gemeldet": "Teil B: dieselben zwei Prognosen, aber jede meldet ihren eigenen Fehler mit (Streuung 2,24). In der vorgerechneten Reihe von Experiment E5 ist dann das Quantilmittel im CRPS besser.",
}
