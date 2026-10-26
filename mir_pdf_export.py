"""PDF-Export (fpdf2, Kernschriften): Einstellungen, die fünf Mitten der Nacht, die Lieferanten-Pläne und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import mir_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "‖": "|", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"',
            "ε": "eps", "ρ": "rho", "σ": "sigma", "∞": "unendlich", "±": "+-", "→": "->", "μ": "mu", "Δ": "Delta"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def centre_rows(a: V.GateAnalysis) -> list[tuple[str, str, str, str]]:
    """(Name, Uhrzeit, Abstand zur wahren Mitte in Minuten oder „-“, Fensterabdeckung in %) der fünf Mitten."""
    rows = []
    for name, h in a.centres.items():
        err = "-" if a.error_min is None else V.de(a.error_min[name], 1)
        rows.append((name, V.hhmm(h), err, V.pct(a.coverage[name], 0)))
    return rows


def build_pdf(a: V.GateAnalysis, summary: dict, verdicts: list[tuple[str, str]]) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Mitte in anderen Räumen: Kreis und Verteilungen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    truth = "keine wahre Mitte (zwei Gipfel)" if a.night.truth is None else f"wahre Mitte {V.hhmm(a.night.truth)} Uhr"
    pdf.multi_cell(0, 5, _latin(f"Teil A: n = {s.n} Ankünfte, {truth}, Streuung {V.de(s.sigma, 2)} h, Anteil zweiter Gipfel {s.share} %, Abstand {V.de(s.gap, 1)} h, Tagesbeginn {s.cut} Uhr, "
                                f"Fenster +- {V.de(s.window, 2)} h, Zufalls-Seed {s.seed}. R = {V.de(a.R, 3)}, Schwelle {V.de(a.threshold, 3)}."))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Die fünf Mitten", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for name, hhmm, err, cov in centre_rows(a):
        pdf.cell(0, 5.5, _latin(f"{name}: {hhmm} Uhr, Abstand zur wahren Mitte {err} min, Fensterabdeckung {cov} %"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Teil B: Lieferanten", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Median A {V.de(10.0, 0)} Tage, Median B {V.de(10.0 * summary['ratio'], 1)} Tage, Log-Streuung {V.de(summary['s'], 2)}, Anteil B {summary['share_b']:.0f} %, "
                                f"Zielservicegrad {summary['service']:.0f} %. Mittel {V.de(summary['mix_mean'], 2)} Tage, Streuung Mischung {V.de(summary['mix_sd'], 2)}, Quantilmittel {V.de(summary['vin_sd'], 2)}."))
    for name, plan in summary["plans"].items():
        pdf.cell(0, 5.5, _latin(f"{name}: Bestellpunkt {V.de(plan, 1)} Tage, tatsächlicher Servicegrad {V.pct(summary['service_at_plan'][name], 1)} %"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Alle Daten sind erzeugt (wrapped normal, Lognormal); das Torzeit- und das Lieferzeit-Szenario sind gewählt, keine Messung an echten Daten."))
    return bytes(pdf.output())
