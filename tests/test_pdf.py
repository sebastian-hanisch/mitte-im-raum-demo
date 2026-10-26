"""PDF-Export: erzeugt gültige PDF-Bytes mit Umlauten und Sonderzeichen, für jedes Preset, mit und ohne wahre Mitte."""

import mir_constants as C
import mir_evaluation as V
import mir_experiments as X
import mir_pdf_export as P


def _pdf(settings):
    a = V.analyse_gate(settings)
    summary = X.supplier_summary(settings.ratio, settings.s, float(settings.wb), float(settings.service))
    verdicts = [V.verdict_gate(a.R, settings.n, a.circle_spread_h, a.arith_gap_h, settings.cut, a.cut_range_min, a.night.n_second), V.verdict_suppliers(summary)]
    return P.build_pdf(a, summary, verdicts)


def test_pdf_for_every_preset_is_a_pdf():
    for name, preset in C.PRESETS.items():
        data = _pdf(V.Settings(**preset))
        assert data[:5] == b"%PDF-" and len(data) > 1500, name


def test_latin_replaces_characters_the_core_fonts_do_not_know():
    assert P._latin("R ≤ √(3/n) – ρ = 0,3 · 45° **fett** ∞ ± 1") == "R <= Wurzel(3/n) - rho = 0,3 * 45 Grad fett unendlich +- 1"
    assert P._latin("Größe ä ö ü ß") == "Größe ä ö ü ß"


def test_centre_rows_by_hand():
    a = V.analyse_gate(V.Settings(mu=14.0, sigma=0.25))
    rows = P.centre_rows(a)
    assert [r[0] for r in rows] == list(a.centres) and rows[1][1] == V.hhmm(list(a.centres.values())[1]) and rows[1][3] == V.pct(a.coverage[rows[1][0]], 0) and rows[1][2] == V.de(a.error_min[rows[1][0]], 1)
    two = P.centre_rows(V.analyse_gate(V.Settings(share=50)))
    assert all(r[2] == "-" for r in two)
