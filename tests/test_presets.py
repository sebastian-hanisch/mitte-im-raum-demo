"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset einzeln erkennbar; Permalink-Begrenzung."""

import pytest

import mir_constants as C
import mir_evaluation as ev
import mir_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
                assert isinstance(value, (int, float)) and (spec.caster is float or float(value).is_integer()), (name, key)


def test_default_preset_changes_nothing():
    assert C.PRESETS[next(iter(C.PRESETS))] == {}


def test_setting_specs_and_preset_keys_are_one_to_one():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len(set(P.PRESET_KEYS)) == len(P.SETTING_SPECS)


def test_presets_are_pairwise_different_in_at_least_one_setting():
    resolved = [tuple(preset.get(k, P.SETTING_SPECS[s].default) for k, s in P.PRESET_KEYS.items()) for preset in C.PRESETS.values()]
    assert len(set(resolved)) == len(resolved)


def test_settings_defaults_agree_with_the_specs():
    s = ev.Settings()
    mapping = {"n_slider": s.n, "mu_slider": s.mu, "sigma_slider": s.sigma, "share_slider": s.share, "gap_slider": s.gap, "cut_slider": s.cut, "window_slider": s.window,
               "seed_input": s.seed, "ratio_slider": s.ratio, "s_slider": s.s, "wb_slider": s.wb, "service_slider": s.service, "se_slider": s.se, "calib_select": s.calib}
    assert set(mapping) == set(P.SETTING_SPECS)
    for key, value in mapping.items():
        assert P.SETTING_SPECS[key].default == value, key


def test_preset_settings_run_through_the_analysis():
    for name, preset in C.PRESETS.items():
        kwargs = {P.PRESET_KEYS[k].replace("_slider", "").replace("_select", "").replace("_input", ""): v for k, v in preset.items()}
        a = ev.analyse_gate(ev.Settings(**kwargs))
        assert 0.0 <= a.R <= 1.0, name


def test_the_four_gate_presets_show_what_their_help_says():
    """Zahlen der Hilfetexte stehen in den Presets: Nachtschicht 23:30 Uhr, Streuung 1 h, 60 Ankünfte; Tagschicht 14 Uhr; Zwei Gipfel 6 und 18 Uhr; Ein Lkw: 41 Ankünfte, 49/51 %."""
    base = ev.Settings()
    assert (base.mu, base.sigma, base.n) == (23.5, 1.0, 60)
    day = ev.analyse_gate(ev.Settings(mu=14.0, sigma=1.0))
    night = ev.analyse_gate(ev.Settings())
    assert day.arith_gap_h < 0.1 and night.arith_gap_h > 5.0                                    # Tagschicht: kein Sprung; Nacht: Schnittpunkt trifft die Häufung
    two = ev.analyse_gate(ev.Settings(**{"mu": 6.0, "share": 50, "gap": 12.0}))
    assert two.R < two.threshold and two.night.n_second == 30
    lorry = C.PRESETS["Ein Lkw mehr (Gegenpunkte)"]
    assert (lorry["n"], lorry["share"]) == (41, 51) and ev.analyse_gate(ev.Settings(n=41, share=51, mu=0.0, sigma=0.5)).night.n_second == 21
    assert ev.analyse_gate(ev.Settings(n=41, share=49, mu=0.0, sigma=0.5)).night.n_second == 20


def test_the_one_lorry_preset_makes_four_of_five_centres_jump_by_more_than_six_hours():
    """Aussage des Hilfetextes zum Preset „Ein Lkw mehr (Gegenpunkte)“ (ein Beispiel, kein Mittel): 49 statt 51 Prozent, sonst gleiche Einstellungen."""
    import mir_circle as K
    kw = dict(n=41, mu=0.0, sigma=0.5, gap=12.0)
    a49, a51 = ev.analyse_gate(ev.Settings(share=49, **kw)), ev.analyse_gate(ev.Settings(share=51, **kw))
    jumps = [float(K.hours_distance(a49.centres[name], a51.centres[name])) for name in a49.centres]
    assert sum(j > 6.0 for j in jumps) == 4 and jumps[0] < 6.0                                       # nur das arithmetische Mittel (Tag ab 0 Uhr) bleibt


def test_the_two_peak_preset_puts_most_centres_into_the_gap():
    a = ev.analyse_gate(ev.Settings(mu=6.0, share=50, gap=12.0))
    import mir_circle as K
    empty = [min(float(K.hours_distance(h, 6.0)), float(K.hours_distance(h, 18.0))) >= 2.0 for h in a.centres.values()]
    assert sum(empty) >= 4 and a.R < a.threshold / 5


def test_the_forecast_preset_help_texts_state_checkable_numbers():
    import math
    over, honest = C.PRESETS["Prognose zu zweit"], C.PRESETS["Prognose ehrlich gemeldet"]
    assert over["se"] == honest["se"] == 2.0 and "calib" not in over and honest["calib"] == "ehrlich"
    assert "mit Streuung 2 neben" in C.PRESET_HELP["Prognose zu zweit"] and ev.de(math.sqrt(1 + 2.0 ** 2), 2) == "2,24" and "2,24" in C.PRESET_HELP["Prognose ehrlich gemeldet"]
    import mir_results as R
    assert R.forecast_cell(2.0, "ehrlich")["diffs"]["mix_minus_vin"]["crps_diff"]["mean"] > 3 * R.forecast_cell(2.0, "ehrlich")["diffs"]["mix_minus_vin"]["crps_diff"]["std"]   # „Quantilmittel im CRPS besser“


def test_the_supplier_preset_help_numbers():
    import mir_experiments as X
    r = C.PRESETS["Zwei Lieferanten"]
    assert r == {"ratio": 3.0, "s": 0.15} and "30 gegen 10 Tage" in C.PRESET_HELP["Zwei Lieferanten"] and "0,15" in C.PRESET_HELP["Zwei Lieferanten"]
    s = X.supplier_summary(r["ratio"], r["s"])
    assert s["mix_mean"] == pytest.approx(s["vin_mean"], rel=1e-12) and s["sd_ratio"] > 3
