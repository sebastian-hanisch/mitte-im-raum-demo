"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen
(einzeln, dazu alle Minima und alle Maxima zugleich) und prüft, dass kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` und Fehler bei Reglern mit
berechneten Grenzen. Der Regler „Abstand des zweiten Gipfels“ ist ohne zweiten Gipfel deaktiviert und wird erst nach dem Aktivieren geprüft."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mir_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e)[:300] for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title[0].value
    assert [h.value.split(":")[0] for h in at.subheader if h.value.startswith("🔬")] == [f"🔬 E{i}" for i in range(1, 7)] and any(h.value.startswith("🚧") for h in at.subheader)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e)[:300] for e in at.exception])


def _set_edge(at, widget, edge):
    value = widget.min if edge == "min" else widget.max
    if isinstance(widget.value, int):
        value = int(value)
    widget.set_value(value)


def test_every_slider_at_its_min_and_max_does_not_raise():
    sliders = [(s.key, s.label) for s in _fresh_app().slider]
    assert sliders, "keine Slider gefunden"
    for key, label in sliders:
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = next(s for s in at.slider if s.key == key)
            if slider.disabled:
                at.slider(key="share_slider").set_value(50).run(timeout=TIMEOUT)         # zweiter Gipfel an: der Abstandsregler wird frei
                slider = next(s for s in at.slider if s.key == key)
                assert not slider.disabled
            _set_edge(at, slider, edge)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e)[:300] for e in at.exception])


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sidebar_sliders_at_the_same_edge_do_not_raise(edge):
    """Eckfall-Matrix: alle Regler der Seitenleiste zugleich am Minimum oder am Maximum (zweiter Gipfel vorher an, damit der Abstand frei ist)."""
    at = _fresh_app()
    at.sidebar.slider(key="share_slider").set_value(50).run(timeout=TIMEOUT)
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.lo is None or state_key == "seed_input":
            continue
        widget = at.sidebar.slider(key=state_key)
        _set_edge(at, widget, edge)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e)[:300] for e in at.exception]


def test_every_select_option_does_not_raise():
    first = _fresh_app()
    boxes = [(s.key, list(s.options)) for s in first.selectbox]
    assert boxes
    for key, options in boxes:
        for option in options:
            at = _fresh_app()
            next(s for s in at.selectbox if s.key == key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e)[:300] for e in at.exception])
    sliders = [(s.key, list(s.options)) for s in first.select_slider]
    assert sliders
    for key, options in sliders:
        for option in options:
            at = _fresh_app()
            next(s for s in at.select_slider if s.key == key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e)[:300] for e in at.exception])


def test_live_buttons_show_a_result_and_the_pdf_download():
    at = _fresh_app()
    at.button(key="forecast_live").click().run(timeout=TIMEOUT)
    assert not at.exception and any("Kein allgemeiner Sieger" in i.value for i in at.info)
    at.button(key="gate_live").click().run(timeout=TIMEOUT)
    assert not at.exception and any(i.value.startswith("Nachgemessen:") for i in at.info)
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert len(at.get("download_button")) == 1


def test_live_result_is_marked_stale_after_a_changed_setting():
    at = _fresh_app()
    at.button(key="forecast_live").click().run(timeout=TIMEOUT)
    at.sidebar.slider(key="se_slider").set_value(3.0).run(timeout=TIMEOUT)
    assert not at.exception and any("Knopf erneut drücken" in c.value for c in at.caption)


def test_permalink_round_trip_and_clamping():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "9999"
    at.query_params["mu"] = "7.5"
    at.query_params["calib"] = "ehrlich"
    at.query_params["wb"] = "abc"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.sidebar.slider(key="n_slider").value == 500 and at.sidebar.slider(key="mu_slider").value == 7.5
    assert at.sidebar.selectbox(key="calib_select").value == "ehrlich" and at.sidebar.slider(key="wb_slider").value == 50
    bad = AppTest.from_file(APP_PATH)
    bad.query_params["calib"] = "egal"
    bad.run(timeout=TIMEOUT)
    assert not bad.exception and bad.sidebar.selectbox(key="calib_select").value == "überzuversichtlich"
