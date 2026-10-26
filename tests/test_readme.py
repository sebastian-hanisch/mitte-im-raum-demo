"""README: Pflichtabschnitte, Hypothesen-Tabelle, Footer als letzter Absatz, Tabellen ohne literales | in Zellen, keine Status-Zusätze und keine Umgebungsaussagen."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_required_sections_in_playbook_order():
    headings = re.findall(r"^## (.+)$", README, flags=re.M)
    wanted = ["Warum dieses Problem", "Modell", "Methodik", "Befunde (gemessen, keine Behauptungen)", "Hypothesen des Einzelplans (Ja/Nein nach eigener Messung)",
              "Befunde und Korrekturen gegenüber dem Plan", "Ehrliche Grenzen", "Verifikation", "Dateistruktur", "Bewusst nicht umgesetzt", "Lokal ausführen", "Tests ausführen"]
    positions = [headings.index(h) for h in wanted]
    assert positions == sorted(positions)


def test_all_thirteen_hypotheses_have_a_verdict():
    rows = re.findall(r"^\| (H\d+) \|.*\| (.+) \|$", README, flags=re.M)
    assert [r[0] for r in rows] == [f"H{i}" for i in range(1, 14)]
    for name, verdict in rows:
        assert verdict.strip().startswith(("**Ja**", "**Teilweise**", "**Nein", "**Offen**")), name


def test_footer_is_the_last_paragraph_and_the_live_link_is_a_comment():
    last = README.strip().split("\n\n")[-1]
    assert last.startswith("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)")
    assert "<!-- TODO nach dem Deploy" in README and README.count("streamlit.app") == 1


def test_no_status_additions_environment_statements_or_forbidden_words():
    lowered = README.lower()
    for word in ("geplant", "kommt noch", "saatwert", "schwanz", "noch nicht angelegt", "läuft lokal", "läuft nur lokal"):
        assert word not in lowered, word
    assert "Zufalls-Seed" in README
    for ascii_umlaut in ("Haeufung", "Raeume", "Maerkte", "Waehrend", "fuer ", "ueber "):
        assert ascii_umlaut not in README, ascii_umlaut


def test_every_table_row_has_a_consistent_number_of_cells():
    """Ein literales | in einer Zelle bricht die Spalten (feedback_markdown_table_literal_pipe)."""
    lines = README.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-| ]+\|$", lines[i + 1]):
            cells = lines[i].count("|")
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == cells, (j + 1, lines[j][:80])
                j += 1
            i = j
        else:
            i += 1


def test_the_readme_names_what_was_not_read_and_what_is_chosen():
    assert "gewählt, keine Messung an echten Daten" in README and "nicht eingesehen" in README and "Von-Mises" in README
    assert "Kein allgemeiner Sieger" in README or "keinen allgemeinen Sieger" in README
