"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen (Kreis, Uhrzeit h gehört zum Winkel 15°·h):
* Uhrzeiten 23, 23,5, 0,5, 1 (Winkel 345°, 352,5°, 7,5°, 15°, also −15°, −7,5°, 7,5°, 15° um die Mitternacht):
  - arithmetisch (Tag ab 0 Uhr) (23 + 23,5 + 0,5 + 1)/4 = 12,00 h; mit Tagesbeginn 12 Uhr (23, 23,5, 24,5, 25)/4 = 24,00 = 0,00 h;
  - Kreismittel 0 h (die Punkte liegen spiegelbildlich zu 0°); R = (cos 15° + cos 7,5°)/2 = 0,97869;
  - Fréchet p = 2: Minimum bei 0 h mit F = (2·15² + 2·7,5²)/4 Grad² = 140,625 Grad²; ein zweites lokales Minimum am Gegenpunkt 12 h (F dort viel größer);
  - zirkulärer Median: mittlere Entfernung von 23 h: (0 + 7,5 + 22,5 + 30)/4 = 15°, von 23,5 h: (7,5 + 0 + 15 + 22,5)/4 = 11,25°, von 0,5 h ebenso 11,25°, von 1 h 15°;
    Gleichstand zwischen 23,5 h und 0,5 h, es gilt der kleinere Winkel, also 0,5 h;
  - kleinster Bogen: größte Lücke von 15° bis 345° (330°), der Bogen 345° bis 15° hat die Länge 30°, Mitte 0 h.
* Zwei Gegenpunkte 0 h, 12 h: R = 0, zwei gleich gute Minima bei 6 h und 18 h mit F = (π/2)² = π²/4.
* Drei gleichmäßige Punkte 0, 8, 16 h: R = 0, drei Minima, jedes auf einem Punkt, F = (0 + 2·(2π/3)²)/3 = 8π²/27.
* Vier gleichmäßige Punkte 0, 6, 12, 18 h: vier Minima bei 3, 9, 15, 21 h mit F = (2·45² + 2·135²)/4 Grad² = 10125 Grad².
* Verteilungen: N(0, 1) und N(10, 1) zu gleichen Teilen: Quantilmittel N(5, 1); Mischung mit Mittel 5 und Varianz 1 + 25 = 26.
  Diskrete {1, 2, 3} und {10, 20, 30}: Quantilmittel (sortiert, paarweise gemittelt) 5,5 / 11 / 16,5; Mischung aus allen sechs Punkten mit Mittel 11.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def uhr4():
    """Uhrzeiten 23, 23,5, 0,5, 1 in Stunden."""
    return np.array([23.0, 23.5, 0.5, 1.0])


@pytest.fixture
def gegenpunkte():
    return np.array([0.0, 12.0])


@pytest.fixture
def drei_punkte():
    return np.array([0.0, 8.0, 16.0])


@pytest.fixture
def vier_punkte():
    return np.array([0.0, 6.0, 12.0, 18.0])


def deg(x):
    return math.radians(x)
