# Mitte in anderen Räumen: Kreis und Verteilungen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mitte-im-raum-demo.streamlit.app/)** -->

Stück 8 der Reihe **„Mittelwerte und Lagemaße“** (optional), Teil der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Reihe fragt, welche Mitte zu welcher Frage passt. Dieses Stück verlässt die Zahlengerade: **Teil A** mittelt **Uhrzeiten** (Lkw-Ankünfte am Tor, ein Kreis von 24 Stunden), **Teil B** mittelt **ganze Verteilungen** (Lieferzeiten zweier Lieferanten, zwei Prognosen). Leitbegriff ist das **Fréchet-Mittel**: die Mitte als Minimierer von Σ d(xᵢ, c)ᵖ in einem beliebigen Raum. Aus der Reihe: [mittelwert-median-demo](https://github.com/sebastian-hanisch/mittelwert-median-demo), [lp-mitte-demo](https://github.com/sebastian-hanisch/lp-mitte-demo), [trimmen-huber-demo](https://github.com/sebastian-hanisch/trimmen-huber-demo), [weiszfeld-demo](https://github.com/sebastian-hanisch/weiszfeld-demo) und [median-of-means-demo](https://github.com/sebastian-hanisch/median-of-means-demo); dieses Stück hat keine Voraussetzung unter ihnen und baut nichts von ihnen nach.

Das Torzeit- und das Lieferzeit-Szenario sind **gewählt, keine Messung an echten Daten**; alle Daten sind erzeugt.

**Abgrenzung** (README-Abgleich vor dem Bau):
[shift-demo](https://github.com/sebastian-hanisch/shift-demo) behandelt Schichten über Mitternacht als Überdeckungsproblem (totale Unimodularität), nicht als Mitte;
[forecast-combination-demo](https://github.com/sebastian-hanisch/forecast-combination-demo) kombiniert Punktprognosen und nennt die Quantilkombination als nicht gebaut;
[forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) bildet Intervalle aus den Fehlern eines Punktmodells;
[bunching-demo](https://github.com/sebastian-hanisch/bunching-demo) misst Takt und Wartezeit, keine Uhrzeit-Mitte.
Rangfolgen als Mitte stehen in `rangaggregation-demo`, Graphen in [centrality-demo](https://github.com/sebastian-hanisch/centrality-demo) und [p-center-demo](https://github.com/sebastian-hanisch/p-center-demo); hier nur als Verweis.

## Warum dieses Problem

Zwischen 23 Uhr und 1 Uhr liegen 2 Stunden, nicht 22. Das arithmetische Mittel der Uhrzeiten 23, 23,5, 0,5 und 1 ist **12,00 Uhr**, das Kreismittel **0 Uhr**. Eine Nachtschicht, die um 23:30 Uhr ihre Mitte hat, bekommt mit dem arithmetischen Mittel (Tag ab 0 Uhr) eine „typische Ankunft“ am Nachmittag.
Der Kreis-Hook allein ist ein Zweizeiler (`arctan2`). **Was die Messreihen darüber hinaus tragen:** wann die Mitte auf dem Kreis nur noch Zufall ist (Schwelle für R), was die Wahl des Mittels im Betrieb kostet (Torzeit-Fenster, Bestellpunkt), und dass eine Mitte bei zwei Häufungen zwar definiert, aber sinnlos ist.

**Kernfrage:** *Wann ist „die Mitte“ in einem anderen Raum als der Zahlengeraden gut definiert, wann ist sie definiert, aber sinnlos, und was kostet es im Betrieb, wenn man trotzdem das gewohnte Mittel nimmt?*

## Modell

* **Teil A (Kreis).** n Ankünfte einer Nacht, Uhrzeit h gehört zum Winkel 2π·h/24. Erzeuger: **wrapped normal** (eine Normalverteilung, auf den Kreis gefaltet) um eine wahre Mitte, optional mit einem zweiten Gipfel; ihre resultierende Länge ist ρ = exp(−σ²/2). Eine Von-Mises-Verteilung wird nicht verwendet.
  Fünf Mitten: arithmetisch (Tag ab c Uhr), **Kreismittel** (Richtung der Resultierenden, extrinsisch), **Fréchet p = 2** (intrinsisch, **exakt aufgezählt**), **zirkulärer Median p = 1**, **kleinster Bogen p = ∞**.
* **Teil B (Verteilungen).** Zwei Lieferanten mit Lognormal-Lieferzeit (Median 10 Tage für A, ein wählbares Vielfaches für B, gleiche Log-Streuung s). Zwei Arten der „Mitte“: die **Mischung** (Wahrscheinlichkeiten mitteln: jeder Auftrag geht zufällig an einen Lieferanten) und das **Quantilmittel** (Quantile mitteln). Beide haben dasselbe Mittel. Für Prognosen: Nachfrage y = m + ε mit bekannter Wahrheit, zwei Normalprognosen mit Lagefehler sₑ, gemeldete Streuung „überzuversichtlich“ (nur die Nachfrage-Streuung 1) oder „ehrlich“ (√(1 + sₑ²)).
* Formeln und Symbole: Expander **📐 Mathematische Formulierung** der App.

## Methodik

Sechs Experimente (**E1 bis E6**), jedes mit Verdikt aus den Zahlen. Vier feste Zufalls-Seeds je Zelle (`SEEDS`), angegeben wird das Mittel (und, wo es zählt, die Streuung zwischen den Seeds); alle verglichenen Verfahren sehen **dieselben Daten** (vier Mitten desselben Datensatzes, Mischung und Quantilmittel derselben Prognosefälle). Die Messreihen sind vorgerechnet (`data/mir_results.json`, Erzeuger-Prüfsumme der Quellen, `tools/make_results.py`); die Knöpfe „Nachmessen“ in der App rechnen mit den Reglern der Seitenleiste neu. Prozentangaben ohne Nachkommastelle sind auf ganze Prozent gerundet: „100 %“ heißt mindestens 99,5 %, „0 %“ höchstens 0,5 %.

| Experiment | Frage | Kern |
|---|---|---|
| E1 | Wann ist die Mitte Zufall? | R unter Gleichverteilung, Güte und Richtungsfehler einer wahren Häufung |
| E2 | Zwei Häufungen, ein Lkw, Sehne gegen Bogen | extrinsisch gegen intrinsisch, Mitte im Leeren, Sprung durch einen einzigen Lkw, Gitterfehler |
| E3 | Der Torzeit-Fall | Fehler und Fensterabdeckung, Nachtschicht, Schnittpunkt, zwei Gipfel |
| E4 | Lieferantenmix und Bestellpunkt | Mischung gegen Quantilmittel in geschlossener Form, tatsächlicher Servicegrad |
| E5 | Zwei Prognosen mitteln | CRPS und Abdeckung, überzuversichtlich gegen ehrlich |
| E6 | Das Quantilmittel hängt an der Skala | x gegen log x |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und werden dort aus `data/mir_results.json` (bzw. von Hand) nachgerechnet.

| Frage | Befund |
|---|---|
| Uhrzeiten 23, 23,5, 0,5, 1 | arithmetisch 12,00 h, Kreismittel 0 h, R = 0,979; die Fréchet-Funktion hat zwei lokale Minima (Handrechnung, `tests/test_circle.py`). |
| E1: R unter reiner Zufallsstreuung | Mittel von R: 0,402 / 0,282 / 0,162 / 0,089 / 0,028 bei n = 5 / 10 / 30 / 100 / 1000. R liegt in 5 % der Nächte über 0,315 bei n = 30; die Schwelle √(−ln 0,05 / n) ≈ √(3/n) (0,316) trifft ab n = 8 auf 0,01 genau (R₉₅ 0,603 gegen 0,612), bei n = 3 **nicht** (0,940 gegen 0,999). |
| E1: Größe des einfachen Rayleigh-Tests (soll 5 %) | n = 3 / 5 / 8 / 10 / 30 / 100 / 1000: 0,1 / 4,3 / 4,5 / 4,6 / 4,9 / 4,8 / 5,2 %. Unter reiner Zufallsstreuung ist R < 0,1 bei n = 10 in 9,1 %, bei n = 30 in 25,7 %, bei n = 100 in 62,9 % der Nächte. |
| E1: Güte (R über der Schwelle) | wahre Länge ρ = 0,125: 34 % bei n = 100, 60 % bei n = 200; ρ = 0,25: 39 % bei n = 30, 91 % bei n = 100; ρ = 0,45: 41 % bei n = 10, 93 % bei n = 30; ρ = 0,7: 94 % bei n = 10. **Die Mitte wird stetig unzuverlässig, nicht plötzlich undefiniert:** bei ρ = 0,125 und n = 30 liegt die Richtung in 43 % der Nächte mehr als 45° daneben (mittlerer Fehler 50°), bei ρ = 0,25 und n = 100 in 1,1 % (13°). |
| E2: ein Gipfel, Sehne gegen Bogen (ρ = 0,45, n = 20) | Kreismittel und Fréchet-Mittel liegen im Mittel 7,4° auseinander (n = 100: 3,5°), in 26 % der Nächte um mehr als 10°. **Das intrinsische Mittel ist nicht genauer:** Abweichung von der wahren Mitte 20,9° gegen 20,9° (n = 100: 8,7° gegen 8,8°); Median 23,9°, kleinster Bogen 38,8°. Bei ρ = 0,7 ist es mit 10,8° gegen 11,3° nur wenig besser. |
| E2: Störankünfte (ρ = 0,9, 20 % gleichverteilt, n = 20) | Abweichung vom Kreismittel 8,7°, vom Median 9,3°, vom **Fréchet-Mittel 13,9°**, vom kleinsten Bogen 53,0° (n = 100: 3,9° / 4,1° / 6,4° / 58,5°). |
| E2: mehrere lokale Minima | Schon bei **einem** Gipfel hat die Fréchet-Funktion in 72 % der Datensätze (ρ = 0,45, n = 20) mehr als ein lokales Minimum (n = 100: 67 %; ρ = 0,7, n = 20: 44 %); das zweite liegt in der Nähe des Gegenpunkts. |
| E2: schiefe Daten (70 % bei 0°, 30 % bei 100°) | Kreismittel und Fréchet-Mittel liegen 5,5° auseinander (n = 100), in 100 % der Datensätze um mehr als 2°; eine gemeinsame wahre Mitte gibt es dort nicht. |
| E2: zwei Häufungen im Abstand 120° (n = 40) | **Kreismittel, Fréchet-Mittel und kleinster Bogen liegen in 100 % der Datensätze im Leeren** (weiter als 20° von beiden Häufungen). Der Median sitzt bei zufälliger 50/50-Aufteilung in 71 % auf einer Häufung, bei fester 60/40-Aufteilung in 99 % auf der größeren; bei fester 50/50-Aufteilung nur in 28 %, weil die Zielfunktion des Medians bei geradem n dort ein flaches Stück hat. |
| E2: zwei Gegenhäufungen (Abstand 180°) | Zufällige 50/50-Aufteilung (R = 0,13): das Kreismittel liegt zu 32 % in A, 33 % in B, 35 % im Leeren; **das Fréchet-Mittel liegt nie in einer Häufung** (100 % im Leeren, bei ±90°). Bei fester 60/40-Aufteilung (R = 0,20) liegt das Kreismittel in 96 % in A, der Median in 98 % in A, das Fréchet-Mittel weiter 100 % im Leeren. |
| E2: ein einziger Lkw wechselt die Häufung (n = 41, 20 gegen 21) | Verschiebung im Mittel bei 120° Abstand: Kreismittel 4,9°, Fréchet 2,9°, **Median 71,0°** (in 88 % der Paare über 60°), kleinster Bogen 0,2°. Bei 180° Abstand (R = 0,04): Kreismittel **97,8°** (79 % der Paare über 60°), Fréchet-Mittel 20,8° (9 %), **Median 143,5°** (100 %), kleinster Bogen 6,6° (4 %). |
| E2: Gitter statt Aufzählung | Ein Gitter mit 720 Punkten zählt die lokalen Minima der Fréchet-Funktion in 6,0 % (±1,5 Prozentpunkte zwischen den Seeds) von 500 Datensätzen je Seed falsch (n = 3 bis 12 Punkte, Datenmix siehe `mir_experiments.grid_miscount`). Die App zählt deshalb exakt. |
| E3: Torzeit, 60 Ankünfte, Mitte irgendwo am Tag | Streuung 0,5 / 1 / 2 / 3 h: mittlerer Fehler des arithmetischen Mittels 26 / 52 / 99 / 147 Minuten, des Kreismittels 3 / 6 / 13 / 19 Minuten. Bei Streuung 1 h liegt das arithmetische Mittel in 14 % der Nächte mehr als eine Stunde daneben (bei 3 h: 43 %), das Kreismittel in 0 %. Fenster ± 1 h: 60 % der Ankünfte (arithmetisch) gegen 69 % (Kreismittel, zirkulärer Median 69 %, kleinster Bogen 66 %); ein Fenster um die wahre Mitte fängt 68 %. Zirkulärer Median 8 Minuten und kleinster Bogen 16 Minuten Fehler. |
| E3: Nachtschicht (Mitte 23:30 Uhr, Streuung 1 h) | Das arithmetische Mittel (Tag ab 0 Uhr) liegt im Median bei **16,15 Uhr**, im Mittel 444 Minuten neben der Mitte, Fensterabdeckung **0 %**; das Kreismittel bei 23,50 Uhr, 6,3 Minuten daneben, Abdeckung 69 %; Median 7,9 Minuten, kleinster Bogen 15,3 Minuten. Mit Tagesbeginn 12 Uhr ist das arithmetische Mittel gleich gut (6,3 Minuten): Der Fehler hängt am Schnittpunkt, nicht am Mittel. |
| E3: Schnittpunkt des Tages (Mitte irgendwo am Tag, Streuung 1 h) | Die arithmetischen Mittel derselben Nacht bei Tagesbeginn 0 / 6 / 12 / 18 Uhr liegen in **58 %** der Nächte mehr als eine Stunde auseinander (mittlere Spannweite 191 Minuten); das Kreismittel kennt keinen Schnittpunkt. |
| E3: zwei Gipfel (6 und 18 Uhr, R = 0,107) | Fensterabdeckung ± 1 h: arithmetisch 0 %, Kreismittel 24 %, Fréchet 0 %, zirkulärer Median 30 %, kleinster Bogen 0 %; das beste Einzelfenster fängt 41 %. Arithmetisch, Fréchet und kleinster Bogen sitzen in 100 % der Nächte im Leeren; das Kreismittel folgt der zufälligen Mehrheit (in 73 % nahe einem Gipfel), der Median in 93 %. |
| E4: zwei Lieferanten (Median 10 und 30 Tage, s = 0,15, je 50 %) | Mischung und Quantilmittel haben das Mittel 20,23 Tage. Streuung **10,67 gegen 3,05 Tage** (Faktor 3,5); Masse im Bereich von 15 bis 25 Tagen: 5,8 % gegen 90,4 %. Die Formeln stimmen mit einer Ziehung aus der Definition überein (10,67 und 3,05). |
| E4: Bestellpunkt (Ziel 95 %, Wahrheit = Mischung) | Plan mit dem 95-%-Quantil des Quantilmittels (25,6 Tage): tatsächlicher Servicegrad **57,2 %** statt 95 %; Mittel + z·Std des Quantilmittels 56,2 %; „typische“ Lieferzeit (Mittel 20,2 Tage) 50,2 %; richtig ist das Mischungsquantil (36,4 Tage). Bei s = 0,3 (Median 10 und 30 Tage): 80,8 % (Quantilmittel-Plan 32,8 Tage, richtig 44,1 Tage). |
| E4: wann fallen beide zusammen? (Servicegrad bei Plan auf dem Quantilmittel, Medianverhältnis 1 / 1,2 / 1,5 / 2 / 3) | s = 0,15: 95,0 / 92,3 / 83,3 / 69,6 / 57,2 %; s = 0,3: 95,0 / 94,4 / 92,1 / 87,6 / 80,8 %; s = 0,5: 95,0 / 94,8 / 94,1 / 92,5 / 89,8 %. Streuung der Mischung durch Streuung des Quantilmittels (s = 0,15): 1,00 / 1,17 / 1,67 / 2,45 / 3,50 (s = 0,5: 1,00 / 1,02 / 1,09 / 1,23 / 1,46). Die Mitten fallen zusammen, wenn die Lieferanten gleich sind oder ihre eigene Streuung die Lücke überdeckt. |
| E5: Prognosen zu zweit, **überzuversichtlich** gemeldet | CRPS-Differenz Mischung minus Quantilmittel (negativ: Mischung besser) bei sₑ = 1 / 2 / 4: +0,0039 ± 0,0029 / +0,0080 ± 0,0094 / −0,0167 ± 0,0228 (± Streuung zwischen den Seeds): **der CRPS unterscheidet beide praktisch nicht.** Die Abdeckung des 80-%-Intervalls tut es: Mischung 77,4 / 72,2 / 64,2 %, Quantilmittel 70,3 / 53,9 / 33,3 % (eine einzelne Prognose bei sₑ = 2: 44,0 %). Das Intervall der Mischung ist bei sₑ = 2 mit 4,11 gegen 2,56 breiter. |
| E5: **ehrlich** gemeldet | Jetzt ist das Quantilmittel besser: CRPS-Differenz +0,0135 ± 0,0018 / +0,0509 ± 0,0036 / +0,1261 ± 0,0052 bei sₑ = 1 / 2 / 4; die Mischung ist in 32 / 26 / 22 % der Fälle besser. Beide decken zu viel ab (80-%-Intervall: Mischung 89,3 / 93,7 / 95,0 %, Quantilmittel 85,7 / 90,5 / 91,8 %). Bei sₑ = 0,25 sind beide gleich (Differenz unter 0,0003). Ungleich gemeldete Streuungen (0,5 und 2, sₑ = 0,5): +0,0128 ± 0,0024, die Mischung ist in 49,2 % der Fälle besser. **Kein allgemeiner Sieger.** |
| E6: Skala | Dasselbe Quantilmittel einmal in x, einmal in log x gemittelt (Lagefehler gleich dem Doppelten der Streuung der Nachfrage, überzuversichtlich): CRPS 0,612 gegen 0,525, Differenz 0,0866 ± 0,0031 (bei Verhältnis 0,5 / 1: 0,0007 / 0,0055). Die Mischung (0,557) ist von der Skala unabhängig: Sie schlägt das Quantilmittel in x (−0,0552), unterliegt dem in log x (+0,0314). Ehrlich gemeldet: 0,0902 / −0,0342 / +0,0560. |

## Hypothesen des Einzelplans (Ja/Nein nach eigener Messung)

| | Hypothese | Stand |
|---|---|---|
| H1 | Uhrzeiten 23, 23,5, 0,5, 1: arithmetisch 12,00 h, Kreismittel 0 h, R = 0,979 | **Ja** (Handrechnung und Orakel gegen `scipy.stats.circmean`/`circvar`) |
| H2 | Unter Gleichverteilung R ≈ √(π/4n), R₉₅ ≈ √(3/n) ab n ≈ 8, Rayleigh-Größe 5 % | **Ja** ab n = 8 (R₉₅ 0,603 gegen 0,612; Größe 4,3 bis 5,2 % ab n = 5); **nein bei n = 3** (Größe 0,1 %) |
| H3 | Die Mitte wird stetig unzuverlässig | **Ja** (E1). Die Zahlen des Plans (Von Mises, κ) sind nicht übernommen: anderer Erzeuger, eigene Messung |
| H4 | Extrinsisch und intrinsisch weichen bei breitem Gipfel ab; das intrinsische Mittel ist **nicht** genauer | **Ja**: 7,4° Abstand bei n = 20; gleich genau (20,9° gegen 20,9°), bei Störankünften schlechter (13,9° gegen 8,7°) |
| H5 | Bei zwei Häufungen sitzen beide Mitten im Leeren, der Median auf einer Häufung | **Ja** bei 120° (100 % im Leeren); der Median nur bei ungleicher Aufteilung zuverlässig auf einer Häufung (99 %), bei zufälliger 50/50-Aufteilung in 71 % |
| H6 | Ein einziger Lkw bei Gegenhäufungen verschiebt Kreismittel, Median und Fréchet-Mittel stark | **Ja**: 97,8° / 143,5° / 20,8° |
| H7 | Torzeit: arithmetische Mitte einer Nachtschicht bei 16 Uhr, Fensterabdeckung 0 gegen 69 % | **Ja** (16,15 Uhr; 0 % gegen 69 %) |
| H8 | Zwei Gipfel: R ≈ 0,1, jede Mitte außer dem Median sitzt im Leeren | **Teilweise**: arithmetisch, Fréchet und kleinster Bogen zu 100 %; das **Kreismittel nicht** (nahe einem Gipfel in 73 %, folgt der Zufallsmehrheit); der Median in 93 % nahe einem Gipfel |
| H9 | Zwei Lieferanten: gleiches Mittel, Streuung 10,67 gegen 3,05, 5,8 % gegen 90,4 % | **Ja** (geschlossene Formel, Ziehung bestätigt) |
| H10 | Bestellpunkt mit dem Quantilmittel: Servicegrad 57,2 % statt 95 % (s = 0,15), 80,8 % (s = 0,3) | **Ja** (rechnerisch aus den Verteilungen) |
| H11 | Prognosekombination: CRPS kaum verschieden, die Mischung deckt bei überzuversichtlichen Prognosen besser, ist bei ehrlichen schlechter | **Ja**, in dieser Modellwelt (zwei Prognosen, gleiche Gewichte, normalverteilt) |
| H12 | Das Quantilmittel hängt an der Skala, die Mischung nicht | **Ja** (CRPS-Differenz 0,0866 ± 0,0031 bei Verhältnis 2) |
| H13 | Ungleiche Gewichte, mehr als zwei Prognosen, echte Daten, zirkulärer Median als Robustheit über n hinaus | **Offen**, nicht gemessen (ungleich gemeldete Streuungen sind ein einzelner Fall) |

## Befunde und Korrekturen gegenüber dem Plan

* **R ≈ 0 ist keine scharfe Schwelle.** Die Unzuverlässigkeit wächst stetig mit sinkendem R·√n; als Schwelle taugt √(−ln 0,05 / n) ≈ √(3/n) (n = 30: 0,316). Bei zwei Häufungen ist die Mitte **definiert, aber sinnlos** (Fréchet bei ±90°), nicht undefiniert.
* **Von Mises wird nicht verwendet**, sondern eine wrapped normal (Abweichung vom Plan, der beides erlaubte); die resultierende Länge ρ = exp(−σ²/2) ist exakt, was das Raster in ρ statt in κ erklärt. Die Zahlen des Plans (κ) sind nicht übernommen und nicht vergleichbar.
* **Das Fréchet-Mittel wird exakt aufgezählt, nicht auf einem Gitter gesucht.** Der Plan fand ein Gitter von 720 Punkten in 87 von 500 Datensätzen falsch; die Demo misst mit eigenem Datenmix 6,0 %. Die beiden Zahlen gelten für verschiedene Datenmixe und sind nicht gleichzusetzen.
* **Die Mitte „Anteil Nächte vor und nach Mitternacht“** der Rohdatei der Vormessung war unbrauchbar und ist nicht verwendet.
* **Schon bei einem Gipfel gibt es oft mehr als ein lokales Minimum** (72 % bei ρ = 0,45, n = 20); das war im Plan ein Einzelfund (Uhrzeiten 23, 23,5, 0,5, 1) und ist hier gemessen.
* **Das intrinsische Mittel ist nicht genauer als das extrinsische** (H4); der Plan hatte keine Zahl für „extrinsisch ≠ intrinsisch“. Der Zusatz „Mehrdeutigkeit bei antipodalen Häufungen“ trifft nur den ausgeglichenen Fall: bei fester 60/40-Aufteilung liegt das Kreismittel in 96 % auf der größeren Häufung.
* **PLAN E7 bestätigt** (Mittel 20,23 Tage, Streuung 10,67 gegen 3,05, 5,8 % gegen 90,4 %). Neu ist die Folge für den Bestellpunkt und das Raster. Die **Mischung** ist die richtige Beschreibung zufälliger Zuordnung; das Quantilmittel hat seinen Ort bei der **Kombination von Prognosen**, nicht bei Lieferanten.
* **Die Prognosekombination hat keinen Sieger.** Der CRPS unterscheidet Mischung und Quantilmittel bei überzuversichtlich gemeldeten Prognosen praktisch nicht (die Differenz liegt innerhalb der Streuung zwischen den Seeds), bei ehrlich gemeldeten ist das Quantilmittel besser. Die Demo gibt keine Empfehlung „Quantile mitteln“ aus.
* **H8 gilt nur teilweise** (siehe Hypothesen).
* **Der Kreis-Hook allein ist ein Zweizeiler.** Tragfähig sind die gemessenen Schwellen, die Torzeit-Folgen (Fehler 52 gegen 6 Minuten, Nachtschicht 444 Minuten, Schnittpunkt-Abhängigkeit 58 %) und der Bestellpunkt (57,2 % statt 95 %); der Rest ist geometrisch naheliegend (zwei Gegenpunkte haben keine Mitte).

## Ehrliche Grenzen

* **Erzeugte Daten.** Alle Befunde gelten für die wrapped normal, die Lognormal-Lieferanten und die Normalprognosen dieser Demo. Echte Torzeiten haben Spitzen, Pausen und Wochentage. Es gibt keinen Datensatz aus einem echten Tor.
* **Vier Seeds je Zelle.** Die Streuung ist die zwischen vier festen Zufalls-Seeds, kein Konfidenzintervall. Einzelne Zellen (zum Beispiel E5 bei sₑ = 4, Differenz −0,0167 ± 0,0228) sind von Null nicht zu trennen.
* **Zirkulärer Median bei geradem n.** Die Zielfunktion ist zwischen zwei Datenpunkten flach (Gleichstand in 100 % der Datensätze bei n = 40); die Demo nimmt den Punkt mit dem kleineren Winkel. Die Zeile „fest 50/50“ in E2 sagt über den Median deshalb wenig.
* **Gleichstände des Fréchet-Mittels** (bei zwei Gegenpunkten zwei gleich gute Minima bei ±90°) werden zugunsten des kleineren Winkels aufgelöst.
* **Der einfache Rayleigh-Test** (P(nR² ≥ z) ≈ e⁻ᶻ) ist eine Näherung; ihre Größe ist gemessen (E1). Die Formelquelle wurde nicht eingesehen.
* **CRPS aus 100 Quantilstufen.** Der Rasterfehler (gegen die geschlossene Formel der Normalverteilung, 2000 Beobachtungen): im Mittel unter 0,0001, höchstens 0,003; er ist für beide Verfahren gleich.
* **Zwei Prognosen, gleiche Gewichte, bekannte Wahrheit.** Mehr Prognosen, ungleiche Gewichte und echte Daten sind nicht gemessen (H13).
* **Literatur nicht eingesehen.** Fréchet-Mittel, Vincent-Mittel (Quantilmittel), lineare Mischung und Rayleigh-Test sind in der Literatur eingeführt; die Quellen wurden für diese Demo nicht eingesehen. Es gibt keinen Neuwert gegenüber der Literatur, den die Demo beanspruchen würde: Der Beitrag sind die gemessenen Folgen für einen Bestellpunkt und einen Torzeit-Plan mit bekannter Wahrheit.
* **Mausklicks** auf die Diagramme sind nicht Teil der Demo; die Bedienung der Regler und Knöpfe ist mit AppTest geprüft.

## Verifikation

* **Handrechnung** (`tests/conftest.py`, `tests/test_circle.py`, `tests/test_dist.py`): Uhrzeiten 23, 23,5, 0,5, 1 (arithmetisch 12,00 h, Kreismittel 0 h, R = 0,979, zwei lokale Minima, Median-Gleichstand, kleinster Bogen), zwei Gegenpunkte (zwei Minima bei ±90°, F = π²/4), drei und vier gleichmäßige Punkte, N(0, 1) und N(10, 1), diskrete Mini-Verteilungen {1, 2, 3} und {10, 20, 30} (Quantilmittel 5,5 / 11 / 16,5), Randfälle (n = 1, n = 2, doppelte Punkte, Punkte am Schnitt).
* **Orakel** (`tests/test_oracle_kreis.py`, `tests/test_oracle_verteilungen.py`): Resultierende Länge und Richtung gegen `scipy.stats.circvar`/`circmean` (200 Datensätze); Fréchet-Mittel gegen die Aufzählung aller Entfaltungen (60 Datensätze, n ≤ 6) und gegen ein Feingitter; zirkulärer Median gegen ein Feingitter; kleinster Bogen gegen die Aufzählung der Startpunkte; Φ gegen `math.erfc` (721 Stellen bis 9 Standardabweichungen, Fehler unter 1e-14), Φ⁻¹ gegen `scipy.stats.norm.ppf`, Mischungsquantil gegen `scipy.optimize.brentq`, Mischungs- und Lognormal-Momente gegen eine Ziehung aus der Definition, CRPS aus dem Quantilraster gegen die geschlossene Formel.
* **Eigenschaften** (`tests/test_eigenschaften_kreis.py`): Drehung dreht jede Mitte mit, Spiegelung spiegelt sie, die Reihenfolge der Punkte ist gleichgültig, Verschiebung und Skalierung der Verteilungen wirken auf beide Mitten gleich, Quantilfunktionen sind monoton, die Mischung vertauscht mit monotonen Umrechnungen, das Quantilmittel in x ist größer als in log x.
* **Zweig-Tests** (`tests/test_zweige.py`, Nullspalten-Regel): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab (arithmetisch gegen Kreismittel, extrinsisch gegen intrinsisch, Median gegen beide bei 60/40, kleinster Bogen gegen Median bei einem Ausreißer, Mischung gegen Quantilmittel, überzuversichtlich gegen ehrlich, x gegen log x) und keine Spalte der Messreihen besteht nur aus Nullen.
* **Szenario und Messreihen** (`tests/test_scenario.py`, `tests/test_copies.py`, `tests/test_experiments.py`, `tests/test_results.py`): der blockweise Zufallsstrom gleich der wörtlich kopierten Klasse `SplitMix64`; resultierende Länge der wrapped normal gleich ρ; die gespeicherten Zellen werden von den Live-Funktionen reproduziert; die Prüfsumme der Erzeuger stimmt.
* **Verdikte, Presets, Oberfläche** (`tests/test_verdicts.py`, `tests/test_presets.py`, `tests/test_layout.py`, `tests/test_app.py`, `tests/test_visualization.py`, `tests/test_pdf.py`): Schwellen und jede Zahl im Text an handgesetzten Werten; jedes Preset einzeln erkennbar und mit Aussagen seines Hilfetextes geprüft; AppTest für jeden Button, jeden Regler am Minimum und Maximum, alle Regler zugleich am Rand, jede Auswahloption, Permalink; Worst-Case am Slider-Maximum im eigenen Prozess mit Zeitgrenze (`tests/test_worst_case.py`).
* **Zahlen der README:** `tests/test_claims.py`.
* **Mutationsprüfung** (`tools/mutate_demo.py` aus dem Website-Repo, `--exclude` für `mir_rng`, `mir_layout`, `mir_pdf_export`, `mir_visualization`, `mir_constants`, `mir_presets`, `mir_scenario`, ohne `test_results_current.py`, `test_layout.py`, `test_pdf.py`; Stand `tests/mutation_state.json`): Das Werkzeug wählt die Module, die die Orakel-Tests erreichen, das sind `mir_circle.py` und `mir_dist.py`. Vollauf über alle **133 Stellen: 103 gefangen, 30 äquivalent, 0 offen.** Im ersten Lauf überlebten 33 Mutanten; drei waren echte Lücken (eine nicht getestete Hilfsfunktion `lognormal_mean`, die Verteilungsfunktion und die Dichte der Lognormalverteilung zwischen 0 und 1) und sind mit Tests in `tests/test_dist.py` geschlossen. Die 30 äquivalenten sind festgehalten: Vorzeichen von ±π beim Gegenpunkt (modulo 2π gleich), ±12 in `hours_distance` (modulo 24 gleich), Toleranzgrenzen und Gleichheitsfälle (`<` gegen `<=` bei exaktem Gleichstand oder Null), die Blockgröße der Zwischenfelder (ändert nur den Speicher), Abbruchbedingung und Schleifenlänge der Newton- und Bisektionsschritte (ändert nur Rechenzeit oder die 17. Stelle) und die Zahl der Reihenglieder in `erfc` (44 statt 45 Glieder, Abweichung unter der Prüfgrenze 1e-14). **Nicht mutiert** sind `mir_evaluation.py` und `mir_experiments.py` (das Werkzeug erreicht sie über die Orakel nicht); sie sind durch `test_verdicts.py`, `test_experiments.py` und `test_claims.py` geprüft, nicht durch Mutation.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets, Zwischenstände der Messreihen (`data/_parts/`) |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Teil A (Uhr-Kreis), Teil B (Lieferanten, Prognosen), Experimente E1 bis E6, Grenzen, Formeln |
| `data/mir_results.json` | vorgerechnete Messreihen mit Prüfsumme der Quellen |
| `mir_circle.py` | Kreis-Einheiten: Winkel und Abstände, Resultierende, Rayleigh, Fréchet exakt, Median, kleinster Bogen, fünf Mitten |
| `mir_constants.py` | Regler-Grenzen, Farben, Presets und ihre Hilfetexte |
| `mir_dist.py` | Verteilungs-Einheiten: Φ und Φ⁻¹ (eigene erfc), Mischung, Quantilmittel, Lognormal-Formeln, CRPS |
| `mir_evaluation.py` | Einstellungen, Analyse der Nacht, Verdikte (reine Funktionen) |
| `mir_experiments.py` | Messreihen E1 bis E6, Live-Messung, Prüfsumme der Erzeuger |
| `mir_layout.py` | Fester Kopf (Stylesheet) |
| `mir_pdf_export.py` | PDF-Export (fpdf2, erst nach Klick) |
| `mir_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `mir_results.py` | Zugriff auf die vorgerechneten Messreihen |
| `mir_rng.py` | SplitMix64, wörtliche Kopie aus `reduktionskette-demo` |
| `mir_scenario.py` | Blockweiser Zufallsstrom, Erzeuger für Nächte, Häufungen, Prognosefälle |
| `mir_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/make_results.py` | erzeugt `data/mir_results.json` |
| `requirements.txt` / `requirements-dev.txt` | Laufzeit (kein scipy) / plus pytest und scipy für die Gegenproben |
| `tests/` | Handrechnung, Orakel, Eigenschaften, Zweige, Randfälle, Messreihen, Claims, Verdikte, Presets, Layout, AppTest, PDF, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **Rangräume** (Borda als Mittel der Ränge, Kemeny als Median): `rangaggregation-demo`. **Graphen** (Closeness als Graph-Median, 1-Center): [centrality-demo](https://github.com/sebastian-hanisch/centrality-demo), [p-center-demo](https://github.com/sebastian-hanisch/p-center-demo). Hier nur als Verweis.
* **Wasserstein-Theorie.** Das Quantilmittel wird als Definition benutzt (Mittel der Quantilfunktionen), nicht als Baryzentrum-Theorie entwickelt.
* **Tori und höhere Sphären**, **Von-Mises-Verteilung**, **echte Torzeit-Daten**, **mehr als zwei Prognosen und ungleiche Gewichte** (H13).
* **Eine Empfehlung „Quantile mitteln“.** Die Messreihe zeigt, dass es keinen allgemeinen Sieger gibt.
* **Zwei Häufungen auflösen.** Dafür ist Clustern da ([kmeans-demo](https://github.com/sebastian-hanisch/kmeans-demo), k = 2); keine Mitte ersetzt die Aussage „zwei Häufungen“.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python: Die CI läuft mit 3.12, lokal laufen die Tests mit 3.14; der Code nutzt nur Sprach- und Bibliotheksmerkmale, die in 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Messreihen neu erzeugen (nach einer Änderung an den Erzeugern; `tests/test_results.py` meldet sie):

```bash
python tools/make_results.py --part m1   # m1 bis m6, die Teile können parallel laufen (m5 und m6 brauchen einige Minuten)
python tools/make_results.py --merge
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
