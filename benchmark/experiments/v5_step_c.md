# v5 Schritt C: Probelauf mit Training und Startbedingung für v5 (Vorab-Festlegung, final)

Status: **final** (Regel vom Nutzer freigegeben am 2026-10-02, vor der ersten Messung von Schritt A).

## Zweck

Vor einem vollen Benchmark v5 (rund 60 Läufe, 9 bis 10 $ brutto, eine Nacht) zeigt ein kleiner Probelauf, ob die Änderungen aus Schritt A (Rolle 4: Pflichtliste, Quellenrang, "gilt für") und Schritt B (Validierungslabel-Prüfung, Kantenrichtung) den Abstand zum Paperwert wirklich verkleinern. Ohne diesen Beleg läuft v5 nicht.

## Probelauf

- Einheiten: `zeng_dlinear_2023` (DLinear), `wu_sgc_2019` (SGC), `shchur_gat_2018` (Shchur-GAT). Auswahl nach `failure_analysis_v4.md`: bei allen dreien nennt die Analyse eine Ursache im Dossier (Repo-Standardwerte, README-Standardisierung, 64 je Kopf) oder eine jetzt geprüfte Ursache (Kantenrichtung).
- Nur Arm B (mit Tavily, gespeicherte Ergebnisse), Produktkonfiguration mit `setup_v2` und den Prüfungen aus Schritt B, LLM-Cache aus, **eine** frische Generierung je Einheit, Seeds wie in `benchmark/papers.yaml`, echter Claim und Placebo.
- Store `runs/v5_step_c.jsonl` (getrennt von jedem Benchmark-Store). Kosten erwartet rund 1 $ brutto, Stopp bei 2 $.
- Vergleich: In v4 war in Arm B **keine** der 9 Generierungen dieser drei Einheiten "reproduziert" (`results/benchmark_v4.md`).

## Startbedingung für v5

v5 wird nur gestartet, wenn **beides** gilt:

1. Schritt A hat alle drei Ziele erreicht (`v5_step_a.md`, Ergebnis `results/v5_step_a.md`).
2. Im Probelauf ist der echte Claim bei **mindestens 2 der 3 Einheiten** "reproduziert", und kein Placebo ist "reproduziert".

Ist die Bedingung nicht erfüllt: kein v5. Der Benchmark bleibt bei v4, das Ergebnis von Schritt A und C steht im README als Entwicklungsergebnis mit dieser Regel. Eine Wiederholung des Probelaufs mit demselben Code, bis er passt, ist nicht erlaubt; nach einer weiteren Code-Änderung ist **ein** neuer Probelauf erlaubt (höchstens zwei Probeläufe insgesamt), beide werden berichtet.

## Ehrlichkeit

- Eine Generierung je Einheit ist eine Stichprobe, kein Beweis. Die Bedingung verhindert einen teuren Lauf ohne Anzeichen einer Verbesserung; sie garantiert kein besseres v5.
- Die Läufe des Probelaufs zählen nicht in v5. v5 erzeugt alle Generierungen neu.
- Die acht bisherigen Einheiten sind Entwicklungs-Einheiten; RLinear und SparseTSF werden in Schritt A bis C nicht angefasst und in v5 getrennt berichtet.
- v5 bekommt eine eigene Vorab-Festlegung (`benchmark_v5.md`) vor dem Start.
