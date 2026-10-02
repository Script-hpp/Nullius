# v5 Schritt A: Dossier-Treue von Rolle 4 (Vorab-Festlegung, final)

Status: **final** (vom Nutzer freigegeben am 2026-10-02). Ziele und Soll-Liste werden nach dem ersten Messlauf nicht mehr geändert; gestrichene Soll-Punkte (nicht belegbar) werden vor dem ersten Lauf vermerkt.

Grundlage: `failure_analysis_v4.md`. Befund: 84 % der Setup-Angaben, die im Dossier stehen, kommen im Code an; die Lücke liegt im Dossier (Angabe fehlt, vor allem Merkmals-Normalisierung in 14 von 21 Läufen, oder sie ist richtig zitiert, gilt aber nicht für den Claim). Schritt A misst deshalb **nur das Dossier**, ohne Training: billig, schnell, und kein Urteil des Benchmarks wird dabei angesehen.

## Änderungen (Vorschläge V1 bis V3 der Analyse, nur Rolle 4 und Plan-Prompt)

- **V1 Pflichtliste je Task-Familie.** Graph: Merkmals-Normalisierung, Adjazenz-Normalisierung, Lernrate, Weight Decay (und worauf), Dropout, Hidden/Köpfe, Epochen, Early-Stopping-Regel. Zeitreihe: Eingabelänge, Lernrate, Lernraten-Plan, Batchgröße, Epochen, Geduld. Fehlt ein Punkt nach der ersten Extraktion, liest Rolle 4 gezielt die Datei, die die Daten lädt, und das Startskript des Experiments. Weiter gilt: jede Angabe braucht ein wörtliches Zitat, sonst wird sie verworfen. "Nicht gefunden" ist ein gültiges Ergebnis und steht im Dossier.
- **V2 Rangfolge der Quellen.** Startskript des Experiments vor Konfigurationsdatei vor `argparse`-Default vor README-Fließtext; README-Sätze mit "optional", "for example", "can" als "optional advice" markiert.
- **V3 Feld "gilt für".** Jede Angabe trägt Modell, Datensatz und Einstellung (transduktiv/induktiv), mit Zitat. Angaben für ein anderes Modell oder einen anderen Datensatz gehen nicht in den Plan (bleiben im Dossier, markiert).

Nicht erlaubt: Methodenwissen oder Werte aus unseren Referenzlösungen in einem Prompt. Die Soll-Liste unten dient nur der Auswertung.

## Messung

- Nur Rolle 4 (Extraktion) je Einheit, Arm B, gespeicherte Tavily-Ergebnisse und Repo-Dateien (`tavily_mode="fixture"`), LLM-Cache aus, **3 Wiederholungen** je Einheit, einmal mit dem Stand vor der Änderung (Basis) und einmal danach. Kein Training, keine Sandbox-Läufe des Agenten-Codes, kein Tavily-Aufruf.
- Auswertung je Soll-Punkt: **korrekt** (Angabe im Dossier, Wert stimmt, gilt für den Claim), **fehlt**, **falsch** (anderer Wert oder Angabe für anderes Modell/Datensatz als gültig geführt). Skript mit festen Mustern je Soll-Punkt; Zweifelsfälle sieht der Reviewer von Hand an und listet sie im Ergebnis.
- Kosten: erwartet unter 2 $ brutto (8 Einheiten × 3 × 2 Extraktionen, Super).

## Soll-Liste (aus Paper und Original-Repo, wie in `tests/reference/` belegt; nur Auswertung)

```yaml
kipf_gcn_2017:
  feature_norm: "Merkmale zeilennormalisiert"
  adj_norm: "D^-1/2 (A+I) D^-1/2"
  lr: 0.01
  weight_decay: "5e-4, nur erste Schicht"
  dropout: 0.5
  hidden: 16
  epochs: 200
  early_stopping: "Validierungsloss, Fenster 10"
wu_sgc_2019:
  feature_norm: "Merkmale zeilennormalisiert (nicht standardisiert)"
  adj_norm: "D^-1/2 (A+I) D^-1/2, K = 2"
  lr: 0.2
  epochs: 100
  weight_decay: "auf Validierung gesucht (hyperopt), kein fester Wert"
velickovic_gat_2018:
  feature_norm: "Merkmale zeilennormalisiert"
  lr: 0.005
  weight_decay: "L2 5e-4"
  dropout: "0.6 auf Eingaben und Attention"
  hidden: "8 Köpfe x 8 Merkmale, Ausgabe 1 Kopf"
  early_stopping: "Validierungsloss und -accuracy, Geduld 100"
velickovic_dgi_2019:
  feature_norm: "Merkmale zeilennormalisiert"
  encoder: "eine GCN-Schicht, 512 Einheiten, PReLU (transduktiv)"
  lr: 0.001
  early_stopping: "Trainingsloss, Geduld 20"
  classifier: "logistische Regression auf den Einbettungen"
  applies_to: "transduktive Einstellung (nicht die induktiven Angaben)"
shchur_pitfalls_2018:   # GCN
  feature_norm: "Merkmale NICHT normalisiert"
  lr: 0.01
  weight_decay: "L2 1e-3"
  dropout: 0.8
  hidden: 64
  early_stopping: "Geduld 50, Gesamt-Validierungsloss"
shchur_gat_2018:
  feature_norm: "Merkmale NICHT normalisiert"
  lr: 0.01
  weight_decay: "L2 0.01"
  dropout: "0.6 Eingaben, 0.3 Attention"
  hidden: "8 Köpfe x 8 Merkmale (Hidden 64 gesamt, nicht 64 je Kopf)"
  early_stopping: "Geduld 50, Gesamt-Validierungsloss"
zeng_dlinear_2023:
  input_length: 336
  lr: 0.005
  lr_schedule: "type1, pro Epoche halbiert"
  batch_size: 32
  epochs: 10
  patience: 3
zeng_nlinear_2023:
  method: "letzten Wert des Fensters abziehen und wieder addieren"
  input_length: 336
  lr: 0.005
  lr_schedule: "type1, pro Epoche halbiert"
  batch_size: 32
  epochs: 10
  patience: 3
```

48 Soll-Punkte. Jeder Punkt wird vor dem ersten Lauf vom Reviewer gegen Paper/Repo-Fixture geprüft; ein Punkt, der in den gespeicherten Quellen nicht wörtlich belegbar ist, wird gestrichen (mit Vermerk), weil Rolle 4 ihn dann nicht liefern kann.

## Ziele (vorab)

1. **Normalisierung/Methode:** der Punkt `feature_norm` (bzw. `method` bei NLinear) ist in mindestens 2 von 3 Wiederholungen korrekt bei **mindestens 5 der 7** Einheiten, die ihn haben (Basis laut Analyse: 7 von 21 Läufen).
2. **Gesamt:** Anteil "korrekt" über alle Soll-Punkte und Wiederholungen **mindestens 80 %**, und höher als die Basis.
3. **Keine irreführende Angabe als gültig:** die vier bekannten Fälle (SGC-Standardisierung als Pflicht, DLinear `argparse`-Default für Eingabelänge/Lernrate, DGI induktive Angaben, Shchur-GAT 64 je Kopf) stehen in **keiner** Wiederholung als gültig für den Claim im Plan-Text.

Schritt A gilt als bestanden, wenn alle drei Ziele erreicht sind. Verfehlt: Ursache berichten, höchstens **zwei** Nachbesserungsrunden an Rolle 4, jede mit eigener Messung im Ergebnis; danach weiter oder stoppen, die Ziele bleiben.

## Ehrlichkeit

- Die acht Einheiten sind ab jetzt **Entwicklungs-Einheiten**: an ihnen wird Rolle 4 verbessert. Das README berichtet v5 für sie getrennt von den **neuen Einheiten** RLinear und SparseTSF, an denen nichts entwickelt wurde (keine Soll-Liste, keine Messung in Schritt A bis C).
- Die Ergebnisse von Schritt A sagen nichts über die Benchmark-Kennzahl. Die misst erst v5.
- Ergebnisdatei: `results/v5_step_a.md`, Rohdaten der Extraktionen unter `runs/v5_step_a/`.
