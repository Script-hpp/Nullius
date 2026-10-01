# Roadmap: Qualität und Produkt bis zur Abgabe

Stand 2026-10-01, vom Nutzer bestätigt. Abgabe 30.10.2026, 18:00 MEZ. Ziel: ein sehr gutes Projekt, nicht nur ein ehrlicher Prototyp. Benchmark bleibt das Messinstrument; jede Änderung am Agenten wird vorab festgelegt und gemessen.

## Ausgangslage

- Benchmark v2 (`d71a8af`) und v3 (`4bab96e`): je 2 von 8 Einheiten stimmen mit dem Referenzurteil überein, in beiden Armen. **Kein Placebo je als "reproduziert" gewertet** (0 von 96 bzw. bisher 0 in v3): Ein "reproduziert" von Nullius ist belastbar, ein "nicht reproduziert" heißt oft "Agent hat es nicht geschafft".
- Fehlermuster mit Belegen: `benchmark/experiments/failure_patterns_v3.md`, automatische Liste `results/failure_log_v3.md`.
- Hauptursachen: Streuung zwischen Generierungen ("keine Mehrheit"), knapp daneben (fehlende Umsetzungsdetails), einzelne Code-Fehler (M1 Zuordnung, M3 verbotene Bibliothek), Repo-Standardwerte statt getunter Werte (M7).

## Reihenfolge

1. **v3 abschließen** (heute): Auswertung `results/benchmark_v3.md`, Fehlerprotokoll neu erzeugen, v2 und v3 im README.
2. **Block A, Agent verbessern** (Reviewer, Tests mit Fakes), dann **Benchmark v4** über Nacht (Vorab-Festlegung vor dem Start).
3. **Block B, Benchmark erweitern** (worker-2 recherchiert parallel, nur Doku und Fixtures, keine echten Aufrufe; Reviewer macht die echten Läufe).
4. **Block C, Produkt** (Deployment, Demo-URL, UI, Video, Devpost).

## Block A: Agent verbessern

| # | Maßnahme | Zielt auf | Leak-Schutz | Kosten pro Lauf |
|---|---|---|---|---|
| A1 | **Mehrere Code-Kandidaten, Auswahl über Validierung.** 3 Kandidaten schreiben, jeder läuft Seed 0; unser Harness misst die **Validierungs**-Metrik (Agent schreibt zusätzlich Vorhersagen für die Validierungs-IDs); der beste Kandidat läuft über alle Seeds. | "keine Mehrheit", knapp daneben | Nur Validierungsdaten, die der Agent ohnehin hat; Testlabels bleiben beim Harness, keine Testzahl in Prompts oder Auswahl. | ca. 3× Code-Schreiben + 2 zusätzliche Seed-0-Läufe |
| A2 | **Abgleich mit dem Original-Code.** Wenn ein Repo existiert: Modell vergleicht unseren Code mit den Repo-Dateien und listet nur Abweichungen (Dropout-Stellen, Normalisierung, Initialisierung, Regularisierung, Abbruchkriterium) mit Zitat aus dem Repo; Abweichungen gehen als Reparaturhinweis an `code`. Modus bleibt "Nachbau" (eigener Code, Original nur als Prüfliste). | GAT, Shchur-GAT, DGI (feine Umsetzungsdetails) | Repo-Code ist öffentlich und Teil von Rolle 4; keine Testzahl. | 1 Vergleichsaufruf + ggf. 1 Reparatur |
| A3a | **M1 Vorhersage-Zuordnung:** Checkliste präzisieren (`pred[test_ids]`), deterministische Prüfung im Smoke-Lauf: Agent sagt auch Trainingsknoten vorher, Abgleich mit Trainingslabels. | 4 Fehlurteile in v3 | Nur Trainingslabels. | gering |
| A3b | **M3 verbotene Bibliothek:** Hinweis schon im Code-Prompt, nicht erst in der Reparatur. | Ausfälle Shchur-GAT | – | keine |
| A3c | **M7 Repo-Standardwerte:** Angaben aus `argparse`-Defaults als "Default, im Paper ggf. getunt" kennzeichnen; Konflikt Paper-Protokoll vs. Repo-Default im Plan sichtbar. | SGC mit Tavily | – | keine |

Messung: **Benchmark v4** wie v3 (8 Einheiten, beide Arme, 3 Generierungen, Placebos, Mehrheitsregel), Vorab-Festlegung `benchmark/experiments/benchmark_v4.md` vor dem Start. Erwartung ehrlich offen; v2, v3, v4 stehen alle im README.

## Block B: Benchmark erweitern

| # | Maßnahme | Begründung |
|---|---|---|
| B1 | **Mehr Einheiten auf vorhandenen Datensätzen** (Cora, Cora-LCC, ETTh1): Kandidaten z. B. APPNP, GraphSAGE, ChebNet (Cora), weitere Zeitreihenmodelle auf ETTh1 (Horizont 96). Zielwerte nur aus dem Paper (Rolle 2), Referenzlösung im eigenen Harness (R). Ziel 12 bis 16 Einheiten. | 8 Einheiten sind zu wenig für eine belastbare Aussage. Neue Datensätze kosten Tage (Task + Harness). |
| B2 | **Echter Negativfall:** ein Paper auf unseren Datensätzen, dessen Zahl nachweislich nicht hält (Kritik-Paper, Reproducibility Challenge, dokumentierter Bug in einem Issue). Belegt per Tavily, eigene Referenzlösung bestätigt den niedrigeren Wert. | Stärkster Beleg, dass Nullius ein echtes Problem erkennt; bisher nur Placebos (konstruiert). |

worker-2 recherchiert B1/B2 (Paperliste mit belegten Zielwerten, Repo-Links, Aufwand je Einheit), nur Doku; der Reviewer prüft und macht echte Läufe. Kontingent des Nutzers schonen: klar begrenzte Aufträge.

## Block C: Produkt

- Deployment (Coolify, `docs/deploy.md`) und Demo-URL, Pflicht, erreichbar bis 15.12.2026.
- "What others found" und "Possible explanations" (Tavily, geplant in `docs/idea_tavily_community_check.md`).
- UI-Feinschliff, Video (`docs/video_script.md`), Devpost (`docs/devpost_submission.md`), Nebius-Feedback.

## Kernaussage fürs README (Arbeitsfassung)

"Nullius never confirmed a false claim in our benchmark; it does not yet confirm every true one." Zahlen jeweils aus dem neuesten vollständigen Benchmark, ältere Läufe bleiben sichtbar.
