# Bauplan Modus „Nachlauf" v1

Stand 2026-10-02, Entwurf des Reviewers. Regeln und Prüfung: `benchmark/experiments/nachlauf_v1.md`. Vorversuch: `benchmark/experiments/results/nachlauf_prestudy.md`.

## Grundsätze

- Neuer Code liegt in **einem neuen Modul** `src/repro_agent/nachlauf.py`. Der bestehende Pfad (`Orchestrator.run`, Nachbau) bleibt bitgleich, damit Benchmark v4 nachrechenbar bleibt. Der Modus wird über einen Schalter gestartet (`mode="nachlauf"`), Default bleibt Nachbau.
- Worker bauen mit Fakes. Echte Läufe (Sandbox, Token Factory) macht nur der Reviewer.
- Der Agenten-Code sieht nie Testlabels (Trennung in Trainings- und Vorhersage-Branch, siehe Vorab-Festlegung).
- Nemotron-Rollen wie bisher: `code` (Super) wählt den Startbefehl und schreibt den Adapter, `fast` liest die Eigenausgabe aus stdout. Kein Ultra nötig.

## Was der Agent liefert

1. **Startbefehl** für den Claim, mit Quelle (Datei, Zeile, wörtliches Zitat aus dem Repo). Eingabe: Dateiliste und die von Rolle 4 gewählten Repo-Dateien (vorhanden: `repo_source.read_repo`).
2. **Seed-Weg**: Schalter oder Zeile im Startskript.
3. **Adapter `predict.py`**: Modellklasse des Repos importieren, Checkpoint laden, `test_inputs` aus `/data/etth1_public.npz` (letzte `seq_len` Schritte), `predictions.csv` schreiben. Kein Zugriff auf Rohdaten (nicht im Branch).
4. Bei Fehlern: Reparatur von Befehl oder Adapter aus dem Ende von stderr.

## Meilensteine

| # | Inhalt | Wer | Dauer (geschätzt) | Abnahme |
|---|---|---|---|---|
| M1 | **Harness und Ablauf ohne LLM.** `harness/independence.py` (P2), Befehlsprüfung P1 (`nachlauf.check_command`), Selbstprüfung P3, zwei Snapshots (Training mit Daten, Vorhersage ohne), Checkpoint-Übertragung, Seed-Kopie mit Diff-Prüfung, Dossier-Felder (`mode`, `nachlauf: {command, command_source, command_diff, env_diff, own_metric, self_check, independence}`), Events. Tests mit Fake-Sandbox. | Worker (Sonnet) | 1,5 Tage | Suite grün; Reviewer fährt den Ablauf mit den **Hand-Adaptern** des Vorversuchs in der echten Sandbox: gleiche Zahlen wie im Vorversuch, P2 und P3 bestanden |
| M2 | **Negativkontrollen.** Drei Adapter bzw. Befehle aus der Vorab-Festlegung als Fixtures, Tests mit Fakes und ein echter Sandbox-Lauf. | Worker, Lauf Reviewer | 0,5 Tage | alle drei abgelehnt, sonst zurück zu M1 |
| M3 | **Agenten-Teil.** Prompts für Startbefehl, Seed-Weg, Adapter, Reparatur; Einbindung der Reparaturschleife und des Kostenzählers. Entwicklung nur an RLinear und DLinear. | Worker; echte Läufe Reviewer | 1,5 Tage | je ein gültiger Lauf für beide Entwicklungs-Einheiten auf Token Factory |
| M4 | **Prüfung** nach Vorab-Festlegung: `src/` einfrieren, vier neue Einheiten, eine Generierung. | Reviewer | 0,5 Tage (Laufzeit) | Ergebnisdatei `results/nachlauf_v1.md`, Erfolgsregel angewandt |
| M5 | **Produkt.** API-Schalter (`POST /api/reproduce` mit `mode`), Zeugnis mit zwei Spalten „Authors' code" und „Rebuilt from the paper", Karte „description gap" (nur wenn beide Urteile vorliegen), Abweichungen des Startbefehls und der Umgebung als Liste, Film-Szene, Export. | Worker (web und Export) | 1,5 Tage | Screenshots hell/dunkel, Desktop/390 px, `npm test` grün |
| M6 | README-Abschnitt, Videoszene, Katalog-Einträge der Einheiten mit Nachlauf. | Reviewer und Worker | 0,5 Tage | |

Summe rund 6 Arbeitstage. M5 kann parallel zu M3 und M4 laufen, sobald die Dossier-Felder aus M1 feststehen.

## Reihenfolge gegenüber den Pflichtteilen der Abgabe

Abgabe ist am 30.10.2026. Deployment mit Demo-URL, Video und Devpost-Text sind Pflicht, der Nachlauf nicht. Vorschlag:

1. Jetzt: Merge der offenen Branches, Deployment starten (Worker bereitet vor, Nutzer richtet Coolify ein).
2. Parallel M1 und M2.
3. Haltepunkt nach M2: Sind die Negativkontrollen nicht alle abgelehnt, wird der Modus nicht weitergebaut.
4. M3, M4, dann M5 und M6.
5. Spätestens am 20.10. steht fest, was im Video gezeigt wird. Ist M4 bis dahin nicht bestanden, zeigt das Video den Vorversuch als „manual check by the team" (RLinear, `drop_last`), klar so beschriftet.

## Kosten

- M1, M2: 0 $ (kein LLM).
- M3: geschätzt unter 1 $ brutto (wenige Aufrufe von Super je Lauf, einige Entwicklungsläufe).
- M4: Stopp 3 $ brutto laut Vorab-Festlegung.
- Tavily: keine Aufrufe in M1 bis M4 (Repo-Adressen vorgegeben). Im Produkt nutzt der Nachlauf dieselbe Repo-Suche wie der Nachbau (Rolle 1), also keine zusätzlichen Credits je Lauf.

## Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Nemotron schreibt keinen lauffähigen Adapter für fremde Repos | Adapter ist kurz (im Vorversuch rund 30 Zeilen eigener Code); Reparaturschleife; Erfolgsregel verlangt nur 2 von 4 |
| P2 lehnt ehrliche Adapter ab (z. B. Modelle mit Batch-Statistik) | Toleranz 1e-4 und `model.eval()` im Prompt; zeigt sich in M1 an den Hand-Adaptern, vor dem Agenten-Teil |
| Repos brauchen alte Paketversionen | nur Pakete aus den Abhängigkeitsdateien des Repos, Abweichung im Zeugnis; sonst „nicht reproduzierbar (Setup)" |
| Modelle brauchen Zeitmerkmale | v1 schließt sie aus; Zeitstempel in der öffentlichen Datei wären eine spätere Erweiterung |
| Zeit | Haltepunkte nach M2 und am 20.10.; Pflichtteile der Abgabe haben Vorrang |

## Offene Entscheidungen für den Nutzer

1. Freigabe von `nachlauf_v1.md` (Regeln, vier neue Einheiten, Erfolgsregel).
2. Reihenfolge wie oben, oder Deployment erst vollständig abschließen.
3. Soll P2 nach der Abgabe-Prüfung auch in den Nachbau (schließt dort die Lücke bei ETTh1, ändert aber den Pfad des Benchmarks)? Für v1 bleibt der Nachbau unverändert.
