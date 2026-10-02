# Vorversuch „Nachlauf": Läuft der Code der Autoren in unserer Sandbox und unter unserem Harness?

Status: **final** (vom Nutzer am 2026-10-02 zur Ausführung freigegeben, Schritt 1 an worker-2 als Auftrag 33). Fragen, Einheiten und Entscheidungsregel werden ab jetzt nicht mehr geändert; Abweichungen werden datiert notiert.

## Worum es geht

Heute gibt es nur den Modus „Nachbau": Der Agent schreibt eigenen Code aus Paper und Repo. Ein verfehlter Zielwert kann dann zwei Ursachen haben, die das Zeugnis nicht trennt: Der Agent hat es falsch gebaut, oder das Paper beschreibt nicht genug.

Der Modus „Nachlauf" führt den **Original-Code** der Autoren aus und misst ihn mit unserem unabhängigen Harness. Zusammen ergeben beide Modi eine neue Aussage je Paper:

| Nachlauf | Nachbau | Aussage |
|---|---|---|
| trifft | trifft | Zahl stimmt, das Paper beschreibt genug |
| trifft | trifft nicht | Zahl stimmt, die Beschreibung hat eine Lücke („description gap") |
| trifft nicht | egal | Der Code der Autoren liefert die Zahl unter unserer Messung nicht |
| läuft nicht | egal | nicht reproduzierbar (Setup) |

Dieser Vorversuch baut **nichts** davon. Er klärt mit Handarbeit an zwei Repos, ob die Idee technisch trägt, bevor `src/` angefasst wird.

## Fragen (vorab festgelegt)

1. **Lauffähigkeit:** Läuft der Original-Code in der Nebius-Sandbox auf CPU, ohne Netz im Trainingslauf, in höchstens 10 Minuten je Seed?
2. **Zuordnung:** Stimmen Testfenster, Reihenfolge und Standardisierung des Repos mit dem Harness überein (2785 Fenster, z-Score mit Trainingsstatistik)?
3. **Adapter:** Wie groß ist der Adapter, der die Vorhersagen des Original-Modells nach `/output/predictions.csv` schreibt, und wie viel davon ist zwischen zwei Repos gleich?
4. **Eingriff:** Kommt der Lauf ohne Änderung an Dateien des Repos aus (nur Kommandozeile und eine neue Adapter-Datei)?
5. **Befund:** Was misst der Harness auf allen Fenstern, und was meldet das Repo selbst?

## Einheiten

| Einheit | Repo und Stand | Startbefehl | Zielwert |
|---|---|---|---|
| RLinear, ETTh1, Horizont 96 | `github.com/plumprc/RTSF`, Commit `0fec00104f754f4fbf795b9b4da5fa2459b32e76` (wie Fixture `benchmark/fixtures/repos/plumprc_RTSF.json`) | laut `script.md:3` (`--batch_size 128 --seq_len 336 --pred_len 96`) | MSE 0,366 (Tabelle 2) |
| DLinear, ETTh1, Horizont 96 | `github.com/cure-lab/LTSF-Linear`, Commit aus Fixture `cure-lab_LTSF-Linear.json` | `scripts/EXP-LongForecasting/Linear/etth1.sh`, Zeile für Horizont 96 | MSE 0,375 (Tabelle 2) |

Warum diese zwei: Beide Repos stammen aus derselben Code-Familie (Autoformer-Struktur mit `data_provider/`, `exp/`, `run_longExp.py`). Das prüft Frage 3 direkt. RLinear ist außerdem der Fall, bei dem `b1_candidates.md` den `drop_last`-Effekt bisher nur **nachgebildet** hat („ein Lauf des Autorencodes fand nicht statt"). Der Vorversuch schließt diese Lücke.

Kein Cora-Repo im Vorversuch. `tkipf/gcn` ist TensorFlow 1; das wäre ein eigener, späterer Schritt.

## Vorgehen

Alles von Hand, ohne LLM, ohne Tavily, ohne Token Factory. Kosten: 0 $ (Nebius-Sandbox ist kostenlos). Kein Eingriff in `src/`, keine Änderung an `benchmark/papers.yaml`.

**Schritt 1, lokal (Worker, Sonnet).** Skripte unter `scripts/prestudy_nachlauf/`, eigene Umgebung mit torch CPU (wie bei den Gegenproben in `failure_analysis_v4.md`).

1. Repo auf dem festen Commit klonen. Abhängigkeiten aus dem Repo ablesen, Liste mit Versionen festhalten.
2. `ETTh1.csv` aus derselben Quelle wie der Harness (`harness/etth1_data.py`, URL dort) an den Ort legen, den das Repo erwartet. SHA-256 der Datei notieren.
3. Startbefehl unverändert ausführen, nur Gerät auf CPU. Achtung: `--use_gpu` hat `type=bool`; prüfen, ob das Repo ohne GPU von selbst auf CPU fällt. Seed des Repos (`fix_seed`) festhalten.
4. Wert notieren, den das Repo selbst ausgibt (Test-MSE des Autorencodes, mit dessen Test-Loader).
5. **Zuordnungsprüfung vor jeder Metrik:** Zielwerte des Repo-Testdatensatzes ohne `drop_last` gegen die Labels des Harness vergleichen. Erwartet: 2785 Fenster, gleiche Reihenfolge, größte absolute Abweichung ≤ 1e-4. Schlägt das fehl, Ursache suchen und berichten, nicht weiterrechnen.
6. Adapter schreiben: trainiertes Modell laden, Test-Loader mit `drop_last=False` und `shuffle=False` bauen, Vorhersagen im ID-Format des Tasks schreiben (`{window}_{step}_{channel}`, 2785 × 96 × 7 Zeilen). Der Adapter ist eine neue Datei; Dateien des Repos bleiben unverändert (`git status` im Repo-Klon als Beleg).
7. Harness auswerten (`etth1_eval.py`, mit `droplast.py`): MSE auf allen Fenstern und die `drop_last`-Werte.

**Schritt 2, Nebius-Sandbox (Reviewer).** Dieselben Skripte im Basis-Snapshot und in einem Branch: Abhängigkeiten im Snapshot, Training und Adapter im Branch unter `unshare -n`, Harness danach frisch in den Branch kopiert. Laufzeit je Seed messen.

**Seeds:** Seed des Repos plus vier weitere (0 bis 3), zusammen fünf, wie in `papers.yaml`. Mittelwert und Streuung berichten, dazu den Einzelwert des Repo-Seeds. Der Seed steht in beiden Repos fest im Startskript (`fix_seed`). Für die vier weiteren Seeds wird das Startskript als **neue Datei** kopiert, in der nur diese eine Zeile anders ist; der Diff steht im Ergebnis. Der Lauf mit dem Repo-Seed nutzt das unveränderte Startskript.

## Was gemessen und berichtet wird

Je Einheit:

- Wert des Autorencodes laut eigener Ausgabe (mit dessen Test-Loader).
- Harness-MSE auf allen 2785 Fenstern, Mittel und Streuung über fünf Seeds.
- Harness-Wert der `drop_last`-Auswertung für die Batchgröße des Startbefehls.
- Abstand zum Zielwert und Urteil nach der bestehenden Regel (Toleranz 0,01, unverändert).
- Vergleich mit unserer Referenzlösung (`tests/reference/`), nur im Ergebnisdokument.
- Laufzeit je Seed lokal und in der Sandbox, Größe des Snapshots.
- Adapter: Zeilenzahl, Anteil gleicher Zeilen zwischen beiden Repos, benötigte Kenntnisse über das Repo (welche Klassen, welche Argumente).
- Arbeitszeit von Hand je Repo und jede Stelle, an der es gehakt hat (Abhängigkeit, Pfad, Gerät, Argument).

## Entscheidungsregel (vorab)

**Weiter mit dem Bau des Modus (nur ETTh1-Familie)**, wenn für **beide** Einheiten gilt:

1. Lauf in der Sandbox auf CPU, Training ohne Netz, höchstens 10 Minuten je Seed.
2. Zuordnungsprüfung bestanden (2785 Fenster, Abweichung ≤ 1e-4).
3. Keine Datei des Repos geändert.
4. Selbstprüfung des Adapters: Die `drop_last`-Auswertung des Harness trifft den Wert, den der Autorencode selbst ausgibt, auf ≤ 0,001 (gleicher Seed). Das zeigt, dass Adapter und Harness dasselbe Modell messen wie das Repo.

**Eingeschränkt weiter**, wenn genau eine Einheit alle vier Punkte erfüllt: Ursache der anderen berichten, der Nutzer entscheidet.

**Nicht bauen**, wenn keine Einheit sie erfüllt.

Das Urteil („reproduziert" oder nicht) ist **kein** Kriterium. Der Vorversuch prüft Machbarkeit, nicht das Ergebnis.

## Bekannte Risiken, die der Vorversuch sichtbar machen soll

- **Leak-Schutz ist im Nachlauf schwächer.** Der Original-Code braucht den vollen Datensatz, die Testwerte liegen also im Branch. Schicht 3 des Leak-Schutzes („Testlabels nie im Agenten-Datensatz") gilt dort nicht. Ersatz, im Vorversuch zu beurteilen: fester Commit mit Hash, unveränderte Repo-Dateien, Adapter als einzige neue Datei mit geprüftem Inhalt. Das Ergebnisdokument sagt, ob das reicht.
- **Abweichende Datenaufbereitung.** Stimmen Split oder Standardisierung nicht, ist der Vergleich nicht fair. Dafür ist die Zuordnungsprüfung da.
- **Versionen.** Die Repos nennen ältere torch-Versionen. Läuft der Code nur mit einer alten Version, steht das im Ergebnis; die Version des Snapshots wird nicht stillschweigend geändert.
- **Verallgemeinerung.** Zwei Repos derselben Familie sagen nichts über Graph-Repos.

## Ehrlichkeit

- Die Zahlen des Vorversuchs sind **Handarbeit des Teams**, kein Ergebnis des Agenten. Sie stehen nicht als Agenten-Ergebnis im README und zählen in keinem Benchmark.
- Aus `b1_candidates.md` gibt es eine Erwartung (RLinear: alle Fenster etwa 0,370, mit `drop_last` etwa 0,3665). Sie ist bekannt und wird nicht als Vorhersage des Vorversuchs ausgegeben. Weicht der Lauf des Autorencodes davon ab, wird das berichtet und `b1_candidates.md` mit Datum ergänzt, nicht umgeschrieben.
- Kein Wert aus dem Vorversuch oder aus `tests/reference/` geht in einen Prompt.
- Benchmark v4 bleibt unverändert und bleibt ein Nachbau-Ergebnis. Ein späterer Nachlauf-Vergleich bekommt eine eigene Vorab-Festlegung.

## Ergebnis und Aufwand

- Ergebnisdatei: `benchmark/experiments/results/nachlauf_prestudy.md`, Rohausgaben daneben als `.txt`.
- Aufwand geschätzt (nicht gemessen): Schritt 1 ein halber Tag, Schritt 2 zwei Stunden.
- Danach, nur bei „weiter": eigener Bauplan für den Modus (Task-Typ, Adapter-Prompt, Diff-Prüfung, Dossier-Felder, zweite Spalte im Zeugnis), geschätzt vier bis sechs Arbeitstage.

## Entscheidungen des Nutzers (2026-10-02)

- Regel freigegeben, Schritt 1 führt worker-2 aus (Auftrag 33, `docs/worker_brief_33_nachlauf_prestudy.md`).
- Offen: Reihenfolge gegenüber Deployment und Demo-URL.
