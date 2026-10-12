# SARAF: lokale Ursachenanalyse und Messungsdarstellung, 2026-10-13

Die hohen Zahlen lassen sich lokal reproduzieren. Das erzeugte Modell und die Trainingslogik weichen erheblich vom Autorencode ab. Es gibt keinen Beleg, dass das Paper seinen Zielwert verfehlt. Die Anzeige vermischte daneben parallele Kandidaten und spaeter verworfene Messungen.

## Lauf und unveraenderte Endmessungen

Untersucht: `disc_2606_04135_f7b4e6db785fdb63--20261012-224202`, offizielles Repository `ShiqiaoZhou/SARAF`, Commit `7b49aa8aabb38e8fe63adbc3146b1890958bf3ca`. Modus laut Ereignissen `nachbau_autorencode`: ein neu geschriebener Nachbau mit Referenzausschnitt, keine Ausfuehrung des Original-Repositories.

- Korrekt gegengepruefter Paperwert: SARAF, ETTh1, Horizont 96, MSE 0.371.
- Erste gewaehlte Version `698f54efaada`: Seed 0/1/2 = 0.8269564189 / 0.8116463064 / 0.9074021173. Mittel 0.8486682809.
- Nach automatischer Reparatur `35d73e6a7201`: 0.4939007280 / 0.4880756517 / 0.4930273299. Mittel **0.4916679032**, Std **0.0031414820**.
- Dossier markiert ausschliesslich die drei letztgenannten Messungen als `chosen`. Methodentreue bleibt `nicht geprueft`, Aussage vorlaeufig.
- Der Reparaturausloeser war **MAE**: urspruenglicher Mittelwert 0.7185956962 > Repeat-Baseline 0.7131813544. MSE lag dagegen unter beiden einfachen MSE-Baselines (Zero 1.1099284101, Repeat 1.2943705948). Die unspezifische UI-Beschreibung konnte das nicht erklaeren.

Die Belege liegen im [veroeffentlichten Dossier](https://nullius.akca.dev/data/disc_2606_04135_f7b4e6db785fdb63/20261012-224202/dossier.json), [Ereignislog](https://nullius.akca.dev/data/disc_2606_04135_f7b4e6db785fdb63/20261012-224202/events.jsonl) und den exportierten Code-Versionen `code_698f54efaada.py` / `code_35d73e6a7201.py` im selben Verzeichnis. Original- und Endmessungen wurden nicht umgeschrieben.

## Lokal isolierte Ursachen

### 1. Der Prompt enthielt die Berechnung des Modells nicht

`spec.focused_excerpt` begrenzt jede Datei auf 6.000 Zeichen, insgesamt 15.000. Der Ausschnitt wurde mit den exakten Autoren-Dateien und `setup_details` des Laufs rekonstruiert: **15.285 Zeichen inklusive Abschnittsmarken**, dieselben sechs `excerpt_docs` wie im Live-Ereignis.

`models/SARAF.py`: 9.738 Zeichen, `def encoder` beginnt bei Zeichen **6.116** (nullbasierter Index). Sichtbar waren nur die ersten 6.000 Zeichen. Initialisierung und ungenutzte Layer waren sichtbar, der tatsaechliche Forward-Pfad mit Offset-Normalisierung und einfacher Fusion fehlte.

`layers/Retrieval.py`: 32.473 Zeichen, im Prompt nur **3.297** Zeichen. `decompose_mg` beginnt bei 15.341, `masked_sim` bei 22.881, `retrieve` bei 29.180. Alle fehlten. Dateinamen im Herkunftslabel belegten somit keine vollstaendige Kenntnis des Kernverfahrens. Der spaetere Repo-Vergleich erhielt mehr Code, lieferte jedoch keine belegte Abweichung und wegen Kuerzungen korrekt nur `incomplete`; er reparierte diese Fehler nicht.

### 2. Der beste Checkpoint wurde ueberschrieben

Die erste gewaehlte Version verwendet `best_state = model.state_dict().copy()`. Die Dictionary-Kopie teilt Tensor-Speicher mit dem weitertrainierten Modell. Ein lokaler PyTorch-Probeversuch bestaetigt: gespeicherte Gewichte aendern sich bei einem spaeteren Update; `copy.deepcopy` bewahrt sie.

Auf den echten ETTh1-Daten, Seed 0, 10 Epochen, identischer Code ausser der genannten Intervention:

| Lokaler Diagnoseversuch | Test-MSE | Harness-kompatibler Val-MSE |
| --- | ---: | ---: |
| Unveraenderte Version `698f54efaada` | 0.8269566150 | 1.2938986153 |
| Nur unabhaengiger bester Checkpoint | **0.5625861601** | 1.0027876907 |
| Checkpoint plus korrigierte Projektionsachsen | 1.0245523274 | 1.4393192502 |

Der erste Wert stimmt bis auf ca. 0.0000002 mit Live-Seed 0 ueberein. Die Checkpoint-Korrektur allein erklaert hier eine MSE-Verbesserung um ca. 0.2644. Die zweite Intervention ist kein Erfolg: trotz realem Achsenfehler wird dieser weiterhin falsche Nachbau schlechter. Kein Parameter wurde anhand der Testzahlen optimiert; es sind Diagnoseversuche, keine veroeffentlichte Paper-Reproduktion.

Auch ein vierter lokaler Versuch mit ausschliesslich korrigierten Projektionsachsen im spaeteren, bereits zentrierten Code `35d73e6a7201` erreicht nur MSE **0.6883316147** und Val-MSE **1.3307076378** (Seed 0, 210.86 s). Damit ist die Achsenkorrektur in beiden untersuchten Varianten keine isolierte Loesung fuer den Zielwert. Die eigentliche SARAF-Berechnung muss vollstaendig erhalten bleiben.

### 3. Der erzeugte Retrieval-/Forward-Pfad ist ein anderes Verfahren

Belegte Unterschiede zum gepinnten Original:

- Original zentriert die Eingabe am letzten Zeitschritt und addiert diesen Offset am Ausgang wieder. Erste Version tut beides nicht. Die spaetere Reparatur verwendet den Fenstermittelwert, weiterhin einen anderen Mechanismus.
- Original berechnet mehrskalige, zentrierte Korrelation, datenabhaengige Stationaritaet und adaptive Auswahl mit MMR. Nachbau nimmt rohe euklidische Distanzen und Top-k. Adaptive Parameter stehen nur in der Konfiguration.
- Original maskiert im Training ueberlappende Zeitfenster. Nachbau darf sein eigenes Trainingsfenster abrufen. Synthetischer Probeversuch: Fenster 40 waehlt sich selbst, abgerufenes Trainingsziel hat MSE **0**. Das beweist den Trainings-Shortcut, nicht einen Zugriff auf versteckte Testlabels.
- Nachbau flacht `[B, Zeit, Kanal]` direkt zu `[B*Kanal, Zeit]` ab. Ohne `permute(0,2,1)` werden Kanal und Zeit falsch zugeordnet. Identitaetsprojektion im Probeversuch ergibt fuer die erste Zeitposition `[0,96,192,288,384,480,576]` statt `[0,1,2,3,4,5,6]`.
- Autoren-Forward nutzt eine einfache 0.5-Fusion ohne Ausfuehrung des deklarierten Gates/LayerNorm. Nachbau fuehrt Gate, LayerNorm und drei skalige Projektionen aus. Die erste Plan-Kurzbeschreibung nennt die deklarierten, teilweise ungenutzten Layer als Architektur.
- `run.py` definiert `n_period=1`, `time_aware_weight=0.5`; das ETTh1-Skript ueberschreibt diese nicht. Nachbau verwendet 3 bzw. 0. Weitere Details wie Zeitmarken fehlen ebenfalls.

### 4. Gemeinsame Trainingslogik wird falsch ausgeschlossen

`setup_details.path_model_exclusion` klassifiziert `exp_long_term_forecasting.py` als anderes Modell. Das Live-Ereignis zeigt den Ausschluss von Adam-/Schedule- und weiteren Trainingsangaben. `Retrieval.py` wird ebenfalls als anderes Modell eingestuft. Der Plan sagt folglich, kein Scheduler sei genannt; Autorentraining ruft aber `adjust_learning_rate` mit dem `type1`-Schedule auf (Halbierung pro Epoche nach dem ersten Update). Der Nachbau haelt die Lernrate konstant.

### 5. Daten und Scoring passen; Validierung ist eine bewusst kleinere Stichprobe

Split 8640/2880/2880, train-only Z-Standardisierung, sieben Kanaele und 96 Testschritte stimmen mit dem Autorendatenloader ueberein. Lokaler Vergleich der erzeugten Val-Ziele mit `ett_data.val_targets`: maximale Float32-Rundungsdifferenz **2.33e-7**, Form `[117,96,7]`. Die fast exakte lokale Wiederholung von Live-Seed 0 spricht gegen einen vertauschten Testsplit oder Skalierungsfehler als Erklaerung von 0.827.

Nullius verwendet 117 Val-Fenster mit Tagesabstand; Autorenvalidierung verwendet dichte Fenster. Das kann die Auswahl des Checkpoints beeinflussen. Ausserdem mittelt der Nachbau Batch-Verluste ungewichtet; der Harness gewichtet jeden Zielwert gleich. Gedruckter `best_val_loss` und Harness-Val-MSE sind deshalb auch bei korrektem Checkpoint nicht identisch. Die groessere Abweichung der ersten Version kommt zusaetzlich vom ueberschriebenen Checkpoint.

## Reparaturprioritaet fuer das Verfahren

1. Vollstaendige Kernberechnung und aufgerufene Abhaengigkeiten statt beliebiger Dateipraefixe in den Autoren-Prompt nehmen; bei nicht abgedecktem Kern explizit stoppen. Ein Dateiname darf Vollstaendigkeit nicht vortaeuschen.
2. Gemeinsame Trainings-/Retrieval-Dateien korrekt zuordnen; Startskript und wirksame CLI-Defaults beruecksichtigen.
3. Checkpoint-Kopie und Achsenvertraege vor einem teuren Lauf pruefen. Trainings-Retrieval gegen eigene/ueberlappende Beispiele pruefen.
4. Erst danach einen originalgetreuen Nachbau mit Val-Auswahl testen. Keine Zusage, dass diese Schritte MSE 0.371 erreichen; das ist mit dieser Untersuchung nicht bewiesen.

## In diesem Arbeitsstand korrigierte Anzeige

Versuche und Code-Versionen nennen den Kandidaten. Vor der Auswahl bleiben Kandidatenwerte in der Versuchshistorie; der Seed-Chart beginnt nach Auswahl. Er verwirft fremde Kandidaten und abgelehnte Repo-Reparaturen. Der Orchestrator publiziert kuenftig `measurements_selected` mit exakt den Messungen des Urteils sowie Annahme/Ablehnung der Reparatur nach allen Seeds. Bei aelteren Logs kann nur eine eindeutige vollstaendige Code-Gruppe mit passendem aufgezeichnetem Endmittel rekonstruiert werden; andernfalls bleibt die Endzuordnung ausdruecklich unbekannt.

Fuer den untersuchten Lauf ergibt der korrigierte Renderer dieselben drei `35d73e6a7201`-Messungen und 0.4916679032 wie das Dossier. Das ist eine Darstellungsreparatur; die oben beschriebenen Verfahrensfehler sind damit noch nicht behoben.

## Lokale Artefakte und Grenzen

`/tmp/nullius-saraf-local-audit/`: `actual_excerpt_reconstructed.txt`, gepinntes `author_repo/`, `probes.json`, `train_probe.py`, `train_{original,deep,axes_deep}.json` und Logs, `live_final/` mit unveraendert heruntergeladenem Dossier/Code/Events. Die Trainingsprobe ersetzt ausschliesslich `nullius_io` durch lokale Arrays und eine Ergebniserfassung; Train/Test-Berechnung des Skripts bleibt erhalten. PyTorch 2.14.1, CPU, vier Threads, Seed 0. Keine neuen Tavily-, Modell- oder Sandbox-Auftraege; nur statische oeffentliche Referenzen und lokale CPU-Berechnungen.

Verifikation der Darstellungsreparatur: 149 Frontend-Tests, 76 Orchestrator-/Reparatur-/Kandidaten-Tests, 76 API-/Export-/Graph-/Val-Tests. Produktionsbuild erfolgreich; Lint ohne Fehler (bestehende Warnungen). Sechs Browser-Faelle mit den aufgenommenen Live-Ereignissen: Desktop/Mobil, vor Auswahl/nach Auswahl/final; korrekte Kandidaten, finale 0.491668, keine Seitenfehler oder horizontaler Ueberlauf, keine POST-Auftraege.

Echte ETTh1-Datei SHA-256: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. Erste erzeugte Version SHA-256: `698f54efaada4ce6748cffca537da646b6657c6f4917e842fb3fb37a23c72fbc`. Diagnostische Werte beweisen keine Kausalwirkung fuer alle Seeds oder beliebige Papers.
