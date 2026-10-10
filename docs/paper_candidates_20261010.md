# Weitere Paper-Kandidaten — 2026-10-10

Recherche aus Paper, offiziellen Repositories und Startskripten. Die Auswahl ist von Hand und nach Entwicklung getroffen; sie ist kein Blindtest. CPU-Eignung ist aus Architektur und Setup abgeleitet, nicht durch eine Laufzeitmessung bewiesen. Keine Autorprogramme auf dem Host ausgefuehrt.

## Prioritaet

| Paper | Fester Einzelclaim | Warum versuchen / offene Huerde |
|---|---|---|
| GLinear | ETTh1, multivariat 7 Kanaele, Eingang 336, Horizont 96, MSE **0.3820**, Paper v1 Table II | Zwei lineare Schichten, GELU und RevIN; konkretes Originalskript. Batch 8 kann die Laufzeit erhoehen. |
| TimeBase | ETTh1, Eingang 720, Horizont 96, MSE **0.349**, ICML 2025 Table 10 | Kleines Modell, passende Originalkonfiguration; zusaetzlicher Orthogonalitaets-Loss muss erhalten bleiben. |
| SOFTS | ETTh1, Eingang 96, Horizont 96, MSE **0.381**, Paper v3 Appendix C.1 | Offizielles Repo und Konfiguration schon geprueft; groesseres Modell, stochastisches Pooling im Training. |

### GLinear: gestarteter Versuch

Nutzerfreigabe: „ja du darfst starten“. Genau ein Produktlauf mit drei Seeds, Toleranz +/-0.01 und dem vorher festgelegten Einzelclaim. Kein Austausch von Ziel, Variante oder Horizont nach Sichtung der Messungen. Paper und Autorencode dienen dem Nachbau; eigene Messung im Harness.

- [Paper v1, Table II](https://arxiv.org/html/2501.01087v1).
- [Offizielles Repo](https://github.com/t-rizvi/GLinear), recherchierter Commit `f54bc961ea9434a6b88b17c42fa4d681e98e9672`; tatsaechlich gelesener Commit wird vom Lauf dokumentiert.
- `scripts/EXP-LookBackWindow_&_LongForecasting/Linear_LongForecasting.sh`: GLinear, ETTh1, `features M`, sieben Kanaele, `seq_len 336`, `pred_len 96`, Batch 8, Lernrate 0.001.
- `run_longExp.py`: 10 Epochen, Patience 3, Schedule `type1`, urspruenglicher Seed 2021. Unser Produktlauf nutzt seine drei Standard-Seeds und ist daher kein identischer Nachlauf.
- `data_loader.py`: 12/4/4-Monatsgrenzen, Standardisierung ausschliesslich auf Trainingsdaten. Test-Loader verwirft keine Rest-Batches.
- Kein vorhandener Laufordner fuer diese arXiv-ID in den betrachteten Workspaces gefunden. Das Paper wurde aber bereits in einer frueheren Tavily-Recherche gefunden (`tavily_experiments_addendum_a.md`); „nie zuvor gesehen“ waere falsch.
- Einmaliger Driver: `/tmp/nullius-paper-search/run_glinear_bounded.py`, Output `glinear_trial_output.log`, vorab eingefrorener Input `GLinear/frozen_trial_input.yaml`. Driver und Budgetspur werden beim Abschluss im Laufordner gesichert.
- Maximal 0.50 $ brutto Inferenz durch Pruefung jedes HTTP-Versuchs vor Versand, einschliesslich Retries; unklare Requests bleiben reserviert. Tavily separat auf 12 geschaetzte Credits begrenzt. Bestehende lokale Budgetbasis bleibt erhalten, keine Aufstockung und keine Freigabe der Demo-Reserve.
- Offline-Pruefung des wiederverwendeten Guards bestanden: Buchung, Vorab-Stopp, unbekannter Preis, Timeout-Reservierung und Retry-Pruefung. Erster Start mit System-Python scheiterte vor jeder Anfrage am fehlenden Tavily-Modul; anschliessend vorhandene Projekt-venv genutzt, keine Pakete installiert.

### TimeBase

[Veroeffentlichtes Paper](https://proceedings.mlr.press/v267/huang25az.html), Table 10 im dort verlinkten Volltext. **0.396** in Table 2 ist der Mittelwert ueber vier Horizonte, kein Einzelclaim. **0.364** aus einem anderen Tabellenkontext ist kein Ersatz fuer standalone TimeBase.

[Offizielles Repo](https://github.com/hqh0728/TimeBase), Commit `369b330f3d77371fcc7e8c75c808d01330c40899`. `scripts/etth1.sh`: Eingang 720, Horizont 96, Periode 24, sechs Basen, Orthogonalitaetsgewicht 0.16, Lernrate 0.1, Batch 64, 30 Epochen, Patience 5. Skriptname `LightTimeBaseTST` ist laut `exp_main.py` ein Alias fuer `models/TimeBase.py`, kein fremdes Modell. Das Modell liefert bei aktiviertem Orthogonalitaets-Loss ein Tupel; der Loss muss im Training verarbeitet werden, die Vorhersage separat ausgewertet werden.

### SOFTS

[Paper v3](https://arxiv.org/html/2404.14197v3), [offizielles Repo](https://github.com/Secilia-Cxy/SOFTS), Commit `f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44`. Details und passende Konfiguration in [paper_round_20261010.md](paper_round_20261010.md). Noch kein neuer Versuch gestartet.

## Reserve und bereits bekannte Versuche

- **S²GC / Simple Spectral Graph Convolution (ICLR 2021)** ist ein anderes Paper als SGC (ICML 2019). [Offizielles Repo](https://github.com/allenhaozhu/SSGC), Commit `fb833a7149f439a15fad45c9ebaca30aac46be3f`: CiteSeer 73.6%, 150 Epochen, lr 0.2, Weight Decay 0.0001, alpha 0.05, Grad 16. Leichte lineare Klassifikation nach Graph-Vorverarbeitung. Allerdings blockiert OpenReview den Volltextabruf; Paper-Spalten noch nicht vollstaendig selbst verifiziert. Cora-Wert im README und gefundenen Paper-Auszug unterscheiden sich. Zudem keine passende arXiv-ID gefunden, waehrend der Produktimport arXiv erwartet. Deshalb derzeit kein bezahlter Produktversuch.
- **FITS:** beide historischen Nachlauf-Versuche scheiterten an der Befehlsauswahl, ohne Training oder Messwert. [Bericht](../benchmark/experiments/results/nachlauf_v1.md). Keine neue Einheit; dieser Nachlauf darf laut damaliger Festlegung nicht ein drittes Mal laufen.
- **SGC:** fruehere Modi teilweise/gescheitert; im spaeteren Autorencode-Benchmark reproduziert: 0.8054 gegen 0.810. [Bericht](../benchmark/experiments/results/recall_cg_v1.md). Entwicklungseinheit, kein neuer Paper-Erfolg.
- **CycleNet:** bereits in dieser Runde gemessen, nicht reproduziert. Erst vorhandene Fenster-/Setup-Abweichung untersuchen, bevor ein weiterer bezahlter Versuch erwogen wird.

## Abschluss GLinear

Lauf `runs/disc_2501_01087/20261010-161916` abgeschlossen: **Zielwert erreicht, Produkturteil reproduziert/vorlaeufig**. MSE **0.3795828852 +/- 0.0002884590** (Standardabweichung), Ziel **0.3820 +/- 0.01**, drei Seeds. Einzelwerte: 0.3793850143, 0.3794497777, 0.3799138635. Ausgefuehrter End-Code `67dae24762de`, gelesenes offizielles Repo entspricht dem vorab recherchierten Commit.

Inferenz **0.204707727 $ brutto**, HTTP-Guard meldet denselben Wert. **11 geschaetzte Tavily-Credits**, separat von Inferenz. Kein weiterer Paper-Versuch gestartet. Auf Basis des letzten Nutzer-Screenshots und der beiden begrenzten Versuche CycleNet/GLinear bleibt planerisch ca. 1.054 $ ueber der 12-$-Demo-Reserve; dies ist kein neuer Konsolen-Kontostand und ersetzt keine aktuellen Abrechnungsdaten. Keine bedingte 10-$-Aufstockung eingerechnet.

Drei Programm-Kandidaten, Auswahl anhand der Validierung (0.7206 / **0.6859** / 0.7080), nicht anhand des besten Testwerts. Zwei Setup-Fehler eines Kandidaten repariert. Anschliessend Repo-Abgleich mit Reparatur des Early-Stopping-Vergleichs, erneut validiert und mit drei Seeds gemessen. Eingefrorener Input, Original-Zitat, Repo-Metadaten und manueller Abschlussvermerk sind neben Dossier, Events, Programmversionen und Budgetspur im Laufordner gesichert.

Grenzen: Automatische Zielwert-Gegenpruefung scheiterte an einer Modellantwort ohne JSON; automatische Spaltenpruefung blieb ungeprueft. Die Quelle wurde vorab von Hand direkt in Table II gelesen (richtige GLinear-Spalte, Horizont 96). Das unveraenderte Produktzeugnis enthaelt die automatischen Warnungen weiterhin. Methodentreue ist nicht geprueft. Der Agent las nicht alle Startargumente, Batch/Epochen blieben im extrahierten Setup offen; ein erreichter Zielwert beweist deshalb keinen identischen Autorennachlauf. Auswahl von Hand nach Entwicklung, ausserhalb der acht Entwicklungseinheiten, aber Paper schon in frueheren Community-Suchtreffern bekannt. Kein Blindtest und keine neue allgemeine Erfolgsquote.

Die anderen Recherchebefunde sind weiterhin Kandidaten, keine eigenen Reproduktionsergebnisse. Der Lauf wurde in dieser Runde noch nicht in Website-Katalog und statische Exporte uebernommen.
