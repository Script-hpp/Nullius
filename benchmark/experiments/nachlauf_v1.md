# Modus „Nachlauf" v1: Regeln und Prüfung (Vorab-Festlegung)

Status: **final** (vom Nutzer am 2026-10-02 freigegeben). Regeln, Einheiten und Erfolgsregel werden nicht mehr geändert, Abweichungen werden datiert notiert. Noch einzutragen, vor dem ersten Agenten-Lauf der Prüfung (M4): Commits der vier neuen Repos und die Bestätigung der Zielwerte gegen das Paper. Bauplan: `docs/plan_nachlauf.md`. Grundlage: Vorversuch `results/nachlauf_prestudy.md` (Regel erfüllt, beide Einheiten).

## Was der Modus prüft

- **Nachbau** (besteht): Der Agent schreibt eigenen Code aus Paper und Repo. Frage: Reicht die Beschreibung?
- **Nachlauf** (neu): Der Code der Autoren trainiert, unser Harness misst. Frage: Liefert der Code der Autoren die Zahl im Paper?

Beide Urteile stehen nebeneinander im Zeugnis. Die Benchmark-Zahlen im README (v4) bleiben Nachbau-Zahlen. Nachlauf-Zahlen werden nie damit vermischt.

Abgeleitete Aussage, nur wenn beide Urteile vorliegen: Nachlauf „reproduziert" und Nachbau nicht „reproduziert" = „Zahl stimmt, Beschreibung reicht dem Agenten nicht" (im UI „description gap"). Das ist eine Aussage über Paper **und** Agent zusammen, nicht über das Paper allein; so steht es im Zeugnis.

## Ablauf und Trennung (fest)

Drei Branches je Seed, der Agenten-Code sieht nie Testlabels:

1. **Trainings-Branch** (Snapshot mit Repo auf festem Commit und vollem Datensatz). Es läuft ausschließlich der Startbefehl der Autoren, unter `unshare -n`. Kein vom Agenten geschriebener Code läuft in diesem Branch. Ergebnis: Checkpoint und Ausgabe des Autorencodes. Danach `git status` und `git diff --stat` im Klon: müssen leer sein, sonst `ungueltig`.
2. **Vorhersage-Branch** (frischer Branch von einem Snapshot **ohne** Rohdatensatz: Repo-Klon ohne `dataset/`, dazu `/data/etth1_public.npz` wie im Nachbau). Der Orchestrator überträgt nur die Checkpoint-Datei. Der Adapter des Agenten (`predict.py`) lädt Modellklasse und Gewichte und schreibt `/output/predictions.csv` aus `test_inputs`. Lauf unter `unshare -n`, statische Codeprüfung wie im Nachbau.
3. **Harness** wie bisher: frisch kopiert, eigene Labels, alle 2785 Fenster.

## Prüfungen (deterministisch, ohne LLM)

- **P1 Startbefehl belegt.** Der Befehl muss wörtlich in einer Datei des Repos stehen (Skript, README, Dokumentation), Quelle mit Datei und Zeile im Dossier. Jede Abweichung davon (Token für Token) wird gelistet. Ohne Meldung erlaubt sind nur: Gerät/GPU-Schalter, Pfade, Log-Umleitung. Jede andere Abweichung (z. B. korrigierter Modellname) steht im Zeugnis als „Startbefehl weicht von der Dokumentation ab". Schalter, die laut Hilfetext oder Name auf Training mit allen Daten zielen (z. B. `train_only`), sind verboten: `ungueltig`.
- **P2 Unabhängigkeit der Vorhersagen (neu, `harness/independence.py`).** Fenster mit Abstand ≥ 816 teilen keine Information (720 Schritte Eingabe, 96 Schritte Ziel). Der Harness lässt den Adapter in frischen Branches noch einmal auf kleinen Teilmengen laufen (die öffentliche Datei im Branch wird dafür durch die Teilmenge ersetzt, die volle Datei ist dort nicht vorhanden): 8 zufällige Versätze, je Versatz die Fenster `r, r+816, r+1632, …`, gemischt mit Fenstern aus Rauschen. Die Vorhersagen für diese Fenster müssen denen des vollen Laufs gleichen (größte Abweichung ≤ 1e-4). Sonst `ungueltig` („Vorhersage hängt von anderen Fenstern ab"). Ein Adapter, der Ziele aus Nachbarfenstern abschreibt, fällt hier durch, weil die Nachbarn in der Probe fehlen. Die Versätze kennt der Adapter nicht (eigener Zufall des Harness je Lauf).
- **P3 Selbstprüfung gegen die Eigenausgabe.** Gibt der Autorencode eine eigene Test-Metrik aus, vergleicht der Harness sie mit seiner Rechnung auf denselben Fenstern (alle, oder die `drop_last`-Teilmenge zur Batchgröße des Startbefehls). Abstand ≤ 0,001 = „Zuordnung geprüft". Größer = `ungueltig` („Harness und Autorencode messen nicht dasselbe"). Gibt das Repo nichts aus: Warnung „Zuordnung nicht geprüft" im Zeugnis, Urteil bleibt.
- **P4 Formprüfung** des Adapters mit `SMOKE=1` wie im Nachbau.

P2 schließt nebenbei eine Lücke, die es im Nachbau für ETTh1 schon gibt: Auch dort stehen die Ziele eines Fensters in den Eingaben späterer Fenster, bisher verbietet das nur ein Satz im Prompt. Der Einbau in den Nachbau ist **nicht** Teil von v1 (würde den Benchmark-Pfad ändern); er steht als offener Punkt im README.

## Umgebung und Seeds

- Snapshot wie der Task (aktuelles torch CPU), dazu `numpy<2`. `requirements.txt` des Repos wird nicht blind installiert. Fehlt ein Paket, darf die Reparatur nur Pakete nachinstallieren, die in den Abhängigkeitsdateien des Repos genannt sind. Jede abweichende Version steht im Dossier (z. B. „torch 2.14 statt 1.9.0").
- Seeds: Seed des Repos mit unverändertem Startskript, dazu vier weitere. Erlaubte Wege: Kommandozeilenschalter des Repos, oder Kopie des Startskripts mit genau einer geänderten Zeile (im Branch gegen das Original geprüft). Geht beides nicht: nur der Repo-Seed, Zeugnis nennt „eine Messung, keine Streuung".
- Urteil wie bisher: Mittel über Seeds gegen Zielwert, Toleranz aus `papers.yaml`, unverändert. Die `drop_last`-Auswertung und der Hinweis dazu bleiben wie gebaut.
- Fehler (Absturz, fehlendes Paket, falscher Modellname): Reparaturschleife wie bisher, höchstens `MAX_ATTEMPTS_PER_CLAIM`, Fehlertext ist das Ende von stderr.

## Grenzen (stehen im Zeugnis und im README)

- Wählt der Autorencode selbst auf Testdaten aus, sieht der Nachlauf das nicht. Der Nachlauf prüft „Code liefert Zahl", nicht „Code ist methodisch sauber".
- Die Testdaten bereitet unser Harness auf, nicht der Autorencode. P3 belegt die Gleichheit nur, wenn das Repo eine eigene Metrik ausgibt.
- v1 gilt nur für ETTh1, Horizont 96, Modelle ohne Zeitmerkmale als Eingabe (die öffentliche Datei enthält keine Zeitstempel). CPU, höchstens 10 Minuten je Seed.
- Die Inferenz läuft mit Modellklasse und Gewichten der Autoren, aber über unseren Adapter, nicht über das Testskript der Autoren.

## Prüfung des Modus (vor dem ersten Agenten-Lauf festgelegt)

**Entwicklungs-Einheiten** (bekannt aus dem Vorversuch, daran wird gebaut): RLinear (`plumprc/RTSF` `0fec00104f`), DLinear (`cure-lab/LTSF-Linear` `0c113668a3`).

**Neue Einheiten** (bis zur Prüfung nicht angefasst: kein Klon, kein Lauf, kein Blick in den Code über `b1_candidates.md` hinaus):

| Einheit | Repo | Zielwert laut `b1_candidates.md` |
|---|---|---|
| SparseTSF, ETTh1-96 | `github.com/lss-1138/SparseTSF` | 0,362 (Tabelle 11) |
| FITS, ETTh1-96 | `github.com/VEWOXIC/FITS` | 0,372 |
| PatchTST/42, ETTh1-96 | `github.com/yuqinie98/PatchTST` | 0,375 |
| NLinear, ETTh1-96 | `github.com/cure-lab/LTSF-Linear` (gleiches Repo wie DLinear, anderes Modell) | 0,374 |

Commits werden beim Einfrieren eingetragen (Stand des Hauptzweigs an dem Tag), Zielwerte vorher gegen das Paper geprüft. Repo-Adressen sind vorgegeben, kein Tavily-Aufruf in der Prüfung.

**Lauf:** Token Factory, LLM-Cache aus, je neue Einheit **eine** Generierung, 5 Seeds (oder Repo-Seed allein, siehe oben), `src/` eingefroren ab dem ersten Lauf. Kostenstopp 3 $ brutto. Store `runs/nachlauf_v1.jsonl`.

**Erfolgsregel:**

1. **Gültiger Nachlauf bei mindestens 2 der 4 neuen Einheiten.** Gültig heißt: Training lief, P1 bis P4 bestanden (P3 „nicht geprüft" zählt als bestanden, wird aber ausgewiesen), Harness-Ergebnis vorhanden. Das Urteil selbst (reproduziert oder nicht) ist kein Kriterium.
2. **Negativkontrollen, alle drei abgelehnt** (echte Sandbox, kein LLM): (a) Adapter, der Ziele aus Nachbarfenstern abschreibt, scheitert an P2; (b) Startbefehl mit `train_only` scheitert an P1; (c) Adapter mit vertauschter Fensterreihenfolge scheitert an P3 oder P2.
3. **Kein gültig gewerteter Lauf mit bekanntem Fehler:** Der Reviewer liest jeden als gültig gewerteten Adapter und Startbefehl von Hand; findet er einen Verstoß, den P1 bis P4 nicht gemeldet haben, gilt Punkt 3 als verfehlt.

Erfüllt: Modus wird in der Demo gezeigt, Ergebnis je Einheit im README, getrennt nach Entwicklungs- und neuen Einheiten. Punkt 1 verfehlt, 2 und 3 erfüllt: Modus bleibt als „funktioniert für die Entwicklungs-Einheiten" im Produkt, README sagt das. Punkt 2 oder 3 verfehlt: Modus wird **nicht** veröffentlicht.

Kein zweiter Prüflauf mit demselben Code. Nach einer Code-Änderung ist genau ein weiterer Lauf erlaubt, auf denselben Einheiten, dann als „zweiter Versuch" berichtet.

## Ehrlichkeit

- Vier Einheiten, eine Generierung: eine Stichprobe. Drei der vier Repos sind mit den Entwicklungs-Repos verwandt (gleiche Code-Familie).
- Die Zahlen des Vorversuchs sind Handarbeit und bleiben so beschriftet.
- Kein Wert aus `tests/reference/`, aus dem Vorversuch oder aus dieser Tabelle steht in einem Prompt. Der Agent bekommt Claim, Repo-Dateien und Fehlermeldungen.

## Nachtrag 2026-10-02 (vor dem ersten Lauf der Prüfung, vom Nutzer freigegeben)

**Anlass.** Im ersten echten Agenten-Lauf an einer Entwicklungs-Einheit (RLinear, Code `3f6051a`) las das Modell die Datei mit den dokumentierten Befehlen nicht und baute in der Reparatur einen Startbefehl aus den `argparse`-Definitionen. Training und alle Prüfungen P1 bis P4 liefen durch, das Urteil war „reproduziert" (0,3746), aber mit einem Befehl, den die Autoren nicht dokumentieren. Nach der Regel oben hätte dieser Lauf als „gültig" gezählt. Der Code ist inzwischen geändert (`f81b35f`: dokumentierte Befehle werden per Programm gesucht, das Modell wählt per Nummer, ohne Kandidaten kein Lauf).

**Verschärfung der Erfolgsregel, Punkt 1.** Ein Nachlauf zählt nur als gültig, wenn zusätzlich gilt: **Der Startbefehl geht auf eine wörtliche Zeile einer Datei des Repos zurück** (Quelle mit Datei und Zeile im Dossier). Abweichungen von dieser Zeile sind erlaubt, wenn P1 sie listet (Beispiel aus der Entwicklung: korrigierter Modellname). Ein Lauf ohne solche Quelle wird im Ergebnis als „Lauf mit nicht dokumentiertem Befehl" ausgewiesen und zählt nicht als gültig. Das ist strenger als die ursprüngliche Regel; sonst ändert sich nichts.

**Commits der neuen Einheiten** (Stand des Hauptzweigs am 2026-10-02, per `git ls-remote` abgefragt, ohne die Repos zu klonen oder zu lesen):

| Einheit | Repo | Commit | Claim in `papers.yaml` |
|---|---|---|---|
| SparseTSF | `github.com/lss-1138/SparseTSF` | `b8c2740eecc84d8095ffce49ba5acafe68e53bb8` | `sparsetsf_2024` / `sparsetsf_etth1_96_mse` |
| FITS | `github.com/VEWOXIC/FITS` | `d040bb015b6299da26d879b90dd19c80fb72c160` | `fits_2023` / `fits_etth1_96_mse` |
| PatchTST/42 | `github.com/yuqinie98/PatchTST` | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` | `patchtst_2023` / `patchtst42_etth1_96_mse` |
| NLinear | `github.com/cure-lab/LTSF-Linear` | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `zeng_nlinear_2023` / `nlinear_etth1_96_mse` |

**Zielwerte gegen das Paper geprüft** (Reviewer, 2026-10-02, HTML-Fassungen auf arXiv; kein Tavily-Aufruf): SparseTSF 0,362 (Tabelle 11 „after fixing code bug", arXiv v2; Tabelle 2 vor der Korrektur nennt 0,359), FITS 0,372 (Tabelle 1, arXiv v3), PatchTST/42 0,375 (Tabelle 3, arXiv v2; PatchTST/64 wäre 0,370), NLinear 0,374 (Tabelle 2 des LTSF-Linear-Papers, bereits per Fixture und Spaltenprüfung belegt, siehe `papers.yaml`).

**Bekannt aus `b1_candidates.md`, vor der Prüfung offengelegt:** Der SparseTSF-Stand `b8c2740eec` hat den Test-Loader bereits korrigiert (`drop_last = False`); die Zielwerte von FITS und PatchTST stammen aus der Zeit vor dem Fund des `drop_last`-Fehlers. Der Reviewer kennt diese Dateien aus der früheren Kandidatensuche, der Agenten-Code wurde nicht an ihnen entwickelt.

**Code-Stand der Prüfung:** `751be6b` (`main` nach dem Merge von `nachlauf-m3`, eingetragen am 2026-10-03 vor dem ersten Lauf der Prüfung). Ab hier ist `src/` eingefroren, bis alle vier Läufe fertig sind; spätere Commits bis zum Ende der Prüfung ändern nur Dokumente.

**Entwicklung bis zum Einfrieren (offengelegt).** An den beiden Entwicklungs-Einheiten gab es fünf Nachbesserungsrunden nach echten Läufen (Token Factory, zusammen 0,31 $ brutto, Store `runs/nachlauf_dev.jsonl`): Absturz bei falsch geformter Modellantwort; erfundener Startbefehl (Anlass dieses Nachtrags); Seed-Zeile falsch gewählt; unklarer Satz im Adapter-Vertrag zum Arbeitsverzeichnis; fehlerhafte eigene Schreiblogik des Modells für die Ergebnisdatei (jetzt feste Funktion im Vertrag). Mit dem eingefrorenen Stand (`283cdf2`, identisch in `751be6b`) laufen beide Einheiten gültig mit fünf Seeds: RLinear 0,3708 ± 0,0013 (4 LLM-Aufrufe), DLinear 0,3781 ± 0,0082 (3 LLM-Aufrufe), Adapter jeweils im ersten Versuch, P2 und P3 bei allen Seeds bestanden. Diese Ergebnisse sind Entwicklungsergebnisse und zählen nicht für die Erfolgsregel.

**Ablauf der Prüfung:** vier Läufe nacheinander in der Reihenfolge SparseTSF, FITS, PatchTST/42, NLinear, gestartet vom Reviewer aus dem Hauptverzeichnis, unbeaufsichtigt über Nacht (vom Nutzer freigegeben). Kein Eingriff, keine Wiederholung eines Laufs mit demselben Code. Bricht ein Lauf an der Infrastruktur ab (Sandbox- oder API-Ausfall ohne Urteil), wird genau dieser Lauf einmal mit unverändertem Code neu gestartet und der Vorfall berichtet.

## Nachtrag 2026-10-03 (zweiter Versuch, vor jeder Code-Änderung festgelegt, vom Nutzer freigegeben)

Die Prüfung hat Punkt 1 verfehlt (0 von 4, `results/nachlauf_v1.md`). Die Vorab-Festlegung erlaubt nach einer Code-Änderung **genau einen** weiteren Lauf auf denselben vier Einheiten. Der Nutzer hat ihn am 2026-10-03 um 00:5x Uhr freigegeben, unbeaufsichtigt über Nacht.

**Was sich gegenüber dem ersten Versuch ändert (nur diese drei Ursachen und ein Hilfsmittel):**
1. Ordnung der Befehls-Kandidaten nach Passung zum Claim (Modellname, Datensatz, Horizont und Eingabelänge nach Auflösung der Skript-Variablen), gruppiert nach Datei, mit der Möglichkeit für das Modell, weitere Kandidaten anzufordern.
2. Ältere Paketversion als erlaubte Reparatur, wenn ein Training in einer installierten Bibliothek abstürzt; jede solche Änderung steht als Abweichung der Umgebung im Dossier und im Zeugnis.
3. P1 strenger: Die Reparatur darf Schalter nicht ändern, die Datensatz, Art der Aufgabe, Zahl der Kanäle, Horizont oder Eingabelänge bestimmen; der Modellname darf nur auf den Namen aus dem Claim geändert werden.
4. Hilfsmittel: Der Fehlertext für die Reparatur enthält zusätzlich die letzte Zeile des Tracebacks, die im Repo liegt.

**Was gleich bleibt:** Einheiten, Commits, Zielwerte, Toleranz, Erfolgsregel (mit dem Nachtrag zur Gültigkeit), Prüfungen P2 bis P4, je eine Generierung, fünf Seeds, Kostenstopp 3 $ brutto (eigener Store `runs/nachlauf_v1_try2.jsonl`). Vor dem Lauf müssen beide Entwicklungs-Einheiten mit dem neuen Code weiter gültig laufen; sonst findet der zweite Versuch nicht statt.

**Ehrlichkeit:** Die Änderungen sind durch die Fehler an den vier Einheiten veranlasst; diese Einheiten sind nicht mehr unberührt. Der zweite Versuch wird immer als „zweiter Versuch" berichtet, getrennt vom ersten, und der erste bleibt das Ergebnis der Prüfung an unberührten Einheiten. Ein dritter Versuch ist ausgeschlossen. Der Code des zweiten Versuchs wird vor dem Lauf hier mit Commit nachgetragen.

**Code des zweiten Versuchs:** `4388a76` (`main` nach dem Merge von `nachlauf-try2`, Änderungen in `7296b36`), eingetragen am 2026-10-03 vor dem Lauf. Bedingung erfüllt: Beide Entwicklungs-Einheiten laufen mit diesem Code gültig, je fünf Seeds (RLinear 0,3708 ± 0,0013, 4 LLM-Aufrufe; DLinear 0,3781 ± 0,0082, 3 LLM-Aufrufe; zusammen 0,04 $ brutto; Store `runs/nachlauf_dev_try2.jsonl` im Worktree `research-try2`). Bekannt vor dem Lauf und **nicht** geändert, weil keine der vier festgelegten Änderungen: Befehle aus Skripten in Unterordnern werden im Stammverzeichnis des Repos gestartet.
