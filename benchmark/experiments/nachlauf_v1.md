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
