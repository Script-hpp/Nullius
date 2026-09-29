# Tavily v2, Baustein 1: Zitatprüfung für PDF-Text (Rolle 4)

Entwicklungsmenge (GCN, SGC, Shchur, DLinear; **nicht** GAT oder DGI). Vorher/Nachher auf **denselben Rohangaben des Modells** (einmal mit LLM gesammelt, `benchmark/fixtures/role4_raw_<paper>.json`, Kosten 0,087 $ netto), danach deterministisch ohne LLM: `scripts/measure_role4_drops.py evaluate alt|neu`.

## Ergebnis
| | Rohangaben | behalten | verworfen | davon Zitat nicht im Quelltext | Zahl des Werts nicht im Zitat | kein Objekt (Modellausgabe fehlerhaft) |
|---|---|---|---|---|---|---|
| alt (Stand v1) | 196 | 166 | 30 (15 %) | **21** | – | 9 |
| neu | 196 | 175 | 18 (9 %) | **6** | 3 | 9 |

Je Paper (verworfen alt → neu): GCN 12 → 7, SGC 7 → 0, Shchur 1 → 1, DLinear 10 → 10.

## Was geändert wurde (`claims_source._norm`, `setup_details.quote_in_text/numbers_in_quote`)
- Schreibweise der Zitate geglättet: NFKC (Ligaturen), nicht trennende Bindestriche und typografische Striche/Anführungszeichen (das Modell schrieb `two‑layer` mit U+2011, die Quelle `two-layer`), weiche Trennstriche, Leerraum vor Satzzeichen (Quelle `transductively .`, Zitat `transductively.`).
- Auslassungspunkte im Zitat: am Rand ignoriert, im Inneren zerlegen sie das Zitat in Stücke, die der Reihe nach vorkommen müssen (jedes mindestens 10 Zeichen).
- **Zahlen und Bezeichner bleiben exakt** (Tests: `0.05` gegen `0.5`, `kernel_size=5` gegen `25`, `lr=0.001` gegen `0.01`, `5e-4` gegen `5e-5`).
- Neu und strenger: Jede Zahl im Wert muss als ganze Zahl im Zitat stehen (`0.005` ist nicht durch `0.05` belegt). Das verwarf drei Angaben, die Variablen statt Zahlen zitierten (`--seq_len $seq_len` für den Wert 336).

## Kontrolle auf falsche Annahmen
Von den 9 neu akzeptierten und 3 zusätzlich verworfenen Angaben habe ich alle einzeln angesehen: Die neu akzeptierten Zitate sind wörtlich bis auf Bindestrich, Leerraum, Auslassungspunkte oder Zeilenumbruch; keine Zahl weicht ab. Es rutschte nichts Falsches durch, soweit die 196 Angaben das zeigen (kleine Stichprobe).

## Was bleibt verworfen (zu Recht)
- Das Modell schreibt Formeln als LaTeX um (`\hat{A} = \tilde{D}^{-1/2} ...`), die Quelle hat Unicode-Formeltext: nicht wörtlich.
- Code-Zitate mit vom Modell eingefügten Escape-Zeichen oder Auslassungen um sehr kurze Stücke.
- 9 Antworten des Modells, die gar kein Objekt sind (`None`): Ausgabefehler, kein Zitatproblem.

## Grenzen
- Nur 196 Rohangaben aus 4 Papers (davon 3 PDF-Texte); die Verwerfungsquote von GAT (17 von 41, ein Paper) habe ich **nicht** nachgemessen, um GAT als Prüffall nicht zu benutzen.
- DLinear verwirft weiter 10, hauptsächlich Code-Zitate und `None`-Antworten, kein PDF-Problem.
- Die gleiche `_norm` gilt jetzt auch für die Zielwert-Prüfung (Rolle 2) und die Zitate der Bausteinliste; die Tests dazu laufen (138 grün).
