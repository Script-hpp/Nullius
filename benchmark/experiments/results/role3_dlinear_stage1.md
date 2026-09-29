# Ergebnis: Rolle 3, Stufe 1 (Einwände sammeln) — DLinear/ETTh1

Vorab festgelegt in `../role3_dlinear.md`. Rohausgabe: `role3_dlinear_stage1.txt`, gesammelte Seiten: `benchmark/fixtures/role3_zeng_objections.json`.
Ein Tavily-Aufruf (`collect_objections`: Search mit Rohtext, `limit=5`, plus Crawl der Issue-Seite des Repos, `limit=5`), Extraktion mit Rolle `extract` (Super), Abgleich mit dem Repo ohne LLM. Kosten LLM 0,008 $ netto.

## Vorab-Erfolgsregel Stufe 1: nicht erfüllt
„Mindestens ein verifizierter Einwand zum Auswertungsprotokoll (Stichwörter: drop_last, letzter Batch, Testset, Leakage, Hyperparameter-Suche auf Testdaten).“
Gefunden wurden 2 verifizierte Einwände (Zitat steht im Quelltext, Seite nennt DLinear/LTSF), **keiner mit einem dieser Stichwörter**:
1. „Comparison fairness“ (Hugging-Face-Blog zu Autoformer): ob Transformer und DLinear unter gleichen Feature-Bedingungen verglichen werden.
2. „input-window mismatch“ (alphaXiv): Die Vergleichsanordnung des Papers nutze 336 Schritte Eingabefenster für lineare Modelle, 96 für die Transformer.
Beide Repo-Abgleiche: „nicht prüfbar“ (kein `drop_last`-Bezug). **Stufe 2 (Test des Einwands mit der Referenz) wird nach der Vorab-Regel nicht ausgeführt.**

## Beobachtungen (post hoc, keine Regeländerung)
- **Ein Fehler in Rolle 3 entdeckt und behoben:** `collect_objections` rief `crawl(urls=..., limit=...)` auf; die Tavily-API kennt nur `url`, der Aufruf wäre mit `TypeError` gescheitert. Er lief vorher nie. Jetzt: Search plus Crawl der Issue-Seite (`url=`, `limit`, `chunks_per_source`, `instructions`).
- **Der Crawl fand Reproduktionsberichte, die die Vorab-Regel herausfilterte:** Issue #117 („etth1 96 48 96 mse = 0.426, not in paper 0.386“) und Issue #120 (ob MSE/MAE normalisiert oder rückskaliert berichtet werden). Die Seiten nennen „DLinear“/„LTSF“ nicht im Text und fielen damit durch die vorab festgelegte Namensregel (5 von 10 Seiten). Andere Issues betreffen Abstürze und Speicher, nicht das Ergebnis.
- Die Einwand-Frage nach unterschiedlichen Eingabefenstern ist im Paper selbst offen benannt; der Harness gibt den Agenten ohnehin frei, wie viele der 720 Schritte sie nutzen (Task-Beschreibung), es ist kein verdeckter Unterschied.
- Der Wert von Rolle 3 hier ist damit ein Negativbefund für `drop_last`, aber ein Beleg, dass Search plus Crawl echte Reproduktionsberichte findet.

## Einschränkungen
Ein Paper, ein Aufruf, Suchergebnisse nicht deterministisch. Die Namensregel ist streng; ohne sie hätte Extraktion Issue #117 und #120 geprüft (nicht ausgeführt, weil vorab festgelegt).
