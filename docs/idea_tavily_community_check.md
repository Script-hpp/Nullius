# Idee: Tavily-Rollen 3/7 und 6 in den Produktlauf einbauen

Stand: 2026-10-02. Status: **gebaut auf Branch `community-check`, nicht gemerged** (erst nach Benchmark v4). Umsetzung: `src/repro_agent/community.py`, Einbindung `Orchestrator._community_round` (nach allen Urteilen), Dossier-Felder `community` und `explanations`, Zeugnis-Abschnitte, Karte `web/src/components/CommunityCard.tsx` direkt nach "Claimed vs. Measured" und vor den Tabs, Live-Phase "What others found". Tests nur mit Fakes (`tests/test_community.py`); der erste echte Tavily-Aufruf steht aus (Reviewer).

Entscheidungen (Reviewer, 2026-10-02): Karte vor dem Branch-Baum; Crawl der Repo-Issues nur, wenn das Repo bekannt ist und die Suche keine Issue-Seite dieses Repos lieferte (limit 3); keine Sprachfilter; Ergebnis je arXiv-ID 7 Tage in `logs/community_cache.json`; Discovery-Community-Treffer (`candidate.community` im Katalog) ersetzen die Suche (ein Extract statt einer Search); Etiketten deterministisch (Protokollhinweis, Reproduktionsbericht, sonstiges), keine LLM-Einstufung; Rolle-6-Erklärungen nie in Plan, Code oder Urteil.

## Ausgangslage

Der Produktlauf ruft Tavily heute für Rolle 2 (Paper-Text, Zielwert), Rolle 4 (README, Setup-Details) und in Discover (Suche nach Papers, `curate_discussion` in `src/repro_agent/discovery.py:810`) auf. Die Werkzeuge für die Rollen 3, 5, 6 und 7 existieren in `src/repro_agent/tavily_tools.py` (`collect_objections`, `resolve_error`, `clarify_discrepancy`, `cite_sources`), werden vom Loop (`src/repro_agent/loop.py`) aber nicht aufgerufen. Rolle 3 wurde einmal als Experiment ausgeführt (`scripts/experiment_role3_stage1.py`, Leser `src/repro_agent/objections.py`).

## Grundsatz für beide Erweiterungen

**Nur berichten, nie das Urteil ändern.** Achse 1 (Zahl) bleibt allein Sache von Harness und Messung. Damit bleiben die Zahlen aus Benchmark v2 für das Produkt gültig, und "Worte vs. Messung" wird in der Evidence-Ansicht sichtbar: links, was andere sagen, rechts, was wir gemessen haben.

## 1. "What others found" (Rollen 3 und 7)

- **Wann:** einmal pro Lauf, nach dem Urteil (Ergebnis darf das Urteil nicht beeinflussen, auch nicht über den Plan).
- **Aufruf:** Tavily Search, Domains `github.com`, `openreview.net`, `arxiv.org`, Suchbegriff Paper-Titel plus "reproduce / reproduction / issue"; Crawl nur mit `limit`, höchstens 5 Quellen. Research höchstens 1× pro Paper (CLAUDE.md, Rolle 7), hier nicht nötig.
- **Auswertung:** vorhandener Leser `objections.py`: jede Aussage braucht ein wörtliches Zitat, das im Quelltext steht, und der Quelltext muss das Paper nennen. Sonst verworfen. Deterministischer Repo-Abgleich wie dort (`repo_check`).
- **Ausgabe:** neues Ereignis in `events.jsonl` (z. B. `community`), Liste im Dossier mit URL, Zitat, Modus `live`/Fixture, Zeitstempel, SHA-256 (wie `Dossier.sources`). Evidence-Ansicht: Karte "What others found" in der linken Spalte, je Fund Quelle, Zitat, Einordnung (bestätigt / widerspricht / Protokollhinweis). Keine Funde: "No public discussion found" statt leerer Karte.
- **Kosten:** 1 Search (+ ggf. 1 Extract) Tavily pro Lauf, ein kleiner `extract`-Aufruf (Super) pro Quelle.
- **Benchmark:** läuft dort nicht (Fixture-Modus, würde Tavily-Kontingent verbrauchen); im README als reine Berichtsfunktion ausweisen.

## 2. Abweichung erklären (Rolle 6)

- **Wann:** nur wenn Achse 1 "teilweise" oder "nicht reproduziert" ist (nicht bei Setup-Fehlern oder "Agenten-Fehler vermutet").
- **Aufruf:** `clarify_discrepancy`: Search nach neuerer arXiv-Version, Errata, Issues zum Zielwert. Ergänzend: Vergleich der arXiv-Versionen (steht der Zielwert in vN noch?).
- **Ausgabe:** Abschnitt "Possible explanations" im Zeugnis, jede Erklärung mit Zitat und Quelle, ausdrücklich als ungeprüfte Hypothese markiert. **Keine neuen Branches** in dieser Stufe (das wäre der volle Loop aus CLAUDE.md, "jede Erklärung wird neue Hypothese mit eigenem Branch", später).
- **Kosten:** höchstens 1 Search pro abweichendem Claim.

## Offene Fragen

- Reihenfolge in der UI: Karte vor oder nach dem Branch-Baum?
- Tavily-Kontingent in der Live-Demo: Tageslimit der API (`MAX_DAILY_REPRO_JOBS`) deckt das mit ab; reicht das Kontingent bis 15.12.?
- Sprache der Funde: nur Englisch zulassen?

## Aufwand (Schätzung, ungeprüft)

Loop-Ereignis + Dossier-Feld + Tests mit Fakes, UI-Karte, einmaliger echter Lauf zur Abnahme. Vorlage für Tests: `tests/test_loop_fakes.py`.
