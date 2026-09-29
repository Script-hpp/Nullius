# Benchmark: Infrastruktur-Abbrüche und Code-Stände

Regel: `../benchmark_protocol.md` (Nachtrag „Infrastruktur-Abbrüche und Code-Stände“). Ein Infrastruktur-Abbruch ist kein Ausfall des Agenten; die abgebrochene Generierung wird vollständig neu gestartet. Halbe Läufe schreiben keine Zeilen in `runs/benchmark.jsonl`.

## Abbrüche (alle: `contree_sdk.sdk.exceptions.api.ApiTimeoutError`, Lese-Timeout beim Statusabfragen einer Sandbox-Operation, SDK 0.3.6)
| # | Prozess | Abbruch (CEST, 29.09.2026) | Betroffene Generierung (vollständig neu erzeugt) | Code-Stand | Bemerkung |
|---|---|---|---|---|---|
| 1 | Shchur | ca. 13:44 | Shchur Arm B, Generierung 0 | `c120911`, Timeout 10 s | `response=None`; Laptop war zeitweise aus |
| 2 | GCN/DLinear/SGC | ca. 13:48 | GCN Arm B, Generierung 1 | `c120911`, Timeout 10 s | `response=None`; Laptop war zeitweise aus |
| 3 | Shchur (Neustart) | ca. 14:16 | Shchur Arm B, Generierung 0 | `4de8caf`, Timeout 10 s | `response=None`; Laptop wach |
| 4 | GCN/DLinear/SGC (Neustart) | ca. 14:56 | DLinear Arm B, Generierung 1 | `4de8caf`, Timeout 10 s | Antwort mit Headern (`chunked`, `gzip`), Body kam nicht in 10 s |

Zeitpunkte aus dem Änderungszeitpunkt der Prozess-Logs (Minutengenauigkeit).
Ursache (nicht bewiesen): `transport_timeout` des SDK ist 10 s; unter Last antwortet die Beta-API teils nicht rechtzeitig. Eine erneute Anfrage auf derselben SDK-Session geht nicht (Session `FAILED`, echter Test).

## Maßnahmen
- Commit `8661d9f`: `transport_timeout` 60 s; ein Versuch wird bei `SandboxUnavailable` bis zu 3× auf einem neuen Branch wiederholt (gleicher Code, gleicher Seed, nichts neu generiert); `MAX_PARALLEL_SEEDS` konfigurierbar (Standard 4, für die Neustarts 3); `code_commit` je Zeile.
- Shchur seit Neustart 3 (`8661d9f`, Parallelität 3), GCN/DLinear/SGC seit Neustart 5 (`f2ae2f7`, Parallelität 3).

## Code-Stand je Zeile in `runs/benchmark.jsonl`
- **Zeilen ohne Feld `code_commit`:** Altstand (Timeout 10 s, kein Wiederholen), Commits `c120911` und `4de8caf`.
- **Zeilen mit Feld `code_commit`:** Neustand (`8661d9f` beziehungsweise `f2ae2f7`, gleicher Infrastrukturcode).
- Die Codestände unterscheiden sich **nur im Umgang mit Sandbox-Timeouts**, nicht in Prompts, Harness oder Messung. Die Zeilen werden deshalb gemeinsam ausgewertet; die Auswertung weist die Aufteilung aus.
