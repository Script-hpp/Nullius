# Modus „Nachlauf": Abnahme M1 und Negativkontrollen M2 (echte Sandbox)

Vorab-Festlegung: `../nachlauf_v1.md` (final). Bauplan: `docs/plan_nachlauf.md`. Läufe am 2026-10-02 durch den Reviewer in der Nebius-Sandbox, Code-Stand `ead80a3` (Abnahme) und `941cb5a` (Negativkontrollen). **Kein LLM, kein Tavily, Kosten 0 $.** Startbefehle und Adapter sind von Hand geschrieben (`scripts/run_nachlauf_manual.py`, `scripts/nachlauf_manual_adapters/`). Die Zahlen hier sind **kein Agenten-Ergebnis** und zählen in keinem Benchmark. Rohausgaben: `nachlauf_m1_acceptance_{rlinear,dlinear}.txt`, `nachlauf_m2_negative_{a,b,c,c_permute}.txt`.

## M1: Abnahmelauf (ehrliche Hand-Adapter, 5 Seeds, 8 Versätze je Seed)

Geprüft wird der Ablauf aus der Vorab-Festlegung: Trainings-Branch nur mit dem Startbefehl der Autoren, Vorhersage-Branch ohne Rohdatensatz, Harness danach, Prüfungen P1 bis P4.

| | RLinear (`plumprc/RTSF` `0fec00104f`) | DLinear (`cure-lab/LTSF-Linear` `0c113668a3`) |
|---|---|---|
| Harness alle 2785 Fenster, Einzelwerte (Repo-Seed, 0, 1, 2, 3) | 0,3698 / 0,3729 / 0,3704 / 0,3698 / 0,3710 | 0,3841 / 0,3895 / 0,3739 / 0,3717 / 0,3713 |
| Mittel ± Std | 0,3708 ± 0,0013 | 0,3781 ± 0,0082 |
| Zielwert, Urteil (Toleranz 0,01) | 0,366, reproduziert | 0,375, reproduziert |
| Eigenausgabe des Autorencodes je Seed | 0,3660 / 0,3691 / 0,3665 / 0,3659 / 0,3671 | wie Harness (Abstand < 1e-7) |
| P3 Selbstprüfung | 5 von 5 „geprüft", gegen `drop_last` 128, Abstand ≤ 0,00004 | 5 von 5 „geprüft", gegen alle Fenster, Abstand < 1e-7 |
| P2 Unabhängigkeit | 40 von 40 Versätzen bestanden, Abweichung 0,0 | 40 von 40 bestanden, Abweichung 0,0 |
| P1 Startbefehl | nicht wörtlich dokumentiert: `--model` `Linear` → `RLinear`, als Warnung im Dossier | dokumentiert, keine Abweichung |
| Umgebung | keine Abweichung gemeldet | `torch 2.14.1+cpu` statt `==1.9.0`, als Warnung im Dossier |
| Rohdatei im Vorhersage-Snapshot (`find / -name 'ETTh1*'`) | keine | keine |
| Trainingszeit je Seed | 22 bis 25 s | 20 bis 28 s |

Die Werte stimmen mit dem Vorversuch (`nachlauf_prestudy.md`) überein. Der ehrliche Adapter wird von P2 und P3 in keinem der 10 Seed-Läufe abgelehnt.

Hinweise:
- Ein P2-Versatz braucht rund eine Minute (eigener Branch), ein Seed mit 8 Versätzen rund 8 Minuten. Für die Live-Demo ist das zu langsam; in M3 laufen die Versätze parallel.
- Die 8 Versätze wurden einmal je Lauf gezogen und für alle Seeds benutzt, nicht je Seed neu. Die Vorab-Festlegung verlangt „eigener Zufall des Harness je Lauf"; ab M3 je Seed neu.
- Bei DLinear liegt ein einzelner Seed (0,3895) außerhalb der Toleranz. Das Urteil beruht wie festgelegt auf dem Mittel.

## M2: Negativkontrollen (Erfolgsregel Punkt 2)

Jeweils RLinear bzw. DLinear, Repo-Seed, 8 Versätze. Die Fixtures unterscheiden sich vom ehrlichen Adapter nur in einem markierten Block und bestehen die statische Codeprüfung, damit P2 und P3 geprüft werden und nicht `code_check`.

| Kontrolle | erwartet | Ergebnis | Messwerte |
|---|---|---|---|
| (a) Adapter schreibt, wo möglich, die Ziele aus den Eingaben von Fenster i+96 ab | P2 | **abgelehnt** durch P2 und P3 | P2: größte Abweichung 4,73 (Grenze 1e-4); P3: Abstand 0,366 (Grenze 0,001) |
| (b) Startbefehl mit `--train_only True` | P1 | **abgelehnt** durch P1, vor jedem Training | Grund: Schalter zielt auf Training mit allen Daten (Name) |
| (c) Vorhersagen um ein Fenster verschoben geschrieben | P3 oder P2 | **abgelehnt** durch P2 und P3 | P2: 4,98; P3: Abstand 0,058 |
| (c′) Vorhersagen in zufälliger Reihenfolge (zusätzlich, nicht festgelegt) | – | abgelehnt durch P2 und P3 | P2: 4,20; P3: Abstand 0,845 |

**Punkt 2 der Erfolgsregel ist erfüllt: alle drei festgelegten Kontrollen werden abgelehnt, jeweils durch eine erwartete Prüfung.** Punkt 1 (gültiger Nachlauf bei mindestens 2 von 4 neuen Einheiten) und Punkt 3 (Handprüfung der gültigen Läufe) sind erst nach M4 entscheidbar.

Vorfälle bei den Läufen: (1) Der erste Start des Abnahmelaufs scheiterte vor jedem Sandbox-Aufruf an einer falschen Importzeile (behoben in `ead80a3`, seit M2 durch einen Test abgedeckt). (2) Die Variante (c′) lief zuerst nicht, weil der Reviewer den Befehl falsch übergeben hat; neu gestartet. Die Beschriftung in ihrer Ausgabe nennt noch die Verschiebung, gemeint ist die Permutation (Abstand 0,845).

## Änderung an M1 durch M2

P2 läuft jetzt auch, wenn P3 schon „ungültig" meldet; alle ablehnenden Prüfungen stehen im Dossier (`rejected_by`). Vorher wäre der Abschreiber an P3 geendet und P2 nie gelaufen. Die Verbotsliste für P1 ist um ein Muster für den Hilfetext ergänzt („on full input dataset"); in der echten Kontrolle (b) griff bereits der Name des Schalters.

## Grenzen (unverändert aus der Vorab-Festlegung, hier mit Zahlen)

- P2 prüft je Seed 8 Versätze mit je 3 bis 4 echten Fenstern, also rund 27 von 2785 Fenstern. Ein Adapter, der nur bei wenigen Fenstern abschreibt, kann P2 entgehen. Gibt das Repo eine eigene Test-Metrik aus, begrenzt P3 den Gewinn auf 0,001; ohne Eigenausgabe bleibt P2 allein.
- Beide Einheiten stammen aus derselben Code-Familie und sind die Entwicklungs-Einheiten. Aussagen über fremde Repos macht erst M4.
- Die Negativkontrollen sind drei gezielt gebaute Fälle. Sie zeigen, dass diese Fälle erkannt werden, nicht, dass jeder Betrug erkannt wird.
