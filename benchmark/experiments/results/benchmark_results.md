# Benchmark-Ergebnis (48 Urteile aus 24 Läufen)

Protokoll: `../benchmark_protocol.md`. Rohdaten: `benchmark.jsonl`. Alle Urteile vorläufig (Achse 2 nicht geprüft).

## Kernzahlen (Übereinstimmung mit dem Referenzurteil)

| | Arm | Übereinstimmungen | Ausfälle | „nah“ (nur echte Claims) | Urteilsverteilung |
|---|---|---|---|---|---|
| echte Claims | A_ohne_tavily | 3 von 12 | 0 | 10 von 12 | teilweise 7, reproduziert 3, nicht reproduziert 2 |
| echte Claims | B_mit_tavily | 6 von 12 | 1 | 10 von 12 | reproduziert 6, teilweise 4, nicht reproduziert 1, nicht reproduzierbar (Setup) 1 |
| Placebo-Claims | A_ohne_tavily | 12 von 12 | 0 | – | nicht reproduziert 12 |
| Placebo-Claims | B_mit_tavily | 12 von 12 | 0 | – | nicht reproduziert 12 |

## Je Paper (Übereinstimmungen von 3 Generierungen; Mittelwerte der Messung)

| Paper | Claim | Referenzurteil | Arm A | Arm B | Mittel A (Gen. 0/1/2) | Mittel B (Gen. 0/1/2) |
|---|---|---|---|---|---|---|
| kipf_gcn_2017 | gcn_cora_acc | reproduziert | 0/3 | 0/3 | 0.5082 / 0.7974 / 0.7897 | 0.8042 / 0.8046 / 0.8020 |
| kipf_gcn_2017 | gcn_cora_acc_placebo | nicht reproduziert | 3/3 | 3/3 | 0.8068 / 0.7964 / 0.7900 | 0.8011 / 0.8151 / 0.8071 |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert | 0/3 | 2/3 | 0.7994 / 0.8013 / 0.8021 | 0.8097 / 0.8053 / 0.8044 |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc_placebo | nicht reproduziert | 3/3 | 3/3 | 0.7995 / 0.8025 / 0.8000 | 0.8091 / 0.8097 / 0.8081 |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert | 1/3 | 1/3 | 0.3823 / 0.3989 / 0.4268 | 0.8640 / – / 0.3834 |
| zeng_dlinear_2023 | dlinear_etth1_96_mse_placebo | nicht reproduziert | 3/3 | 3/3 | 0.3797 / 0.6070 / 0.3840 | 0.8640 / 0.4440 / 0.4788 |
| wu_sgc_2019 | sgc_cora_acc | reproduziert | 2/3 | 3/3 | 0.8116 / 0.8190 / 0.8280 | 0.8017 / 0.8153 / 0.8146 |
| wu_sgc_2019 | sgc_cora_acc_placebo | nicht reproduziert | 3/3 | 3/3 | 0.8060 / 0.8386 / 0.8060 | 0.8285 / 0.7910 / 0.8124 |

## Graph-Ablation und Zielwert-Prüfung

- A_ohne_tavily: Graph-Ablation {'Graph genutzt': 17, 'nicht messbar': 1}
- B_mit_tavily: Graph-Ablation {'Graph genutzt': 17, 'unklar': 1}
- Belegter Zielwert weicht von `papers.yaml` ab: 0 Fälle

## Code-Stand je Zeile (siehe `benchmark_incidents.md`)

| Stand | Zeilen (Urteile) | Übereinstimmungen echt | Übereinstimmungen Placebo | Ausfälle |
|---|---|---|---|---|
| 8661d9f | 10 | 2 von 5 | 5 von 5 | 0 |
| Altstand (ohne Feld, Timeout 10 s) | 20 | 1 von 10 | 10 von 10 | 0 |
| f2ae2f7 | 18 | 6 von 9 | 9 von 9 | 1 |

## Aufwand: brauchen Placebo-Läufe auffällig mehr Versuche oder Tuning?

| Claim-Art | Arm | Versuche je Claim (Median) | Code-Versionen (Median) | Reparaturen gesamt | Code mit Tuning-Stichwort | Codelänge (Median, Zeilen) |
|---|---|---|---|---|---|---|
| paper | A_ohne_tavily | 10.0 | 1.0 | 6 | 11 von 12 | 146.0 |
| paper | B_mit_tavily | 11.0 | 1.5 | 9 | 12 von 12 | 158.0 |
| placebo | A_ohne_tavily | 10.5 | 1.0 | 7 | 11 von 12 | 147.0 |
| placebo | B_mit_tavily | 10.0 | 1.0 | 5 | 11 von 12 | 162.0 |
