# Benchmark v3 (96 Urteile aus 48 Läufen)

Vorab-Festlegung: `../benchmark_v2.md`. Rohdaten: `runs/benchmark_v2.jsonl`. Alle Urteile vorläufig: **method fidelity not checked** (Achse 2). Die Zahlen messen „Zielwert laut Harness erreicht“, nicht „Methode reproduziert“. Keine Signifikanzaussagen.

## Primär: Mehrheit je Paper-Claim-Einheit (8 Einheiten, je 3 Generierungen)

| Arm | Echte Claims: stimmt mit Referenzurteil überein | Mehrheit Ausfall | keine Mehrheit | unvollständig | Placebos als „nicht reproduziert“ erkannt |
|---|---|---|---|---|---|
| A_ohne_tavily | 2 von 8 | 1 | 1 | 0 | 8 von 8 |
| B_mit_tavily | 2 von 8 | 0 | 3 | 0 | 6 von 8 |

## Je Einheit (Urteile der Generierungen 0/1/2, Mehrheit, Mittelwerte der Messung)

| Einheit | Claim | Referenzurteil (Mittel, Herkunft) | Arm | Urteile | Mehrheit | Übereinstimmung | Mittel je Generierung | Placebo: Mehrheit |
|---|---|---|---|---|---|---|---|---|
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | A_ohne_tavily | rep / Ausfall (Agenten-Fehler vermutet) / rep | reproduziert | ja | 0.8059 / 0.1745 / 0.8146 | nicht reproduziert |
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | B_mit_tavily | rep / rep / rep | reproduziert | ja | 0.8151 / 0.8180 / 0.8191 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | A_ohne_tavily | rep / rep / rep | reproduziert | ja | 0.8184 / 0.8060 / 0.8050 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | B_mit_tavily | nicht / nicht / nicht | nicht reproduziert | nein | 0.7269 / 0.7800 / 0.7204 | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | A_ohne_tavily | nicht / teil / Ausfall (nicht reproduzierbar (Setup)) | keine Mehrheit | nein | 0.7549 / 0.3928 / – | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | B_mit_tavily | nicht / nicht / rep | nicht reproduziert | nein | 0.4293 / 0.4257 / 0.3764 | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | A_ohne_tavily | teil / teil / teil | teilweise | nein | 0.7955 / 0.7954 / 0.7990 | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | B_mit_tavily | rep / Ausfall (Agenten-Fehler vermutet) / nicht | keine Mehrheit | nein | 0.8086 / 0.1729 / 0.7629 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | A_ohne_tavily | Ausfall (nicht reproduzierbar (Setup)) / Ausfall (nicht reproduzierbar (Setup)) / nicht | Ausfall | nein | – / – / 0.7999 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | B_mit_tavily | nicht / teil / Ausfall (nicht reproduzierbar (Setup)) | keine Mehrheit | nein | 0.7802 / 0.8033 / – | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | A_ohne_tavily | nicht / Ausfall (nicht reproduzierbar (Setup)) / nicht | nicht reproduziert | nein | 0.7083 / – / 0.7449 | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | B_mit_tavily | nicht / Ausfall (nicht reproduzierbar (Setup)) / rep | keine Mehrheit | nein | 0.7892 / – / 0.8328 | Ausfall |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | A_ohne_tavily | nicht / Ausfall (nicht reproduzierbar (Setup)) / nicht | nicht reproduziert | nein | 0.7796 / – / 0.7871 | nicht reproduziert |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | B_mit_tavily | nicht / teil / nicht | nicht reproduziert | nein | 0.7153 / 0.7885 / 0.7562 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | A_ohne_tavily | nicht / teil / nicht | nicht reproduziert | nein | 0.4446 / 0.3888 / 0.8189 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | B_mit_tavily | rep / teil / rep | reproduziert | ja | 0.3808 / 0.3918 / 0.3805 | Ausfall |

## Sekundär: Rohzählung (einzelne Urteile)

| Claim-Art | Arm | Übereinstimmungen mit Referenzurteil | Ausfälle | „nah“ (nur echt) | Urteilsverteilung |
|---|---|---|---|---|---|
| echt | A_ohne_tavily | 5 von 24 | 6 | 10 von 24 | nicht reproduziert 8, reproduziert 5, teilweise 5, nicht reproduzierbar (Setup) 5, Agenten-Fehler vermutet 1 |
| echt | B_mit_tavily | 7 von 24 | 3 | 11 von 24 | nicht reproduziert 10, reproduziert 8, teilweise 3, nicht reproduzierbar (Setup) 2, Agenten-Fehler vermutet 1 |
| Placebo | A_ohne_tavily | 20 von 24 | 4 | – | nicht reproduziert 20, Agenten-Fehler vermutet 2, nicht reproduzierbar (Setup) 2 |
| Placebo | B_mit_tavily | 15 von 24 | 9 | – | nicht reproduziert 15, Agenten-Fehler vermutet 5, nicht reproduzierbar (Setup) 4 |

Median der Abweichung vom Zielwert (gültige echte Generierungen, post hoc): A_ohne_tavily 0.0196 (n=19), B_mit_tavily 0.0297 (n=22)

## Ausfälle nach Ursache

- A_ohne_tavily: {'nicht reproduzierbar (Setup)': 7, 'Agenten-Fehler vermutet': 3}
- B_mit_tavily: {'Agenten-Fehler vermutet': 6, 'nicht reproduzierbar (Setup)': 6}

## Graph-Ablation und Zielwert-Prüfung

- A_ohne_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {'Graph genutzt': 9, 'nicht messbar': 5, None: 4}
- B_mit_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {'Graph genutzt': 13, 'Graph nicht genutzt': 1, 'nicht messbar': 2, None: 2}
- Belegter Zielwert weicht von `papers.yaml` ab: 0 Fälle

## Kosten und Code-Stand

- A_ohne_tavily: 24 Läufe, 0.662 $ netto, 0.788 $ brutto
- B_mit_tavily: 24 Läufe, 1.942 $ netto, 2.311 $ brutto

| Stand (`code_commit`, `config`) | Urteile | Übereinstimmungen echt | Ausfälle |
|---|---|---|---|
| 4bab96e / v3_fixes_role5 | 96 | 12 von 48 | 22 |
