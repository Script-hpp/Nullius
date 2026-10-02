# Benchmark v4 (96 Urteile aus 48 Läufen)

Vorab-Festlegung: `../benchmark_v2.md`. Rohdaten: `runs/benchmark_v4.jsonl`. Alle Urteile vorläufig: **method fidelity not checked** (Achse 2). Die Zahlen messen „Zielwert laut Harness erreicht“, nicht „Methode reproduziert“. Keine Signifikanzaussagen.

## Primär: Mehrheit je Paper-Claim-Einheit (8 Einheiten, je 3 Generierungen)

| Arm | Echte Claims: stimmt mit Referenzurteil überein | Mehrheit Ausfall | keine Mehrheit | unvollständig | Placebos als „nicht reproduziert“ erkannt |
|---|---|---|---|---|---|
| A_ohne_tavily | 1 von 8 | 0 | 0 | 0 | 8 von 8 |
| B_mit_tavily | 2 von 8 | 0 | 0 | 0 | 8 von 8 |

## Je Einheit (Urteile der Generierungen 0/1/2, Mehrheit, Mittelwerte der Messung)

| Einheit | Claim | Referenzurteil (Mittel, Herkunft) | Arm | Urteile | Mehrheit | Übereinstimmung | Mittel je Generierung | Placebo: Mehrheit |
|---|---|---|---|---|---|---|---|---|
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | A_ohne_tavily | nicht / nicht / teil | nicht reproduziert | nein | 0.7365 / 0.7476 / 0.7987 | nicht reproduziert |
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | B_mit_tavily | rep / rep / rep | reproduziert | ja | 0.8194 / 0.8170 / 0.8178 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | A_ohne_tavily | teil / nicht / teil | teilweise | nein | 0.8215 / 0.8530 / 0.7851 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | B_mit_tavily | nicht / teil / nicht | nicht reproduziert | nein | 0.7489 / 0.7970 / 0.7336 | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | A_ohne_tavily | teil / rep / rep | reproduziert | ja | 0.3863 / 0.3834 / 0.3840 | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | B_mit_tavily | teil / teil / Ausfall (nicht reproduzierbar (Setup)) | teilweise | nein | 0.3866 / 0.3930 / – | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | A_ohne_tavily | teil / teil / nicht | teilweise | nein | 0.7957 / 0.7942 / 0.7849 | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | B_mit_tavily | teil / rep / rep | reproduziert | ja | 0.7984 / 0.8086 / 0.8092 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | A_ohne_tavily | teil / teil / nicht | teilweise | nein | 0.8122 / 0.8022 / 0.7801 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | B_mit_tavily | teil / teil / teil | teilweise | nein | 0.8148 / 0.8440 / 0.8014 | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | A_ohne_tavily | nicht / nicht / nicht | nicht reproduziert | nein | 0.6722 / 0.7506 / 0.7811 | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | B_mit_tavily | nicht / nicht / nicht | nicht reproduziert | nein | 0.6897 / 0.7754 / 0.6898 | nicht reproduziert |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | A_ohne_tavily | teil / nicht / teil | teilweise | nein | 0.7917 / 0.7778 / 0.7913 | nicht reproduziert |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | B_mit_tavily | teil / teil / nicht | teilweise | nein | 0.8043 / 0.7938 / 0.7062 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | A_ohne_tavily | nicht / nicht / nicht | nicht reproduziert | nein | 0.8423 / 0.7689 / 0.8385 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | B_mit_tavily | teil / nicht / nicht | nicht reproduziert | nein | 0.3854 / 0.4304 / 0.4595 | nicht reproduziert |

## Sekundär: Rohzählung (einzelne Urteile)

| Claim-Art | Arm | Übereinstimmungen mit Referenzurteil | Ausfälle | „nah“ (nur echt) | Urteilsverteilung |
|---|---|---|---|---|---|
| echt | A_ohne_tavily | 2 von 24 | 0 | 12 von 24 | nicht reproduziert 12, teilweise 10, reproduziert 2 |
| echt | B_mit_tavily | 5 von 24 | 1 | 15 von 24 | teilweise 10, nicht reproduziert 8, reproduziert 5, nicht reproduzierbar (Setup) 1 |
| Placebo | A_ohne_tavily | 23 von 24 | 1 | – | nicht reproduziert 23, Agenten-Fehler vermutet 1 |
| Placebo | B_mit_tavily | 24 von 24 | 0 | – | nicht reproduziert 24 |

Median der Abweichung vom Zielwert (gültige echte Generierungen, post hoc): A_ohne_tavily 0.0290 (n=24), B_mit_tavily 0.0166 (n=23)

## Ausfälle nach Ursache

- A_ohne_tavily: {'Agenten-Fehler vermutet': 1}
- B_mit_tavily: {'nicht reproduzierbar (Setup)': 1}

## Graph-Ablation und Zielwert-Prüfung

- A_ohne_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {'Graph genutzt': 15, 'nicht messbar': 3}
- B_mit_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {'nicht messbar': 3, 'Graph genutzt': 15}
- Belegter Zielwert weicht von `papers.yaml` ab: 0 Fälle

## Kosten und Code-Stand

- A_ohne_tavily: 24 Läufe, 1.686 $ netto, 2.007 $ brutto
- B_mit_tavily: 24 Läufe, 4.851 $ netto, 5.772 $ brutto

| Stand (`code_commit`, `config`) | Urteile | Übereinstimmungen echt | Ausfälle |
|---|---|---|---|
| 2b13b31 / v4_block_a | 22 | 0 von 11 | 1 |
| bf520c1 / v4_block_a | 74 | 7 von 37 | 1 |
