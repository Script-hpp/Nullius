# Benchmark v2 (96 Urteile aus 48 Läufen)

Vorab-Festlegung: `../benchmark_v2.md`. Rohdaten: `runs/benchmark_v2.jsonl`. Alle Urteile vorläufig: **method fidelity not checked** (Achse 2). Die Zahlen messen „Zielwert laut Harness erreicht“, nicht „Methode reproduziert“. Keine Signifikanzaussagen.

## Primär: Mehrheit je Paper-Claim-Einheit (8 Einheiten, je 3 Generierungen)

| Arm | Echte Claims: stimmt mit Referenzurteil überein | Mehrheit Ausfall | keine Mehrheit | unvollständig | Placebos als „nicht reproduziert“ erkannt |
|---|---|---|---|---|---|
| A_ohne_tavily | 2 von 8 | 1 | 2 | 0 | 6 von 8 |
| B_mit_tavily | 2 von 8 | 1 | 4 | 0 | 7 von 8 |

## Je Einheit (Urteile der Generierungen 0/1/2, Mehrheit, Mittelwerte der Messung)

| Einheit | Claim | Referenzurteil (Mittel, Herkunft) | Arm | Urteile | Mehrheit | Übereinstimmung | Mittel je Generierung | Placebo: Mehrheit |
|---|---|---|---|---|---|---|---|---|
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | A_ohne_tavily | teil / Ausfall (nicht reproduzierbar (Setup)) / rep | keine Mehrheit | nein | 0.7897 / – / 0.8070 | Ausfall |
| kipf_gcn_2017 | gcn_cora_acc | reproduziert (0.8136, R) | B_mit_tavily | rep / teil / rep | reproduziert | ja | 0.8063 / 0.8020 / 0.8063 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | A_ohne_tavily | rep / nicht / rep | reproduziert | ja | 0.8182 / 0.6100 / 0.8039 | nicht reproduziert |
| wu_sgc_2019 | sgc_cora_acc | reproduziert (0.8081, R) | B_mit_tavily | rep / teil / teil | teilweise | nein | 0.8181 / 0.7994 / 0.8201 | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | A_ohne_tavily | rep / rep / teil | reproduziert | ja | 0.3815 / 0.3823 / 0.3947 | nicht reproduziert |
| zeng_dlinear_2023 | dlinear_etth1_96_mse | reproduziert (0.3809, R) | B_mit_tavily | rep / rep / Ausfall (Agenten-Fehler vermutet) | reproduziert | ja | 0.3779 / 0.3763 / 1.3485 | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | A_ohne_tavily | teil / teil / teil | teilweise | nein | 0.8017 / 0.8015 / 0.8018 | nicht reproduziert |
| shchur_pitfalls_2018 | shchur_gcn_cora_acc | reproduziert (0.8107, R+H) | B_mit_tavily | nicht / teil / rep | keine Mehrheit | nein | 0.7783 / 0.8049 / 0.8051 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | A_ohne_tavily | nicht / nicht / teil | nicht reproduziert | nein | 0.7919 / 0.7951 / 0.8157 | nicht reproduziert |
| velickovic_gat_2018 | gat_cora_acc | reproduziert (0.828, R) | B_mit_tavily | Ausfall (nicht reproduzierbar (Setup)) / teil / nicht | keine Mehrheit | nein | – / 0.8068 / 0.7926 | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | A_ohne_tavily | nicht / nicht / teil | nicht reproduziert | nein | 0.5069 / 0.5531 / 0.7988 | nicht reproduziert |
| velickovic_dgi_2019 | dgi_cora_acc | teilweise (0.8047, R) | B_mit_tavily | Ausfall (nicht reproduzierbar (Setup)) / Ausfall (nicht reproduzierbar (Setup)) / rep | Ausfall | nein | – / – / 0.8191 | Ausfall |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | A_ohne_tavily | Ausfall (nicht reproduzierbar (Setup)) / Ausfall (nicht reproduzierbar (Setup)) / Ausfall (nicht reproduzierbar (Setup)) | Ausfall | nein | – / – / – | Ausfall |
| shchur_gat_2018 | shchur_gat_cora_acc | reproduziert (0.814, R+H) | B_mit_tavily | nicht / Ausfall (nicht reproduzierbar (Setup)) / teil | keine Mehrheit | nein | 0.7001 / – / 0.8059 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | A_ohne_tavily | Ausfall (Agenten-Fehler vermutet) / rep / nicht | keine Mehrheit | nein | 0.9037 / 0.3726 / 0.4680 | nicht reproduziert |
| zeng_nlinear_2023 | nlinear_etth1_96_mse | reproduziert (0.3739, R) | B_mit_tavily | Ausfall (Agenten-Fehler vermutet) / teil / nicht | keine Mehrheit | nein | 1.3389 / 0.3873 / 0.5212 | nicht reproduziert |

## Sekundär: Rohzählung (einzelne Urteile)

| Claim-Art | Arm | Übereinstimmungen mit Referenzurteil | Ausfälle | „nah“ (nur echt) | Urteilsverteilung |
|---|---|---|---|---|---|
| echt | A_ohne_tavily | 7 von 24 | 5 | 13 von 24 | teilweise 7, reproduziert 6, nicht reproduziert 6, nicht reproduzierbar (Setup) 4, Agenten-Fehler vermutet 1 |
| echt | B_mit_tavily | 6 von 24 | 6 | 14 von 24 | reproduziert 7, teilweise 7, nicht reproduziert 4, nicht reproduzierbar (Setup) 4, Agenten-Fehler vermutet 2 |
| Placebo | A_ohne_tavily | 17 von 24 | 6 | – | nicht reproduziert 17, nicht reproduzierbar (Setup) 5, Agenten-Fehler vermutet 1, teilweise 1 |
| Placebo | B_mit_tavily | 18 von 24 | 6 | – | nicht reproduziert 18, nicht reproduzierbar (Setup) 5, Agenten-Fehler vermutet 1 |

Median der Abweichung vom Zielwert (gültige echte Generierungen, post hoc): A_ohne_tavily 0.0170 (n=20), B_mit_tavily 0.0113 (n=20)

## Ausfälle nach Ursache

- A_ohne_tavily: {'nicht reproduzierbar (Setup)': 9, 'Agenten-Fehler vermutet': 2}
- B_mit_tavily: {'Agenten-Fehler vermutet': 3, 'nicht reproduzierbar (Setup)': 9}

## Graph-Ablation und Zielwert-Prüfung

- A_ohne_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {None: 4, 'Graph genutzt': 12, 'nicht messbar': 2}
- B_mit_tavily: Graph-Ablation (echte Claims, Graph-Tasks) {'Graph genutzt': 11, 'nicht messbar': 3, None: 4}
- Belegter Zielwert weicht von `papers.yaml` ab: 0 Fälle

## Kosten und Code-Stand

- A_ohne_tavily: 24 Läufe, 0.873 $ netto, 1.039 $ brutto
- B_mit_tavily: 24 Läufe, 2.154 $ netto, 2.563 $ brutto

| Stand (`code_commit`, `config`) | Urteile | Übereinstimmungen echt | Ausfälle |
|---|---|---|---|
| d71a8af / v2_graph_repair | 96 | 13 von 48 | 23 |
