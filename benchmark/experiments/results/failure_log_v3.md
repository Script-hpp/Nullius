# Fehlerprotokoll `benchmark_v3.jsonl` (96 Urteile)

Automatisch aus `dossier.json`/`events.jsonl` (`scripts/failure_log.py`). Erfasst: echte Claims, die nicht "reproduziert" sind, und Placebos, die nicht "nicht reproduziert" sind. Die Ursache ist eine Einordnung aus Erklärung und Ereignissen, kein Beweis.

| Einheit | Arm | Gen | Art | Urteil | Ursache | Reparaturen (Fehler/Graph/Code) | Rolle 5 | Lauf |
|---|---|---|---|---|---|---|---|---|
| zeng_dlinear_2023 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.7549, Ziel 0.375) | 0/0/0 | 0 | `runs/zeng_dlinear_2023/20261001-101151` |
| shchur_pitfalls_2018 | A | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7955, Ziel 0.815) | 0/0/1 | 0 | `runs/shchur_pitfalls_2018/20261001-101151` |
| shchur_pitfalls_2018 | B | 0 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: Genauigkeit 0.171 nahe Zufall (Vorhersage-Zuordnung oder kein Lernen) | 2/1/2 | 2 | `runs/shchur_pitfalls_2018/20261001-102906` |
| shchur_pitfalls_2018 | A | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7954, Ziel 0.815) | 0/0/1 | 0 | `runs/shchur_pitfalls_2018/20261001-104554` |
| zeng_dlinear_2023 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.4293, Ziel 0.375) | 0/0/0 | 0 | `runs/zeng_dlinear_2023/20261001-101655` |
| zeng_dlinear_2023 | B | 0 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: schlechter als triviale Baseline (1.3210) | 2/0/5 | 4 | `runs/zeng_dlinear_2023/20261001-101655` |
| zeng_dlinear_2023 | A | 1 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3928, Ziel 0.375) | 0/0/1 | 0 | `runs/zeng_dlinear_2023/20261001-110423` |
| zeng_dlinear_2023 | B | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.4257, Ziel 0.375) | 0/0/0 | 0 | `runs/zeng_dlinear_2023/20261001-110925` |
| zeng_dlinear_2023 | A | 2 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (RuntimeError: permute(sparse_coo) | 0/0/3 | 0 | `runs/zeng_dlinear_2023/20261001-111429` |
| zeng_dlinear_2023 | A | 2 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (ValueError: Since axis is `None`, must provide window_shape for all dimensions of `x`; got 1 window_shape elements and `) | 0/0/3 | 0 | `runs/zeng_dlinear_2023/20261001-111429` |
| shchur_pitfalls_2018 | B | 1 | paper | Agenten-Fehler vermutet | Agenten-Fehler: Genauigkeit 0.173 nahe Zufall (Vorhersage-Zuordnung oder kein Lernen) | 2/1/3 | 1 | `runs/shchur_pitfalls_2018/20261001-110404` |
| shchur_pitfalls_2018 | A | 2 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7990, Ziel 0.815) | 0/0/0 | 0 | `runs/shchur_pitfalls_2018/20261001-112508` |
| zeng_nlinear_2023 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.4446, Ziel 0.374) | 1/0/0 | 0 | `runs/zeng_nlinear_2023/20261001-114128` |
| shchur_gat_2018 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7796, Ziel 0.818) | 0/0/1 | 0 | `runs/shchur_gat_2018/20261001-101151` |
| zeng_nlinear_2023 | A | 1 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3888, Ziel 0.374) | 0/0/1 | 0 | `runs/zeng_nlinear_2023/20261001-115306` |
| shchur_pitfalls_2018 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7629, Ziel 0.815) | 0/0/0 | 0 | `runs/shchur_pitfalls_2018/20261001-114430` |
| zeng_nlinear_2023 | B | 1 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3918, Ziel 0.374) | 0/0/1 | 1 | `runs/zeng_nlinear_2023/20261001-115739` |
| zeng_nlinear_2023 | B | 1 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: schlechter als triviale Baseline (1.1258) | 2/0/0 | 0 | `runs/zeng_nlinear_2023/20261001-115739` |
| zeng_nlinear_2023 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.8189, Ziel 0.374) | 0/0/0 | 0 | `runs/zeng_nlinear_2023/20261001-120610` |
| shchur_gat_2018 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7153, Ziel 0.818) | 0/0/1 | 1 | `runs/shchur_gat_2018/20261001-115103` |
| kipf_gcn_2017 | A | 0 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: Genauigkeit 0.179 nahe Zufall (Vorhersage-Zuordnung oder kein Lernen) | 2/1/0 | 0 | `runs/kipf_gcn_2017/20261001-120000` |
| kipf_gcn_2017 | A | 1 | paper | Agenten-Fehler vermutet | Agenten-Fehler: Genauigkeit 0.174 nahe Zufall (Vorhersage-Zuordnung oder kein Lernen) | 2/1/0 | 0 | `runs/kipf_gcn_2017/20261001-122158` |
| shchur_gat_2018 | A | 1 | paper | nicht reproduzierbar (Setup) | Setup: statische Prüfung (verbotener Import o. ä.) (Statische Pruefung: Dataset-Bibliothek: 'torch_geometric' in Zeile 37) | 0/0/3 | 0 | `runs/shchur_gat_2018/20261001-121217` |
| zeng_nlinear_2023 | B | 2 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (Smoke run (SMOKE=1) | 0/0/3 | 3 | `runs/zeng_nlinear_2023/20261001-121149` |
| shchur_gat_2018 | B | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7885, Ziel 0.818) | 1/0/1 | 0 | `runs/shchur_gat_2018/20261001-123256` |
| shchur_gat_2018 | B | 1 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (RuntimeError: The size of tensor a (12623) | 0/0/3 | 1 | `runs/shchur_gat_2018/20261001-123256` |
| wu_sgc_2019 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7269, Ziel 0.81) | 0/0/0 | 0 | `runs/wu_sgc_2019/20261001-130319` |
| wu_sgc_2019 | B | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7800, Ziel 0.81) | 0/0/0 | 0 | `runs/wu_sgc_2019/20261001-131311` |
| shchur_gat_2018 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7871, Ziel 0.818) | 0/0/1 | 0 | `runs/shchur_gat_2018/20261001-130132` |
| wu_sgc_2019 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7204, Ziel 0.81) | 0/0/0 | 0 | `runs/wu_sgc_2019/20261001-132313` |
| shchur_gat_2018 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7562, Ziel 0.818) | 1/0/2 | 0 | `runs/shchur_gat_2018/20261001-132132` |
| velickovic_gat_2018 | A | 0 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (TypeError: scatter_reduce() | 0/0/3 | 0 | `runs/velickovic_gat_2018/20261001-141021` |
| velickovic_gat_2018 | A | 0 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (RuntimeError: The size of tensor a (2866) | 0/0/3 | 0 | `runs/velickovic_gat_2018/20261001-141021` |
| velickovic_gat_2018 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7802, Ziel 0.83) | 0/0/2 | 2 | `runs/velickovic_gat_2018/20261001-142526` |
| velickovic_gat_2018 | A | 1 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (Smoke run (SMOKE=1) | 0/0/3 | 0 | `runs/velickovic_gat_2018/20261001-144521` |
| velickovic_gat_2018 | B | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8033, Ziel 0.83) | 0/0/1 | 1 | `runs/velickovic_gat_2018/20261001-145403` |
| velickovic_gat_2018 | B | 1 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (RuntimeError: einsum() | 0/0/3 | 0 | `runs/velickovic_gat_2018/20261001-145403` |
| velickovic_gat_2018 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7999, Ziel 0.83) | 0/0/2 | 0 | `runs/velickovic_gat_2018/20261001-151858` |
| velickovic_gat_2018 | B | 2 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (AttributeError: 'int' object has no attribute 'cross_entropy') | 0/0/3 | 0 | `runs/velickovic_gat_2018/20261001-155941` |
| velickovic_dgi_2019 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7083, Ziel 0.823) | 0/0/1 | 0 | `runs/velickovic_dgi_2019/20261001-160423` |
| velickovic_dgi_2019 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7892, Ziel 0.823) | 0/0/2 | 1 | `runs/velickovic_dgi_2019/20261001-164311` |
| velickovic_dgi_2019 | B | 0 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: schlechter als triviale Baseline (0.4066) | 2/0/3 | 2 | `runs/velickovic_dgi_2019/20261001-164311` |
| velickovic_dgi_2019 | A | 1 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (PythonDispatcher: registered at /__w/pytorch/pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:196 [backend fallback]) | 0/0/3 | 0 | `runs/velickovic_dgi_2019/20261001-170336` |
| velickovic_dgi_2019 | A | 1 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: Genauigkeit 0.176 nahe Zufall (Vorhersage-Zuordnung oder kein Lernen) | 2/0/1 | 0 | `runs/velickovic_dgi_2019/20261001-170336` |
| velickovic_dgi_2019 | B | 1 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (ModuleNotFoundError: No module named 'sklearn') | 0/0/3 | 1 | `runs/velickovic_dgi_2019/20261001-171520` |
| velickovic_dgi_2019 | B | 1 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: schlechter als triviale Baseline (0.4343) | 2/0/3 | 0 | `runs/velickovic_dgi_2019/20261001-171520` |
| velickovic_dgi_2019 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7449, Ziel 0.823) | 1/0/3 | 0 | `runs/velickovic_dgi_2019/20261001-173752` |
| velickovic_dgi_2019 | B | 2 | placebo | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, ) | 0/0/3 | 0 | `runs/velickovic_dgi_2019/20261001-175228` |

## Zählung nach Art

- Abweichung (nicht reproduziert): 18
- Setup: 13
- Agenten-Fehler: 9
- Abweichung (teilweise): 8
