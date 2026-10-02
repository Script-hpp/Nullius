# Fehlerprotokoll `benchmark_v4.jsonl` (96 Urteile)

Automatisch aus `dossier.json`/`events.jsonl` (`scripts/failure_log.py`). Erfasst: echte Claims, die nicht "reproduziert" sind, und Placebos, die nicht "nicht reproduziert" sind. Die Ursache ist eine Einordnung aus Erklärung und Ereignissen, kein Beweis.

| Einheit | Arm | Gen | Art | Urteil | Ursache | Reparaturen (Fehler/Graph/Code) | Rolle 5 | Lauf |
|---|---|---|---|---|---|---|---|---|
| zeng_dlinear_2023 | A | 0 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3863, Ziel 0.375) | 0/0/0 | 0 | `runs/zeng_dlinear_2023/20261001-203448` |
| shchur_pitfalls_2018 | A | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7957, Ziel 0.815) | 0/0/3 | 0 | `runs/shchur_pitfalls_2018/20261001-203448` |
| shchur_gat_2018 | A | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7917, Ziel 0.818) | 1/0/2 | 0 | `runs/shchur_gat_2018/20261001-203448` |
| zeng_dlinear_2023 | B | 0 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3866, Ziel 0.375) | 0/0/6 | 3 | `runs/zeng_dlinear_2023/20261001-204505` |
| shchur_pitfalls_2018 | B | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7984, Ziel 0.815) | 0/0/3 | 3 | `runs/shchur_pitfalls_2018/20261001-205908` |
| shchur_pitfalls_2018 | A | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7942, Ziel 0.815) | 0/0/0 | 0 | `runs/shchur_pitfalls_2018/20261001-220311` |
| shchur_gat_2018 | B | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8043, Ziel 0.818) | 0/0/4 | 3 | `runs/shchur_gat_2018/20261001-210628` |
| zeng_dlinear_2023 | B | 1 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3930, Ziel 0.375) | 0/0/4 | 0 | `runs/zeng_dlinear_2023/20261001-215952` |
| zeng_dlinear_2023 | B | 2 | paper | nicht reproduzierbar (Setup) | Setup: Versuche erschöpft (ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) | 0/0/9 | 0 | `runs/zeng_dlinear_2023/20261001-223820` |
| zeng_nlinear_2023 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.8423, Ziel 0.374) | 0/0/1 | 0 | `runs/zeng_nlinear_2023/20261001-225043` |
| shchur_pitfalls_2018 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7849, Ziel 0.815) | 0/0/4 | 0 | `runs/shchur_pitfalls_2018/20261001-224641` |
| shchur_gat_2018 | A | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7778, Ziel 0.818) | 1/0/7 | 0 | `runs/shchur_gat_2018/20261001-222635` |
| zeng_nlinear_2023 | B | 0 | paper | teilweise | Abweichung: teilweise, MSE zu hoch (gemessen 0.3854, Ziel 0.374) | 3/0/5 | 4 | `runs/zeng_nlinear_2023/20261001-230318` |
| zeng_nlinear_2023 | A | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.7689, Ziel 0.374) | 3/0/1 | 0 | `runs/zeng_nlinear_2023/20261001-231908` |
| kipf_gcn_2017 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7365, Ziel 0.815) | 0/0/0 | 0 | `runs/kipf_gcn_2017/20261001-233305` |
| kipf_gcn_2017 | A | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7476, Ziel 0.815) | 0/0/5 | 0 | `runs/kipf_gcn_2017/20261001-235153` |
| shchur_gat_2018 | B | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7938, Ziel 0.818) | 2/0/4 | 0 | `runs/shchur_gat_2018/20261001-231903` |
| kipf_gcn_2017 | A | 2 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7987, Ziel 0.815) | 0/0/1 | 0 | `runs/kipf_gcn_2017/20261002-001630` |
| zeng_nlinear_2023 | B | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.4304, Ziel 0.374) | 1/0/2 | 0 | `runs/zeng_nlinear_2023/20261001-232931` |
| wu_sgc_2019 | A | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu hoch (gemessen 0.8215, Ziel 0.81) | 0/0/3 | 0 | `runs/wu_sgc_2019/20261002-003527` |
| wu_sgc_2019 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7489, Ziel 0.81) | 0/0/1 | 1 | `runs/wu_sgc_2019/20261002-004306` |
| wu_sgc_2019 | A | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu hoch (gemessen 0.8530, Ziel 0.81) | 0/0/1 | 0 | `runs/wu_sgc_2019/20261002-010019` |
| zeng_nlinear_2023 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.8385, Ziel 0.374) | 0/0/2 | 0 | `runs/zeng_nlinear_2023/20261002-003205` |
| wu_sgc_2019 | B | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7970, Ziel 0.81) | 0/0/1 | 1 | `runs/wu_sgc_2019/20261002-010524` |
| zeng_nlinear_2023 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, MSE zu hoch (gemessen 0.4595, Ziel 0.374) | 0/0/3 | 0 | `runs/zeng_nlinear_2023/20261002-010849` |
| wu_sgc_2019 | A | 2 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7851, Ziel 0.81) | 0/0/4 | 0 | `runs/wu_sgc_2019/20261002-011230` |
| wu_sgc_2019 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7336, Ziel 0.81) | 0/0/3 | 1 | `runs/wu_sgc_2019/20261002-012056` |
| shchur_gat_2018 | A | 2 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.7913, Ziel 0.818) | 0/0/6 | 0 | `runs/shchur_gat_2018/20261002-001352` |
| shchur_gat_2018 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7062, Ziel 0.818) | 0/0/8 | 0 | `runs/shchur_gat_2018/20261002-012908` |
| velickovic_gat_2018 | A | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8122, Ziel 0.83) | 0/0/5 | 0 | `runs/velickovic_gat_2018/20261002-015850` |
| velickovic_dgi_2019 | A | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.6722, Ziel 0.823) | 1/0/9 | 0 | `runs/velickovic_dgi_2019/20261002-032153` |
| velickovic_dgi_2019 | B | 0 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.6897, Ziel 0.823) | 3/0/5 | 3 | `runs/velickovic_dgi_2019/20261002-033734` |
| velickovic_gat_2018 | B | 0 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8148, Ziel 0.83) | 1/0/2 | 2 | `runs/velickovic_gat_2018/20261002-032153` |
| velickovic_dgi_2019 | A | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7506, Ziel 0.823) | 0/0/6 | 0 | `runs/velickovic_dgi_2019/20261002-035319` |
| velickovic_gat_2018 | A | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8022, Ziel 0.83) | 0/0/6 | 0 | `runs/velickovic_gat_2018/20261002-035810` |
| velickovic_dgi_2019 | B | 1 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7754, Ziel 0.823) | 1/0/7 | 0 | `runs/velickovic_dgi_2019/20261002-040607` |
| velickovic_dgi_2019 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7811, Ziel 0.823) | 2/0/2 | 0 | `runs/velickovic_dgi_2019/20261002-044340` |
| velickovic_dgi_2019 | A | 2 | placebo | Agenten-Fehler vermutet | Agenten-Fehler: schlechter als triviale Baseline (0.3691) | 5/0/9 | 0 | `runs/velickovic_dgi_2019/20261002-044340` |
| velickovic_dgi_2019 | B | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.6898, Ziel 0.823) | 1/0/9 | 0 | `runs/velickovic_dgi_2019/20261002-050651` |
| velickovic_gat_2018 | B | 1 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu hoch (gemessen 0.8440, Ziel 0.83) | 0/0/5 | 0 | `runs/velickovic_gat_2018/20261002-043629` |
| velickovic_gat_2018 | A | 2 | paper | nicht reproduziert | Abweichung: nicht reproduziert, Genauigkeit zu niedrig (gemessen 0.7801, Ziel 0.83) | 1/0/7 | 0 | `runs/velickovic_gat_2018/20261002-054502` |
| velickovic_gat_2018 | B | 2 | paper | teilweise | Abweichung: teilweise, Genauigkeit zu niedrig (gemessen 0.8014, Ziel 0.83) | 0/0/6 | 0 | `runs/velickovic_gat_2018/20261002-063447` |

## Zählung nach Art

- Abweichung (teilweise): 20
- Abweichung (nicht reproduziert): 20
- Setup: 1
- Agenten-Fehler: 1
