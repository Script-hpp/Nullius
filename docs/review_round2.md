# Review Runde 2 an Rolle 4 (Auftrag 32)

Stand 2026-10-02. Diff `5e9e301..073a442` (`src/repro_agent/setup_details.py`, eine Zeile in `loop.py`, `tests/test_setup_v2.py`). Nur gelesen und mit Attrappen geprüft, kein `src/` geändert, kein LLM-, Tavily- oder Sandbox-Aufruf. Neue Tests: `tests/test_setup_round2.py`.

## Kurzfazit

6 Befunde, die der Reviewer entscheiden muss (R1 bis R6), alle mit `xfail(strict=True)`-Test. R1 und R2 bestätigen den Verdacht aus Aufgabe 2. R1 ist die schwerste: die Merkmals-Normalisierung gilt fast immer als abgedeckt, der gezielte zweite Durchgang bleibt aus. Kein Methodenwissen und kein Wert aus Soll-Liste oder `tests/reference/` in einem Prompt gefunden; ein Prompt-Satz ist aber aus dem DGI-Fall der Soll-Liste abgeleitet (R6, Stopp-Kandidat nach Ermessen des Reviewers).

## Aufgabe 1: Befunde in `src/`

**R1 (hoch) `setup_details.py:459`, `FEATURE_NORM_RE` um `preprocess|standardi|l2` erweitert.** `missing_topics` wertet Thema und Wert. Muster `l2` trifft jede L2-Weight-Decay-Angabe: `v2("weight decay", "L2 5e-4")` deckt "feature normalisation" ab. Fast jedes Graph-Dossier hat so eine Angabe, der zweite Durchgang für Normalisierung bleibt aus, genau die Hauptursache laut `failure_analysis_v4.md`. Auch `preprocess` allein deckt ab: `v2("data preprocessing", "removing self-loops")`. `FEATURE_NORM_EXCLUDE` kennt weder `self-loop` noch `decay|regulari`. Tests: `test_l2_weight_decay_does_not_cover_feature_normalisation`, `test_self_loop_removal_does_not_cover_feature_normalisation` (xfail).

**R2 (hoch) `setup_details.py:673`, `plan_details_v2` filtert `d.optional`.** Optionale README-Angaben kommen nicht mehr in den Plan. Zugleich zählt `missing_topics` sie als Abdeckung (nur `excluded` wird ausgeschlossen). Beispiel SGC: der Satz "Practical advice: ... normalize the features ..." ist die einzige Normalisierungsangabe; er gilt als abgedeckt, steht nicht im Plan, wird nicht als fehlend gemeldet, kein zweiter Durchgang. Still verloren. Der Zweig `[optional advice, not part of the experiment]` in `format_for_prompt_v2` (Zeile 703) ist toter Code; V2 verlangt Markierung, nicht Entfernung. Tests: `test_optional_advice_mark_reaches_plan_text`, `test_optional_only_feature_normalisation_is_not_silently_covered` (xfail). Ob Entfernen gewollt war, ist Entscheidung des Reviewers; dann muss `missing_topics` optionale Angaben ignorieren.

**R3 (mittel) `setup_details.py:294`, `RANKS`.** Neue Reihenfolge `start script, config, paper, code, readme text, argparse default`. "Paper vor argparse und README" ist laut Auftrag gewollt; aber `argparse default` steht jetzt hinter `readme text`. V2 legt fest: Startskript vor Konfiguration vor argparse-Default vor README-Fließtext. Abweichung von der Vorab-Festlegung. Test: `test_rank_order_follows_preregistration_v2` (xfail). Die geänderte Prüfung in `test_source_rank` (Zeile 52) vergleicht eine Liste mit ihrer eigenen Sortierung über dieselben `RANKS`-Indizes, also immer wahr, wenn die Elemente in `RANKS` stehen und in dieser Reihenfolge; sie ist bis auf die Reihenfolge der Liste selbst kein Test. Ersetzt durch `test_paper_ranks_above_…`, `test_conflict_paper_value_wins_…`.

**R4 (mittel) Akronym-Abgleich `path_model_exclusion` (Zeile 413 ff.) und `scope_exclusion` (Zeile 436 ff.).**
- Akronym wird aus den gefilterten Wörtern gebildet (`GENERIC_MODEL_WORDS` enthält "network"): "Graph Convolutional Network" ergibt `gc`, nicht `gcn`; die GCN-Angabe wird bei einem GCN-Claim weiter ausgeschlossen (Test `test_acronym_with_generic_word_in_full_name`, xfail). "Simplifying Graph Convolution" für SGC funktioniert.
- `claim_method in model` ist ein Teilstring: Angabe für "GCNII" gilt für einen GCN-Claim (Test `test_substring_of_other_model_name_is_not_a_match`, xfail). Gleiches gilt für `t in claim_method` im Pfadzweig.
- `claim_method` kommt aus dem deutschen Muster " erreicht " (`catalog.py:155` erzeugt es); ohne dieses Muster fällt der Abgleich still auf den alten Weg zurück (Test `test_acronym_does_not_apply_without_erreicht_claim_format`, grün, dokumentiert das Verhalten).
- Dreifach kopierte Zeile `claim_method = ...` (auch in `collect_middle_ground`); eine Hilfsfunktion wäre sauberer.

**R5 (mittel) `setup_details.py:295`, `OPTIONAL_RE` um `advice|helpful` erweitert.** Der SGC-Satz wird getroffen (Test grün). Die Muster gelten aber für das ganze Zitat (bis 300 Zeichen), nicht den Satz: `can`/`may`/`could` (schon vorher) in einem Nebensatz markieren eine verbindliche Angabe als optional ("Run python train.py --lr 0.01 --epochs 200; the script can be run on CPU."). Seit R2 heißt "optional" "aus dem Plan entfernt", die Wirkung ist damit stärker als im Stand vor Runde 2. Test: `test_optional_re_binding_value_with_unrelated_modal_verb` (xfail). Negativfälle ("Weight decay is 5e-4 ...", "recommended", Hidden/Epochen-Sätze) sind grün; "recommended" wird nicht getroffen.

**R6 (mittel, Leak-Nähe) `setup_details.py:152`, `collect_middle_ground`.** Prompt-Satz "If it's for the transductive setting (e.g., Cora), ignore inductive batch sizes." steht fest im Prompt, auch für Zeitreihen (ETTh1). Er enthält keinen Wert und kein Methodenwissen aus `tests/reference/`, entspricht aber wörtlich dem Soll-Eintrag `applies_to` für DGI ("transduktive Einstellung, nicht die induktiven Angaben") und ist damit auf den Messfall zugeschnitten. Prompt-Vorlage sonst zahlenfrei (Test `test_first_pass_prompts_contain_no_numbers_outside_claim_and_excerpt` grün). Einordnung als Stopp-Befund oder erlaubte Rolle-4-Anweisung entscheidet der Reviewer. Test: `test_inductive_hint_only_for_graph_family` (xfail).

**Nebenbefunde (kein Test):**
- `collect_middle_ground` ersetzt `collect_raw` in `extract_setup_details_v2`: eine Frage je Thema × Dokument, also (8+1) × Dokumente statt Stücke × Dokumente, bei kleinen Dokumenten (<15000 Zeichen) jeweils der ganze Text. Mehr Aufrufe und Kosten, nicht in `v5_step_a.md` abgeschätzt (erwartet < 2 $). Gleiche Angabe kommt aus mehreren Themen-Jobs mehrfach; `verify_raw` dedupliziert nicht, erst `merge` im zweiten Durchgang.
- Timeout 120 s statt 600 s in `_ask`; bei langsamen Free-Endpunkten Ausfälle möglich. `import ThreadPoolExecutor` doppelt, Leerzeilen/Trailing-Whitespace in neuem Code.
- `target_docs`: Paper nimmt einen der `MAX_TARGET_DOCS = 8` Plätze; bei vielen Skripten fällt README (zuletzt eingefügt) aus dem Ziel.
- `other settings`-Muster `initiali|optimi|momentum|clip` ist ein Themenwunsch, kein Wert, unkritisch.

## Aufgabe 1: Prüfpunkte

| Punkt | Ergebnis |
|---|---|
| Leak (Werte aus `tests/reference/`, Soll-Liste im Prompt) | Keine Werte. R6 siehe oben. |
| Zitatpflicht | Unverändert: `verify_raw_v2` verwirft Angaben ohne wörtliches Zitat, auch für fensterweise Ausschnitte (Quelle wird gegen das ganze Dokument geprüft). Tests `test_quote_check_drops_item_with_quote_missing_in_document`, `test_quote_from_windowed_excerpt_is_verified_against_whole_document`. |
| Nur mit Schalter `setup_v2` | Ja. `loop.py` übergibt `paper_title` nur im v2-Zweig; `collect_raw`, `extract_setup_details`, `_extract_chunk` unverändert. Alle geänderten Funktionen (`RANKS`, `OPTIONAL_RE`, `target_docs`, `plan_details_v2`, `scope_exclusion`, …) werden nur aus v2-Pfaden gerufen (grep). Benchmark v4 bleibt nachrechenbar. Test `test_old_path_unchanged_without_switch`. |
| `OPTIONAL_RE` trifft SGC-Satz | Ja (wörtlich getestet). Zu breit: R5. |

## Aufgabe 2: Geänderte Prüfungen in `tests/test_setup_v2.py`

| Stelle | Urteil |
|---|---|
| `test_source_rank` (RANKS-Reihenfolge) | Folge von Runde 2 (Paper nach vorn), aber die neue Prüfung ist tautologisch und verdeckt R3. Ersatz in der neuen Testdatei. |
| `test_target_docs_…` (`paper` jetzt im Ziel, `names[1:4]`) | Gewollt: Runde 2 nimmt das Paper in den zweiten Durchgang. |
| `test_two_pass_…` (`We use dropout` jetzt in gezielten Prompts) | Gewollt, gleiche Begründung; neuer Test `test_second_pass_finds_topic_stated_only_in_paper`. |
| `test_prompt_format_rank_marks_and_conflicts`: `"optional advice" in text` entfernt | **Regression (R2).** Die Markierung kann nicht mehr ankommen, weil optionale Angaben vorher gefiltert werden; die Prüfung wurde passend zum Verhalten gestrichen statt das Verhalten zu prüfen. Test `test_optional_advice_mark_reaches_plan_text` (xfail). |
| `test_feature_normalisation_coverage_…`: `assert not cover(... "removing self-loops")` zu `assert cover(...)` | **Regression (R1).** Kleinstes Beispiel: `missing_topics([v2("data preprocessing", "removing self-loops")], "graph")` meldet "feature normalisation" nicht mehr als fehlend; die Normalisierung wird nicht gesucht. Zusätzlich `l2`-Fall. Tests (xfail) beider Fälle. |

Die Prüfungen in `tests/test_setup_v2.py` habe ich nicht zurückgedreht; Vorschlag: nach Behebung von R1 und R2 die beiden Zeilen auf die ursprüngliche Form zurücksetzen, die xfail-Marker entfernen.

## Aufgabe 3: Neue Tests (`tests/test_setup_round2.py`)

37 grün, 9 xfail (strict). Gegenprobe: gegen `src/` von `5e9e301` sind 24 Tests rot (die Runde-2-Tests; die xfail-Fälle wechseln dort die Richtung).

- Rangfolge: `test_paper_ranks_above_argparse_default_and_readme_text`, `test_conflict_paper_value_wins_over_readme_text`, `…_over_argparse_default`, `test_start_script_and_config_still_above_paper`, xfail `test_rank_order_follows_preregistration_v2`.
- Akronym: `test_acronym_of_full_model_name_matches_claim_method`, `test_other_model_still_excluded_after_acronym_matching`, `test_acronym_does_not_apply_without_erreicht_claim_format`, `test_path_model_exclusion_uses_acronym_of_file_name`, xfail `test_acronym_with_generic_word_in_full_name`, `test_substring_of_other_model_name_is_not_a_match`.
- Paper im zweiten Durchgang: `test_target_docs_put_paper_first_and_keep_readme`, `test_second_pass_finds_topic_stated_only_in_paper`, `test_second_pass_prompts_name_missing_topics_and_carry_paper_text`.
- Gezielte Fragen je Pflichtthema: `test_first_pass_asks_one_question_per_mandatory_topic_graph`, `…_forecast_topics_for_time_series`, `test_first_pass_jobs_scale_with_documents`, `test_first_pass_prompt_has_title_and_target_method`, `test_big_document_without_topic_match_is_skipped_and_hits_are_windowed`, `test_first_pass_prompts_contain_no_numbers_outside_claim_and_excerpt` (Leak-Wache).
- Zitatpflicht und alter Pfad: `test_quote_check_drops_item_…`, `test_quote_from_windowed_excerpt_is_verified_against_whole_document`, `test_old_path_unchanged_without_switch`.
- Induktiv: `test_first_pass_prompt_has_inductive_hint`, `test_inductive_setting_detail_is_excluded_for_graph_claim`, xfail `test_inductive_hint_only_for_graph_family`.
- `OPTIONAL_RE`: SGC-Satz wörtlich (`test_optional_re_matches_sgc_readme_sentence_verbatim`, `test_sgc_readme_sentence_is_marked_optional_after_verification`), Negativfälle (parametrisiert, 5), alte Muster (3), `test_optional_mark_only_for_readme_text`, xfail `test_optional_re_binding_value_with_unrelated_modal_verb`.
- Optional im Plan: `test_optional_detail_is_not_in_plan_and_not_in_conflicts`, xfail `test_optional_advice_mark_reaches_plan_text`, `test_optional_only_feature_normalisation_is_not_silently_covered`.
- Normalisierungsabdeckung: drei grüne Tests (Preprocess-Aufruf/Zeilenwert, unverwandte Themen, Graph-Standardisierung), xfail `test_self_loop_removal_…`, `test_l2_weight_decay_…`.

Suite: `uv run pytest --deselect tests/test_harness_baselines.py --deselect tests/test_nebius_sandbox.py` → `593 passed, 2 skipped, 10 deselected, 9 xfailed`.
