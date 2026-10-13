# Unabhaengige Gegenpruefung des allgemeinen Zuverlaessigkeitsaudits

Stand **2026-10-13**, Basis **`e945e01c2e02ddf60635bf44b31698f859bb9cf9`** (Worktree `analyst`). Geprueft wurde der Bericht
`ceo/docs/general_reliability_audit_20261013.md` (E01–E23, R01–R04) samt seiner Proben. Dieses Dokument ist der Analyseteil.
Die danach vom Nutzer freigegebenen Fixes fuer E07–E12 und E15–E19 sind am Ende in einem eigenen Abschnitt beschrieben und
aendern nichts an der hier dokumentierten Ausgangslage.

## Kurzfassung

- **Alle 23 Befundgruppen sind im Kern reproduzierbar.** Keine ist widerlegt. Bei fuenf Befunden korrigiere ich Reichweite oder
  Begruendung (E01, E04, E05, E06, E20), bei vier erweitere ich sie (E01, E10, E13, E23).
- **Wichtigste Korrektur zu E01:** Der Docker-Beleg des Erstaudits lief *nicht* durch den Produktpfad `LocalSandbox`. Dieser Pfad
  ist in einer Standard-Docker-Installation gar nicht lauffaehig (siehe N01/N02). Der Mechanismus selbst ist aber ueber den
  echten `Orchestrator._run_and_eval` bestaetigt und schaerfer als beschrieben: Ein Startup-Hook kann die vom Harness geladenen
  Zielwerte ersetzen. Der Harness gibt dann **eine einzige, formal korrekte JSON-Zeile** aus (gemessen 0.3703).
- **Neuer Befund N03:** Der Repo-Abgleich (`_repo_diff_round`) ersetzt den ausgewaehlten Code nach der Auswahl, ohne den
  Val-Label-Check fuer die neue Version zu wiederholen. Im Fake-Pipelinelauf wird ein Code, der auf Val-Labels trainiert,
  uebernommen, mit allen Seeds gemessen und als `chosen` markiert. Die Ereignisbegruendung lautet dabei
  „Fehler-Check, Ablation und Validierung bestanden“, obwohl vor der Uebernahme keine Ablation lief.
- **Echte Live-Artefakte stuetzen E07, E08, E09 und E10 direkt:** Im gespeicherten SARAF-Job wurde Dropout 0.1 erzwungen, obwohl
  das Autorenmodell keinen Dropout hat. Ein Kandidat bekam daraufhin eine zusaetzliche Dropout-Schicht „as per spec“. Der aus
  `layers/Retrieval.py` korrekt uebernommene Retrieval-Batch 1024 wurde als Konflikt gemeldet. Batch 32 stammt aus einem Satz ueber
  Inferenzgeschwindigkeit. `time_aware_weight` verlor seinen Default, weil nur Aufrufe fuer andere Datensaetze ihn setzen.

Kein Befund belegt eine fruehere absichtliche Manipulation. Alle Aussagen betreffen fehlende oder fail-open Schutzschichten.

## Vorgehen und Abgrenzung

| Schritt | Befehl / Artefakt |
| --- | --- |
| Hash aller getrackten Produktdateien vor Beginn | `git ls-files src tests web/src benchmark deploy Dockerfile pyproject.toml \| xargs sha256sum` (1126 Dateien, Scratchpad) |
| Erstaudit-Proben unveraendert gegen `analyst` | Kopie mit angepasstem `ROOT` im Scratchpad: 25/25 reproduziert. Die JSON-Datei in `ceo` wurde nicht ueberschrieben. |
| Frontend-Proben | Kopie im Scratchpad mit Symlink auf `analyst/web`, Node 22.23: 3/3 reproduziert. |
| Eigene Offline-Proben | `uv run python docs/audits/opus_independent_probes_20261013.py`, Ergebnis `opus_independent_probes_20261013.json` (8 Proben, Netz gesperrt, Produkt-Hashes vorher/nachher gleich) |
| Eigene Container-Probe (echter Produktpfad) | `uv run python docs/audits/opus_runtime_probe_20261013.py`, Ergebnis `opus_runtime_probe_20261013.json` |
| Echte Public-Daten fuer E02 | `/data/etth1_public.npz` aus dem vorhandenen lokalen Image `repro-snapshot:gate-etth1` (nur kopiert, Image-ID unveraendert) |
| Echte Live-Artefakte | `ceo/logs/verification/saraf-20261013-080931/` (Events, Code). Autorenrepo-Kopie `/tmp/nullius-saraf-local-audit/author_repo` (`run.py` sha256 `f4fe3ab4…`, `scripts/ETTh_720.sh` sha256 `b073aa3e…`) |

Die Container-Probe startet jeden Branch-Container mit `--network none`. Sie mountet nichts vom Host. Container werden danach
entfernt, das Image wird nie committet. Das Host-Python wird nicht veraendert; der Host steuert nur Docker. Die Zielwerte in der
Probe sind synthetisch (Seed-Rauschen), es gibt keine echten ETTh1-Ziele. Es gab keine Aufrufe an OpenRouter, Token Factory,
Tavily oder Nebius und keinen Live-Lauf.

**Wie belastbar sind die Fakes?** `FakeSandbox`, `ValSandbox` und `MaskSandbox` ersetzen nur Sandbox und Harness-Zahlen. Die
Entscheidungslogik (`run_claim`, `_run_candidates`, `_repo_diff_round`, `val_label_check`, `select_candidate`, `_verdict`) laeuft
unveraendert. Fake-Proben belegen also Entscheidungen bei gegebenen Messwerten, nicht die Haeufigkeit solcher Messwerte bei
echten Modellen. Fuer E01 habe ich den Fake deshalb durch echte Container und echte Harness-Dateien ersetzt.

## Bewertungsmatrix

Schwere: **S1** = kann eine falsche Zahl oder ein falsches Paperurteil erzeugen. **S2** = verfaelscht Auswahl, Methode oder
Konfiguration und damit indirekt die Zahl. **S3** = Fehlalarm, Budgetverbrauch oder Darstellung. Reichweite = wo der Pfad im
heutigen Produkt (API, `product_kwargs()`, Nebius-Backend, 3 Kandidaten) tatsaechlich wirkt.

| ID | Status | Eigener Beleg | Pfad | Vorbedingung | Schwere | Reichweite |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | **bestaetigt, Reichweite korrigiert und erweitert** | Container-Probe durch echten `_run_and_eval`: B = `ok`, MSE 0.3703 aus synthetischen Hook-Zielen; C ohne Hook = `ungueltig` (Download offline unmoeglich), also stammen die Ziele aus dem Hook. Static Guards leer. | `loop.py:797,831` Training mit `unshare -n`; `loop.py:848-856` Harness im selben Branch, ohne `unshare`, letzte Zeile; `ett_eval.py`/`planetoid_eval.py` laden Ziele per `urllib` | Agentencode kann `site-packages` beschreiben | S1 | Nebius: **offen** (UID, Capabilities, Schreibrechte nicht belegt). LocalSandbox: in Standard-Docker nicht erreichbar (N01/N02). |
| E02 | bestaetigt | Echte Public-NPZ: `test_inputs[1:, :-1] == test_inputs[:-1, 1:]` exakt (max. Abweichung 0.0); 98.2585 % der Zielwerte stehen in Eingaben spaeterer Fenster. Indexalgebra `ett_data.py:101,124`. | `ett_data.prepare_public`, `tasks.py:275-280` (nur Textverbot) | nicht kausaler oder fensteruebergreifender Code | S1 | alle ETT-Tasks im Produkt |
| E03 | bestaetigt, realistischere Variante | Label-Clamp im Stil von Label Propagation (`pred[labels >= 0] = labels[...]`): statisch `none`, Maskenabfall 1.0 -> 0.14, gewaehlt, `excluded=false` | `loop.py:752` (Ausschluss nur bei statisch `training`) | Code nutzt sichtbare Val-Labels fuer Ausgaben statt Gradienten | S2 | Cora, CiteSeer, Cora-LCC |
| E04 | bestaetigt, praezisiert | Ridge-Forecaster auf synthetischer AR-Serie mit Produkt-Harnessfunktionen: train+val gewinnt `select_candidate`, Val-Vorteil 0.042 gegen Testvorteil 0.011 | `loop.py:1424` (`val_label_check` nur mit `mask_val_cmd`), `tasks.py:353` | Kandidat fittet auf denselben Val-Fenstern, die die Auswahl bewertet | S2 | alle ETT-Tasks |
| E05 | bestaetigt, Wahrscheinlichkeit eingeschraenkt | Code: Rueckgabe von `sb.run(branch, task.mask_val_cmd, 120)` wird verworfen (`loop.py:813`); Absturz des maskierten Laufs -> `nicht messbar` -> nicht ausgeschlossen (`loop.py:752`). Erstaudit-Probe reproduziert. | `val_label_check` | Infrastrukturfehler bzw. Absturz nur mit maskierten Labels | S2 | Graph-Tasks. Fehlschlag des festen Maskenbefehls ist selten; der Absturzfall ist vom Agentencode steuerbar. |
| E06 | bestaetigt, Reichweite praezisiert | `loop.py:810` ignoriert Exit-Code. Ablation laeuft im Produkt nach allen Seeds trotz `allow_test_feedback=False` (`loop.py:1313-1324`) und setzt `methode="nein"`. | `ablation_check` | Ablationsbefehl scheitert | S3 (nur Achse 2) | Graph-Tasks; Zahlurteil unberuehrt |
| E07 | **bestaetigt durch echtes Live-Artefakt** | Live-Spec Job 175724ef: `batch_size=32` mit Zitat „inference-time efficiency … batch size 32“; Autor-Default 8 (`run.py:100`). `dropout 0.1` erzwungen, Autormodell hat keinen Dropout; Kandidat 2 bekam `nn.Dropout(configs.dropout)  # Apply dropout as per spec` (`code_ab1e99ecf73f.py:351`). | `spec.verify_entries`, `spec.resolve`, `spec_check.config_check` | LLM-Extraktion waehlt Zahl mit falscher Rolle | S2 | Autorencode-Modus (Standard) |
| E08 | **bestaetigt durch echtes Live-Artefakt** | Echte Skriptzeilen: `time_aware_weight` wird nur fuer ETTh2/ETTm gesetzt, ETTh1/96 nutzt Default 0.5; Default wird trotzdem entfernt. Live-Event `dropped_defaults` enthaelt `time_aware_weight`. | `spec.drop_run_script_defaults` | Startskript mit mehreren Aufrufen | S2 | Autorencode-Modus |
| E09 | bestaetigt (Live-Mechanismus, Wert zufaellig richtig) | Echtes `ETTh_720.sh`: lr 0.001 (h96) und 0.0001 (h336) -> `lr` fehlt in `raw_items`. Live wurde der argparse-Default 0.001 erzwungen; fuer h96 zufaellig identisch. | `spec.raw_items` | mehrere Aufrufe mit verschiedenen Werten | S2 | Autorencode-Modus |
| E10 | **bestaetigt durch echtes Live-Artefakt** | Live `spec_check`: Kandidat 1 und 2 `literal 1024` fuer `batch_size`; 1024 stammt aus Autorcode `layers/Retrieval.py:685`. Erstaudit: `F.normalize(p=2)` als Dropout. | `spec_check._keyword_conflicts` | gleichnamiges Keyword anderer Funktion | S3 (Reparatur, Budget), S2 wenn Reparatur das Verfahren aendert | Steckbrief-Modus |
| E11 | bestaetigt, erweitert | Eigene Varianten: `CONFIG['lr'] *= 1000`, `CONFIG={**CONFIG,'lr':1.0}`, `CONFIG.__setitem__(...)` -> Score 1/1, wirksame lr 1.0 | `spec_check._config_mutations` | Schreibweise ausserhalb der bekannten Muster | S2 | Steckbrief-Modus |
| E12 | bestaetigt (Code und Probe) | `resolve(..., corroborate=paper_numbers(...))`: globale Zahlenmenge, keine Schluesselbindung (`spec.py:461ff`) | `spec.resolve` | gleichrangiger Konflikt, Zahl kommt zufaellig im Paper vor | S2 | Autorencode-Modus |
| E13 | bestaetigt, **erweitert** | Eigene Probe: Spec-Score vor `_run_seed`; Smoke-Absturz -> Reparatur liest `CONFIG['lr']` nicht mehr -> Auswahlgrund `spec (1) + val`, tatsaechlicher Score 0/1; der reparierte Hash wurde nie spec-geprueft | `loop.py:1398-1400` | jede Code-Aenderung nach `_spec_conform` (Absturzreparatur, Qualitaetsschleife, Repo-Abgleich) | S2 | Produkt mit Steckbrief |
| E14 | bestaetigt, Reichweite eingeschraenkt | Eigene Probe: letzte Zeile gewinnt (`5,1` dann `5,2` -> `2`) | `harness/verify.py:29-39` | Agent schreibt CSV selbst | S3 | Im Standard schreibt `nullius_io` die Datei aus Arrays (eindeutige IDs). Betrifft Paper-Modus und Agenten, die den Helfer umgehen. |
| E15 | bestaetigt (Code und Erstaudit-Probe) | `source_context.py:50-51`: eine Ebene, nur `list(repo)` statt Listing | `author_excerpt` | Trainer importiert Helfer in Ketten | S2 | Autorencode-Modus |
| E16 | bestaetigt (Code und Probe) | `parse_github_url` verwirft den Ref; Ziele werden zu `tree/HEAD/...` (`repo_source.py:68,213,468-472`) | Repo-Aufloesung | Paper verlinkt Branch/Tag | S2 | alle Repo-Laeufe |
| E17 | bestaetigt als Restluecke | Eigene Variante `return 0 * x` unerkannt | `method_guard._placeholder` | — | S2 | Guard beansprucht keine Vollstaendigkeit |
| E18 | bestaetigt als Restluecke | Eigene Variante: Dict-Comprehension ueber `state_dict()` unerkannt | `training_check.checkpoint_issues` | — | S2 | dto. |
| E19 | bestaetigt | Eigene Variante: Wort `socket` in Docstring -> „Netzwerk-Modul“ | `harness/code_check.py` (Regex auf Rohtext) | — | S3 | alle Laeufe |
| E20 | bestaetigt, Reichweite eingeschraenkt | `n_candidates=1` setzt nur `scripts/review/cg_gate.py:191`; Produkt 3 (`loop.py:355`) | `run_claim` Einzelpfad | Review-/Experimentskript | S3 | nicht im Produkt |
| E21 | bestaetigt | `phases.ts:151` ohne Bezug zu `running`; Text `RunTimeline.tsx:307` | UI | Stopp vor `candidates`-Event | S3 | Live/Replay |
| E22 | bestaetigt an echten Daten | `kipf_gcn_2017/20261002-110746`: gezeigt Ziel 0.815 mit Placebo-Urteil | `phases.ts:142,148` (erstes `claim_start`, letztes `candidates`/`verdict`) | Lauf mit mehreren Claims | S3 (inhaltlich irrefuehrend) | Replay, StepGraph |
| E23 | bestaetigt, **erweitert** | Neben `setup_details` rufen auch `claims_source.py:308`, `repo_source.py:352`, `community.py:311`, `objections.py:50`, `fidelity.py`, `setup_details.py:534,609` `llm.complete` direkt auf | `phases.ts:590` zaehlt nur `llm_call` | — | S3 | alle Live-Laeufe |
| R01 | bestaetigt als Risiko | `ett_data.py:21` (`main`), `planetoid_data.py:21` (`master`), `pip install` ohne Versionen (`tasks.py:196,358,431`); Snapshot hat z. B. torch 2.14.1 | Setup und Eval | Upstream aendert sich zwischen Snapshot und Eval | S2 | kein tatsaechlicher Drift beobachtet |
| R02 | bestaetigt als Politikfolge | `loop.py:1380-1393` erzeugt alle Kandidaten vor dem ersten Lauf | Kandidatenphase | Budget knapp | S3 | Live-Job brach spaeter ab (in Reparaturen), nicht in der Generierung |
| R03 | bestaetigt | `api.py:287` `JOB_KEEP_SECONDS = 3600`, Registry nur im Speicher | API | Job aelter als 1 h oder Neustart | S3 | oeffentliche Nachvollziehbarkeit |
| R04 | bestaetigt als Heuristik | `verify.py:agent_fault_suspected` prueft auch Nicht-Claim-Metriken; `_verdict` ersetzt das Zahlurteil (`loop.py:1781`) | Urteil | MSE trifft, MAE schlechter als Baseline | S3 | Politikentscheidung, kein Codefehler |

## Korrekturen am Erstaudit

1. **E01, Docker-Fallback.** Das Erstaudit nennt den Mechanismus „Belegt im Docker-Fallback“. Die Probe lief aber in einem
   nackten `python:3.11-slim` ohne `unshare -n` und ohne Orchestrator. Der echte `LocalSandbox`-Pfad scheitert vorher zweimal:
   `docker cp` legt `/work` nicht an (N02), und `unshare -n` braucht `CAP_SYS_ADMIN`, das ein Standard-Container nicht hat (N01).
   Belastbar ist: In jeder Laufzeit, in der der Agentencode die Python-Installation beschreiben darf und der Harness danach im
   selben Dateisystem laeuft, ist das Ergebnis manipulierbar. Das zeigt meine Probe durch den echten `_run_and_eval`.
2. **E01, Nebius.** Ich stimme der Praezisierung aus `ceo` zu: Dass die Evaluation ohne `unshare -n` laeuft, beweist keine
   oeffentliche Netzroute in Nebius. `CAP_SYS_ADMIN` fuer `unshare` ist nicht gleich UID 0 und nicht gleich Schreibrecht auf
   `site-packages`. Ob Nebius-Sessions beides erlauben, ist **offen**. Kein vorhandenes Log belegt es. Sicher ist nur: Training
   und Evaluation laufen in derselben Session (`nebius.py:run` mutiert die Session, `loop.py:848-852`).
3. **E01, Schaerfe.** Das Erstaudit beschreibt eine angehaengte zweite JSON-Zeile. Ein Fix „erste Zeile lesen“ oder „genau eine
   Zeile verlangen“ wuerde das nicht beheben: Der Hook kann den Datenzugriff des Harness ersetzen und den Harness selbst rechnen
   lassen. Der Val-Harness laeuft im selben Branch und ist ebenso betroffen (Kandidatenauswahl).
4. **E04.** Die Verfuegbarkeit des `val`-Arrays ist gewollt und kein Fehler. Training auf train+val ist zulaessig, wenn es erst nach
   abgeschlossener Auswahl geschieht. Der Fehler ist allein: dieselben Val-Fenster dienen dem Fit **und** der Kandidaten- oder
   Checkpointwahl, und nichts prueft das fuer Zeitreihen.
5. **E05.** Die Erstaudit-Probe erzwingt einen Fehlschlag des festen Maskenbefehls. Das ist im Produkt unwahrscheinlich, weil der
   Befehl vor dem Agentencode auf einem frischen Branch laeuft. Die eigentliche Luecke ist die fail-open-Politik: fehlende Evidenz
   wird wie bestandene Unabhaengigkeit behandelt.
6. **E06.** Betrifft nur Achse 2 (`methode`), nie das Zahlurteil.
7. **E20.** Betrifft nur `scripts/review/cg_gate.py`, keinen Produktpfad.

## Neue Befunde

### N01 — LocalSandbox kann in Standard-Docker kein Training ausfuehren

`LocalSandbox.branch` startet Container ohne Capabilities (`sandbox/local.py:37-41`). Jeder Agentenlauf beginnt mit
`unshare -n` (`loop.py:674,797,831`). Beleg (Container-Probe, Fall A, nur Netzflag ergaenzt):
`Smoke run (SMOKE=1) crashed: unshare: unshare failed: Operation not permitted (exit code 1)`. Mit `--cap-add SYS_ADMIN` laeuft
derselbe Befehl. Folge: Der dokumentierte Fallback bei Nebius-Problemen ist nicht einsatzfaehig, und jeder Versuch wuerde als
Agentenfehler in die Reparaturschleife gehen (Budgetverbrauch). Schwere S3, Produkt nutzt standardmaessig Nebius.

### N02 — LocalSandbox.put_file legt Zielverzeichnisse nicht an

`docker cp` scheitert mit `Could not find the file /work in container …`. `/work` und `/opt/harness-<zufall>` existieren im
Snapshot nicht. Nebius `apply_files` legt sie offenbar an (Produktion lief). Zusammen mit N01: Fallback zweifach defekt.
Ausserdem raeumt `LocalSandbox` Branch-Container nie ab (nur der Setup-Container wird entfernt).

### N03 — Repo-Abgleich uebernimmt Code ohne erneuten Val-Label-Check

Pfad: `run_claim` (`loop.py:1267`) -> `_repo_diff_round` -> `finish("adopted", "Fehler-Check, Ablation und Validierung bestanden")`
(`loop.py:1542`) -> `run_rest`. Mit `allow_test_feedback=False` laeuft vor der Uebernahme nur `_val_fault_check` und der
Val-Vergleich „gleich oder besser“. Ein Code, der auf Val-Labels trainiert, verbessert die Val-Metrik und besteht genau diesen
Vergleich. Eigene Probe `repo_diff_adopts_unchecked_code`: neuer Hash statisch `training`, Val 0.80 -> 0.95, uebernommen, alle
drei Seeds mit diesem Hash, `chosen=True`, Urteil `reproduziert`; `val_label_check` enthaelt nur die drei urspruenglichen
Kandidaten, kein maskierter Lauf fuer den neuen Code. Die Ablation laeuft erst spaeter in `run_claim` und kann nur `methode`
herabstufen, nie die Uebernahme rueckgaengig machen. Die Begruendung „Ablation … bestanden“ ist fuer den Standardpfad falsch.
Schwere S2, Reichweite: Produkt mit Autorencode, wenn der Abgleich verifizierte Abweichungen findet.

### N04 — Spec-Erzwingung kann das Autorenverfahren aktiv veraendern (Live-Beleg, Teil von E07)

Der Steckbrief verlangt jeden erzwungenen Schluessel als gelesen. Fehlt der Mechanismus im Autormodell (Dropout bei SARAF), ist
der Befund `unused` eine Reparaturaufforderung, die eine nicht vorhandene Schicht einfuehrt. Live belegt (`spec_check` Kandidat 2,
Runde 0–2, `code_ab1e99ecf73f.py:351`). Ich fuehre das als eigenen Punkt, weil es die Richtung des Schadens zeigt: nicht nur
falsche Werte, sondern erzwungene Abweichung vom Referenzcode.

## Allgemeine Loesungswege

Reihenfolge nach Wirkung auf die Glaubwuerdigkeit einer Zahl. Jeder Punkt nennt den kleinsten verantwortlichen Layer.

### L1 — Vertrauenswuerdige Messumgebung (E01, Voraussetzung fuer L2/L3)

**Empfehlung:** Evaluation nie im Branch des Agenten. Nach dem Training nur `/output/*.csv` (Groesse begrenzt) aus dem Branch
lesen und in einem **frischen Branch vom Basis-Snapshot** auswerten, mit `python -I -S` (kein `sitecustomize`, kein
Benutzer-Site, kein `PYTHONPATH`) und Harness in einem neuen Zufallsverzeichnis. Die Zielwerte gehoeren in diesen frischen Branch
oder in den Orchestrator, nicht in eine Umgebung, die der Agent je beschrieben hat.

- *Alternative A:* Auswertung im Orchestrator-Prozess (numpy, keine Sandbox). Am einfachsten, aber Harness-Code laeuft dann mit
  Produktionsrechten und Netz; nur mit festen, kleinen Eingaben vertretbar.
- *Alternative B:* Agent als unprivilegierter Benutzer mit read-only `site-packages`. Haengt von Nebius-Laufzeitdetails ab, die
  offen sind; ohne L1 nicht ausreichend, weil weitere Persistenzwege bleiben (z. B. `.pth`, `$HOME`, geaenderte Binaries).
- **Abnahme:** Ein Agentenprogramm, das `sitecustomize`, `.pth` oder `urllib` patcht, aendert weder Test- noch Val-Ergebnis.
  Regressionstest: Fake-Sandbox prueft, dass `eval_cmd` und `val_eval_cmd` nie auf dem Trainingsbranch laufen und `-I -S` tragen;
  plus die Container-Probe dieses Audits als manueller Integrationstest (erwartet dann `ungueltig` oder echten Wert).

### L2 — Kausaler Forecast-Vertrag (E02)

L1 schuetzt den Harness, aber nicht die oeffentlichen Testfenster im Agentenbranch. Zwei tragfaehige Wege:

1. **Mechanischer Inferenzvertrag (bevorzugt).** Das Agentenskript trainiert und speichert einen Checkpoint und eine Funktion
   `predict(window[L,7]) -> [96,7]`. Der Harness-Runner (aus L1, frischer Branch) laedt Checkpoint und Funktion und ruft sie
   **pro Fenster einzeln** auf, ohne dass `test_inputs` als Ganzes im Prozess liegt. Damit ist fensteruebergreifende Information
   strukturell ausgeschlossen. Kosten: neuer `nullius_io`-Vertrag, Prompt-Aenderung, einmalige Neuvalidierung der Referenzen.
2. **Infer-only-Unabhaengigkeitsprobe mit festem Checkpoint** (wie `harness/independence.py` im Nachlauf): gleicher Checkpoint,
   Probe-Datei mit Fenstern im Abstand >= 816 plus Rauschfenstern, Vergleich <= 1e-4. Nur gueltig, wenn Training und Inferenz
   getrennt sind. Ein erneutes Training auf der Probe vermischt Nichtdeterminismus mit Leckage und ist **nicht** geeignet.

Beide Wege duerfen nur stoppen oder ausschliessen. Ein Ergebnis dieser Pruefung darf nie in einen Reparatur-Prompt gelangen, der
Testzahlen enthaelt. **Abnahme:** Ein Programm, das `test_inputs[i+h+1][-1]` abschreibt, erhaelt kein Paperurteil; ein kausales
Referenzmodell (DLinear-Referenz) besteht unveraendert. Solange keiner der Wege umgesetzt ist, sollte jedes ETT-Urteil den Hinweis
„Kausalitaet der Vorhersagen nicht erzwungen“ tragen.

### L3 — Unabhaengige Auswahl (E03, E04, E05, N03, Datenvertrag)

Grundregel: **Daten, die eine Auswahl entscheiden, duerfen nicht in den Fit des bewerteten Kandidaten eingehen.** Das Endmodell
darf nach abgeschlossener Auswahl auf train+val nachtrainieren.

- **Getrennte Auswahldaten durch den Harness.** Fuer ETT: Der Harness haelt einen Auswahl-Abschnitt (z. B. die letzten Wochen des
  Val-Zeitraums) im Agentenbranch zurueck und wertet die Val-Vorhersagen nur dort aus. Der Agent kann diese Ziele nicht fitten,
  weil er sie nicht hat. Fuer Graph: Auswahl-Labels als eigener, verborgener Knotensatz (aus dem bisherigen Val-Satz), Labels nur
  im Harness. Das ersetzt die heuristische Masken-/Statik-Kombination durch eine strukturelle Garantie.
- **Zweiphasiges Protokoll fuer train+val:** Phase 1 Auswahl auf verborgenen Auswahldaten; Phase 2 optional Nachtraining des
  gewaehlten Codes mit identischen Hyperparametern auf train+val, gemessen mit allen Seeds. Phase 2 darf die Auswahl nicht
  rueckwirkend aendern.
- **Kontrollen fail-closed:** Kontrollbefehl muss Exit 0 liefern, und eine Pruefsumme der veraenderten Datei muss sich aendern.
  `nicht messbar` bei statisch `training` schliesst aus. Gilt fuer Maske und Ablation (E05, E06).
- **Jede Codeversion, die gemessen wird, durchlaeuft dieselben Auswahlkontrollen** (N03, E13). Technisch: eine Funktion
  `admit(code_hash)`, die Spec-Score, Val-Label-Kontrolle und Platzhalterpruefung fuer genau diesen Hash liefert und vor
  `run_rest` verlangt wird. Ereignistexte nennen nur tatsaechlich gelaufene Pruefungen.
- **Datenvertrag fuer Val-Fenster:** `nullius_io` liefert fertige Val-Eingabefenster (`val_inputs[w]`) statt nur der Serie. Das
  beseitigt die im SARAF-Lauf beobachteten Nulltargets/Tuple-Fehler bei der Val-Konstruktion durch den Agenten.
- **Abnahme:** Label-Clamp, Fit auf Val und Repo-Abgleich-Ersatz koennen in den Proben dieses Audits nicht mehr gewinnen; ehrliche
  Referenzen mit Early Stopping werden nicht ausgeschlossen (bestehende Referenztests `tests/reference/` bleiben gruen).

### L4 — Zweckgebundene Parameter (E07–E12, N04)

- Jeder Steckbriefeintrag traegt eine **Rolle** (`train`, `inference`, `eval`, `model`) und, wo vorhanden, **Aufruferkontext**
  (Modell, Datensatz, Horizont). Erzwungen werden nur `train`/`model`-Eintraege. Zitate mit Inferenz-/Effizienz-/Laufzeitkontext
  werden deterministisch herabgestuft (nicht erzwingen, nur zeigen).
- Startskripte werden **pro Aufruf** gelesen. Der passende Aufruf wird ueber Datensatz und Horizont bestimmt; nur seine Flags
  ersetzen Defaults (E08), und nur seine Werte gelten (E09). Gibt es keinen eindeutigen passenden Aufruf, bleibt der Schluessel
  offen statt global gemittelt.
- Paper-Bestaetigung nur ueber Zitate, deren Text den Schluessel nennt (E12).
- `unused` nur erzwingen, wenn der Referenzcode den Mechanismus nachweislich benutzt (N04).
- Keyword-Konflikte nur fuer den passenden Aufrufer (Optimizer fuer `lr`/`weight_decay`, `DataLoader` des Trainings fuer
  `batch_size`, `nn.Dropout`/`F.dropout` fuer `dropout`) (E10).
- CONFIG-Mutationen: jede Schreiboperation an `CONFIG` oder einem Alias ausserhalb des Literalblocks, egal wo (E11). Was nicht
  sicher auswertbar ist, gilt als Konflikt.
- Score immer fuer den finalen Hash neu berechnen (E13, Teil von L3).

### L5 — Exakte Revisionen (E15, E16, R01)

Angeforderter Git-Ref aus dem Paperlink bleibt erhalten und wird geklont; ist er nicht erfuellbar, bricht der Lauf mit klarer
Meldung ab. Trainingshelfer werden transitiv wie der Modellkern aufgeloest, gegen das vollstaendige Listing. Datensatz-URLs auf
Commit festlegen und Hash der Rohdatei im Snapshot und im Harness pruefen; `pip install` mit festen Versionen. Abnahme: Daten- und
Codehash stehen im Dossier und stimmen zwischen Vorbereitung und Auswertung ueberein.

### L6 — Budget und Darstellung (R02, R03, E21–E23)

Kandidaten nacheinander erzeugen und sofort smoke-pruefen, statt alle vorab; Reparaturbudget je Kandidat begrenzen. Abgebrochene
Jobs als Dossier mit stabilem Link exportieren. UI: Unterbereiche aus dem terminalen Jobzustand ableiten; Modell je `claim_id`
bauen; Aufrufzaehler aus dem LLM-Client (zentral) statt aus einzelnen Events.

## Priorisierung

1. **L1** (E01): ohne vertrauenswuerdige Auswertung sind alle anderen Garantien umgehbar. In `ceo` laut Nachricht bereits in Arbeit.
2. **L3** (N03, E03–E05, E13): betrifft die Standardauswahl jedes Laufs.
3. **L2** (E02): groesster Umbau; bis dahin klarer Hinweis im Zeugnis.
4. **L4/L5** (E07–E12, E15, E16, N04): live belegte Fehlkonfigurationen und Reparaturkosten.
5. Guards (E17–E19), N01/N02, Darstellung (E21–E23, R03), Politik R04.

## Freigegebene Fixes in diesem Worktree

Der Nutzer hat nach Abschluss der Analyse Fixes fuer E07–E12 und E15–E19 freigegeben, beschraenkt auf `spec.py`, `spec_check.py`,
`source_context.py`, `repo_source.py`, `harness/code_check.py`, `harness/training_check.py`, `method_guard.py` und deren Tests.
Nicht in diesem Worktree: E01–E06, E13, E14, E20–E23, N01–N03, R01–R04 (Orchestrator, Tasks, Sandbox, UI; laut `ceo` dort in Arbeit).
Kein Befund aus diesem Block war widerlegt, deshalb gibt es zu jedem eine Aenderung.

### Umsetzung

| ID | Aenderung (kleinster Layer) | Grenze |
| --- | --- | --- |
| E07 | `spec.verify_entries` verwirft Zahlen aus Zitaten mit Inferenz-/Effizienzkontext, wenn das Zitat das Training nicht nennt (`not_training_context`). Grund steht im Dossier (`spec.dropped`). | Wortliste; ein Satz „trained … inference speed“ bleibt gueltig (gewollt). |
| N04 (Teil E07) | `spec.inactive_mechanisms`: `dropout` faellt aus dem Steckbrief, wenn Modelldatei und ihre transitiv importierten, gelesenen Module keinen Dropout enthalten. Kein Erzwingen, keine S3-Frage, keine Reparatur. | Nur Mechanismus `dropout`; andere Schluessel bleiben wie bisher. |
| E08 | `drop_run_script_defaults(excerpt, dataset, horizon, model)`: nur Flags aus Aufrufen, die zu Datensatz, Horizont (ETT: 96) und Modell passen (`call_matches`). Shell-Variablen (`$h`) gelten als passend; nur ein ausdruecklich anderer Literalwert schliesst aus. | Ohne Kontext (alte Aufrufer, Erstaudit-Probe) unveraendertes Verhalten. |
| E09 | `raw_items(..., dataset, horizon)`: `--flag Zahl` je passendem Aufruf; nur Mehrdeutigkeit innerhalb passender Aufrufe verwirft den Wert. Zitat ist das Flag selbst. | Datensatzvergleich mit Praefix (Cora / Cora-LCC). |
| E10 | `spec_check._keyword_applies`: `p=` zaehlt nur fuer Dropout-Aufrufe; `batch_size=` nur fuer Trainings-Loader (Name mit `train` oder `shuffle=True`). `dropout=` und `lr=` weiterhin ueberall. | Heuristik ueber Namen; ein Trainings-Loader ohne `train`-Namen und ohne `shuffle=True` wird nicht geprueft. |
| E11 | `spec_check._config_mutations`: Alias, Neuzuweisung des ganzen Dicts (auch `{**CONFIG, ...}`), `+=`/`*=`, `__setitem__`, `pop`/`del`/`clear`, `update(<nicht literal>)`, in `if`/`with`/`try`. Nicht auswertbare Schreibzugriffe gelten als Konflikt. Eigene Reparaturmeldung „changes CONFIG … after the CONFIG block“. | Schreibzugriffe in Schleifen und Funktionen bleiben erlaubt (Lernraten-Plan, bewusst unveraendert). |
| E12 | `paper_numbers` liefert Zahlen je Parameter: ein Zitat mit Parameterwort stuetzt nur diesen (Horizont bindet 96/32 an `horizon`), Tabellenzeilen ohne Parameterwort bleiben ungebunden. `resolve` akzeptiert weiterhin eine flache Menge (alte Aufrufer). | Wortliste je Schluessel. |
| E15 | `author_excerpt`: Trainingshelfer transitiv gegen das vollstaendige Listing; gelistet, aber nicht gelesen = `SourceContextBlocked` wie beim Modellkern. Andere Modelle neben der Modelldatei (Registry wie `exp_basic`) werden nie verlangt. `read_repo` liest bis zu 5 statt 2 Importebenen, begrenzt durch das bestehende Zeichenbudget. | **Verhaltensaenderung:** Laeufe, die frueher mit unvollstaendigem Protokoll starteten, stoppen jetzt vor dem Experiment. |
| E16 | `requested_ref`, `RepoLink.ref`, `RepoDump.ref`: jeder ausdrueckliche Ref aus `tree/<ref>/...` eines Paperlinks wird geklont und ausgecheckt (`git checkout -q <ref>`), **auch `main` und `master`** (Nachbesserung: der Standard-Branch kann ein anderer sein). Nur `HEAD` bezeichnet den Standard-Branch. Fehlt der Ref, scheitert das Setup sichtbar, ohne zweiten Klon. | `clone_plan` behaelt seine Signatur; Repo-Suchtreffer (Tavily) sind keine Anforderung des Papers und bleiben beim Standard-Branch. Katalogeintrag TSMixer (`tree/HEAD/tsmixer`) und alle Fixtures unveraendert. Folge: Ein Paperlink auf einen inzwischen umbenannten `master` scheitert jetzt sichtbar statt still HEAD zu lesen. |
| E17 | `method_guard`: konstante Nullen auch ueber eine nie veraenderte Variable, `0 * x`, `full_like(x, 0)`; umbenannte Retrieval-Methode mit Eingabeparameter wird gegen die Retrieval des Repos insgesamt geprueft. Hilfsfunktionen `init*`, `reset*`, `pad*` usw. bleiben erlaubt. | Weiterhin keine Methodentreue-Zertifizierung. |
| E18 | `training_check`: Wiederherstellung per `state_dict=`-Keyword und ueber Attribute (`self.best`); Trainingsschritt = `backward()` oder `step()` eines beliebig benannten Optimierers (ausser Scheduler); flache Kopien auch per Dict-Comprehension, `detach()` ohne `clone()`, `OrderedDict(...)`. | Wiederherstellung in einer anderen Funktion bleibt ungeprueft (wie bisher). |
| E19 | `code_check.executable_text`: Kommentare und Docstrings werden vor den Regeln positionsgleich ausgeblendet; `fetch_*` nur noch fuer bekannte Datensatz-Lader. Strings im Code (z. B. URL) bleiben verboten; nicht parsebarer Code wird unveraendert (streng) geprueft. | Netzsperre `unshare -n` bleibt die erste Schicht. |

### Testbelege

| Befehl | Ergebnis |
| --- | --- |
| Fokussierte Suite vor den Aenderungen (23 Dateien: Spec, Raw, Check, Config, Repo, Source, Guards, Kandidaten, Val-Label, Referenzen, Nachlauf) | 427 passed, 3 skipped |
| Dieselbe Suite nach den Aenderungen plus `tests/test_reliability_spec_guards.py` und `tests/test_author_code_mode.py` | 491 passed, 3 skipped (nach E16-Nachbesserung 493 passed, 3 skipped) |
| E16-Nachbesserung | `test_requested_main_is_not_replaced_by_a_different_default_branch` (Repo-Default master, angefordert main: sichtbarer `SandboxSetupError`, genau ein Klonversuch), `test_paper_link_keeps_every_explicit_ref`; `tests/test_repo_resolution.py`: TSMixer-Paperlink erwartet jetzt `tree/master/tsmixer` statt `tree/HEAD/tsmixer` |
| Neue Datei `tests/test_reliability_spec_guards.py` gegen den unveraenderten Stand `e945e01` (git archive im Scratchpad) | 34 failed, 17 passed: alle Befundtests rot, alle Erhaltungstests gruen |
| Geaenderter Test | `tests/test_spec_raw.py::test_raw_items_shell_flags_ambiguous_runs_are_skipped`: Der Aufruf fuer Modell B verdraengt den Wert fuer Modell A nicht mehr (neue Invariante), echte Mehrdeutigkeit wird zusaetzlich geprueft. |
| Echte Artefakte | Referenzloesungen `tests/reference/*.py`: keine Treffer der Guards; vorher meldete `code_check` bei `gcnii_cora_reference.py` faelschlich „Planetoid“ aus dem Docstring. Sechs gespeicherte SARAF-Codeversionen: Platzhalter-, Checkpoint- und Code-Guard unveraendert (keine neuen Fehlalarme). |

Erstaudit-Proben nach den Fixes: `training_vs_inference_batch`, `unrelated_keyword_conflicts`, `silent_config_overwrites`,
`placeholder_blind_spots`, `checkpoint_blind_spots`, `protocol_dependency_omission`, `comments_rejected_as_operations` und
`inactive_model_setting_enforced` bestaetigen das alte Verhalten nicht mehr. `unrelated_default_stripping`, `multi_command_values`,
`paper_number_collision` und `requested_repo_ref_lost` laufen weiter gruen, weil sie die Funktionen ohne Experimentkontext, mit einer
flachen Zahlenmenge oder ueber die unveraenderte `clone_plan`-Signatur aufrufen. Der Produktpfad (`build_spec`, `paper_numbers`,
`requested_ref`/`read_repo`) ist durch eigene Tests abgedeckt, darunter `test_build_spec_passes_the_experiment_to_script_parsing`.
Die vollstaendige Suite laeuft nach der Integration in `ceo`.
