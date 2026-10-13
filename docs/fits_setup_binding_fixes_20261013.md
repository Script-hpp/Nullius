# FITS: Ursachenfix fuer den autonomen Lauf vom 13. Oktober 2026

Der Live-Lauf `ee039f0f-5a4c-47d3-83a7-ad7388473738` hatte das richtige Paperziel
0.372, aber falsche Trainingsparameter und den falschen Autoren-Unterpfad gelesen.
Sein Mittelwert 0.7035825614 bleibt als historische Messung erhalten. Die
Ursachenbelege stehen in [fits_live_setup_verification_20261013.md](fits_live_setup_verification_20261013.md).

## Umgesetzt

1. Der aktive LaTeX-Text fuer Setup-Angaben ignoriert auch Inline-Kommentare.
   Originaltext, Quellenhash und Target-Bindung bleiben unveraendert. Ein
   Mehrfachheader mit 90/180/360/720 oder eine Suchliste belegt keinen Einzelwert.
2. Quellengepruefte Discovery-Details erreichen die Spec unabhaengig davon,
   ob der zweite Extraktor sie erinnert. `discovery` ist Herkunft; das woertliche
   Zitat muss weiterhin im aktiven Paper stehen.
3. Die Forecast-Dateiauswahl zieht das kanonische `models/FITS.py`, das
   ETTh1-Startskript und dessen Python-Einstieg hinzu. Ein im README ausdruecklich
   als finales Preset beschriebenes `_best`-Skript ersetzt Ablationsskripte.
   Eine falsche AD-Auswahl wird korrigiert; Modellregister fuehren nicht zum
   Einlesen saemtlicher fremder Baselines.
4. Ein begrenzter statischer Interpreter liest Shell-Bindungen, argparse-Werte
   und bekannte nachfolgende Arithmetik. Autorenprogramme werden dabei nie
   ausgefuehrt. Bei FITS werden Batch64 ohne Augmentation zu Batch128 und
   `int(seq_len // base_T + 1) * H_order + 10` zu cut_freq106.
5. Identitaet umfasst Methode, Datensatz, Horizont96 und multivariate FeaturesM.
   Identitaetsschleifen werden auf ihren belegten Zielwert eingeschraenkt;
   Hyperparametergitter werden nicht geraten. Die belegte Eingabelänge waehlt
   kompatible Aufrufe aus. Ungeloeste aktive Einstellungen und gleichrangige
   widerspruechliche Presets stoppen vor dem Experiment. `--feature` wird als
   eindeutige argparse-Abkuerzung beruecksichtigt.
6. Mutationen in Helfern/Klassen/importierten Trainern, `setattr`, Namespace-
   Ersatz sowie `vars`-/`__dict__`-Updates lassen unbekannte Werte nicht als
   vermeintlich gueltige CLI-Defaults stehen. Numerische Zusatzflags werden
   nur verpflichtend, wenn der gelesene Modell-/Trainingspfad sie verwendet.
7. Im Produktpfad mit unabhaengiger Auswahl muss ein Kandidat alle belegten
   Pflichtwerte verwenden. Ein unbenutzter CONFIG-Key wird vor Sandbox-Start
   erkannt; eine unvollstaendige Spec wird auch bei finaler Zulassung blockiert.
   Damit wird aus einem bereits bekannten Setupfehler kein Paperurteil.
8. Der Forecast-Worker meldet die Primaerausnahme ueber einen separaten Pipe-FD.
   Der Broker zeigt bei EOF/BrokenPipe diese Ursache als letzten Fehlerblock,
   statt sie mit seinem eigenen Traceback zu verdecken. Rechte, kausale
   Inferenz, Wiederholungsproben und Transfergrenzen bleiben bestehen.

## Belegte Parameterkette

| Einstellung | Falscher Live-Lauf | Aus Originalquellen abgeleitet |
|---|---:|---:|
| Eingabelänge | 90 | 360 |
| Effektive Batchgroesse | 32 | 128 |
| Lernrate | 0.001 | 0.0005 |
| Maximale Epochen | 10 | 50 |
| Early-stopping patience | 3 | 20 |
| Frequenzgrenze | 3 | 106 |
| Prognosehorizont / Kanaele | 96 / 7 | 96 / 7 |
| train_mode | unzuverlaessig | 1 |

Die Werte stammen aus `etth1_best.sh`, `run_longExp_F.py` und der belegten
Discovery-Eingabelänge. Kein Testscore waehlt die Werte. Die FITS-Quellfixture
enthaelt die Apache-2.0-Lizenz des gepinnten Autoren-Repos.

## Validierung

- Neue Regressionen reproduzierten vor dem Fix sechs Fehler bei sieben ersten
  Proben. Settings-/Spec-/Repo-Regressionen: 143 Tests bestanden, inklusive unbekannter
  und sicher univariater Feature-/Kanalidentitaet vor dem Rohwert-Fallback.
- Echte gecachte SparseTSF-/SegRNN-Skripte testen Horizontschleifen und
  Eingabelänge720. Opus hat ausserdem DLinear, NLinear, MoHETS und GCN anhand
  realer gecachter Repo-Kontexte gegengeprueft.
- Der komplette FITS-Dateipfad liest 13 Dateien einschliesslich importierter
  Trainer/DataLoader. Eine absichtlich falsche AD-Auswahl fuehrt zu den neun
  richtigen numerischen Pflichtwerten. Auch die Feature-/Kanalidentitaet wird effektiv geprueft, inklusive
  post-parse Mutationen und argparse-Defaults. Die Quellenfragmente sind echte Zitate
  aus den jeweils angegebenen Dateien.
- Produktionsimage `nullius:settings-20261013` erfolgreich gebaut; derselbe
  13-Dateien-/Spec-/CONFIG-Pfad darin netzlos erfolgreich geprueft. [Produktionsbeleg](audits/fits_setting_production_20261013.json): finales
  Image und Checkout quellhashidentisch, daraus erzeugter Adapter hashidentisch
  mit den drei Trainingslaeufen, unbekannte Features blockiert. Git-Metadaten
  wurden aus dem gepinnten lokalen Checkout uebergeben, Autorenquellen als
  schreibgeschuetzter Mount. Kein externer Download und kein Modellaufruf. Containerstart, /api/health
  und /api/status lieferten HTTP200.
- Der tatsaechlich gespeicherte schlechte Live-Kandidat `947d5088ab2c`
  wird mit der neuen Spec vor jedem Sandbox-Aufruf statisch zurueckgewiesen:
  [Gegenbeleg](audits/fits_historical_setting_guard_20261013.json).
- Worker-Fehler und Sicherheitsvertrag: 32 netzlose Docker-Tests bestanden,
  einschliesslich Primaerausnahme, sys.exit, SIGKILL, EOF, kausaler Inferenz,
  frischer Bewertung und Kalenderdaten.

Drei neue lokale Trainingslaeufe mit dieser automatisch abgeleiteten CONFIG
liefern 0.372051543 / 0.371997134 / 0.372089070; Mittelwert **0.372045916**
bei Stichproben-Standardabweichung **0.000046225**. Alle drei liegen innerhalb
der unveraenderten Toleranz 0.372 +/-0.01. Abschliessender Backend-Gate: **1567 passed, 10 skipped, 8 deselected,
5 xfailed** (364.82s), keine Fehler. Die bewusst ausgelassenen Integrations-/
Baseline-Tests sind durch diesen Gate nicht neu belegt; relevante echte
Sandbox-Vertraege wurden separat mit Docker getestet. Maschinenlesbare Belege:
[fits_setting_handoff_20261013.json](audits/fits_setting_handoff_20261013.json),
[Settings/Quellen](audits/fits_setting_handoff_20261013.settings.json),
[eingefrorener Adapter](audits/fits_setting_handoff_20261013.adapter.py).
Reproduktion des lokalen Checks: `uv run python scripts/review/fits_setting_handoff.py`.

## Aussagegrenze

Der lokale Trainingscheck verwendet einen festen menschlich geschriebenen
Referenzadapter um den Autoren-Modellkern. Neu ist die automatisch abgeleitete
Source->Spec->CONFIG-Kette. Dies beweist nicht, dass Nemotron bereits einen
korrekten autonomen Neuaufbau erzeugt. Dafuer braucht es nach Deployment einen
neuen Produktlauf; ein Erfolg wird nicht vorweggenommen. Unbekannte Python-
Konfigurationen werden konservativ gestoppt. Es wurde kein historisches
Ergebnis umgeschrieben und kein Parameter anhand des Testwertes optimiert.
Fuer Nullius hat diese Fixrunde null zusaetzliche Nebius-/OpenRouter-/Tavily-
Aufrufe verursacht. Opus beriet separat ueber den freigegebenen Orca-Worker.

Unabhaengige Reviews: [Worker](opus_forecast_worker_error_fix_20261013.md) und
[Settings](opus_setting_fixes_review_20261013.md).
