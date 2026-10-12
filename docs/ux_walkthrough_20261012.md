# UI/UX-Durchgang aus Erstnutzersicht — 12. Oktober 2026

**Ergebnis vor den Reparaturen:** Der Einstieg und die Ergebnisdarstellung sind stark. Der Weg vom Ergebnis zu einer nachvollziehbaren Originalquelle und zum erzeugten Code ist noch zu umstaendlich. Die naechsten Verbesserungen sollten diese Wege erleichtern, bevor weitere Funktionen dazukommen.

Agentengestuetzter Durchgang der deployten Seite https://nullius.akca.dev/, kein abgeschlossener menschlicher Nutzertest. Der Nutzer verschiebt den menschlichen Test auf spaeter. Beobachtete Bedienwege und konkrete UI-Zustaende werden von meiner Einschaetzung ihrer Wirkung getrennt. Keine neuen kostenpflichtigen Laeufe, keine Kontaktaufnahme und keine Produktcode-Aenderung in dieser Runde.

## Durchgang

1. Startseite: Problem lesen, Primaeraktion `Watch a check in 30 seconds` auswaehlen.
2. TSMixer-Wiedergabe ansehen, Ergebnis und numerische Einschraenkung lesen, `Inspect sources & setup` oeffnen.
3. Zum Katalog wechseln, GLinear ueber den Titel oeffnen, Quellen- und Code-Abkuerzungen benutzen, eingeklappten Code auffinden.
4. What-if oeffnen, den Zielwert veraendern und zuruecksetzen; pruefen, ob das gespeicherte Urteil unveraendert bleibt.
5. `Back to runs` benutzen, weitere Navigation ueber Discover, aufgezeichnete Discovery bei 4x, Benchmark und Contact.
6. Den Kernweg bei **390 × 844 Pixeln** wiederholen: Startseite → Katalog → GLinear → Quellen → Code → What-if → mobiles Menue → Benchmark/Discover.
7. Export-Erlebnis mit der direkt vorausgehenden Deploy-Abnahme abgleichen: tatsaechlich geoeffnete Druckvorschau und gespeichertes dreiseitiges A4-PDF mit Quellen, Einschraenkungen und drei finalen Seeds.

Die sichtbaren Klickwege wurden zuerst benutzt; DOM-Ausgaben dienten danach der Belegsicherung. Screenshots und mobile Beobachtungen liegen unter `/tmp/nullius-ux-*`; das gespeicherte Produktions-PDF unter `/tmp/nullius-deploy-glinear-certificate.pdf`.

## Was bereits gut funktioniert

- Die Startseite vermittelt das Problem in wenigen Saetzen und bietet eine eindeutige erste Aktion ohne Anmeldung.
- Die Wiedergabe ist klar als Recorded beschriftet und zeigt Quelle, Code, Reparatur, Sandbox und Ergebnis. Historische Laufdauer und komprimierte Wiedergabe werden unterschieden.
- Claim und Messung stehen nebeneinander. `Target value reached · method not verified` ist nun sichtbar; kleine Streuungen bleiben lesbar.
- Gesamtergebnisse, Fehlversuche und ungepruefte Kandidaten sind ueber die Katalogfilter erreichbar.
- What-if aendert den hypothetischen Zielwert und sein Urteil; die gemessenen Punkte und das gespeicherte Haupturteil bleiben gleich. Reset funktioniert. Das ist ein gutes erklaerendes Beispiel.
- Der Druckexport liefert ein eigenstaendiges, lesbares Zertifikat mit Limits und Quellen. Der Kontaktweg ist kurz und verstaendlich.
- Die mobile Bedienung funktioniert. Die wesentlichen Probleme sind Reihenfolge, Textmenge und Beschriftung, keine im Durchgang beobachteten generellen Bedienblockaden.

## Priorisierte Befunde

### P1 — 1. GLinear-Quellenansicht endet ohne benutzbaren Quellenweg

**Beobachtet:** Vom Katalog aus GLinear geoeffnet, dann `Inspect sources & setup`. Die Quellenansicht zeigt `No verified target evidence found.` und `What others found — 0 notes`. Im Panel ist **kein Quellenlink**. Auch der Paper-Link am Kopf fehlt bei diesem Lauf. Die im Dossier/PDF vorhandene manuelle Quelle ist hier nicht erreichbar. GLinear hat weiterhin dokumentierte Grenzen der automatischen Zielwert- und Spaltenpruefung.

**Wirkung:** Nach einem gruenen Ergebnis fuehrt die zentrale Belegaktion in eine Sackgasse. Die fehlende automatische Pruefung kann wie voellig fehlende Evidenz wirken.

**Verbesserung:** Vorhandene Originalquelle, aufgezeichnete Fundstelle und Herkunft zeigen, deutlich als manuell uebernommen bzw. automatisch nicht bestaetigt markieren. Keine fehlende Verifikation erfinden. Eine direkte Aktion zur Originalquelle und zu den einschlaegigen Einschraenkungen anbieten.

**Abnahme:** Nach einem Klick auf Quellen kann man GLinears Originalquelle oeffnen und versteht zugleich, welche automatische Pruefung fehlt.

### P1 — 2. Code-Abkuerzung zeigt den Code nicht

**Beobachtet:** `Inspect code & measurements` oeffnet den Tab `Code & branches`. Zuerst kommen Branch-Baum, Seed-Tabelle und Chart. `Plan and code` bleibt **eingeklappt**. Mobil liegt dessen Anfang nach dem Sprung ungefaehr am unteren Viewportrand; erst nach weiterem Suchen und einem weiteren Klick erscheinen `Agent code` und `Plan`.

**Wirkung:** Eine Aktion mit "Code" im Namen liefert zunaechst keinen Code. Der Nutzer muss die Informationsarchitektur selbst entschluesseln.

**Verbesserung:** Direkte Aktionen `View generated code` und `View measurements`; jeweils das passende Element oeffnen und anspringen. Den Branch-Baum als vertiefende Versuchshistorie erreichbar halten. Diese Abkuerzungen nahe beim Urteil anbieten, ohne erst die neun Schritte lesen zu muessen.

**Abnahme:** Ein Klick auf Generated code zeigt das erzeugte Programm, ein Klick auf Measurements die finalen Seeds. Die Versuchshistorie bleibt erreichbar.

### P1 — 3. "Code" und "Replay" meinen je nach Stelle etwas anderes

**Beobachtet:** `Code` auf GLinears Katalogkarte fuehrt zum **Autorencode** unter `github.com/t-rizvi/GLinear`. Die Ergebnisansicht zeigt dagegen Nullius' erzeugten Code. `Watch replay` im Katalog verlinkt auf `?tab=log` (die Ereignis-/Logansicht), waehrend die Startseiten-Wiedergabe `?play=1` benutzt (den 30-Sekunden-Film).

**Wirkung:** Der gleiche allgemeine Begriff laesst unterschiedliche Ziele erwarten. Gerade die Unterscheidung zwischen Autorencode und erzeugtem Experiment gehoert zum Kernversprechen.

**Verbesserung:** `Authors' repository`, `Generated experiment`, `30-second replay` und `Event log` konsistent unterscheiden. Den Film unter derselben Film-Beschriftung ueberall oeffnen; fuer den Log einen eigenen Begriff verwenden.

**Abnahme:** Nutzer koennen vor dem Klick erkennen, wessen Code sie sehen und welche Form der Wiedergabe sie oeffnen.

### P2 — 4. Mobil kommt der Inhalt nach zu viel Einleitung

**Beobachtet:** Im Katalog beginnt die erste Karte bei **y ≈ 848 px** im **844 px** hohen Viewport. Titel, `What you see here`, `What to look at first`, erklaerende Absaetze, Herstellungshinweis und Filter verbrauchen den gesamten ersten Bildschirm. Das Muster wiederholt sich auf anderen Inhaltsseiten.

**Wirkung:** Auf einem Telefon muss man scrollen, bevor man ein Paper auswaehlen kann. Die Seite erklaert ihre Bedienung laenger, als die Bedienung selbst dauern sollte.

**Verbesserung:** Mobil Titel, einen knappen Kontextsatz und Filter vor die Karten stellen. Ausfuehrliche Anleitung in die vorhandene aufklappbare Erklaerung verschieben. Abstaende im mobilen PageIntro reduzieren. Desktop-Stil und gut lesbare Typografie erhalten.

**Abnahme:** Beim Eintritt in den mobilen Katalog sind Filter und mindestens der Anfang einer Paperkarte ohne Scrollen sichtbar.

### P2 — 5. Zurueckweg verliert den Auswahlkontext

**Beobachtet:** Katalog → GLinear → `Back to runs` fuehrt zu `#overview`, also zur Runs-Sammlung, nicht zur vorherigen Katalogauswahl.

**Wirkung:** Zum Vergleichen weiterer Papers muss man zur richtigen Sammlung zuruecknavigieren; Filter und Position sind nicht Teil des Rueckwegs. Die Beschriftung ist woertlich korrekt, aber fuer diese Reise unpassend.

**Verbesserung:** Rueckkehr zum Ursprung, etwa `Back to catalog`, mit vorheriger Filter-/Scrollposition. Bei Einstieg aus Runs weiterhin dorthin zurueck. Direkte Links brauchen einen nachvollziehbaren Standardweg.

### P2 — 6. Benchmarks beantworten die Nutzerfrage zu indirekt

**Beobachtet:** Schon der erste Abschnitt zeigt 0/48, 0/9, 7/7, 2/8, 7/8 und 0/1 mit Begriffen wie `reference verdict`, `pre-registered`, `post hoc`, `in-sample` und `held-out`. Die Einordnungen sind vorhanden, verlangen aber Aufmerksamkeit und Vorwissen.

**Wirkung:** Die Frage `Does it work?` wird fuer eine nichtwissenschaftliche Jury nicht schnell genug beantwortet. 7/8 Entwicklung und 0/1 separate Gegenprobe koennen trotz Fussnoten unterschiedlich gewichtet oder als allgemeine Erfolgsquote gelesen werden.

**Verbesserung:** Erst eine kurze Aussage in Alltagssprache, dann die drei getrennten Fragen: Entwicklungsbeispiele, separate Probe an einem nicht dafuer abgestimmten Paper und falsche Zielwerte. Bedeutungen direkt neben die Zahlen schreiben; historische Versionen, Generationen und Tabellen darunter zur Vertiefung. Keine allgemeine Erfolgsrate ableiten und den Held-out-Miss nicht verstecken.

### P2 — 7. Eigene Paper und neue Laeufe haben keinen eindeutigen Einstieg

**Beobachtet:** `Check a claim yourself`/`Try it` fuehrt zum Katalog. Discover bietet Datensatz-Checkboxen und Suche, aber **kein arXiv-Eingabefeld**. Reproduktionskarten sagen `Quick check, 3 seeds, watched live`; eine konkrete Dauer wird dort nicht genannt. Der TSMixer-Replay nennt seinen historischen Lauf mit **45 Minuten**.

**Wirkung:** Wer mit einem bestimmten Paper ankommt, weiss nicht, ob es hier eingegeben werden kann. `Quick` erzeugt eine Erwartung, ohne die Wartezeit einzuordnen.

**Verbesserung:** Den tatsaechlichen Weg explizit machen: vorhandenes Paper auswaehlen oder Papers auf unterstuetzten Datensaetzen finden. Ein eigenes Eingabefeld nur anbieten, wenn es durchgaengig unterstuetzt ist. Neue Messung, gespeicherter Check und Replay klar unterscheiden; vorhandene Laufzeiten zur ehrlichen Zeiteinordnung verwenden, ohne eine unbelegte Dauer zu versprechen.

## Kleinere Klarheitsprobleme

- Die TSMixer-Quelle ist eine Zahlenzeile ohne sichtbare Methoden-/Metrikueberschriften. `The third number is our target` ist fuer Erstnutzer wenig hilfreich. Vorhandene Tabellenstruktur bzw. Kontext und ungepruefte Spaltenzuordnung klar zeigen; keine automatische Spaltenbestaetigung behaupten.
- `Inspect sources & setup` zeigt bei TSMixer keine Setup-Liste, obwohl der Name diese erwarten laesst. Wenn keine belegten Setup-Angaben vorhanden sind, das ausdruecklich sagen und die naechste sinnvolle Quelle anbieten.
- GLinear zeigt in den neun Schritten `3 of 3 seeds finished` und `6 runs scored`. Beides beschreibt unterschiedliche Mengen, diese Unterscheidung fehlt aber in der Beschriftung. Klarer: sechs gemessene Versuche, drei fuer das Endurteil ausgewaehlt.
- `Jury ready` und `Ran through the loop` auf Produktkarten sind Begriffe fuer die Projektpraesentation. Nutzerbezogene Labels wie `Evidence available` und `Checked` waeren schneller verstaendlich.
- Die mobile Startseite hat andere Navigationsmoeglichkeiten als die Unterseiten mit Hamburger-Menue. Eine konsistente Navigation wuerde den Zugang zu Discover und Benchmark erleichtern.
- Beim ersten Eintritt in Runs war kurzfristig gleichzeitig `Loading checks…` und `No checks yet` sichtbar. Ladezustand und echte leere Sammlung sollten unterschiedliche Texte haben.

## Empfohlener naechster Arbeitsumfang

1. GLinear-Quellenweg wieder benutzbar machen, ohne seine Verifikationsgrenzen zu verdecken.
2. Die Code-/Messungs-Abkuerzungen an ihr versprochenes Ziel fuehren und Autorencode, erzeugten Code, Film und Log eindeutig benennen.
3. Mobile Einleitungen verkuerzen und den Katalog-Rueckweg erhalten.

Danach Benchmark-Einordnung und Einstiegsbeschreibung vereinfachen. Keine weiteren Datensaetze, Modelle, automatischen Kalibrierungen oder neuen bezahlten Experimente fuer diese UX-Hebel erforderlich.

## Umsetzung nach dem Durchgang — 12. Oktober 2026

Der Nutzer hat die Reparaturen anschliessend beauftragt. Die Beobachtungen oben beschreiben die vorherige Deployment-Version; die folgenden Punkte sind im Produktcode umgesetzt.

- [x] **Quellenweg:** GLinear zeigt den gespeicherten arXiv-HTML-Link und Originalauszug auch ohne `target_evidence`. Fehlgeschlagene Ziel-/Spaltenverifikation bleibt sichtbar; Daten und Verifikationsstatus werden nicht veraendert. TSMixer zeigt Originalzeile, Tabellenort und ungepruefte Spaltenzuordnung statt einer vermeintlich aussagekraeftigen Nummernposition.
- [x] **Setup:** Vorhandene `setup_details` sind jetzt sichtbar, mit Quelle, Code-Commit, Anwendungsbereich und getrennten ausgeschlossenen Referenzen. Es handelt sich um gesammelte Referenzen einschliesslich Defaults und Alternativen, nicht um eine verifizierte Liste der finalen Programmeinstellungen.
- [x] **Direkte Aktionen:** Quellen/Setup, erzeugter Code und Messwerte sind vom Urteil erreichbar. Die Code-Aktion waehlt das erzeugte Experiment, oeffnet dessen Bereich und bringt es ins Bild; Messwerte oeffnet die drei ausgewaehlten Seeds.
- [x] **Replay und Codebegriffe:** Katalog und Startseite verwenden denselben 30-Sekunden-Film. Das Ereignisprotokoll hat einen eigenen Link; die Autoren-GitHub-Seite heisst `Authors’ repository`, der erzeugte Code `Generated experiment`.
- [x] **Rueckweg:** Ein Check merkt sich den Herkunftskatalog samt Filter und Scrollposition; direkte Einstiege haben Runs als Standard. Wiederherstellung wartet auf geladene Karten und bekannten API-Zustand, damit die unterschiedlich lange Live-Einleitung die Position nicht nachtraeglich verschiebt. Der Ursprung gilt innerhalb der geoeffneten App; er wird nicht zwischen Browser-Neuladungen gespeichert.
- [x] **Mobile Einleitung:** Katalog, Runs, Discover und Benchmark haben kurze Einleitungen; ausfuehrliche Hinweise sind mobil aufklappbar und auf Desktop weiter sichtbar. Auf der Startseite gibt es nun ebenfalls ein mobiles Menue. Ein Konflikt zwischen `hidden` und der externen Link-Komponente, der den Menueknopf hinausschob, wurde im Browser gefunden und behoben.
- [x] **Benchmark:** Vor den Detailzahlen stehen die drei getrennten Fragen: Entwicklungsbeispiele (7/8), neues RLinear-Paper (0/1) und falsche Ziele (0/48 bzw. separate 0/9). Kein gemeinsamer Erfolgswert; der neue Fehlschlag bleibt ausdruecklich sichtbar.
- [x] **Einstieg und Dauer:** Vorhandene Auswahl/Discover und fehlender Paper-URL-Eingang sind benannt. `Start new check` verspricht keine schnelle Fertigstellung; gespeicherte Laufdauer kommt aus den aufgezeichneten Ereignissen und wird vom komprimierten Film unterschieden.
- [x] **Kleine Klarstellungen:** `Checked` und `Evidence available` ersetzen interne Praesentationsbegriffe. GLinear unterscheidet sechs gemessene Versuche von drei fuer das Urteil ausgewaehlten Messungen. Runs zeigt waehrend des Ladens keinen falschen Leerzustand mehr.

### Validierung der Reparaturen

- **129 Frontend-Tests bestanden**, darunter sechs neue Regressionen fuer gespeicherte Quellen, ungepruefte Spalten, sichere Quellenlinks und Herkunft/Filter/Scrollposition.
- Produktionsbuild und Lint erfolgreich; vorhandene Bundle-/React-Warnungen bleiben, keine Lint-Fehler.
- Browserdurchgang auf Desktop (1440 × 1000) und mobil (390 × 844): Menue → Katalog → alle Eintraege → GLinear → Quelle/Setup → erzeugter Code → drei Endmessungen → Katalog zurueck → Ereignislog → Film → Benchmark → aufgezeichnete Discovery. Keine JavaScript-Seitenfehler.
- Mobil beginnt die erste Karte bei rund **606 Pixeln**, vorher rund **848 Pixel**. Der Rueckweg stellte die beim Klick gespeicherte Scrollposition wieder her (Desktop 262 → 262, mobil 541 → 541); `All entries` blieb aktiv.
- Lokale GET-Antwort fuer den freien Job-Zustand in der Browserpruefung simuliert, damit die neuen Start-Beschriftungen sichtbar sind; Nicht-GET-Anfragen blockiert. Keine frischen Live-Laeufe oder neuen wissenschaftlichen Messergebnisse.
- Neuer menschlicher Test bleibt offen. Diese Reparaturen benoetigen ein neues Deployment durch den Nutzer; die vorherige Deploy-Abnahme von `1458688` bleibt ein separater Nachweis.
