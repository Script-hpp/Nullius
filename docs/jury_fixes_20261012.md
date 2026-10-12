# Jury-Fixes — Aufgaben und Stand, 12. Oktober 2026

Auftrag: Befunde aus [jury_review_20261012.md](jury_review_20261012.md) notieren und mit der Reparatur beginnen. Basis `792d116` nach Fetch/Pull. Die historische Bewertung bleibt erhalten; dieser Arbeitsstand beschreibt die anschliessenden Aenderungen.

## Lokal umgesetzt

- [x] **Endmessungen statt aller Kandidaten anzeigen.** Ein gemeinsamer Selektor verwendet die vorhandene `chosen`-Markierung des Orchestrators. Ergebnisband, Detaildiagramm, Seed-Tabelle, What-if, Replay und Druckexport verwenden diese Auswahl. Kandidaten, Fehler und Reparaturen bleiben im Branch-Baum. Alte Laeufe ohne Markierung behalten ihre aufgezeichneten Messungen. Weder Zielwerte noch Urteile oder Laufdaten werden neu berechnet oder veraendert.
- [x] **GLinear-Regression absichern.** Drei finale Seeds mit Code `67dae24762de`; ihr Mittel stimmt mit dem gespeicherten Urteil `0.37958288517117206` ueberein. Die drei anderen gemessenen Codeversionen gehen nicht mehr in die Endergebnisgrafik ein. Auch die uebrigen veroeffentlichten Laeufe werden auf Uebereinstimmung von ausgewahlten Messungen und gespeichertem Mittel geprueft.
- [x] **MSE/MAE als Dezimalwerte formatieren.** Quellen und Seed-Tabelle verwenden die Metrik. Nichtprozentuale Ergebnisse stehen mit vier Nachkommastellen; kleine Streuungen wie `0.0003` bleiben sichtbar. Bei noch kleineren Streuungen verhindert eine wissenschaftliche Schreibweise das Wegformatieren zu null. Accuracy bleibt Prozent. Hoehere Fehlerwerte bekommen die entsprechende negative Delta-Farbe.
- [x] **Autorenzuordnung reparieren.** Die Startseite nennt das Paper des geladenen Belegs statt fest Kipf & Welling. Zahl und Titel stammen nun aus demselben Beispiel.
- [x] **Numerisches Ergebnis und Methodengrenze klarer zeigen.** REPRODUCED bleibt gemaess der dokumentierten Nutzerentscheidung erhalten. Direkt unter dem Urteil steht bei ungepruefter Methode deutlich `Target value reached · method not verified` bzw. `Numerical check only · method not verified`. Katalogmarker und Druckvorschau nennen die ungepruefte Methode ausdruecklich.
- [x] **Pauschale Praezisionsversprechen begrenzen.** Startseite, Runs und Zweimodusansicht beschreiben die beobachteten Benchmark-Ergebnisse; 0/48 wird nicht als Garantie fuer weitere Papers dargestellt. Die Ablaufbeschreibung unterscheidet Paperquellen von Tavily-Recherche zur Laufzeit.
- [x] **Katalog vollstaendig erreichbar machen.** Kuratierter Einstieg bleibt `Featured checks`; daneben `All entries`, `Failed checks` und `Unchecked`. Filter haben teilbare URLs (`#catalog?view=all`, `failed`, `unchecked`). `Explore all outcomes` oeffnet alle Eintraege. Kandidaten ohne veroeffentlichte Evidenz erhalten keinen erfundenen Ergebnislink.
- [x] **Demo-Adresse und Dokumentation synchronisieren.** Badge-Snippets verwenden `https://nullius.akca.dev`. README und Devpost-Entwurf nennen die am 12. Oktober gepruefte Demo. Neue Video-/Devpost-Veroeffentlichung bleibt offen. README und Einreichungstexte nennen den nachfolgenden Kosten-/Timeout-Fix mit seinen verbleibenden Grenzen; historische Reviews bleiben als historische Befunde bestehen. Der Tavily-Detail-Link zeigt auf den existierenden Abschnitt in `docs/evaluation_notes.md`.

## Validierung

- Produktionsbuild und Lint erfolgreich. Bekannte Warnungen zu grossen Bundles, React-Fast-Refresh und bestehenden Discovery-Renderpfaden bleiben; keine Lint-Fehler.
- **123 Frontend-Tests bestanden**, einschliesslich fuenf neuer Regressionen mit echten exportierten GLinear-/Katalogdaten, Legacy-Laeufen und Metrikformatierung.
- Lokaler Browser auf Port 5183: GLinear-Urteil und drei Diagrammpunkte, drei finale Seed-Zeilen, TSMixer-Quelle `0.3610 MSE`, dynamischer Papername und eingeschraenkte Benchmark-Texte geprueft.
- Druckvorschau tatsaechlich geoeffnet und ihr Dokument geprueft: drei finale GLinear-Seeds, passender Mittelwert und Streuung, Quellen und Einschraenkungen. Nativer Druckdialog und als Datei gespeichertes PDF sind hier nicht erneut abgenommen.
- Katalog im Browser: 17 Eintraege insgesamt; vier erfolglose Checks; sieben ungepruefte Kandidaten ohne Ergebnislinks. Neue All-Entries-URL oeffnet die Katalogansicht und den richtigen Filter.
- Keine neue belastbare Mobilabnahme: die Orca-Viewport-Anfrage veraenderte die effektive native Fensterbreite nicht. Keine bezahlten Anbieter- oder Sandbox-Laeufe gestartet. Backend und gespeicherte Evidenzdaten blieben unveraendert.

## Noch offen

- [ ] **Deployment dieser Aenderungen.** Der Nutzer hat den Push der geprueften Reparaturen auf `main` zum Deployment beauftragt. Ein Push aktualisiert noch nicht automatisch die Demo; Deployment liegt laut CLAUDE.md beim Nutzer. Danach Startseite, GLinear, TSMixer, Katalogfilter und Druckexport auf dem Server pruefen.
- [ ] **Menschlichen Produkttest abschliessen.** [Bestehender Testplan](product_test_20261010.md): ohne Coaching Ergebnis erklaeren, Beleg finden, PDF speichern und naechste Entscheidung nennen. Beobachtungen statt erfundener Zeitersparnis berichten. Die technische Browserpruefung ersetzt diesen Test nicht.
- [ ] **Video neu schneiden/rendern und veroeffentlichen.** [Brief](jury_walkthrough_and_video_next_steps.md#nächster-videoschnitt): ein durchgehender, belegter Fall mit Quelle, festem Ziel, eigenem Code, echtem Fehler und Reparatur, Sandbox und unabhaengiger Messung. Aufzeichnung kennzeichnen; Methodengrenze lesbar zeigen. Kein neues MP4 in dieser Runde erstellt.
- [ ] **Devpost final eintragen.** Demo vorhanden; Video-Link und eigentliche Einreichung weiterhin offen. Texte, Video und fertiges Produkt vor Abgabe abgleichen.
- [ ] **Spaetere Forschungsarbeit getrennt planen.** Allgemeine Leistung auf neuen Papers und Methodentreue bleiben ungeprueft. Keine neue Kalibrierung oder bezahlte Benchmark-Kampagne begonnen. Weitere Live-Demo-/Outreach-Schritte bleiben gemaess Nutzerentscheidung hinter dem menschlichen Produkttest.
