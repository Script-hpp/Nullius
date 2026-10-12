# Unabhaengiger Jury-Review — 12. Oktober 2026

Bewertung des aktuellen Produkts, keine offizielle Juryentscheidung oder Platzierungsprognose. Devpost und ein neues veroeffentlichtes Video werden hier nicht bewertet.

## Stand und Methode

- `git fetch --all --prune`, danach `git pull --ff-only origin main`: 14 Commits von `228634f` auf `792d116`. Arbeitsbaum vorher sauber; kein Anwendungscode fuer diesen Review geaendert.
- CLAUDE.md, README, Einreichungsentwuerfe, Evaluierungen, Kostenreview und bisherige Produktreviews gelesen; ihre alten Bewertungen nicht als eigene Verifikation uebernommen.
- https://nullius.akca.dev im Orca-Browser geprueft: Startseite, TSMixer-Replay, Quellen, Code und Messungen, GLinear, Katalog und Discover. API-Health erfolgreich; Status meldete Reproduktion und Discovery als verfuegbar. Kein neuer bezahlter Lauf gestartet.
- Effektive Browserbreite 1419 Pixel. Eine angefragte 390-Pixel-Breite wurde vom nativen Browser nicht wirksam angewendet; daher keine neue Mobilabnahme behauptet.
- PDF-Aktion meldete eine geoeffnete Druckvorschau. Das neue Fenster war ueber die Browser-Tab-Liste nicht erreichbar; keine vollstaendige PDF-/Druckabnahme behauptet.
- GitHub-API: Repo oeffentlich, MIT erkannt. Live-Bundle enthaelt noch `https://nullius.example`.
- Frontend: Produktionsbuild, Lint und 118 Tests bestanden. Gezielt ausgewaehlte Backend-Tests: 119 bestanden, einer uebersprungen, einer scheiterte an fehlender Provider-Konfiguration. Letzterer bestand separat mit Dummy-Konfiguration und simuliertem Versuch. Keine Aussage ueber die gesamte Python-Suite.

Offizielle Grundlage: vier gleich gewichtete Kriterien, je 1–5 Punkte, laut [Veranstalter](https://nebiusglobalaihackathon.devpost.com/updates/46204-here-s-how-judging-works).

## Bewertung

| Kriterium | Punkte | Begruendung |
|---|---:|---|
| Technological Implementation | 4/5 | Substanzieller Agentenworkflow: eigener Code, Reparaturen, Sandbox-Branches und unabhaengige Auswertung. Schwachstellen liegen in Quellzuordnung, Methodennachweis und konsistenter Darstellung der finalen Messungen. |
| Design | 4/5 | Eigenstaendige Gestaltung, klare Claim-/Messungsansicht und beschrifteter Replay. Falsche Einheiten und uneindeutige Ergebnisbegriffe beeintraechtigen das Vertrauen; Detailansichten bleiben anspruchsvoll. |
| Potential Impact | 3/5 | Plausibles Problem und konkrete Zielgruppe. Kein abgeschlossener Nutzertest, keine gemessene Zeitersparnis oder belastbare Leistung auf neuen Papers. |
| Quality of the Idea | 4/5 | Das Paper als Testsuite, ein Agent der Vorhersagen statt seiner eigenen Bewertung liefert, und nachvollziehbare experimentelle Evidenz sind ein starker Kern. Verwandte Reproduktionsagenten existieren; die konkrete Produktumsetzung ist der Unterschied. |

**15/20, entsprechend 7,5/10 als subjektive Orientierung.** Ernstzunehmender Track-Kandidat; fuer eine Spitzenbewertung fehlen vor allem widerspruchsfreie Evidenz und beobachteter Nutzwert. Konkurrenz wurde nicht bewertet, daher keine Gewinnwahrscheinlichkeit.

## Konkrete Befunde, nach Prioritaet

### 1. Diagramme vermischen Kandidaten und finale Messungen

GLinear `disc_2501_01087--20261010-161916`: sechs gueltige Messungen im Dossier, vier davon Seed 0 aus verschiedenen Codeversionen. Das Endurteil verwendet korrekt drei Messungen von `67dae24762de`: Mittel **0.3795828852**, Streuung **0.0002884590**. Mittel ueber alle sechs: **0.3822087750**.

`web/src/components/ResultChart.tsx:24` zeichnet alle gemessenen Versuche als Seed-Punkte, stellt daneben aber den gespeicherten finalen Mittelwert. `web/src/components/Gauge.tsx:7` nimmt ebenfalls alle Versuche und berechnet deren Mittel selbst. Damit beschreibt die Detailgrafik einen anderen Mittelwert als das Urteil. Die Beschriftung suggeriert Wiederholungen einer finalen Implementierung.

**Abnahme:** Endergebnis, Diagramme, Seed-Tabelle und PDF verwenden dieselbe explizite Menge finaler Versuch-IDs. Andere Kandidaten und Reparaturen bleiben in der Versuchshistorie erreichbar und als solche beschriftet. GLinear liefert im Ergebnis drei finale Punkte und den gespeicherten Mittelwert; kein nachtraegliches Auswaehlen nach Zielnaehe.

### 2. MSE wird als Prozent formatiert

TSMixer: Quellenansicht zeigt `0.361` als **36.1%**, Seed-Tabelle etwa `0.3625` als **36.2%**. `web/src/components/Evidence.tsx:26` und `:60` rufen `toPercent` ohne Metrik auf. MSE ist hier keine Accuracy.

**Abnahme:** Metrik entscheidet ueber Format und Einheit. Hauptansicht, Quellen, Tabelle, Charts, Replay und Export muessen uebereinstimmen. Kleine Streuungen wie **0.0003** nicht als **0.000** wegformatieren.

### 3. Falsche Autorenzuordnung auf der Startseite

`web/src/components/LandingSections.tsx:56` nennt fest **Kipf & Welling**, die Zahl darunter stammt aus dem dynamischen TSMixer-Beispiel (**0.361 MSE**).

**Abnahme:** Papername, Autorenzuordnung, Zahl und Metrik stammen aus demselben Beleg. Fuer ein Produkt, das falsche Behauptungen erkennen soll, ist diese Inkonsistenz besonders schaedlich.

### 4. Ergebnisbegriffe versprechen mehr als gemessen wurde

Grosses **REPRODUCED**, darunter **Method fidelity not checked**. Die Einschraenkung ist sichtbar, aber viele Nichtfachleute werden den grossen Begriff als Bestaetigung des Papers verstehen. Ein korrekt gemessener Zielwert belegt nicht die korrekte Umsetzung seiner Methode oder seines Protokolls.

**Empfehlung:** Haupturteil **Target reached** mit unmittelbar angrenzendem **Method not verified**. Soll REPRODUCED als bewusste Produktentscheidung bleiben, beide Dimensionen gleichwertig zeigen. **A reached target is a real one** und pauschale Praezisionsversprechen begrenzen: 0/48 falsche Ziele sind ein positives Kontrollergebnis dieses Benchmarks, keine universelle Zuverlaessigkeitsgarantie.

### 5. Katalog und Texte widersprechen sich

Live sichtbar **7 of 17 shown**. `web/src/components/Catalog.tsx:138` filtert zwingend auf jury-ready mit veroeffentlichtem Lauf; es gibt keinen Schalter fuer alle Eintraege. SL-GCN als negativer numerischer Befund bleibt sichtbar. Setup-Probleme und ungepruefte Kandidaten wie HaKAN/Acacia werden dagegen ausgeblendet. Die Startseite verlinkt unter **Explore all outcomes** auf diesen gefilterten Katalog; README und fruehere Dokumente beschreiben einen umfassenderen Katalog.

**Abnahme:** Kuratierte Standardsicht darf bleiben; ein klarer Weg **All checks / Failed / Unchecked** erschliesst die anderen Kategorien. Kurationsentscheidung deutlich von Leistungsbewertung trennen.

### 6. Veroeffentlichungsstand synchronisieren

README behauptet noch, die Demo sei nicht veroeffentlicht und kein funktionierender Link. `web/src/config.ts:3` erzeugt Badge-Snippets fuer `https://nullius.example`; auch das Live-Bundle enthaelt diesen Platzhalter. Devpost-Entwurf kennt die URL ebenfalls noch nicht als verifiziert. Kosten-/Timeout-Luecken werden in Teilen der Dokumentation noch als aktuell beschrieben, obwohl der neuere Kostenreview und der Code einen nachfolgenden Fix dokumentieren.

**Abnahme:** Echte Demo-Adresse, funktionsfaehige Badge-Ziele und ein einheitlicher aktueller Status. Historische Befunde mit Datum bewahren. Die Reparatur der Kostenkontrollen nicht als verifizierte Anbieterrechnung oder absolute Begrenzung aller Kosten darstellen.

## Was wissenschaftlich noch fehlt

- 7/8 misst Referenzurteil-Uebereinstimmung auf den Entwicklungseinheiten, einschliesslich eines teilweise-Urteils. Kein allgemeiner Erfolgsanteil.
- RLinear 0/1 verfehlt die vorab festgelegte Held-out-Bedingung. Das belegt keine universelle Untauglichkeit, verhindert aber die behauptete allgemeine Recall-Verbesserung.
- TSMixer, SparseTSF und GLinear sind positive, nach Entwicklung ausgewaehlte Beispiele, kein Blindtest. Die Quelle-/Protokollwarnungen sind relevant.
- Methodentreue bleibt nach drei verfehlten Kalibrierungen ungeprueft. Ein unabhaengiger Metrik-Harness loest diese andere Frage nicht.
- Neue Papers auf bereits unterstuetzten Datensaetzen sind ein glaubwuerdiger, begrenzter Produktscope. Beliebige arXiv-Papers sind damit nicht abgedeckt.

Die Ehrlichkeit der gespeicherten Evaluierungen ist eine Staerke. Die sichtbaren Schlagzeilen muessen denselben Massstab erfuellen.

## Hoechster Nutzen bis zur Abgabe

1. Die drei nachgewiesenen Evidenzfehler reparieren; gezielte Regression mit den vorhandenen GLinear-/TSMixer-Daten. Keine neuen bezahlten Experimente dafuer noetig.
2. Ergebnisbegriffe, Katalogwege, URL und Dokumentationsstand vereinheitlichen.
3. Den bereits vorgesehenen unbegleiteten Nutzertest abschliessen: Person erklaert das Urteil, findet den Beleg und sagt, welche konkrete Entscheidung es unterstuetzt. Schwierigkeiten und Hilfe dokumentieren. Zeitersparnis erst behaupten, wenn tatsaechlich verglichen.
4. Video mit einem einzigen klaren Fall: Quelle und fester Zielwert, Nemotron schreibt Code, echter gespeicherter Fehler, Reparatur, Nebius-Branch, unabhaengige Messung, Einschraenkung und Certificate. Wiederholung als Recorded kennzeichnen. Neue Live-Schritte gemaess bestehender Nutzerentscheidung erst nach Auswertung des Produkttests planen.
5. Tavily-Beitrag anhand einer konkreten gefundenen Information und ihrer nachpruefbaren Wirkung zeigen. Die gespeicherten Live-Aufrufe machen die Integration glaubwuerdig. Kein gemessener allgemeiner Genauigkeitsgewinn; LaTeX-Direktbezug und Tavily-Fallback korrekt unterscheiden.

Ein groesseres Methodentreue-System, weitere Datensaetze oder eine neue Benchmark-Kampagne sind keine kurzfristige Voraussetzung fuer die obigen Verbesserungen. Spaetere Generalisierungstests brauchen vorab festgelegte neue Papers und getrennte Berichterstattung; dieser Review autorisiert oder startet sie nicht.
