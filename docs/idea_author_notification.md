# Idee: Autoren über das Reproduktions-Ergebnis informieren

Stand: 2026-09-30. Status: **Idee, nicht gebaut, nicht vor der Abgabe.** Für README und Video als Ausblick.

## Idee

Wenn Nullius ein Paper aus dem Katalog nachrechnet, erfahren die Autoren davon nichts. Eine Nachricht an die Autoren ("Eine unabhängige Nachimplementierung erreichte Ihre Zahl, hier die Belege") könnte

- Autoren einen unabhängigen Beleg für ihr Ergebnis geben, den sie verlinken können,
- Nullius bekannt machen und Rückmeldungen zu Fehlern im Agenten liefern (Autoren kennen ihre Methode am besten),
- die Zielgruppe erweitern: nicht nur Leser, die auf einem Paper aufbauen, sondern auch Autoren, die ihr Ergebnis absichern wollen.

## Warum nicht jetzt

1. **Kein Paper hat bisher bestanden.** Stand der ersten Katalog-Läufe (Token Factory, 2026-09-30):
   - SL-GCN (2511.08094): gemessen 0.783 ± 0.018 statt 0.8292, "nicht reproduziert", Gesamtaussage "vorläufig". Der Agent baute ein tiefes GCN ohne die Stuart-Landau-Dynamik, also nicht die Methode des Papers.
   - SL-GCN, erster Lauf: reines MLP ohne Graph (0.552), von der Graph-Ablation erkannt, "nicht bewertbar".
   - MoHETS (2601.21866): MSE 2.32 statt 0.350, "Agenten-Fehler vermutet" (schlechter als die Null-Baseline, vermutlich fehlende Normalisierung).
2. **Achse 2 (Methodentreue) ist nicht geprüft.** Die Eichungen v1–v3 des Prüfers sind gescheitert (CLAUDE.md). Auch ein "reproduziert" belegt nur, dass unser Code die Zahl erreicht, nicht dass er die Methode des Papers umsetzt. Eine Nachricht "Ihr Paper hat bestanden" würde mehr versprechen, als wir belegen können.
3. **Negative Ergebnisse sind heute meist Agenten-Fehler.** Eine Nachricht "nicht reproduziert" an Autoren wäre in den bisherigen Fällen falsch und schadet dem Ruf von Nullius und der Autoren.
4. **Unaufgeforderte automatische E-Mails an Fremde sind Spam** und nach dem Versand nicht zurückzuholen.

## Bedingungen, falls es gebaut wird

- Nur bei Gesamtaussage **"Paper reproduziert"**, also Achse 1 reproduziert **und** Achse 2 geprüft mit "ja" (setzt einen geeichten Methodentreue-Prüfer voraus).
- **Manuelle Prüfung vor jedem Versand** (Code, Zeugnis, Warnungen), kein automatischer Versand.
- Nur positive Ergebnisse aktiv melden. Negative Ergebnisse nie ungefragt verschicken; sie stehen höchstens öffentlich im Zeugnis mit der Formulierung "mit unserem Setup nicht reproduziert" und allen Versuchen.
- Solange die Voraussetzungen im Abschnitt "Langfristiges Ziel: Prüfsiegel" nicht erfüllt sind: Formulierung als Angebot, nicht als Siegel: "Eine unabhängige Nachimplementierung (Modus Nachbau, eigener Code, N Seeds) erreichte X ± s gegenüber Ihren Y. Belege: <Link zum Zeugnis>. Rückmeldung willkommen." Kein "zertifiziert", kein Badge im Namen einer Organisation.
- Kontakt nur über öffentlich angegebene Adresse aus dem Paper, eine Nachricht pro Paper, Abmeldemöglichkeit.
- Einschränkungen gehören in die Nachricht: Achse-2-Methode, Toleranz, Seeds, Zahl der Versuche, Graph-Reparaturen.

## Risikoärmere Varianten

1. **Teilbares Zeugnis** (existiert schon): Jede Evidence-Seite ist per Link teilbar. Autoren oder Leser können sie selbst verlinken.
2. **Selbst-Einreichung durch Autoren:** Autoren geben ihr Paper ein (arXiv-Link) und erhalten das Zeugnis. Kein unaufgeforderter Kontakt, Autoren kennen die Grenzen, weil sie selbst prüfen lassen.
3. **Rückmeldekanal im Zeugnis:** "Sind Sie Autor*in? Melden Sie Fehler in unserem Nachbau" (z. B. GitHub-Issue-Vorlage). Verbessert den Agenten, ohne dass wir zuerst schreiben.

## Für README / Video

Als Ausblick, ein Satz: "Next: authors can submit their own paper and get a shareable reproduction report; once method fidelity is calibrated, we plan to offer successful reproductions to authors as independent evidence."

## Langfristiges Ziel: Prüfsiegel nach TÜV-Prinzip (Stand 2026-10-01, Wunsch des Nutzers)

Ziel: Ein Paper, das unsere Prüfung bestanden hat, bekommt ein Nullius-Prüfsiegel, vergleichbar mit einer TÜV-Plakette: ein festgelegter Prüfkatalog, eine unabhängige Prüfstelle, ein klares "bestanden", ein Datum und eine nachprüfbare Prüfnummer. Status: **nicht gebaut, nicht vor der Abgabe.** Heute darf es kein Siegel geben; dieser Abschnitt hält fest, was dafür fehlt.

### Was ein Siegel aussagen würde

"Das Paper-Ergebnis X auf Datensatz Y wurde von Nullius mit eigenem Code (Modus Nachbau) nachgebaut; die Zahl liegt in der vorab festgelegten Toleranz, und der Code setzt die Methode des Papers um." Also Gesamtaussage **"Paper reproduziert"** (Achse 1 reproduziert **und** Achse 2 "ja", CLAUDE.md "Zweidimensionales Urteil"). Nicht ausgesagt: dass das Paper insgesamt stimmt, dass andere Claims stimmen, dass das Original-Repo läuft.

### Was fehlt (Voraussetzungen, in dieser Reihenfolge)

1. **Geeichter Methodentreue-Prüfer (Achse 2).** Ohne ihn belegt "reproduziert" nur den Zielwert. Eichungen v1-v3 gescheitert (5/10, 7/10, 8/11; `benchmark/experiments/results/calibration_fidelity_v1.md`, `_v2.md`, `_v3.md`). Braucht: neue Vorab-Festlegung mit Schwelle, größere Eichmenge, Ablationstests auch für Nicht-Graph-Modelle (heute nur Graph-Ablation für GCN, `ablation_gcn.md`).
2. **Festgeschriebener Prüfkatalog pro Datensatz/Task.** Welche Prüfungen ein Lauf bestehen muss (Harness-Plausibilität, Leak-Schutz-Schichten, Formprüfung, Baselines, Ablation, Mindestzahl Seeds, Toleranzregel), versioniert (z. B. "Prüfkatalog v1"). Ein Siegel nennt die Katalog-Version.
3. **Nachgewiesene Zuverlässigkeit der Prüfstelle.** Benchmark mit Kennzahlen, die für ein Siegel zählen: Übereinstimmung mit Referenzurteilen (Benchmark v2) und vor allem **Fehlalarmrate für "bestanden"** (Placebos: falscher Zielwert darf nie bestehen). Dazu echte Negativfälle (heute keiner, Dacrema nicht gebaut) und mehr als 8 Einheiten. Vorab festgelegte Mindestwerte, ab denen Siegel vergeben werden.
4. **Mehr unterstützte Datensätze.** Heute Cora, Cora-LCC, ETTh1 (README "Limitations"). Ein Siegel ist nur für Papers auf unterstützten Benchmarks möglich; das muss auf dem Siegel stehen.
5. **Manuelle Abnahme vor Vergabe.** Mensch prüft Code, Zeugnis, Warnungen (wie oben bei "Bedingungen"). Kein automatisches Siegel.
6. **Nachprüfbarkeit und Fälschungsschutz.** Prüfnummer = Hash aus Dossier, `code_hash`, `code_commit`, Prüfkatalog-Version; Siegel ist Link auf die gehostete Zeugnis-Seite (Wer das Bild ohne Seite kopiert, fällt beim Klick auf). Offen: reicht Link + Hash, oder braucht es eine Signatur?
7. **Gültigkeit und Widerruf.** Gebunden an Paper-Version (arXiv vN) und Prüfkatalog-Version. Widerruf bei gefundenem Fehler in Harness oder Agent: Seite zeigt "widerrufen" mit Grund, Siegelbild wechselt. Öffentliche Widerrufsliste.
8. **Dauerhaftes Hosting.** Siegel ohne erreichbare Zeugnis-Seite ist wertlos; Betrieb über die Hackathon-Frist (15.12.2026) hinaus nötig.
9. **Antrag statt ungefragter Prüfung (Pull).** Autor*innen reichen ein und beantragen das Siegel; Nullius schreibt niemanden ungefragt an (siehe "Warum nicht jetzt", Punkt 4). Kein Siegel und keine Plakette für "nicht bestanden"; negative Ergebnisse stehen nur im Zeugnis mit "mit unserem Setup nicht reproduziert" und allen Versuchen.
10. **Sprache und Recht.** Kein "zertifiziert" oder "TÜV" im Siegeltext (geschützte Marke und falscher Eindruck einer akkreditierten Stelle); Name z. B. "Nullius checked". Haftungshinweis auf der Zeugnis-Seite.

### Mögliche Zwischenstufe (vor Punkt 1)

Stufe "Target value reached, method fidelity not checked" als teilbares Zeugnis-Badge, nur auf Antrag und nach manueller Abnahme. Risiko: wird als Methodenbeleg missverstanden. Entscheidung offen; bis dahin nur das teilbare Zeugnis (Risikoärmere Variante 1).

### Für README / Video (Ausblick, ein Satz)

"Long term, Nullius aims to be an inspection body for ML claims: papers that pass a versioned test catalogue, including a calibrated method-fidelity check, get a verifiable seal that links to their certificate."
