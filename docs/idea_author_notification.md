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
- Formulierung als Angebot, nicht als Siegel: "Eine unabhängige Nachimplementierung (Modus Nachbau, eigener Code, N Seeds) erreichte X ± s gegenüber Ihren Y. Belege: <Link zum Zeugnis>. Rückmeldung willkommen." Kein "zertifiziert", kein Badge im Namen einer Organisation.
- Kontakt nur über öffentlich angegebene Adresse aus dem Paper, eine Nachricht pro Paper, Abmeldemöglichkeit.
- Einschränkungen gehören in die Nachricht: Achse-2-Methode, Toleranz, Seeds, Zahl der Versuche, Graph-Reparaturen.

## Risikoärmere Varianten

1. **Teilbares Zeugnis** (existiert schon): Jede Evidence-Seite ist per Link teilbar. Autoren oder Leser können sie selbst verlinken.
2. **Selbst-Einreichung durch Autoren:** Autoren geben ihr Paper ein (arXiv-Link) und erhalten das Zeugnis. Kein unaufgeforderter Kontakt, Autoren kennen die Grenzen, weil sie selbst prüfen lassen.
3. **Rückmeldekanal im Zeugnis:** "Sind Sie Autor*in? Melden Sie Fehler in unserem Nachbau" (z. B. GitHub-Issue-Vorlage). Verbessert den Agenten, ohne dass wir zuerst schreiben.

## Für README / Video

Als Ausblick, ein Satz: "Next: authors can submit their own paper and get a shareable reproduction report; once method fidelity is calibrated, we plan to offer successful reproductions to authors as independent evidence."
