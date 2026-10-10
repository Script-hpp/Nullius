# Jury-Durchgang und naechster Videoschnitt — 2026-10-10

## Massstab

Hackathon-Jury, kein vorausgesetztes ML- oder Statistik-Fachwissen. Nach einer Minute soll jemand beantworten koennen: Welches Problem loest Nullius? Was macht der Agent selbst? Wo sehe ich ein echtes Ergebnis und seinen Beleg?

Abnahme dieser Verbesserung: klarer Einstieg, zwei belegte neue Beispiele, funktionierende Wege zu Quelle/Code/Messung, sichtbare fehlgeschlagene Checks, verstaendliche Begriffe und benutzbare Desktop-/Mobilansicht. Bestehende Gestaltung und gespeicherte Produktionslaeufe verwenden; keine neuen Dienste, Modelle oder kostenpflichtigen Experimente.

## Urteil nach dem lokalen Durchgang

Subjektive Hackathon-Einschaetzung, keine offizielle Bewertung und keine Vorhersage der Platzierung:

| Kriterium | Aktuell | Warum / naechster Hebel |
|---|---:|---|
| Technische Umsetzung | 8/10 | Eigenes Programm, Sandbox-Reparatur, wiederholte Messungen und unabhaengige Auswertung sind nachpruefbar. Neue Evidenz ist nun durchgaengig erreichbar. Fuer mehr Punkte: harte Kostenbegrenzung und Worker-Lebenszyklus aus der Kostenpruefung reparieren; das ist noch offen. |
| Potenzieller Nutzen | 8/10 | Klarer Einsatz fuer Forschende, die eine Behauptung pruefen oder ein Paper als Ausgangspunkt waehlen. Konkrete Beispiele zeigen den Nutzen. Fuer mehr Punkte: reale Nutzerbeobachtung und gemessener Vergleich mit manueller Pruefung; bisher keine belegte Zeitersparnis. |
| Design | 8.5/10 | Wiedererkennbare Typografie, ruhige Farbgebung und lesbarer Vergleich von Behauptung und Messung. Desktop und Mobil, hell und dunkel funktionieren im geprueften Hauptweg. |
| Produkterlebnis | 8.5/10 | Direkter Einstieg in eine 30-Sekunden-Wiedergabe, neue Ergebnisse frueh sichtbar, Belege mit einem Klick. Quellen-/Code-Abkuerzungen geben eine sinnvolle naechste Aktion. Eine echte neue Live-Pruefung wurde hier nicht gestartet. |

Die Umsetzung wird besser demonstriert; die zugrunde liegende Reproduktionsleistung wurde durch diese Website-Aenderungen nicht neu verbessert oder evaluiert. Das staerkere Framing kommt aus belegbaren Beispielen und leichterer Navigation.

## Naechste Hebel nach den offiziellen Kriterien

Am 2026-10-10 erneut geprueft: [Veranstalter zur Bewertung](https://nebiusglobalaihackathon.devpost.com/updates/46204-here-s-how-judging-works). Vier gleich gewichtete Kriterien, je 1 bis 5 Punkte: Technological Implementation, Design, Potential Impact und Quality of the Idea. Die obige 10er-Skala ist ausschliesslich unsere subjektive Orientierung; Produkterlebnis ist hier eine Unterdimension von Design, kein fuenftes offizielles Kriterium.

Prioritaet nach den bereits umgesetzten Website-Aenderungen:

1. Nach Nutzerentscheidung testet eine Person am Ende den Hauptweg: Ergebnis finden, Bedeutung des Urteils erklaeren, Beleg oeffnen und PDF exportieren. Schwierigkeiten dokumentieren, dann die groesste Huerde beheben. Keine erfundenen Nutzerstimmen oder Zeitersparnisse.
2. Video und Devpost auf den aktuellen Stand bringen. Im Video ein echter Fehler mit begruendeter Reparatur und unabhaengiger Messung; Erfolg, Kosten und exportiertes Zeugnis lesbar zeigen. Nebius Token Factory, NVIDIA Nemotron und Sandbox-Rolle konkret und auch im Ton nennen. [Veranstalterhinweise](https://nebiusglobalaihackathon.devpost.com/updates/46205-how-to-build-a-winning-project).
3. Weitere Live-Demo-Schritte warten nach Nutzerentscheidung auf die abschliessenden Testergebnisse. Bekannte Kosten-/Worker-Luecken bleiben dokumentiert; der begrenzte Einmal-Driver eines Paper-Versuchs repariert nicht automatisch die allgemeinen Produktpfade.
4. Quality of the Idea durch den bestehenden Kern erklaeren: „Das Paper ist die Testsuite“. Ein Claim wird in einen eigenen messbaren Versuch uebersetzt. Quellen, Autorencode und erzeugtes Experiment sichtbar auseinanderhalten; reale Messung macht den Nutzen glaubhaft.

Ein weiterer erfolgreicher Paper-Versuch hilft als zusaetzlicher Beleg. Er ersetzt weder Produktverstaendlichkeit noch reale Nutzerbeobachtung. Weitere grosse Features sind fuer diese Hebel nicht noetig.

Spaeterer Outreach-Gedanke des Nutzers: Autorinnen/Autoren passender Papers mit einem konkreten Reproduktionsbefund auf Nullius aufmerksam machen und das Projekt per Mail vorstellen. Erst nach Auswertung des Befunds und Produkttest weiter konkretisieren. Keine Kontaktaufnahme in dieser Runde beauftragt; der entfernte Email-Draft-Knopf bleibt entfernt.

## Auf der Website umgesetzt

- Einstieg nennt die Entscheidung: ein Paper pruefen, bevor man darauf aufbaut. Primaeraktion heisst „Watch a check in 30 seconds“ und ist als aufgezeichnet, mit echten Messungen und ohne Anmeldung beschrieben.
- TSMixer und SparseTSF stehen direkt nach dem Einstieg und auch auf der Benchmark-Seite. Behauptung, Messung, Streuung, Wiederholungen, Toleranz und dokumentierte Inferenzkosten kommen aus dem exportierten Katalog. MSE wird als Vorhersagefehler erklaert.
- SparseTSF-Dossier, Ereignislog, Plan und Programmversionen aus dem gespeicherten Lauf wurden veroeffentlicht. Originalquellen: Google Research fuer TSMixer, `lss-1138/SparseTSF` fuer SparseTSF; Links anhand der Dossiers korrigiert.
- Auswahl nach Entwicklung und vorlaeufige Einordnung stehen am Ergebnis; Details zur teilweise eigenen Auswahl und zur Aussagekraft lassen sich aufklappen. Die beiden Beispiele werden nicht in 7/8 hineingerechnet.
- Der Katalog zeigt standardmaessig alle Eintraege, einschliesslich Setup-Fehlern und ungeprueften Kandidaten. Die Beschreibung unterscheidet diese Faelle. Eine automatische Behauptung „Found by Tavily“ fuer jeden Nicht-Referenzeintrag wurde durch das belegte Pruefdatum ersetzt.
- Ein Kataloglauf ohne veroeffentlichte Evidenz wird als solcher markiert. Fehlende Laufdateien fuehren nicht mehr zu einem irrefuehrenden Evidenz-Link; der Filter beruecksichtigt das.
- Ergebnisansicht: einfache Erklaerung des praktischen Nutzens mit direkten Aktionen fuer Quellen/Setup und Code/Messungen. Wiedergabe dauerhaft als „Recorded run“ beschriftet. Unveroeffentlichte Video-/Devpost-Links erscheinen nicht als TBD im Footer.
- Jury-Verstaendlichkeit ist als dauerhafte Nutzerpraeferenz in `CLAUDE.md` dokumentiert.
- Nach Screenshot-Feedback: Karten heben den Datensatz oben hervor. Mehrere Checks desselben Papers erklaeren explizit „Same paper, separate dataset check“ (GCN auf Cora bzw. CiteSeer). Sortierung: reproduziert, teilweise, nicht reproduziert, Agenten-/Setup-Probleme, ungeprueft; innerhalb eines Ergebnisses neuere Checks zuerst. Build, Lint und Browserpruefung fuer Reihenfolge und Mobilbreite bestanden.

## Nächster Videoschnitt

Den bestehenden visuellen Stil behalten. Das alte Video liegt im Reviewer-Workspace unter `video/out/film.mp4`; die dortige Kritik steht in `docs/video_review_20261010.md`. Hier wurde die Website verbessert und der naechste Schnitt geplant; ein neues MP4 ist noch zu produzieren.

Ziel: ca. 2:30 Minuten, ein klarer Hauptweg. Pro Einstellung eine lesbare Aussage; Details gehoeren ins anklickbare Produkt.

| Zeit | Bild / konkreter Weg | Aussage fuer die Jury |
|---|---|---|
| 0:00–0:15 | Bestehender Hook, Paper-Claim und Messpunkte | „Before your team builds on a machine learning paper, can you reach its reported result? Nullius turns a paper's claim into a test. Here, the paper is the test suite.“ Zielgruppe und Kernidee zusammen erklaeren. |
| 0:15–0:35 | Startseite → aufgezeichneter Check | Ein echtes Ergebnis in wenigen Klicks. Aufzeichnung erkennbar kennzeichnen. |
| 0:35–1:05 | Quelle → eigenes Programm → dokumentierter Crash/Reparatur → Messung | „Nemotron, running on Nebius Token Factory, reads the paper and the authors' code, then writes its own experiment. Nebius Sandboxes run it. Our independent checker measures its predictions against a target fixed before the run.“ Original-Autorencode wird als Text gelesen; nur echte dokumentierte Reparatur zeigen. |
| 1:05–1:20 | TSMixer und SparseTSF, jeweils ein Ergebnis gross | Zwei weitere Papers ausserhalb der acht Entwicklungseinheiten: 0.3624 vs. 0.361 und 0.3653 vs. 0.359. Drei Wiederholungen je Beispiel, Toleranz ±0.01. Lesbarer kurzer Hinweis: „Selected after development · preliminary“. |
| 1:20–1:35 | Zahl 7/8 und separate Held-out-Zahl | „7 of 8 development units matched the reference verdict.“ In-sample sichtbar neben 7/8; RLinear-Held-out 0/1 getrennt. Das ist Uebereinstimmung mit dem Referenzurteil, keine allgemeine Erfolgsquote. |
| 1:35–1:50 | Claim-Regler; Messpunkte bleiben stehen | „Change the claim. The measurements stay.“ Dann: „48 planted false targets. None confirmed. Benchmark v4.“ |
| 1:50–2:15 | Quelle, Code, Messungen, PDF-Export und Kostenbeleg | Die Jury kann den Beleg selbst verfolgen und das Zeugnis mitnehmen. Praktische Entscheidung erklaeren: das Paper weiter untersuchen, bevor man darauf aufbaut. Inferenzkosten als solche beschriften; keine Garantie eines harten Gesamtbudgets suggerieren. |
| 2:15–2:30 | Motto + echte Demo-Adresse | Eine eindeutige Aktion: aufgezeichneten Check ansehen oder belegte Ergebnisse erkunden. |

Die Zeitfenster sind Schnittziele, keine bereits gerenderte Timeline. 10–15 Sekunden der bisherigen wiederholten Ablauferklaerungen kuerzen und fuer neue Ergebnisse einsetzen. Veralteten Zugangscode entfernen. Aktuelle Modellrollen verwenden; historische Kostenbelege mit historischen Modellen als historischen Lauf zeigen.

Naechste Arbeit in dieser Reihenfolge:

1. Aktuelle Website-Aenderungen in den Video-Workspace uebernehmen, inklusive neuer Exportdateien und Originalquellen. Dort die beiden Ergebnislinks oeffnen, bevor Szenen aufgenommen werden.
2. Die neuen Szenen aufnehmen und Texte gemaess Tabelle aktualisieren. Kleine App-Texte gezielt vergroessern, ein Fokus pro Einstellung.
3. Devpost-Entwurf an aktuelles Produkt und dieselben belegten Zahlen angleichen. Demo-/Videoadresse erst nach tatsaechlicher Veroeffentlichung einsetzen.
4. Neuen Film rendern und vollstaendig mit Ton anschauen; zusaetzlich in kleiner eingebetteter Playergroesse pruefen. Vorheriger Videoreview beruhte auf Frames und Timeline, nicht auf einer Audiobewertung.

## Pruefung und verbleibende Grenzen

- Frontend-Build und Lint erfolgreich; 108 bestehende Frontend-Tests bestanden.
- 20 fokussierte Python-Tests fuer Site-Export und Katalog-Referenzen bestanden.
- Playwright: 1440- und 390-Pixel-Ansichten, jeweils hell und dunkel; Startseite → SparseTSF → vier Tabs → Quellen-Abkuerzung → Katalog → Benchmark → Wiedergabe. Keine JavaScript-Seitenfehler oder Dokument-Ueberbreite im geprueften Weg. Screenshots der neuen Ergebnisansicht visuell geprueft.
- Zusaetzlich: mobiles Menue, Tastaturwechsel zwischen Tabs, fehlende veroeffentlichte Evidenz sowie Claim-Regler und Reset geprueft.
- Lokal mit statischen Produktionsbelegen getestet. Keine neuen kostenpflichtigen Modell-/Sandbox-Laeufe und keine Veroeffentlichung auf dem Demo-Server. Live-Budget-, Busy- und Netzwerkfehlerzustaende in dieser Runde nicht neu end-to-end geprueft.
- Bekannte Build-Warnung: grosses JavaScript-Bundle. Fuer gemessene schlechte Ladezeiten waere Aufteilung ein naechster Hebel; daraus wird hier keine unbelegte Performance-Zusage.
- Kosten-/Worker-Baustellen bleiben in [cost_policy_review.md](cost_policy_review.md) dokumentiert. Aus den zwei neuen Beispielen folgt weiterhin keine allgemeine Reproduktionsquote oder bestaetigte Methodentreue.
