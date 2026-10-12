# Discovery: Fehler beim Live-Start — 12. Oktober 2026

## Beobachtung und Ursache

Der Nutzer startete Discovery auf der Live-Demo. Start und SSE-Stream antworteten mit 200, danach zeigte die UI `Finished`, `Runtime NaNs` und null Tavily-/Modellaufrufe. Die Access-Logs enthielten keine erklaerende Fehlermeldung.

Vor Aenderungen `git fetch origin` und `git pull --ff-only origin main`: fuenf neue Frontend-Commits bis `12c1406` uebernommen. Mit einem simulierten fruehen Serverfehler den Screenshotzustand auf diesem Stand exakt nachgestellt.

Der Nutzer pruefte im laufenden Coolify-Container ausschliesslich `DiscoveryAgent()` und meldete die fehlende Variable `OPENROUTER_API_KEY`. Der Container verwendet OpenRouter (Standard bei fehlendem `LLM_PROVIDER`) statt Token Factory; die Initialisierung bricht vor dem ersten Suchaufruf ab. Ohne `TAVILY_MODE=fixture` verwendet Discovery bereits Live-Aufrufe, der fehlende Wert erklaert den Fehler nicht.

Betriebsreparatur: in Coolify `LLM_PROVIDER=tokenfactory` setzen und mit vorhandenen Nebius-Zugangsdaten und `TF_MODEL_*`-Variablen neu deployen. Kein OpenRouter-Key fuer die Produktionsdemo erforderlich. Falls der angenommene Fehlstart das Tageskontingent verbraucht hat, globale und IP-Discovery-Grenze fuer einen weiteren Test jeweils um eins anheben. Der reine Initialisierungstest verbraucht keinen Versuch. API-Starts zaehlen bei Annahme; Logs und Zaehler bleiben erhalten.

## Reparatur der Fehlerdarstellung

- API sendet Start, Fehler, Timeout und expliziten Abschluss mit Zeitstempeln. Bisher wurde `done` vor der Ausgabe an den Client verworfen.
- Worker-/Initialisierungsfehler erscheinen redigiert im Serverlog mit Job-ID; Schluessel werden nicht ausgegeben.
- Discover zeigt Serverfehler oben als Alert. Fehlgeschlagene Starts heissen `Failed`; ein Stream-Abbruch ohne `done` gilt nicht mehr als erfolgreicher Abschluss. Bereits eingegangene Ergebnisse bleiben sichtbar.
- Alte Fehlerereignisse ohne Zeitstempel werden beim Empfang normalisiert. Die Laufzeit bleibt bei unvollstaendigen gespeicherten Daten endlich.
- Warnungen werden sichtbar; Replay bleibt als Fallback erreichbar. Budget-, Tages- und IP-Limits werden durch den Codefix nicht aufgehoben.

## Nachweis

- Zwei neue API- und zwei Frontend-Regressionen schlugen vor dem Fix fehl und bestehen danach. Eine weitere Frontend-Regression prueft Ereignisvalidierung und alte Fehlerereignisse.
- **51 API-/Live-Hook-Tests bestanden**, einschliesslich Erfolgsabschluss, Konstruktorfehler, Geheimnisredaktion und Timeout mit weiter belegtem Worker-Platz.
- **132 Frontend-Tests bestanden**, Produktionsbuild und Lint erfolgreich. Bestehende React-/Bundle-Warnungen bleiben.
- Browser mit simuliertem Backend auf Desktop und mobil: frueher Fehler, erfolgreicher Abschluss, Verbindungsabbruch und Replay-Fallback. Fehler sichtbar, keine NaN-Laufzeit, keine JavaScript-Seitenfehler. Keine bezahlte Suche gestartet.
- Breitere Discovery-/API-Auswahl: **157 bestanden, zwei bestehende Suchlimit-Tests fehlgeschlagen** (`test_max_searches_and_dedup`, `test_search_covers_all_datasets_and_stays_bounded`). Sie erwarten drei Suchen; der unveraenderte Code auf main erlaubt fuenf. Suchverhalten und diese Tests wurden hier nicht veraendert.
- Deployment und echter Suchlauf nach der Konfigurationskorrektur noch ausstehend. HTTP 200 beim Start allein belegt keinen erfolgreichen Discovery-Lauf.
