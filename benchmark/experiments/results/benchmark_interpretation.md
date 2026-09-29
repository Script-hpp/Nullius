# Benchmark: Deutung (von Hand, nach Sicht der Zahlen in `benchmark_results.md`)

Die Zählregeln und Zahlen stehen in `../benchmark_protocol.md` und `benchmark_results.md` (vorab festgelegt, Skript vor den Daten committet). Dieser Text deutet die Zahlen; alles, was nicht im Protokoll steht, ist **post hoc** und so gekennzeichnet. Keine Signifikanzaussagen: 12 gegen 12 Urteile sind Zählwerte.

## Kernsatz (Berichtsvorlage des Protokolls)
Bei echten Claims stimmten **mit Tavily 6 von 12** Urteilen mit dem Referenzurteil überein, **ohne Tavily 3 von 12**. Bei Placebo-Claims erkannte der Agent **mit Tavily 12 von 12, ohne Tavily 12 von 12** als „nicht reproduziert“. Ausfälle: mit Tavily 1 (DLinear, Setup), ohne Tavily 0. Alle Urteile vorläufig, Methodentreue nicht geprüft. Sekundär: „nah“ (reproduziert oder teilweise) an echten Claims 10 von 12 in beiden Armen.

## Was die Zahlen tragen und was nicht
- **Placebos (12/12 in beiden Armen):** Wie geplant bestätigen sie, dass Urteilslogik und Isolation falsche Zielwerte nicht durchlassen: Die Messwerte der Placebo-Läufe liegen bei denselben Werten wie die der echten Läufe (Mediane je Paper und Arm liegen meist auf 0,01 genau beieinander), es gibt keinen Hinweis, dass Agenten auf 88 % hinarbeiten, und kein Leck. Sie messen **nicht**, wie gut ein Agent knapp falsche Behauptungen erkennt (Placebos liegen zu weit weg). Post hoc: Placebo-Läufe brauchten nicht mehr Versuche als echte (Median 10 bis 10,5 gegen 10 bis 11 Versuche je Claim, Code-Versionen 1 gegen 1 bis 1,5).
- **Echte Claims (6/12 mit gegen 3/12 ohne Tavily) sind grenzabhängig.** Die Toleranz ist ±1 Prozentpunkt, und viele Urteile liegen knapp neben der Grenze: Arm B GCN 0,8042 / 0,8046 / 0,8020 (Abstand 0,0108 / 0,0104 / 0,0130) wurden „teilweise“, Shchur Arm B 0,8044 (0,0106) ebenso. Arm A und B erreichen gleich oft „nah“ (10 von 12). Die Agenten liegen bei GCN in beiden Armen etwa einen Punkt unter der menschlichen Referenz (0,8136).
- **Je Paper (Übereinstimmungen, Arm A gegen B, je von 3):** Shchur 0 gegen 2, SGC 2 gegen 3, GCN 0 gegen 0, DLinear 1 gegen 1. Die Differenz von 3 stammt aus Shchur und SGC; bei GCN und DLinear gibt es keinen Unterschied. Bei Shchur liegen alle drei B-Werte über den A-Werten (gleiche 30 Splits, gepaart), was zur Beobachtung im Shchur-Experiment passt (Rolle 4 setzt die Werte aus Anhang C um).
- **Post hoc, sekundär:** Median der Abweichung vom Zielwert über die gültigen echten Generierungen: Arm A 0,0166 (12 Werte), Arm B 0,0097 (11 Werte). Die Richtung stimmt mit den beiden Rolle-4-Experimenten überein, gestützt durch dieselben zwei Paper und ohne Test.

## Auffälligkeiten (belegt in den Dossiers)
- **GCN Arm A, Generierung 0:** Der Code war über Seeds instabil (Seeds 0,113 bis 0,808; 8 von 10 Seeds weit unter der Toleranz), Mittel 0,508, „nicht reproduziert“. Ein Fehler im Agenten-Code, kein Befund über das Paper.
- **DLinear Arm B, Generierung 0:** Mittel 0,864, in beiden Claims (echt und Placebo) auf sechs Nachkommastellen gleich, obwohl die Code-Hashes verschieden sind (nicht untersucht).
- **DLinear Arm B, Generierung 1 (Ausfall):** vier Versuche mit `RuntimeError: The size of tensor a (7) must match the size of tensor b (96) at non-singleton dimension 2`. Genau die Fehlerart, für die die Formprüfung existiert; der Agent konnte sie trotz präziser Meldung nicht beheben.
- **DLinear allgemein:** 1 von 3 in beiden Armen; die Werte 0,38 bis 0,43 liegen wie im Rolle-4-Experiment am Rand oder außerhalb der Toleranz.
- **SGC:** Agenten erreichten teils höhere Werte (bis 0,828) als Zielwert und menschliche Referenz (0,8081), ohne dass die Ablation etwas Auffälliges zeigt.

## Grenzen (zusätzlich zum Protokoll)
- Der Code-Stand ist mit Paper und Arm **vermengt**: Die Zeilen ohne `code_commit` (Altstand, 20) enthalten sämtliche GCN-Läufe (0 von 6 echt übereinstimmend); die 6 von 9 im Neustand stammen vor allem aus SGC. Ein Vergleich der Stände sagt nichts über die Infrastruktur; die Änderung betraf nur Timeouts, nicht die Messung.
- Vier Infrastruktur-Abbrüche (`benchmark_incidents.md`); betroffene Generierungen wurden vollständig neu erzeugt.
- Referenzurteile stammen aus unseren Referenzlösungen im eigenen Harness. Alle vier Papers haben das Referenzurteil „reproduziert“; der echte negative Fall fehlt (Dacrema).
- Drei Generierungen je Paper und Arm; die Ergebnisse streuen stark (GCN Arm A: 0,51 bis 0,80).
