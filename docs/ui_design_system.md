# Nullius - UI/UX Design System & Coding Rules (claude.md)

## 1. Core Identity & Vibe
- **Produkt:** Ein AI-Benchmark-Tool, das arxiv-Paper auf Reproduzierbarkeit prüft ("Nullius in verba").
- **Zielgruppe:** Eine nicht-technische Jury. Keine Entwickler, keine Data Scientists.
- **Look & Feel:** Premium B2B SaaS (wie Linear, Vercel, Raycast, Tavily). Clean, minimalistisch, hochpräzise.
- **Anti-Pattern:** KEIN "AI-Slop", KEINE Neon-Farben, KEINE leuchtenden Gradienten, KEINE Emojis, KEIN roher "Vibe-Coded"-Look (wie Bootstrap-Template oder Standard-Chart-Bibliothek).

## 2. Sprache (STRIKT)
- **100% Englisch:** Das gesamte UI, alle Logs, alle Metriken und alle Source-Files müssen auf Englisch sein. Es darf kein einziges deutsches Wort sichtbar sein.

## 3. Typografie & Farbpalette
- **Schriften:** 
  - UI-Text: Inter oder Geist Sans.
  - Zahlen, Metriken, Code: JetBrains Mono (font-mono) für absolute Präzision.
- **Farben (Tailwind Kontext):**
  - Background: `bg-zinc-950` (Body), `bg-zinc-900` (Cards/Panels).
  - Borders: `border-zinc-800` (fein, niemals klobig).
  - Text: `text-zinc-50` (Headlines), `text-zinc-400` (Body), `text-zinc-500` (Meta-Daten).
  - Akzente: Emerald (`text-emerald-400`, `fill-emerald-500/50`) für "Verified/Pass", Amber (`text-amber-400`, `fill-amber-500/50`) für "Partial/Out of Range", Rose (`text-rose-400`) für "Fail".
- **Keine Hintergrundfarben für Status-Tags:** Nur Text und Icon in Farbe, niemals ein farbiger Kasten im Hintergrund.

## 4. UI-Komponenten & Layout
- **Allgemein:** Großzügiger Whitespace. Cards mit `p-6` Padding. Hover-Effekte auf Cards (`hover:border-zinc-700 hover:bg-zinc-900/80 transition-all`).
- **Status-Badges (Dashboard):** Minimalistisch. Lucide-Icon + Text in Großbuchstaben (`text-[10px] uppercase tracking-wider font-medium`). Keine Pills.
  - Beispiel "Verified": CheckCircle2 Icon + "VERIFIED" in `text-emerald-400`.
  - Beispiel "Partial": CircleDashed Icon + "PARTIAL" in `text-amber-400`.
- **Status-Badges (Detail-Tabellen):** 
  - "PASSED" (CheckCircle2, `text-emerald-400`)
  - "OUT OF RANGE" (CircleDashed, `text-amber-400`)
  - "FAILED" (XCircle, `text-rose-400`)
- **Detailseite:** Zweispaltiges Layout oben (Links: Verdict, Rechts: Chart). Tabs für tiefere Informationen (Data & Evidence, Execution Log, Branch Tree, Raw Files).
- **Tabellen:** Moderne UI-Tabelle. Kein CSV-Dump. Padding (`py-3 px-4`), feine Trennlinien (`border-b border-zinc-800/50`), Monospace für Zahlen, rechtsbündige Ausrichtung der Zahlen.
- **Execution Log:** Kein roher Terminal-Text. Timeline oder Accordion. Überschriften fett, Details kleiner. Für die Jury in verständliche Sätze übersetzen (keine rohen Zitate oder technisches Jargon).

## 5. Daten- & Metrik-Präsentation (Das Herzstück)
- **Prozent-Format (STRIKT):** ALLE Zahlen im gesamten UI müssen als Prozentwert formatiert sein (z.B. `81.5%`). Niemals Dezimalzahlen wie `0.81` oder `0.815` verwenden. Ausnahme: Nur wenn es sich um absolute Zahlen handelt (wie "10 Runs").
- **Klarheit für die Jury:**
  - Keine vertikalen Trennlinien oder verwirrenden Pfeile zwischen Werten.
  - Nutze klare Labels: "CLAIMED" vs. "MEASURED".
  - **Delta-Indikator:** Füge immer eine Differenz zum Claimed-Wert hinzu (z.B. `+0.6%` in Grün oder `-3.0%` in Amber/Rot).
  - **Charts:** Nutze halbtransparente Füllfarben (z.B. `fill-amber-500/50`). Füge immer eine gestrichelte horizontale Ziellinie ("Target: 81.5%") und eine Linie für den Mittelwert ("Mean: 82.1%") ein, damit die Jury sofort den visuellen Kontext hat.
  - **Verdict Card:** Großer, unübersehbarer Status (z.B. `text-3xl font-bold`). Darunter die klare Gegenüberstellung von "Claimed" und "Measured". Kein technisches Rauschen (wie "(std 0.010)").

## 6. Verbotene Elemente (Absolute No-Gos)
- Deutsche Texte im UI.
- Dezimalzahlen wie `0.81` statt `81.0%`.
- Vertikale Linien, kleine Pfeile (→) oder unruhige Boxen in Metrik-Bereichen.
- Neon-Farben, Glow-Effekte oder AI-Gradienten.
- Emojis.
- Rohe Markdown- oder CSV-Dumps (müssen gerendert oder versteckt werden).
- Standard-Chart-Farben (Gelb/Grün aus der Bibliothek).
- Technisches Jargon in Logs oder Beschreibungen (muss für Laien verständlich sein).
