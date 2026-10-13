# Gepruefte primaere Quellenreserve

`2307.03756v3.json.gz` enthaelt den unveraenderten, vom normalen Quellreader
gelesenen LaTeX-Volltext von [FITS v3](https://arxiv.org/html/2307.03756v3),
Xu Zhijian, Zeng Ailing und Xu Qiang. Lizenz: CC BY 4.0, wie auf der
arXiv-HTML-Seite angegeben. Die Vorlage wurde erneut kostenfrei von der
offiziellen versionierten Quelle abgerufen; echte Download-URL und Zeitpunkt
stehen im komprimierten PaperText-Dokument.

Text-SHA-256:
`9022b12b628fc1807f0ef11bfd7d1266f25b8575c1418afeb7bb90c251a00d64`.
Dieser Hash stimmt mit dem Quelltext aus dem ersten Produktionslauf und der
unabhaengigen lokalen Tabellenpruefung ueberein.

Die Reserve wird nur verwendet, wenn kein lesbarer neuer LaTeX-Download
verfuegbar ist. Paper-ID, Version, Quellenart und Text-Hash werden geprueft;
ein vorhandener Discovery-Hash muss ebenfalls passen. Jeder Cache-Einsatz
wird als gespeicherte Quelle mit Version und urspruenglichem Abrufdatum
angezeigt. Zielwert, Tabellenzuordnung und Experiment werden weiterhin
geprueft. Die Reserve enthaelt weder Modellantworten noch Messwerte.

Weitere erfolgreiche Downloads werden unter `logs/paper_sources/` im
vorhandenen persistenten Logs-Volume gespeichert. Live-Aufrufe ueberschreiben
keine Dateien unter `benchmark/fixtures/`. Eine Reserve fuer andere Papers
ist hier nicht vorgetaeuscht; ohne passenden Cache oder lesbare Quelle
stoppt die Pruefung weiterhin mit dem letzten Downloadfehler.
