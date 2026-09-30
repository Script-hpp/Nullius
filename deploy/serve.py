"""Produktions-Einstieg: die FastAPI-App aus repro_agent.api plus die gebaute Web-App (web/dist) in einem Prozess.

Liegt bewusst ausserhalb von src/: api.py bleibt unveraendert. /api-Routen der App gelten zuerst, danach
  /data  -> REPRO_EXPORT_DIR (persistentes Volume; dort schreibt site_export neue Laeufe, index.json/catalog.json)
  /      -> web/dist (Single-Page-App, Hash-Routing)
Ist REPRO_EXPORT_DIR leer, wird es beim Start aus dem Stand im Image (web/dist/data) befuellt.
Start: uvicorn deploy.serve:app --host 0.0.0.0 --port 8000 (genau ein Worker: Jobs und Semaphor liegen im Speicher).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from repro_agent.api import app

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(os.environ.get("WEB_DIST") or ROOT / "web" / "dist")
EXPORT_DIR = Path(os.environ.get("REPRO_EXPORT_DIR") or DIST / "data")


def seed_export_dir(export_dir: Path = EXPORT_DIR, image_data: Path | None = None) -> bool:
    """Volume beim ersten Start mit den Exporten des Images fuellen. True, wenn kopiert wurde. Vorhandene Dateien bleiben unberuehrt."""
    image_data = image_data or DIST / "data"
    if not image_data.is_dir() or image_data.resolve() == export_dir.resolve():
        return False
    export_dir.mkdir(parents=True, exist_ok=True)
    if (export_dir / "index.json").exists():
        return False
    shutil.copytree(image_data, export_dir, dirs_exist_ok=True)
    return True


def mount_static() -> None:
    if not DIST.is_dir():
        raise RuntimeError(f"{DIST} fehlt: zuerst `npm run build` in web/ (im Container erledigt das Dockerfile)")
    seed_export_dir()
    app.mount("/data", StaticFiles(directory=EXPORT_DIR), name="data")
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")


mount_static()
