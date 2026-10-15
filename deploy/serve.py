"""Produktions-Einstieg: die FastAPI-App aus repro_agent.api plus die gebaute Web-App (web/dist) in einem Prozess.

Liegt bewusst ausserhalb von src/: api.py bleibt unveraendert. /api-Routen der App gelten zuerst, danach
  /data  -> REPRO_EXPORT_DIR (persistentes Volume; dort schreibt site_export neue Laeufe, index.json/catalog.json)
  /      -> web/dist (Single-Page-App, Hash-Routing)
Ist REPRO_EXPORT_DIR leer, wird es beim Start aus dem Stand im Image (web/dist/data) befuellt.
Start: uvicorn deploy.serve:app --host 0.0.0.0 --port 8000 (genau ein Worker: Jobs und Semaphor liegen im Speicher).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
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
        merge_image_runs(export_dir, image_data)
        return False
    shutil.copytree(image_data, export_dir, dirs_exist_ok=True)
    return True


def _load_list(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def merge_image_runs(export_dir: Path, image_data: Path) -> int:
    """Add-only: run directories the image references (index.json, catalog.json run_id) but the volume lacks are copied,
    and their index refs appended. Nothing in the volume is overwritten or removed. Returns the number of copied runs."""
    from repro_agent.site_export import write_json_atomic

    wanted: dict[tuple[str, str], dict | None] = {}
    for ref in _load_list(image_data / "index.json"):
        if ref.get("paper_id") and ref.get("timestamp"):
            wanted[(ref["paper_id"], ref["timestamp"])] = ref
    for row in _load_list(image_data / "catalog.json"):
        run_id = row.get("run_id")
        if isinstance(run_id, str) and "--" in run_id:
            paper_id, ts = run_id.split("--", 1)
            wanted.setdefault((paper_id, ts), None)
    index_path = export_dir / "index.json"
    index = _load_list(index_path)
    known = {(r.get("paper_id"), r.get("timestamp")) for r in index}
    copied, added = 0, False
    for (paper_id, ts), ref in wanted.items():
        if Path(paper_id).name != paper_id or Path(ts).name != ts or paper_id.startswith("."):
            continue
        src, dest = image_data / paper_id / ts, export_dir / paper_id / ts
        if not src.is_dir():
            continue
        if not dest.exists():
            shutil.copytree(src, dest)
            copied += 1
        if ref is not None and (paper_id, ts) not in known:
            index.append(ref)
            known.add((paper_id, ts))
            added = True
    if added:
        write_json_atomic(index_path, index)
    return copied


def page_csp(dist: Path = DIST) -> str:
    """CSP fuer die HTML-Seite. Der Theme-Schnipsel in index.html ist inline: sein Hash wird beim Start aus der gebauten Datei gelesen,
    so bleibt script-src ohne 'unsafe-inline' und folgt jeder Aenderung am Schnipsel."""
    hashes = []
    index = dist / "index.html"
    if index.exists():
        for body in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", index.read_text(), flags=re.S):
            hashes.append("'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'")
    return ("default-src 'self'; script-src 'self' " + " ".join(hashes) + "; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'")


def mount_static() -> None:
    if not DIST.is_dir():
        raise RuntimeError(f"{DIST} fehlt: zuerst `npm run build` in web/ (im Container erledigt das Dockerfile)")
    seed_export_dir()
    from repro_agent import site_export
    site_export.refresh_catalog_metadata(site_export.CATALOG_YAML, EXPORT_DIR)
    app.state.page_csp = page_csp()
    app.mount("/data", StaticFiles(directory=EXPORT_DIR), name="data")
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")


mount_static()
