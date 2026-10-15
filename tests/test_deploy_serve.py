"""deploy/serve.py: API plus statische Web-App in einem Prozess (Subprozess, damit die geteilte App der anderen Tests unberuehrt bleibt)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROBE = textwrap.dedent("""
    import sys
    sys.path.insert(0, %r)
    from fastapi.testclient import TestClient
    import deploy.serve as s
    c = TestClient(s.app)
    job = c.get("/api/jobs/current")
    assert job.status_code == 200 and job.json() == {"busy": False}, job.text
    page = c.get("/")
    assert page.status_code == 200 and "NULLIUS-PAGE" in page.text
    assert c.get("/data/index.json").json() == [{"id": "from-volume"}]    # Volume gewinnt gegen den Stand im Image
    assert c.get("/data/new.json").json() == {"seeded": True}              # fehlende Dateien kommen aus dem Image nur beim ersten Start
    print("ok")
""" % str(ROOT))


def test_serve_mounts_api_data_volume_and_web(tmp_path):
    dist, export = tmp_path / "dist", tmp_path / "export"
    (dist / "data").mkdir(parents=True)
    (dist / "index.html").write_text("<html>NULLIUS-PAGE</html>")
    (dist / "data" / "index.json").write_text('[{"id": "from-image"}]')
    export.mkdir()
    (export / "index.json").write_text('[{"id": "from-volume"}]')   # Volume ist schon befuellt: kein Ueberschreiben
    (export / "new.json").write_text('{"seeded": true}')
    env = {"WEB_DIST": str(dist), "REPRO_EXPORT_DIR": str(export), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-800:]


def test_seed_copies_image_data_only_into_empty_volume(tmp_path):
    src = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from pathlib import Path
        import deploy.serve as s
        img, vol = Path(sys.argv[1]), Path(sys.argv[2])
        assert s.seed_export_dir(vol, img) is True and (vol / "index.json").read_text() == "[1]"
        (vol / "index.json").write_text("[2]")
        assert s.seed_export_dir(vol, img) is False and (vol / "index.json").read_text() == "[2]"   # zweiter Start ueberschreibt nichts
        print("ok")
    """ % str(ROOT))
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "index.html").write_text("x")
    (dist / "data" / "index.json").write_text("[1]")
    env = {"WEB_DIST": str(dist), "REPRO_EXPORT_DIR": str(tmp_path / "vol0"), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run([sys.executable, "-c", src, str(dist / "data"), str(tmp_path / "vol")], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-800:]


def test_page_csp_hashes_inline_script_and_headers_reach_the_page(tmp_path):
    src = textwrap.dedent("""
        import sys, hashlib, base64
        sys.path.insert(0, %r)
        from fastapi.testclient import TestClient
        import deploy.serve as s
        body = "var x = 1"
        want = "'sha256-" + base64.b64encode(hashlib.sha256(body.encode()).digest()).decode() + "'"
        csp = s.page_csp()
        assert want in csp and "'unsafe-inline'" not in csp.split("style-src")[0] and "frame-ancestors 'none'" in csp, csp
        c = TestClient(s.app)
        page = c.get("/")
        assert page.headers["content-security-policy"] == csp and page.headers["x-content-type-options"] == "nosniff"
        for path in ("/", "/index.html", "/data/index.json"):
            original = c.get(path)
            for validator in (("if-none-match", "etag"), ("if-modified-since", "last-modified")):
                cached = c.get(path, headers={validator[0]: original.headers[validator[1]]})
                assert cached.status_code == 304, cached.text
                # A 304 must preserve the cached representation's CSP, not replace it with the API policy.
                assert "content-security-policy" not in cached.headers, (path, dict(cached.headers))
                assert cached.headers["x-content-type-options"] == "nosniff"
        assert c.get("/data/index.json").headers["content-security-policy"].startswith("default-src 'none'")   # Nicht-HTML: strenge Policy
        assert c.get("/docs").status_code == 404 and c.get("/api/health").json() == {"ok": True}
        print("ok")
    """ % str(ROOT))
    dist = tmp_path / "dist"
    (dist / "data").mkdir(parents=True)
    (dist / "index.html").write_text('<html><head><script>var x = 1</script></head><body><script type="module" src="/a.js"></script></body></html>')
    (dist / "data" / "index.json").write_text("[]")
    env = {"WEB_DIST": str(dist), "REPRO_EXPORT_DIR": str(tmp_path / "vol"), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-800:]


def test_seed_merges_new_image_runs_into_existing_volume_add_only(tmp_path):
    src = textwrap.dedent("""
        import json, sys
        sys.path.insert(0, %r)
        from pathlib import Path
        import deploy.serve as s
        img, vol = Path(sys.argv[1]), Path(sys.argv[2])
        before = {p: p.read_bytes() for p in vol.rglob("*") if p.is_file()}
        assert s.seed_export_dir(vol, img) is False
        assert (vol / "bangs" / "T1" / "dossier.json").read_text() == "new"
        idx = json.loads((vol / "index.json").read_text())
        assert [r["id"] for r in idx] == ["old--T0", "live--T9", "bangs--T1"], idx
        for p, data in before.items():
            if p.name != "index.json":
                assert p.read_bytes() == data
        assert (vol / "old" / "T0" / "dossier.json").read_text() == "volume-old"   # volume copy wins over image copy
        assert (vol / "live" / "T9" / "dossier.json").exists() and not (vol / "evil").exists()
        snap = {p: p.read_bytes() for p in vol.rglob("*") if p.is_file()}
        assert s.seed_export_dir(vol, img) is False     # idempotent
        assert snap == {p: p.read_bytes() for p in vol.rglob("*") if p.is_file()}
        print("ok")
    """ % str(ROOT))
    img, vol = tmp_path / "dist" / "data", tmp_path / "vol"
    def run(base, pid, ts, text):
        (base / pid / ts).mkdir(parents=True)
        (base / pid / ts / "dossier.json").write_text(text)
    ref = lambda p, t: {"id": f"{p}--{t}", "paper_id": p, "timestamp": t}
    run(img, "old", "T0", "image-old"); run(img, "bangs", "T1", "new")
    (img / "index.json").write_text(json.dumps([ref("old", "T0"), ref("bangs", "T1"), ref("../evil", "T1")]))
    (img / "catalog.json").write_text(json.dumps([{"id": "x", "run_id": "bangs--T1"}]))
    run(vol, "old", "T0", "volume-old"); run(vol, "live", "T9", "live")
    (vol / "index.json").write_text(json.dumps([ref("old", "T0"), ref("live", "T9")]))
    env = {"WEB_DIST": str(tmp_path / "dist"), "REPRO_EXPORT_DIR": str(vol), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}
    (tmp_path / "dist" / "index.html").write_text("x")
    res = subprocess.run([sys.executable, "-c", src, str(img), str(vol)], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-800:]
