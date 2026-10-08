"""Sondierung der Nebius-Sandbox fuer den Messweg von G1 (kein Modell-Aufruf, kein Agenten-Code, nur feste Kommandos).

  uv run python scripts/review/nebius_probe.py --out <Datei.json> [--branches 16] [--mem-test]

Beantwortet: (1) CPU-Zahl und Speicher je Branch, (2) was ein per Timeout beendeter Lauf zurueckgibt und ob seine Dateien noch lesbar sind,
(3) `timeout -s ABRT` + faulthandler: Stack in stderr, Exit-Code, (4) Parallelitaet (n Branches gleichzeitig, Wanduhr gegen Summe).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

IMAGE = "python:3.11-slim"


def main() -> int:
    argv = sys.argv[1:]
    arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
    out = Path(arg("--out", "nebius_probe.json"))
    n_par = int(arg("--branches", "16"))
    from repro_agent.sandbox.nebius import NebiusSandbox

    sb = NebiusSandbox()
    res: dict = {}
    t0 = time.time()
    snap = sb.base_snapshot(IMAGE, [], label="g1-probe")
    res["snapshot_seconds"] = round(time.time() - t0, 1)
    print("snapshot", snap, res["snapshot_seconds"], flush=True)

    b = sb.branch(snap)
    r = sb.run(b, "nproc; free -g | head -2; cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/cpu.max 2>/dev/null; "
                  "python -V; df -h /tmp | tail -1", 60)
    res["resources"] = {"exit": r.exit_code, "stdout": r.stdout, "stderr": r.stderr[-300:]}
    print(r.stdout, r.stderr[-200:], flush=True)

    # (3) timeout -s ABRT + faulthandler im selben Kommando
    cmd = ("cd /tmp && printf 'import time\\nprint(\"start\", flush=True)\\ndef work():\\n    while True:\\n        sum(range(1000))\\nwork()\\n' > loop.py "
           "&& PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 unshare -n timeout -s ABRT 8 python loop.py; echo exit=$?")
    t1 = time.time()
    r = sb.run(b, cmd, 60)
    res["abrt"] = {"seconds": round(time.time() - t1, 1), "exit": r.exit_code, "timed_out": r.timed_out, "stdout": r.stdout[-400:], "stderr": r.stderr[-600:]}
    print("ABRT", res["abrt"], flush=True)

    # (2) Sandbox-Timeout (hartes Kill): Datei vorher schreiben, Dateien danach lesbar?
    t1 = time.time()
    r = sb.run(b, "echo marker > /tmp/before_timeout.txt; python -c \"import time; print('x', flush=True); time.sleep(120)\"", 12)
    res["hard_timeout"] = {"seconds": round(time.time() - t1, 1), "exit": r.exit_code, "timed_out": r.timed_out, "stdout": r.stdout[-200:], "stderr": r.stderr[-200:]}
    try:
        res["hard_timeout"]["file_after"] = sb.get_file(b, "/tmp/before_timeout.txt").decode()[:50]
    except Exception as exc:   # noqa: BLE001
        res["hard_timeout"]["file_after"] = f"FEHLER {type(exc).__name__}: {str(exc)[:200]}"
    try:
        r2 = sb.run(b, "echo still-usable", 30)
        res["hard_timeout"]["branch_after"] = {"exit": r2.exit_code, "stdout": r2.stdout.strip()}
    except Exception as exc:   # noqa: BLE001
        res["hard_timeout"]["branch_after"] = f"FEHLER {type(exc).__name__}: {str(exc)[:200]}"
    print("HARD", res["hard_timeout"], flush=True)

    # Speicher (optional): reserviere g GB und beruehre sie
    if "--mem-test" in argv:
        res["mem"] = {}
        b2 = sb.branch(snap)
        for gb in (4, 8, 12, 16):
            r = sb.run(b2, f"python -c \"a=bytearray({gb}*1024**3); a[::4096]=b'x'*len(a[::4096]); print('ok {gb}')\"; echo exit=$?", 120)
            res["mem"][gb] = {"exit": r.exit_code, "stdout": r.stdout.strip()[-80:], "stderr": r.stderr.strip()[-120:]}
            print("MEM", gb, res["mem"][gb], flush=True)
            if "ok" not in r.stdout:
                break

    # (4) Parallelitaet: n Branches gleichzeitig, jeder rechnet 20 s CPU-gebunden (gleichzeitig `nproc` melden)
    results, lock = [], threading.Lock()
    work = ("python - <<'E'\nimport time, os\nt=time.time(); n=0\nwhile time.time()-t<15:\n    n+=sum(i*i for i in range(2000))\nprint(os.cpu_count(), n)\nE")

    def one(i: int) -> None:
        s = time.time()
        try:
            bb = sb.branch(snap)
            t_b = time.time() - s
            r = sb.run(bb, work, 120)
            row = {"i": i, "branch_s": round(t_b, 1), "total_s": round(time.time() - s, 1), "exit": r.exit_code, "out": r.stdout.strip()[-60:]}
        except Exception as exc:   # noqa: BLE001
            row = {"i": i, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "total_s": round(time.time() - s, 1)}
        with lock:
            results.append(row)

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=n_par) as pool:
        list(pool.map(one, range(n_par)))
    res["parallel"] = {"n": n_par, "wall_s": round(time.time() - t1, 1), "errors": sum(1 for r in results if "error" in r),
                       "rows": sorted(results, key=lambda r: r["i"])}
    print("PAR", {k: v for k, v in res["parallel"].items() if k != "rows"}, flush=True)
    for row in res["parallel"]["rows"][:4] + res["parallel"]["rows"][-2:]:
        print("  ", row, flush=True)

    out.write_text(json.dumps(res, indent=1))
    print("geschrieben:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
