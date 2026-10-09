"""recall cg v1: Mechanik-Pruefung des Produkt-Loops auf Nebius OHNE Modellaufruf und ohne Tavily.

Der echte Orchestrator (graph_repair, io_helper, code_grounded) laeuft gegen die echte Nebius-Sandbox; `_ask` ist durch ein Skript ersetzt, das einen
Persistenz-Vorhersager schreibt (ueber nullius_io). Der erste Code-Aufruf haengt im Voll-Lauf (Schlaf), um den weichen Timeout (F2: timeout -s ABRT,
Stack im Reparaturtext) im Produktpfad zu pruefen. Geprueft wird die Mechanik (Branches, Hilfsdatei, Formpruefung, Timeout-Text, Harness, Urteil),
nicht die Genauigkeit. Kostet 0 $ Modell; die Sandbox ist kostenlos.

  uv run python scripts/run_recall_cg_v1.py --mechanics
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

CODE_OK = '''import os
import numpy as np
import nullius_io as io

d = io.load()
test_pred = np.repeat(d["test_inputs"][:, -1:, :], 96, axis=1)
n_val = io.CONFIG["val_windows"]
val_pred = np.repeat(d["val"][:n_val][:, None, :], 96, axis=1)
io.write_predictions(test_pred, val_pred)
'''
CODE_SLOW = CODE_OK + '''
import time
if os.environ.get("SMOKE") != "1":
    time.sleep(999)   # haengt im Voll-Lauf: weicher Timeout
'''


class Script:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.n_code = 0
        self.lock = threading.Lock()

    def ask(self, role, system, user, max_tokens=None, temperature=0.2):
        with self.lock:
            role = str(getattr(role, "value", role))
            self.calls.append((role, user))
            if role == "code":
                self.n_code += 1
                body = CODE_SLOW if self.n_code == 1 else CODE_OK
                content = f"```python\n{body}```"
            else:
                content = "Plan: persistence forecast (last value repeated)."
        return SimpleNamespace(content=content, finish_reason="stop", cost_usd=0.0, model="script", cached=False, prompt_tokens=0,
                               completion_tokens=0, status="ok", reasoning_tokens=None)


def main() -> int:
    os.environ.setdefault("LLM_PROVIDER", "tokenfactory")
    os.environ["SANDBOX_BACKEND"] = "nebius"
    os.environ["LLM_CACHE"] = "off"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import recall_cg_v1 as R
    from repro_agent.loop import Orchestrator
    from repro_agent import claims_source, repo_source

    kw = dict(R.ORCH_KWARGS, use_tavily=False, setup_details=False, seeds=2, claim_ids=[R.CLAIMS["rlinear_2023"][0]])
    orch = Orchestrator(**kw)
    script = Script()
    orch._ask = script.ask
    orch._paper_summary_round = lambda *a, **k: None
    orch.train_timeout_s = 90   # weicher Timeout bei 75 s
    d = orch.run("rlinear_2023")
    events = [json.loads(x) for x in (orch.last_run_dir / "events.jsonl").read_text().splitlines()]
    types = sorted({e.get("type") or e.get("event") for e in events})
    attempts = [e for e in events if e.get("type") == "attempt"]
    timeouts = [e for e in attempts if e.get("status") == "timeout"]
    repairs = [e for e in events if e.get("type") == "repair"]
    stack_repairs = [e for e in repairs if "train.py" in (e.get("error") or "") and "File " in (e.get("error") or "")]
    v = d.verdicts[0]
    print(json.dumps({"verdict": v.urteil, "mean": v.measured_mean, "std": v.measured_std, "claims": [c.id for c in d.claims],
                      "event_types": types, "code_calls": script.n_code, "attempts": len(attempts), "soft_timeouts": len(timeouts),
                      "repairs": len(repairs), "repairs_with_train_py_frame": len(stack_repairs), "run_dir": str(orch.last_run_dir.relative_to(ROOT)),
                      "spent_usd": orch.spent_usd}, indent=1, ensure_ascii=False))
    for e in repairs[:1]:
        print("--- Reparaturtext nach Timeout (Ende) ---")
        print((e.get("error") or "")[-1200:])
    assert FIXTURES_UNTOUCHED(), "benchmark/fixtures veraendert"
    assert orch.spent_usd == 0.0
    assert timeouts and stack_repairs, "weicher Timeout mit Stack im Reparaturtext nicht beobachtet"
    return 0


def FIXTURES_UNTOUCHED() -> bool:
    import subprocess
    return not subprocess.run(["git", "status", "--porcelain", "benchmark/fixtures"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
