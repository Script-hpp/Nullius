import shutil
import subprocess

import pytest

from repro_agent.loop import AGENT_OUTPUT_TAIL_BYTES, keep_output_tail


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs a POSIX shell")
def test_long_agent_output_keeps_the_final_exception_and_exit_code():
    # Nebius kappt jeden Strom nach 65535 Bytes am Kopf; die Ausnahme steht am Ende eines langen Trainingslogs
    script = ("import sys; print('x' * 100000); sys.stderr.write('\\u00e4' * 50000 + '\\nValueError: boom\\n'); "
              "sys.exit(3)")
    cmd = keep_output_tail(f'cd /tmp && SEED=0 python3 -c "{script}"')
    r = subprocess.run(["sh", "-c", cmd], capture_output=True)
    assert r.returncode == 3
    assert len(r.stdout) <= AGENT_OUTPUT_TAIL_BYTES and len(r.stderr) <= AGENT_OUTPUT_TAIL_BYTES < 65535
    assert r.stderr.decode("ascii").endswith("ValueError: boom\n")   # strikt dekodierbar, auch nach Schnitt im UTF-8-Zeichen


def test_wrapper_keeps_network_isolation_prefix_visible():
    cmd = keep_output_tail("mkdir -p /output && cd /work && SEED=0 unshare -n timeout -s ABRT 585 python train.py")
    assert "unshare -n timeout -s ABRT 585 python train.py" in cmd and "mktemp -d" in cmd


def test_errors_in_own_data_helper_skip_web_search():
    from repro_agent.loop import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.use_tavily, o.tavily_mode = True, "live"
    o.tavily = None   # ein Suchaufruf wuerde hier scheitern
    summary = ("train.py line 40: x = nio.forecast_data('train')\n  at nullius_io.py line 40 in forecast_data: "
               "raise ValueError('history length must be 1..720')")
    assert o._error_hints(None, summary) == ""
