import pytest

from repro_agent.harness.code_check import find_violations


@pytest.mark.parametrize("code", [
    "import urllib.request",
    "import pandas as pd\ndf = pd.read_csv('https://x.org/a.csv')",
    "from sklearn.datasets import fetch_openml",
    "import os\nos.system('curl x')",
    "m = __import__('socket')",
    "exec('print(1)')",
    "from torch_geometric.datasets import Planetoid",
])
def test_violations_detected(code):
    assert find_violations(code)


def test_clean_gcn_code_passes():
    code = (
        "import numpy as np, torch\nimport torch.nn.functional as F\n"
        "data = np.load('/data/cora_public.npz')\nmodel.eval()\n"
        "with torch.no_grad():\n    out = model(x)\n"
    )
    assert find_violations(code) == []
