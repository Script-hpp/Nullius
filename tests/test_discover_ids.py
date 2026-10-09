import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("discover_ids", Path(__file__).resolve().parent.parent / "scripts" / "discover_ids.py")
di = importlib.util.module_from_spec(spec)
spec.loader.exec_module(di)


def test_split_ids_refuses_known():
    ids, refused = di.split_ids("2405.00946v2, 2412.99999,2412.99999", {"2405.00946": "benchmark/papers.yaml"})
    assert ids == ["2412.99999"]
    assert refused == {"2405.00946": "benchmark/papers.yaml"}


def test_known_ids_reads_benchmark():
    assert "2305.10721" in di.known_ids()


def test_main_passes_seed_ids(capsys):
    class FakeAgent:
        def discover(self, datasets, seed_ids=None, on_event=None):
            self.got = (datasets, seed_ids)
            return []
    a = FakeAgent()
    assert di.main(["--ids", "2305.10721,2412.99999", "--dataset", "ETTh1"], agent=a) == 0
    assert a.got == (["ETTh1"], ["2412.99999"])
    assert "SKIP 2305.10721" in capsys.readouterr().out
