from repro_agent.objections import docs_from_collect, repo_check

FACTORY = "def data_provider(args, flag):\n    if flag == 'test':\n        shuffle_flag = False\n        drop_last = False\n    else:\n        drop_last = True\n"
FILES = {"data_provider/data_factory.py": FACTORY}


def test_repo_check_confirms_and_refutes_drop_last_claims():
    assert repo_check("the test loader uses drop_last=False so all windows are evaluated", FILES) == "bestaetigt"
    assert repo_check("the test loader sets drop_last=True and drops the last batch", FILES) == "widerlegt"
    assert repo_check("results look strange", FILES) == "nicht pruefbar"
    assert repo_check("drop_last=True", None) == "nicht pruefbar"
    assert repo_check("drop_last is mentioned but no value", FILES) == "nicht pruefbar"


def test_docs_from_collect_needs_paper_mention_and_dedupes():
    collected = {
        "search": {"results": [
            {"url": "u1", "raw_content": "About DLinear evaluation"},
            {"url": "u2", "raw_content": "Unrelated page"},
            {"url": "u1", "raw_content": "About DLinear evaluation"},
        ]},
        "crawl": {"results": [{"url": "u3", "raw_content": "issue on LTSF-Linear"}, {"url": "u4", "raw_content": ""}]},
    }
    assert [d.name for d in docs_from_collect(collected, ("dlinear", "LTSF"))] == ["u1", "u3"]
    assert docs_from_collect({"search": None, "crawl": {"error": "x", "results": []}}, ("dlinear",)) == []
