"""Planetoid-Datensaetze (Cora, CiteSeer, PubMed): eine parametrisierte Logik, Cora-Verhalten unveraendert, Leak-Schutz je Datensatz.

Ohne Netz: synthetische Rohdateien im Planetoid-Format. Echte Daten (Hashes, Groessen) nur mit PLANETOID_NETWORK_TESTS=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from repro_agent.harness import code_check, edge_direction
from repro_agent.harness import planetoid_data as pd_
from repro_agent.tasks import HARNESS_DIR, PLANETOID, TASKS, citeseer_task, cora_task, planetoid_data_description, planetoid_task, pubmed_task

NAMES = ("cora", "citeseer", "pubmed")
N_CLS, N_FEAT, N_TRAIN, N_VAL, N_TEST_SMALL = 3, 12, 6, 8, 10


@pytest.fixture
def tiny(monkeypatch):
    """Kleine Spec fuer alle drei Namen (gleiche Rohlogik, kleine Zahlen)."""
    for n in NAMES:
        monkeypatch.setitem(pd_.SPECS, n, {"nodes": 0, "features": N_FEAT, "classes": N_CLS, "n_train": N_TRAIN, "n_test": N_TEST_SMALL})
    monkeypatch.setattr(pd_, "N_VAL", N_VAL)


def make_raw(raw: Path, name: str, gaps: tuple[int, ...] = ()):
    """Rohdateien wie kimiyoung/planetoid. Knoten: [train | val | rest(allx) | test (mit Luecken `gaps`)]. Rueckgabe: (features[N,F], labels[N], test ids)."""
    rng = np.random.default_rng(0)
    n_allx = N_TRAIN + N_VAL + 7
    test_ids = [n_allx + i for i in range(N_TEST_SMALL + len(gaps)) if i not in gaps]
    n_total = n_allx + N_TEST_SMALL + len(gaps)
    feats = (rng.random((n_total, N_FEAT)) < 0.4).astype(np.float64) + 0.1      # nie ganz Null
    labels = rng.integers(0, N_CLS, n_total)
    for g in gaps:
        feats[n_allx + g] = 0
    onehot = np.eye(N_CLS)[labels]
    for g in gaps:
        onehot[n_allx + g] = 0
    order = rng.permutation(len(test_ids))                     # test.index ist durchmischt
    tx = sp.csr_matrix(feats[np.array(test_ids)[order]])
    ty = onehot[np.array(test_ids)[order]]
    graph = {i: [(i + 1) % n_total, (i + 3) % n_total] for i in range(n_total)}   # absichtlich einseitig
    graph[0].append(0)                                           # Selbstschleife
    objs = {"x": sp.csr_matrix(feats[:N_TRAIN]), "y": onehot[:N_TRAIN], "tx": tx, "ty": ty,
            "allx": sp.csr_matrix(feats[:n_allx]), "ally": onehot[:n_allx], "graph": graph}
    raw.mkdir(parents=True, exist_ok=True)
    for k, v in objs.items():
        with open(raw / f"ind.{name}.{k}", "wb") as f:
            pickle.dump(v, f)
    (raw / f"ind.{name}.test.index").write_text("\n".join(str(test_ids[i]) for i in order))
    return feats, labels, sorted(test_ids)


def old_cora_load_raw(raw_dir: Path):
    """Eingefrorene Kopie der Cora-Logik vor der Parametrisierung (cora_data.load_raw bis 2026-10-04), nur zum Vergleich."""
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    x, y, tx, ty, allx, ally, graph = (pickle.load(open(raw_dir / f"ind.cora.{n}", "rb"), encoding="latin1") for n in names)
    test_idx_reorder = [int(line) for line in (raw_dir / "ind.cora.test.index").read_text().split()]
    test_idx_range = np.sort(test_idx_reorder)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    edges = sorted({(int(u), int(v)) for u, nbrs in graph.items() for v in nbrs if u != v})
    return np.asarray(features.todense(), dtype=np.float32), labels, np.array(edges), test_idx_range


def test_cora_binding_matches_frozen_old_logic(tiny, tmp_path):
    feats, labels, test_ids = make_raw(tmp_path, "cora")
    # Fake-Graph symmetrisch machen, wie der echte Cora-Rohgraph (dort sind beide Richtungen schon enthalten)
    g = pickle.load(open(tmp_path / "ind.cora.graph", "rb"))
    sym = {u: set(v) for u, v in g.items()}
    for u, vs in list(g.items()):
        for v in vs:
            sym.setdefault(v, set()).add(u)
    pickle.dump({u: sorted(v) for u, v in sym.items()}, open(tmp_path / "ind.cora.graph", "wb"))
    from repro_agent.harness import cora_data
    new = cora_data.load_raw(tmp_path)
    old = old_cora_load_raw(tmp_path)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)
    assert np.array_equal(new[0][test_ids], feats[test_ids].astype(np.float32)) and cora_data.N_TRAIN == 140


def test_cora_module_api_unchanged():
    from repro_agent.harness import cora_data
    for fn in ("download_raw", "load_raw", "prepare_public", "majority_baseline", "no_graph_baseline",
               "val_labels", "val_majority_baseline", "test_labels"):
        assert callable(getattr(cora_data, fn))
    assert (cora_data.N_TRAIN, cora_data.N_VAL) == (140, 500)
    assert (pd_.SPECS["citeseer"]["n_train"], pd_.SPECS["pubmed"]["n_train"]) == (120, 60)
    assert {n: (s["nodes"], s["classes"]) for n, s in pd_.SPECS.items()} == {"cora": (2708, 7), "citeseer": (3327, 6), "pubmed": (19717, 3)}


def test_citeseer_isolated_test_nodes_zero_padded(tiny, tmp_path):
    gaps = (2, 5)
    feats, labels, test_ids = make_raw(tmp_path, "citeseer", gaps)
    x, y, edges, test_idx = pd_.load_raw("citeseer", tmp_path)
    n_allx = N_TRAIN + N_VAL + 7
    assert x.shape[0] == n_allx + N_TEST_SMALL + len(gaps) == y.shape[0]
    assert list(test_idx) == test_ids and len(test_idx) == N_TEST_SMALL
    for g in gaps:                                  # Luecken: Nullvektoren fuer Merkmale und Labels, in keinem Split
        assert x[n_allx + g].sum() == 0 and y[n_allx + g].sum() == 0 and (n_allx + g) not in test_idx
    assert np.array_equal(x[test_idx], feats[test_idx].astype(np.float32))        # Umsortierung (test.index) korrekt
    assert np.array_equal(y[test_idx].argmax(1), labels[test_idx])


@pytest.mark.parametrize("name", NAMES)
def test_edges_undirected_without_selfloops_or_duplicates(tiny, tmp_path, name):
    make_raw(tmp_path, name)
    _, _, edges, _ = pd_.load_raw(name, tmp_path)
    pairs = {tuple(e) for e in edges.tolist()}
    assert len(pairs) == len(edges) and all(u != v for u, v in pairs) and pairs == {(v, u) for u, v in pairs}


@pytest.mark.parametrize("name", NAMES)
def test_public_file_hides_test_labels(tiny, tmp_path, name):
    feats, labels, test_ids = make_raw(tmp_path / "raw", name)
    out = tmp_path / "pub.npz"
    pd_.prepare_public(name, tmp_path / "raw", out)
    d = np.load(out)
    assert (d["labels"][d["test_idx"]] == -1).all() and (d["labels"][N_TRAIN + N_VAL:] == -1).all()
    assert (d["labels"][:N_TRAIN + N_VAL] >= 0).all()
    assert list(d["train_idx"]) == list(range(N_TRAIN)) and list(d["val_idx"]) == list(range(N_TRAIN, N_TRAIN + N_VAL))
    assert not set(d["test_idx"]) & set(range(N_TRAIN + N_VAL))
    # Testlabels stehen nur im Harness-Pfad, nicht im Agenten-Datensatz
    truth = pd_.test_labels(name, tmp_path / "raw")
    assert set(truth) == {str(i) for i in test_ids} and truth[str(test_ids[0])] == str(labels[test_ids[0]])
    assert sorted(d.files) == ["edges", "features", "labels", "test_idx", "train_idx", "val_idx"]
    assert pd_.val_labels(name, tmp_path / "raw") == {str(i): str(labels[i]) for i in range(N_TRAIN, N_TRAIN + N_VAL)}


@pytest.mark.parametrize("name", NAMES)
def test_harness_eval_accuracy_and_validity(tiny, tmp_path, monkeypatch, name, capsys):
    _, labels, test_ids = make_raw(tmp_path / "raw", name)
    monkeypatch.syspath_prepend(str(HARNESS_DIR))
    import planetoid_data as flat_pd   # flache Importe wie im Branch (dieselbe Datei, andere Modul-Identitaet)
    import planetoid_eval
    for mod in (flat_pd,):
        monkeypatch.setitem(mod.SPECS, name, pd_.SPECS[name])
    monkeypatch.setattr(flat_pd, "N_VAL", N_VAL)
    monkeypatch.setattr(flat_pd, "download_raw", lambda n, target: [(target / p.name).write_bytes(p.read_bytes()) for p in (tmp_path / "raw").iterdir()])
    pred = tmp_path / "predictions.csv"
    monkeypatch.setattr(planetoid_eval, "PREDICTIONS", str(pred))
    wrong = [(labels[i] + 1) % N_CLS for i in test_ids]
    for preds, expect in ((list(labels[test_ids]), 1.0), (wrong, 0.0)):
        pred.write_text("id,prediction\n" + "".join(f"{i},{p}\n" for i, p in zip(test_ids, preds)))
        planetoid_eval.main(name)
        res = json.loads(capsys.readouterr().out)
        assert res["status"] == "ok" and res["metrics"]["accuracy"] == expect and set(res["baselines"]) == {"majority", "no_graph"}
    pred.write_text("id,prediction\n" + "".join(f"{i},0\n" for i in test_ids[:-1]))
    planetoid_eval.main(name)
    assert json.loads(capsys.readouterr().out)["status"] == "ungueltig"       # unvollstaendig


@pytest.mark.parametrize("name,task", [("cora", cora_task), ("citeseer", citeseer_task), ("pubmed", pubmed_task)])
def test_task_definition_per_dataset(name, task):
    t = task()
    label, n, f, c, n_train, _, _ = PLANETOID[name]
    path = f"/data/{name}_public.npz"
    d = t.data_description
    assert path in d and f"[{n}, {f}]" in d and f"class 0..{c - 1}" in d and f"train_idx ({n_train})" in d and "val_idx (500)" in d and "test_idx (1000)" in d
    assert t.edge_probe == (path, n, 500, 1000) and t.probe_dims == (c, f)
    for cmd in (t.ablation_cmd, t.mask_val_cmd, t.smoke_ids_cmd, t.val_smoke_ids_cmd):
        assert path in cmd and "cora_public" not in cmd.replace(path, "") if name != "cora" else path in cmd
    assert path in t.code_notes and "SMOKE=1" in t.code_notes and t.setup_cmds[-1].startswith(f"test -s {path}")
    assert t.eval_cmd == f"python {{harness_dir}}/{name}_eval.py" and t.val_eval_cmd == f"python {{harness_dir}}/{name}_val_eval.py"
    for fname in t.harness_files + list(t.val_harness_files):
        assert (HARNESS_DIR / fname).exists(), fname
    assert "planetoid_data.py" in t.harness_files and f"{name}_data.py" in t.harness_files
    # Testlabels nie im Agenten-Datensatz: Rohdaten-Zugriff im Agenten-Code ist verboten (statische Pruefung)
    assert code_check.find_violations(f"x = open('ind.{name}.ty', 'rb')")


def test_code_check_blocks_raw_files_of_all_planetoid_sets():
    for n in NAMES:
        assert code_check.find_violations(f"pickle.load(open('/tmp/ind.{n}.ty','rb'))")
    assert not code_check.find_violations("d = np.load('/data/citeseer_public.npz')")


def test_cora_description_unchanged_and_notes_per_dataset():
    assert "float32 [2708, 1433] bag-of-words (binary, not normalised)" in planetoid_data_description("cora")
    assert "isolated" not in planetoid_data_description("cora") and "isolated" in planetoid_data_description("citeseer")
    assert "TF-IDF" in planetoid_data_description("pubmed") and "sparse" in planetoid_data_description("pubmed")


@pytest.mark.parametrize("name", NAMES)
def test_edge_probe_in_dataset_format(name):
    t = planetoid_task(name)
    _, n, f, c, _, _, _ = PLANETOID[name]
    npz, truth = edge_direction.make_probe(*t.edge_probe[1:], seed=0, n_classes=t.probe_dims[0], n_features=t.probe_dims[1])
    assert npz["features"].shape == (n, f) and npz["labels"].max() < c
    assert len(npz["train_idx"]) == 20 * c and (npz["labels"][npz["test_idx"]] == -1).all()


def test_only_validated_datasets_are_registered():
    # datasets_v2_validation.md: CiteSeer bestanden; PubMed (GAT) und ETTh2/ETTm1/ETTm2 (DLinear) verfehlen die Regel
    assert set(TASKS) == {"Cora", "CiteSeer", "ETTh1", "Cora-LCC"}


@pytest.mark.skipif(not os.environ.get("PLANETOID_NETWORK_TESTS"), reason="braucht Netz (kimiyoung/planetoid), PLANETOID_NETWORK_TESTS=1")
def test_real_data_cora_hashes_and_sizes(tmp_path):
    """Cora-Ausgabe identisch mit der vor der Parametrisierung (Hashes aus dem Lauf vom 2026-10-04) + Groessen der neuen Datensaetze."""
    golden = {"edges": "16f877fb5e5a260a88205571a58ba3391484f48daddd61df7627b2316132e4bb",
              "features": "f0faab5177bcc12f5688f042c8e0ed24ffb9baa8efc3ae7cde440d42524c9075",
              "labels": "547336be7429cb5cd6cf92ee7643b6d3c0daac0024a909353239d3ef4f22cc46",
              "test_idx": "16680ae981534bf8b04697b2c29ae6aa105d20fb580182d87346c5a44cd58634",
              "train_idx": "a103a53f219cde44e3628ccf636c012fb99435a924ae2e947a3f04db437d03d0",
              "val_idx": "ade6bc10376e996ed21ae31286a4d8dfb651629945f2f5cea2437cd2f117d5f6"}
    from repro_agent.harness import cora_data
    cora_data.download_raw(tmp_path / "cora")
    cora_data.prepare_public(tmp_path / "cora", tmp_path / "c.npz")
    d = np.load(tmp_path / "c.npz")
    assert {k: hashlib.sha256(np.ascontiguousarray(d[k]).tobytes()).hexdigest() for k in golden} == golden
    assert hashlib.sha256(json.dumps(sorted(cora_data.test_labels(tmp_path / "cora").items())).encode()).hexdigest() == \
        "a49378ce46fc4a40d707bd372432ddd9e10711740f6fcdf0615a3882f05af364"
    assert (cora_data.majority_baseline(tmp_path / "cora"), cora_data.no_graph_baseline(tmp_path / "cora")) == (0.319, 0.506)
    for name, spec in pd_.SPECS.items():
        pd_.download_raw(name, tmp_path / name)
        x, y, _, test_idx = pd_.load_raw(name, tmp_path / name)
        assert x.shape == (spec["nodes"], spec["features"]) and y.shape == (spec["nodes"], spec["classes"]) and len(test_idx) == 1000
