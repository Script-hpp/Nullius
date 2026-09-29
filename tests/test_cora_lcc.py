import numpy as np

from repro_agent.harness.cora_lcc_data import lcc_from_arrays, make_split


def test_lcc_keeps_largest_component_and_renumbers():
    feats = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.array([0, 1, 0, 1, 0, 1])
    edges = np.array([[0, 1], [1, 2], [3, 4]])          # Komponenten {0,1,2}, {3,4}, {5}
    f, y, e = lcc_from_arrays(feats, labels, edges)
    assert f.shape[0] == 3 and y.tolist() == [0, 1, 0]
    assert set(map(tuple, e.tolist())) == {(0, 1), (1, 0), (1, 2), (2, 1)}   # symmetrisch, neu nummeriert


def test_split_sizes_disjoint_deterministic_and_differs_per_k():
    labels = np.repeat(np.arange(7), 100)
    tr, va, te = make_split(labels, 0)
    assert len(tr) == 140 and len(va) == 210 and len(te) == 700 - 350
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert all((labels[tr] == c).sum() == 20 and (labels[va] == c).sum() == 30 for c in range(7))
    tr2, _, _ = make_split(labels, 0)
    assert (tr == tr2).all()
    assert not (make_split(labels, 1)[0] == tr).all()


def test_split_rejects_too_small_class():
    import pytest

    with pytest.raises(ValueError):
        make_split(np.array([0] * 100 + [1] * 10), 0)
