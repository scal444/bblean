import os
from pathlib import Path
import pickle
import typing as tp

import numpy as np
import pytest

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.merges import DiameterMerge

from tree_cases import CASES, SEED, flat_cluster_ids  # type: ignore

if bblean.bitbirch._cpp_bitbirch is None and os.getenv("BITBIRCH_CANT_SKIP_CPP_TESTS"):
    raise ImportError("The C++ BitBirch tree is not available")

_REFERENCE = np.load(Path(__file__).parent / "tree-reference.npz")
_TREES = ["python"]
if bblean.bitbirch._cpp_bitbirch is not None:
    _TREES.append("cpp")


@pytest.fixture(params=_TREES)
def tree_kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    # The python tree is used if the C++ extension is not available
    if request.param == "python":
        monkeypatch.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
    return request.param


@pytest.mark.parametrize("name", list(CASES))
def test_tree_reference(name: str, tree_kind: str) -> None:
    # Same clusters, cluster order and molecule order as the reference results
    tree = CASES[name]()
    ids, offsets = flat_cluster_ids(tree)
    assert ids.tolist() == _REFERENCE[f"{name}/ids"].tolist()
    assert offsets.tolist() == _REFERENCE[f"{name}/offsets"].tolist()
    assert tree.num_fitted_fps == _REFERENCE[f"{name}/num_fitted"]


def _state(tree: BitBirch) -> tuple[tp.Any, ...]:
    return (
        tree.get_cluster_mol_ids(),
        tree.get_cluster_mol_ids(sort=False),
        tree.get_assignments().tolist(),
        tree.get_assignments(sort=False).tolist(),
        [c.tolist() for c in tree.get_centroids()],
        [c.tolist() for c in tree.get_centroids(packed=False)],
        [(bf._buffer.dtype.name, bf._buffer.tolist()) for bf in tree._get_leaf_bfs()],
        tree.num_fitted_fps,
    )


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
def test_tree_state_python_and_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    # Centroids, linear sums and assignments are the same for both trees
    fps = make_fake_fingerprints(800, n_features=2048, seed=SEED, pack=True)
    cpp = BitBirch(threshold=0.3, branching_factor=7).fit(fps)
    with monkeypatch.context() as m:
        m.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
        py = BitBirch(threshold=0.3, branching_factor=7).fit(fps)
    assert cpp._cpp_tree is not None and py._cpp_tree is None
    assert _state(cpp) == _state(py)


@pytest.mark.skipif("cpp" not in _TREES, reason="C++ BitBirch tree not available")
def test_tree_selection() -> None:
    fps = make_fake_fingerprints(50, n_features=2048, seed=SEED, pack=True)
    assert BitBirch().fit(fps)._cpp_tree is not None
    assert BitBirch(merge_criterion=DiameterMerge()).fit(fps)._cpp_tree is not None
    assert BitBirch().fit(list(fps))._cpp_tree is not None
    unpacked = unpack_fingerprints(fps)
    assert BitBirch().fit(unpacked, input_is_packed=False)._cpp_tree is not None

    # The python tree is used for user defined merge criteria, non-binary or non
    # uint8 inputs, and n_features that are not a multiple of 8
    class Custom(DiameterMerge):
        pass

    assert BitBirch(merge_criterion=Custom()).fit(fps)._cpp_tree is None
    assert BitBirch().fit(unpacked * 2, input_is_packed=False)._cpp_tree is None
    assert BitBirch().fit(fps.astype(np.int64))._cpp_tree is None
    assert bblean.bitbirch._packed_for_cpp(fps, True, 2044) is None

    # A C++ tree can't continue with a user defined criterion or non-binary input
    tree = BitBirch().fit(fps)
    with pytest.raises(ValueError):
        tree.fit(unpacked * 2, input_is_packed=False)
    tree.set_merge(Custom())
    with pytest.raises(ValueError):
        tree.fit(fps)


def test_tree_pickle_and_continue(tree_kind: str) -> None:
    fps = make_fake_fingerprints(600, n_features=2048, seed=SEED, pack=True)
    expect = BitBirch(threshold=0.4, branching_factor=10).fit(fps).get_cluster_mol_ids()
    tree = BitBirch(threshold=0.4, branching_factor=10).fit(fps[:300])
    tree = pickle.loads(pickle.dumps(tree))
    assert tree.fit(fps[300:]).get_cluster_mol_ids() == expect


def test_tree_delete_internal_nodes(tree_kind: str) -> None:
    fps = make_fake_fingerprints(300, n_features=2048, seed=SEED, pack=True)
    tree = BitBirch(threshold=0.4, branching_factor=5).fit(fps)
    ids = tree.get_cluster_mol_ids()
    tree.delete_internal_nodes()
    assert tree.get_cluster_mol_ids() == ids
    with pytest.raises(ValueError):
        tree.fit(fps)
    tree.reset()
    assert not tree.is_init
    assert tree.fit(fps).get_cluster_mol_ids() == ids
