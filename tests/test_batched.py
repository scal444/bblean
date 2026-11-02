import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import bblean.bitbirch
from bblean.bitbirch import BitBirch
from bblean.fingerprints import make_fake_fingerprints, unpack_fingerprints
from bblean.similarity import centroid_from_sum

if bblean.bitbirch._cpp_bitbirch is None and os.getenv("BITBIRCH_CANT_SKIP_CPP_TESTS"):
    raise ImportError("The C++ BitBirch tree is not available")

pytestmark = pytest.mark.skipif(
    bblean.bitbirch._cpp_bitbirch is None, reason="C++ BitBirch tree not available"
)

_SEED = 12620509540149709235
# Labels from nvmolkit.clustering.bitbirch (GPU) for the first 20k
# make_fake_fingerprints(..., seed=_SEED), as uint16. Name: (threshold,
# branching_factor, batch_size)
_NVMOLKIT_LABELS = Path(__file__).parent / "nvmolkit-batched-labels.npz"
_CASES = {
    "n20k_t03_bf50_b1024": (0.3, 50, 1024),
    "n20k_t03_bf7_b256": (0.3, 7, 256),
    "n20k_t05_bf50_b777": (0.5, 50, 777),
    "n20k_t03_bf254_b4096": (0.3, 254, 4096),
}


def _fps(n: int = 20000) -> np.ndarray:
    return make_fake_fingerprints(n, n_features=2048, seed=_SEED, pack=True)


def _first_member_labels(tree: BitBirch, n: int) -> np.ndarray:
    # Clusters numbered by their first member (as nvMolKit numbers them)
    labels = np.empty(n, dtype=np.int64)
    for c, members in enumerate(tree.get_cluster_mol_ids()):
        labels[members] = c
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inverse]


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("n_threads", [1, 3])
def test_batched_equals_nvmolkit(name: str, n_threads: int) -> None:
    threshold, bf, batch_size = _CASES[name]
    fps = _fps()
    tree = BitBirch(
        threshold=threshold,
        branching_factor=bf,
        batch_size=batch_size,
        n_threads=n_threads,
    ).fit(fps)
    expect = np.load(_NVMOLKIT_LABELS)[name].astype(np.int64)
    assert np.array_equal(_first_member_labels(tree, len(fps)), expect)
    assert tree._tree._tree._check_invariants() == ""


def test_batched_input_formats() -> None:
    fps = _fps(3000)
    kwargs = dict(threshold=0.3, branching_factor=20, batch_size=128, n_threads=2)
    expect = BitBirch(**kwargs).fit(fps).get_cluster_mol_ids()  # type: ignore
    # Unpacked
    tree = BitBirch(**kwargs).fit(unpack_fingerprints(fps), input_is_packed=False)
    assert tree.get_cluster_mol_ids() == expect
    # List of arrays
    tree = BitBirch(**kwargs).fit(list(fps))  # type: ignore
    assert tree.get_cluster_mol_ids() == expect
    # Not 8-byte aligned (copied internally)
    buf = np.empty(fps.size + 1, dtype=np.uint8)[1:].reshape(fps.shape)
    buf[:] = fps
    assert BitBirch(**kwargs).fit(buf).get_cluster_mol_ids() == expect  # type: ignore
    # Reinsert indices
    tree = BitBirch(**kwargs).fit(fps, reinsert_indices=range(10, 3010))  # type: ignore
    assert tree.get_cluster_mol_ids() == [[i + 10 for i in c] for c in expect]


def test_batched_file_input(tmp_path: Path) -> None:
    fps = _fps(3000)
    np.save(tmp_path / "fps.npy", fps)
    kwargs = dict(threshold=0.3, branching_factor=20, batch_size=128, n_threads=2)
    expect = BitBirch(**kwargs).fit(fps).get_cluster_mol_ids()  # type: ignore
    tree = BitBirch(**kwargs).fit(tmp_path / "fps.npy")  # type: ignore
    assert tree.get_cluster_mol_ids() == expect


@pytest.mark.parametrize("n_features", [8, 1032, 1024])
def test_batched_feature_sizes(n_features: int) -> None:
    fps = make_fake_fingerprints(2000, n_features=n_features, seed=_SEED, pack=True)
    trees = [
        BitBirch(threshold=0.4, branching_factor=10, batch_size=64, n_threads=t).fit(
            fps, n_features=n_features
        )
        for t in (1, 4)
    ]
    assert trees[0].get_cluster_mol_ids() == trees[1].get_cluster_mol_ids()
    assert trees[0]._tree._tree._check_invariants() == ""


def test_batched_tree_accessors_and_followups() -> None:
    fps = _fps(4000)
    tree = BitBirch(threshold=0.3, branching_factor=20, batch_size=256, n_threads=2)
    tree.fit(fps)
    n = tree.num_fitted_fps
    assert n == 4000
    ids = tree.get_cluster_mol_ids()
    assert sorted(i for c in ids for i in c) == list(range(n))
    # Molecules in each cluster in input order
    assert all(c == sorted(c) for c in ids)
    assignments = tree.get_assignments()
    for k, c in enumerate(ids, 1):
        assert (assignments[c] == k).all()
    # Centroids are the majority vote of the members
    for c, cent in zip(ids, tree.get_centroids()):
        ls = unpack_fingerprints(fps[c]).sum(axis=0, dtype=np.uint64)
        assert np.array_equal(cent, centroid_from_sum(ls, len(c), pack=True))
    for bf, c in zip(tree._get_leaf_bfs(), ids):
        assert bf.n_samples == len(c)
    # Pickling
    copy = pickle.loads(pickle.dumps(tree))
    assert copy.get_cluster_mol_ids() == ids
    # A second batched fit needs reset()
    with pytest.raises(ValueError):
        tree.fit(fps)
    # Serial insertion into the batched tree
    more = _fps(5000)[4000:]
    tree.batch_size = None
    copy.batch_size = None
    tree.fit(more)
    copy.fit(more)
    assert tree.num_fitted_fps == 5000
    assert tree.get_cluster_mol_ids() == copy.get_cluster_mol_ids()
    assert tree._tree._tree._check_invariants() == ""
    # Refinement / reclustering (serial reinsertion)
    tree.refine_inplace(_fps(5000))
    tree.recluster_inplace(iterations=1)
    assert sorted(i for c in tree.get_cluster_mol_ids() for i in c) == list(range(5000))
    tree.delete_internal_nodes()
    assert len(tree.get_cluster_mol_ids()) > 0
    tree.reset()
    tree.batch_size = 256
    assert tree.fit(fps).get_cluster_mol_ids() == ids


def test_batched_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fps = _fps(100)
    with pytest.raises(ValueError):
        BitBirch(batch_size=0)
    with pytest.raises(ValueError):
        BitBirch(batch_size=16, n_threads=0)
    with pytest.raises(ValueError):
        BitBirch(batch_size=16, merge_criterion="radius").fit(fps)
    with pytest.raises(ValueError):
        BitBirch(batch_size=16, branching_factor=2).fit(fps)
    with pytest.raises(ValueError):
        BitBirch(batch_size=16).fit(fps, weights=[2] * len(fps))
    with pytest.raises(ValueError):
        BitBirch(batch_size=16).fit(
            unpack_fingerprints(fps) * 2, input_is_packed=False
        )
    assert "batch_size=16" in repr(BitBirch(batch_size=16))
    # A batched fit needs the C++ tree
    monkeypatch.setattr(bblean.bitbirch, "_cpp_bitbirch", None)
    with pytest.raises(ValueError):
        BitBirch(batch_size=16).fit(fps)


def test_batched_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from bblean.cli import app

    fps = _fps(3000)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    np.save(one / "fingerprints.npy", fps)
    np.save(two / "a.npy", fps[:1000])
    np.save(two / "b.npy", fps[1000:])
    expect = (
        BitBirch(threshold=0.3, branching_factor=50, batch_size=256, n_threads=2)
        .fit(fps)
        .get_cluster_mol_ids()
    )
    runner = CliRunner()
    for d in (one, two):
        out = d / "out"
        args = ["run", str(d), "-o", str(out), "-b", "50", "-t", "0.3"]
        args += ["--batch-size", "256", "--threads", "2", "--no-monitor-mem"]
        result = runner.invoke(app, args + ["--refine-num", "0"])
        assert result.exit_code == 0, result.output
        with open(out / "clusters.pkl", "rb") as f:
            assert pickle.load(f) == expect
    # With a (serial) refinement round
    out = one / "out-refine"
    result = runner.invoke(
        app,
        ["run", str(one), "-o", str(out), "-b", "50", "-t", "0.3"]
        + ["--batch-size", "256", "--refine-num", "1", "--no-monitor-mem"],
    )
    assert result.exit_code == 0, result.output
    with open(out / "clusters.pkl", "rb") as f:
        ids = pickle.load(f)
    assert sorted(i for c in ids for i in c) == list(range(len(fps)))
