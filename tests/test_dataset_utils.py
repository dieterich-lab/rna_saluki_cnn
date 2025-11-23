# This test module avoids requiring pytest at import time so the helper checks
# can be executed directly in environments that do not have pytest installed.

from biolm_utils.dataset_utils import check_batchsize, make_subsets, split_indices


def test_split_indices_two_way():
    idx = list(range(10))
    train, val, test = split_indices(idx, [80, 20])
    assert len(train) == 8
    assert len(val) == 2
    assert test is None


def test_split_indices_three_way():
    idx = list(range(20))
    train, val, test = split_indices(idx, [50, 25, 25])
    assert len(train) == 10
    assert len(val) == 5
    assert len(test) == 5


def test_make_subsets_and_check_batchsize(tmp_path):
    class Dummy:
        def __init__(self, n=10):
            self._data = list(range(n))

        def __len__(self):
            return len(self._data)

        def __getitem__(self, item):
            return self._data[item]

    ds = Dummy(5)
    train, val, test = make_subsets(ds, [0, 1, 2], [3], [4], dev=False)
    assert len(train) == 3 and len(val) == 1 and len(test) == 1

    # too large batch raises
    try:
        check_batchsize(train, 10, "train")
    except ValueError:
        # expected
        return
    raise AssertionError("Expected ValueError for too-large batchsize")
