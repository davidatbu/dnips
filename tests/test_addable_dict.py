import numpy as np  # type: ignore
import pytest
from dnips.addable_dict import AddableDict, Addable, dict_to_numpy
from typing import Counter, Union


def test_addable_dict() -> None:
    test1: AddableDict[str, Union[AddableDict[str, int], int]] = AddableDict(
        outer_1=AddableDict(inner_1=2, inner_2=3), c=5
    )
    test2: AddableDict[str, Union[AddableDict[str, int], int]] = AddableDict(
        outer_1=AddableDict(inner_1=2),
        outer_2=AddableDict(inner_1=5, inner_2=7),
        c=4,
    )

    res: AddableDict[str, Union[AddableDict[str, int], int]] = AddableDict(
        outer_1=AddableDict(inner_1=4, inner_2=3),
        outer_2=AddableDict(inner_1=5, inner_2=7),
        c=9,
    )

    assert dict(test1 + test2) == res


def test_addable_dict_with_counters() -> None:
    test1 = AddableDict(
        outer_1=AddableDict(inner_1=Counter({"hi": 2}), inner_2=Counter({"hello": 3})),
        outer_2=AddableDict(inner_1=Counter({"hi": 1}), inner_2=Counter({"howdy": 3})),
    )
    test2 = AddableDict(
        outer_1=AddableDict(
            inner_1=Counter({"whoops": 9, "hi": 5}), inner_2=Counter({"hi": 1})
        ),
        outer_2=AddableDict(inner_1=Counter({"hi": 3}), inner_2=Counter({"howdy": 2})),
    )

    assert dict(test1 + test2) == AddableDict(
        outer_1=AddableDict(
            inner_1=Counter({"whoops": 9, "hi": 7}),
            inner_2=Counter({"hi": 1, "hello": 3}),
        ),
        outer_2=AddableDict(
            inner_1=Counter({"hi": 4}),
            inner_2=Counter({"howdy": 5}),
        ),
    )


class TestDictToNumpy:
    def test_simple(self) -> None:
        dict1 = {
            "dim1_a": {"dim2_a": {"dim3_a": 1, "dim3_b": 2}, "dim2_c": {"dim3_b": 3}},
            "dim1_b": {"dim2_a": {"dim3_a": 4}, "dim2_b": {"dim3_b": 5}},
        }

        orderings = [
            ["dim1_a", "dim1_b"],
            ["dim2_a", "dim2_b", "dim2_c"],
            ["dim3_a", "dim3_b"],
        ]
        res, keys_not_found = dict_to_numpy(dict1, orderings)

        exp = np.array(
            [[[1, 2], [0, 0], [0, 3.0]], [[4, 0], [0, 5], [0, 0]]], dtype=np.float
        )

        np.testing.assert_equal(res, exp)
        assert keys_not_found == [None] * 3

    def test_ignore_unfound_keys(self) -> None:
        dict1 = {
            "DOESNT EXIST": {
                "dim2_a": {"dim3_a": 1, "dim3_b": 2},
                "dim2_c": {"dim3_b": 3},
            },
            "dim1_b": {"dim2_a": {"dim3_a": 4}, "dim2_b": {"dim3_b": 5}},
        }

        orderings = [
            ["dim1_a", "dim1_b"],
            ["dim2_a", "dim2_b", "dim2_c"],
            ["dim3_a", "dim3_b"],
        ]

        with pytest.raises(ValueError):
            _ = dict_to_numpy(dict1, orderings)

        res, keys_not_found = dict_to_numpy(
            dict1, orderings, ignore_not_found_keys=[True, False, False]
        )
        exp = np.array(
            [[[0, 0], [0, 0], [0, 0]], [[4, 0], [0, 5], [0, 0]]], dtype=np.float
        )
        assert keys_not_found == [["DOESNT EXIST"], None, None]

        np.testing.assert_equal(res, exp)

    def test_transform_func(self) -> None:
        dict1 = {
            "DOESNT EXIST": {
                "dim2_a": {"dim3_a": 1, "dim3_b": 2},
                "dim2_c": {"dim3_b": 3},
            },
            "dim1_b": {"dim2_a": {"dim3_a": 4}, "dim2_b": {"dim3_b": 5}},
        }

        orderings = [
            ["dim1_a", "dim1_b"],
            ["dim2_a", "dim2_b", "dim2_c"],
            ["dim3_a", "dim3_b"],
        ]

        res, keys_not_found = dict_to_numpy(
            dict1,
            orderings,
            ignore_not_found_keys=[True, False, False],
            transform_key_funcs=[
                lambda x: x if x != "DOESNT EXIST" else "dim1_a",
                lambda x: x,
                lambda x: x,
            ],
        )
        exp = np.array(
            [[[1, 2], [0, 0], [0, 3.0]], [[4, 0], [0, 5], [0, 0]]], dtype=np.float
        )
        assert keys_not_found == [[], None, None]

        np.testing.assert_equal(res, exp)
