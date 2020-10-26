from dnips.addable_dict import AddableDict, Addable
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
