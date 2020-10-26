from typing import (
    TypeVar,
    Dict,
    Any,
    Iterable,
    Mapping,
    Tuple,
    overload,
    Callable,
    Sequence,
    List,
)
from typing_extensions import Protocol
import numpy as np  # type: ignore

_T = TypeVar("_T")


_Addable = TypeVar("_Addable", bound="Addable")


class Addable(Protocol):
    def __add__(self: _Addable, other: _Addable) -> _Addable:
        ...


_K = TypeVar("_K")


class AddableDict(Dict[_K, _Addable]):
    def __add__(
        self, other: "AddableDict[_K, _Addable]"
    ) -> "AddableDict[_K, _Addable]":
        other_keys = set(other.keys())
        self_keys = set(self.keys())
        intersected_keys = self_keys & other_keys
        res: "AddableDict[_K, _Addable]" = AddableDict(
            {k: self[k] + other[k] for k in intersected_keys}
        )
        for key in other_keys - intersected_keys:
            res[key] = other[key]
        for key in self_keys - intersected_keys:
            res[key] = self[key]
        return res


def _fill_array_with_dict(
    d: Dict[str, Any], orderings: Sequence[List[str]], array_to_fill: np.ndarray
) -> None:
    """Convert a dict of any depth into a numpy array, where keys at a specific depth
    correspond to a specific numpy dimension."""
    ordering = orderings[0]
    if len(orderings) == 1:
        assert isinstance(d, dict)
        for k, v in d.items():
            try:
                k_idx = ordering.index(k)
            except IndexError:
                continue
            else:
                array_to_fill[k_idx] = v
    else:
        for k, v in d.items():
            try:
                k_idx = ordering.index(k)
            except IndexError:
                continue
            else:
                _fill_array_with_dict(v, orderings[1:], array_to_fill[k_idx])


def dict_to_numpy(
    d: Any,
    orderings: Sequence[List[str]],
    fill_value: np.generic = 0.0,
    dtype: np.dtype = np.float,
) -> np.ndarray:
    """Convert a dict of any depth into a numpy array, where keys at a specific depth
    correspond to a specific numpy dimension."""
    shape = [len(ordering) for ordering in orderings]
    res = np.full(shape=shape, fill_value=fill_value, dtype=dtype)
    _fill_array_with_dict(d, orderings, res)
    return res
