from typing import TypeVar, Dict, Any, Iterable, Mapping, Tuple, overload
from typing_extensions import Protocol

_T = TypeVar("_T")


_Addable = TypeVar("_Addable", bound='Addable')
class Addable(Protocol):
    def __add__(self: _Addable, other: _Addable) -> _Addable:
        ...


_K = TypeVar("_K")


class AddableDict(Dict[_K, _Addable]):

    def __add__(self, other: "AddableDict[_K, _Addable]") -> "AddableDict[_K, _Addable]":
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
