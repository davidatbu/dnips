from typing import (
    Iterable,
    Type,
    TypeVar,
    Tuple,
    Callable,
    Optional,
    Dict,
    Union,
    Any,
    Sequence,
    List,
)
from typing_extensions import Protocol
import tqdm
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


TransformKeyFunc = Callable[[_T], _T]


def _fill_array_with_dict(
    d: Union[Iterable[Tuple[_T, Any]], Dict[_T, Any]],
    orderings: Sequence[List[_T]],
    array_to_fill: np.ndarray,
    ignore_not_found_keys: Sequence[bool],
    transform_key_funcs: Sequence[TransformKeyFunc],
    tqdm_pos: Optional[int],
    transform_value_func: Optional[Callable[[Any], Any]],
    tqdm_prompt:str=None,
) -> List[Optional[List[_T]]]:
    """Convert a dict of any depth into a numpy array, where keys at a specific depth
    correspond to a specific numpy dimension.

    Returns:
        keys_not_found: A list of the *original* keys(before transform_key_func was
        applied) that were not found in the dict, at each specific level. Returned only
        when ignore_not_found_keys is True for each level.
    """
    ordering = orderings[0]
    cur_ignore_not_found_key = ignore_not_found_keys[0]
    cur_transform_key_func = transform_key_funcs[0]
    if cur_ignore_not_found_key:
        cur_keys_not_found: Optional[List[_T]] = []
    else:
        cur_keys_not_found = None
    iterable: Iterable[Tuple[_T, Any]]

    if isinstance(d, dict):
        iterable = d.items()
    else:
        iterable = d

    if tqdm_pos is not None and tqdm_pos < 1:
        cur_tqdm_prompt=f"At level {tqdm_pos+1}: {tqdm_prompt or ''}" if tqdm_pos else None
        iterable = tqdm.tqdm(iterable, position=tqdm_pos, desc=cur_tqdm_prompt, leave=False)

    further_keys_not_found = []
    if len(orderings) == 1:
        assert isinstance(d, dict)
        for k, v in iterable:
            try:
                k = cur_transform_key_func(k)
                k_idx = ordering.index(k)
            except ValueError:
                if cur_ignore_not_found_key:
                    assert cur_keys_not_found is not None
                    cur_keys_not_found.append(k)
                    continue
                else:
                    raise ValueError(
                        f"Key {k} not found in ordering, but ignore_not_found_keys is False."
                    )
            else:
                if transform_value_func:
                    v = transform_value_func(v)
                array_to_fill[k_idx] = v
    else:
        for k, v in iterable:
            try:
                k = cur_transform_key_func(k)
                k_idx = ordering.index(k)
            except ValueError:
                if cur_ignore_not_found_key:
                    assert cur_keys_not_found is not None
                    cur_keys_not_found.append(k)
                    continue
                else:
                    raise ValueError(
                        f"Key {k} not found in ordering, but ignore_not_found_keys is False."
                    )
            else:
                further_keys_not_found = _fill_array_with_dict(
                    v,
                    orderings[1:],
                    array_to_fill[k_idx],
                    ignore_not_found_keys[1:],
                    transform_key_funcs[1:],
                    transform_value_func=transform_value_func,
                    tqdm_pos=tqdm_pos + 1 if tqdm_pos is not None else None,
                    tqdm_prompt=f"{k}" if tqdm_pos else None
                )
    return [cur_keys_not_found] + further_keys_not_found


def dict_to_numpy(
    d: Union[Iterable[Tuple[_T, Any]], Dict[_T, Any]],
    orderings: Sequence[List[Any]],
    ignore_not_found_keys: Union[bool, Sequence[bool]] = False,
    fill_value: Any = 0.0,
    dtype: 'np._Dtype' = np.float32, #type: ignore
    transform_key_funcs: Union[
        TransformKeyFunc, Sequence[TransformKeyFunc]
    ] = lambda x: x,
    transform_value_func: Callable[[Any], Any] = None,
    show_tqdm: bool = False,
) -> Tuple[np.ndarray, List[Optional[List[_T]]]]:
    """Convert a dict of any depth into a numpy array, where keys at a specific depth
    correspond to a specific numpy dimension."""
    if callable(transform_key_funcs):
        transform_key_funcs = [transform_key_funcs] * len(orderings)
    if isinstance(ignore_not_found_keys, bool):
        ignore_not_found_keys = [ignore_not_found_keys] * len(orderings)

    if len(transform_key_funcs) != len(orderings):
        raise ValueError(
            f"Length of transform_key_funcs ({len(transform_key_funcs)}"
            "not equal to length of orderings ({len(orderings)})."
        )
    if len(ignore_not_found_keys) != len(orderings):
        raise ValueError(
            f"Length of ignore_not_found_keys ({len(ignore_not_found_keys)}"
            "not equal to length of orderings ({len(orderings)})."
        )
    shape = [len(ordering) for ordering in orderings]
    res: np.ndarray = np.full(shape=shape, fill_value=fill_value, dtype=dtype)
    keys_not_found = _fill_array_with_dict(
        d,
        orderings,
        res,
        ignore_not_found_keys=ignore_not_found_keys,
        transform_key_funcs=transform_key_funcs,
        transform_value_func=transform_value_func,
        tqdm_pos=0 if show_tqdm else None,
        tqdm_prompt="All" if show_tqdm else None
    )
    return (res, keys_not_found)
