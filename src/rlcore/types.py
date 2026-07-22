"""Core typed data structures shared across the platform.

This module defines :class:`Batch`, the single container in which transition
data flows from collection to update (see ``docs/ARCHITECTURE.md`` §5), and
:class:`PolicyOutput`, the result of a policy acting on observations.

Canonical field names are exported as constants so that collectors, buffers,
transforms, and algorithms agree on spelling. Algorithms may add keys beyond
the canonical set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NoReturn, overload

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

__all__ = [
    "ACTION",
    "CANONICAL_KEYS",
    "LOGPROB",
    "NEXT_OBS",
    "OBS",
    "REWARD",
    "TERMINATED",
    "TRUNCATED",
    "VALUE",
    "Batch",
    "PolicyOutput",
]

OBS: Final = "obs"
ACTION: Final = "action"
REWARD: Final = "reward"
#: MDP-true episode end (absorbing state): bootstrap with value 0.
TERMINATED: Final = "terminated"
#: Time-limit cutoff, not an MDP event: bootstrap from the value function.
TRUNCATED: Final = "truncated"
NEXT_OBS: Final = "next_obs"
LOGPROB: Final = "logprob"
VALUE: Final = "value"

CANONICAL_KEYS: Final[frozenset[str]] = frozenset(
    {OBS, ACTION, REWARD, TERMINATED, TRUNCATED, NEXT_OBS, LOGPROB, VALUE}
)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """The result of a policy acting on a (possibly batched) observation.

    Attributes:
        action: The selected action(s).
        extras: Algorithm-specific per-step tensors that the collector stores
            alongside the transition (e.g. ``logprob``, ``value``).
    """

    action: Tensor
    extras: Mapping[str, Tensor] = field(default_factory=dict)


class Batch:
    """An immutable container of tensors sharing a leading (batch) dimension.

    ``Batch`` is the one structure transition data travels in, from collection
    through buffers and transforms to agent updates. All fields are
    :class:`torch.Tensor` (conversion policy belongs to the collector, not
    here), share the same size along dimension 0, and live on the same device.
    Field dtypes are deliberately heterogeneous (float observations, integer
    actions, boolean termination flags).

    Access patterns::

        batch.obs                 # attribute access to a field
        batch["obs"]              # equivalent key access
        batch[2:5]                # slicing the leading dim -> Batch
        batch[idx_tensor]         # integer or boolean indexing -> Batch
        len(batch)                # leading-dimension size
        batch.to("cuda")          # new Batch on another device
        batch.with_fields(adv=a)  # derived Batch with fields added/replaced
        for mb in batch.minibatches(64, generator=g): ...

    Immutability is structural: fields cannot be added, removed, or rebound in
    place (derive a new ``Batch`` instead), though tensor *contents* are as
    mutable as any tensor. Direct iteration is refused to avoid ambiguity
    between iterating keys and iterating transitions.
    """

    __slots__ = ("_fields", "_length")

    _fields: dict[str, Tensor]
    _length: int

    def __init__(self, fields: Mapping[str, Tensor] | None = None, /, **kwargs: Tensor) -> None:
        """Build a batch from a mapping, keyword arguments, or both.

        Args:
            fields: Optional mapping of field name to tensor. Useful for names
                that are not valid keyword arguments and for programmatic
                construction.
            **kwargs: Fields given as keyword arguments.

        Raises:
            ValueError: If no fields are given, a name is duplicated between
                ``fields`` and ``kwargs``, a name is not a valid identifier or
                collides with a ``Batch`` attribute, a tensor has no leading
                dimension, or leading sizes/devices disagree.
            TypeError: If a field value is not a :class:`torch.Tensor`.
        """
        merged: dict[str, Tensor] = dict(fields) if fields is not None else {}
        for key in kwargs:
            if key in merged:
                raise ValueError(f"Field {key!r} given both in the mapping and as a keyword.")
        merged.update(kwargs)
        if not merged:
            raise ValueError("Batch requires at least one field.")

        for key, value in merged.items():
            _validate_field_name(key)
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Field {key!r} must be a torch.Tensor, got {type(value).__name__}."
                )
            if value.ndim == 0:
                raise ValueError(f"Field {key!r} must have a leading batch dimension.")

        lengths = {key: value.shape[0] for key, value in merged.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Fields disagree on leading-dimension size: {lengths}.")
        devices = {str(value.device) for value in merged.values()}
        if len(devices) > 1:
            raise ValueError(f"Fields must live on one device, found: {sorted(devices)}.")

        object.__setattr__(self, "_fields", merged)
        object.__setattr__(self, "_length", next(iter(lengths.values())))

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        """Return the leading-dimension (batch) size."""
        return self._length

    @property
    def batch_size(self) -> int:
        """The leading-dimension size, alias of ``len(batch)``."""
        return self._length

    @property
    def device(self) -> torch.device:
        """The device shared by every field."""
        return next(iter(self._fields.values())).device

    def keys(self) -> KeysView[str]:
        """Return a view of the field names."""
        return self._fields.keys()

    def values(self) -> ValuesView[Tensor]:
        """Return a view of the field tensors."""
        return self._fields.values()

    def items(self) -> ItemsView[str, Tensor]:
        """Return a view of ``(name, tensor)`` pairs."""
        return self._fields.items()

    def as_dict(self) -> dict[str, Tensor]:
        """Return the fields as a plain dict (shallow copy; tensors shared)."""
        return dict(self._fields)

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` names a field."""
        return key in self._fields

    def __repr__(self) -> str:
        """Summarize length, device, and per-field shape/dtype."""
        inner = ", ".join(
            f"{key}: {tuple(value.shape)} {value.dtype}" for key, value in self._fields.items()
        )
        return f"Batch(len={self._length}, device={self.device}, {inner})"

    # -- field access -------------------------------------------------------

    def __getattr__(self, name: str) -> Tensor:
        """Resolve unknown attributes as field lookups."""
        if name.startswith("_"):  # avoid recursion during unpickling/copying
            raise AttributeError(name)
        fields: dict[str, Tensor] = object.__getattribute__(self, "_fields")
        try:
            return fields[name]
        except KeyError:
            raise AttributeError(
                f"Batch has no field {name!r}; available fields: {sorted(fields)}."
            ) from None

    @overload
    def __getitem__(self, index: str) -> Tensor: ...
    @overload
    def __getitem__(self, index: int | slice | list[int] | Tensor) -> Batch: ...
    def __getitem__(self, index: str | int | slice | list[int] | Tensor) -> Tensor | Batch:
        """Select a field by name, or transitions by leading-dim index.

        Integer indexing returns a length-1 ``Batch`` (the leading dimension
        is retained, so downstream code never meets 0-dim tensors). Slices,
        index lists/tensors, and boolean masks select along dimension 0.
        """
        if isinstance(index, str):
            try:
                return self._fields[index]
            except KeyError:
                raise KeyError(
                    f"Batch has no field {index!r}; available fields: {sorted(self._fields)}."
                ) from None
        if isinstance(index, int):
            if not -self._length <= index < self._length:
                raise IndexError(f"Index {index} out of range for Batch of length {self._length}.")
            index = slice(index, index + 1) if index != -1 else slice(-1, None)
        return Batch({key: value[index] for key, value in self._fields.items()})

    def __iter__(self) -> NoReturn:
        """Refuse direct iteration (ambiguous); see class docstring."""
        raise TypeError(
            "Batch is not directly iterable; iterate batch.keys()/items(), index the "
            "leading dimension, or use batch.minibatches(...)."
        )

    # -- derivation ---------------------------------------------------------

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Batch:
        """Return a batch with every field moved to ``device``.

        Tensors already on ``device`` are returned as-is by torch, so a
        same-device ``to`` is cheap and shares storage.
        """
        return Batch(
            {key: value.to(device, non_blocking=non_blocking) for key, value in self.items()}
        )

    def with_fields(self, **fields: Tensor) -> Batch:
        """Return a new batch with ``fields`` added or replaced.

        New fields are validated against the existing leading size and device;
        the original batch is unchanged.
        """
        return Batch({**self._fields, **fields})

    def minibatches(
        self,
        size: int,
        *,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
    ) -> Iterator[Batch]:
        """Yield minibatches over the leading dimension.

        Args:
            size: Transitions per minibatch (the final one may be smaller).
            shuffle: Visit transitions in a random order. Every transition is
                yielded exactly once either way.
            generator: Seeded generator for the shuffle, for reproducibility.
            drop_last: Skip a final minibatch smaller than ``size``.

        Yields:
            ``Batch`` views of at most ``size`` transitions.

        Raises:
            ValueError: If ``size`` is not a positive integer.
        """
        if size < 1:
            raise ValueError(f"Minibatch size must be >= 1, got {size}.")
        if shuffle:
            order = torch.randperm(self._length, generator=generator)
        else:
            order = torch.arange(self._length)
        for start in range(0, self._length, size):
            indices = order[start : start + size]
            if drop_last and indices.shape[0] < size:
                return
            yield self[indices]

    # -- immutability and serialization -------------------------------------

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Refuse attribute assignment; derive via :meth:`with_fields`."""
        raise AttributeError(
            f"Batch is immutable; cannot set {name!r}. Use with_fields() to derive a new Batch."
        )

    def __delattr__(self, name: str) -> NoReturn:
        """Refuse attribute deletion; Batch is immutable."""
        raise AttributeError(f"Batch is immutable; cannot delete {name!r}.")

    def __getstate__(self) -> dict[str, Tensor]:
        """Pickle as the plain field mapping."""
        return dict(self._fields)

    def __setstate__(self, state: dict[str, Tensor]) -> None:
        """Restore from the field mapping, revalidating invariants."""
        restored = Batch(state)
        object.__setattr__(self, "_fields", restored._fields)
        object.__setattr__(self, "_length", restored._length)


def _validate_field_name(key: str) -> None:
    """Reject field names that would break attribute access."""
    if not isinstance(key, str) or not key.isidentifier():
        raise ValueError(f"Field name {key!r} must be a valid Python identifier.")
    if hasattr(Batch, key):
        raise ValueError(f"Field name {key!r} collides with a Batch attribute.")
