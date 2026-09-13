"""The latent database being inferred — mirrors ``src/model/trace.jl``.

This is the thing PClean is actually reasoning about: not a cleaned table, but a
*database of entities* that would explain the dirty table if you wrote it down. The
clean table is a by-product, read off at the end by following each Record's reference
slots [§2.1].
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .model import ClassID, PCleanModel, VertexID


@dataclass
class LatentObject:
    """One entity: a hospital, a city, a physician.

    `values` maps vertex id -> value for this object's own class. Reference slots live
    in the same dict; their values are ids of objects in another table.
    """
    oid: int
    cls: ClassID
    values: Dict[VertexID, Any] = field(default_factory=dict)
    # How many reference slots currently point at me. This is the CRP's `n_r`, and the
    # reason it is stored rather than recomputed is that the proposal reads it once per
    # candidate per row -- see structure_prior.py.
    ref_count: int = 0


class Table:
    """All objects of one class, plus the exact-match index over its hash keys."""

    def __init__(self, cls: ClassID):
        self.cls = cls
        self.objects: Dict[int, LatentObject] = {}
        self._next_oid = 0
        # key tuple -> object id. This is [App. D.4] observation hashing: the single
        # optimisation without which large data is hopeless, because it replaces
        # "consider every object of this class" with a dict lookup.
        self.index: Dict[Tuple, int] = {}

    def new_object(self) -> LatentObject:
        oid = self._next_oid
        self._next_oid += 1
        obj = LatentObject(oid=oid, cls=self.cls)
        self.objects[oid] = obj
        return obj

    def key_of(self, obj: LatentObject, hash_keys: List[VertexID]) -> Optional[Tuple]:
        if not hash_keys:
            return None
        if any(k not in obj.values for k in hash_keys):
            return None
        return tuple(obj.values[k] for k in hash_keys)

    def register(self, obj: LatentObject, hash_keys: List[VertexID]) -> None:
        key = self.key_of(obj, hash_keys)
        if key is not None:
            self.index[key] = obj.oid

    def lookup(self, key: Tuple) -> Optional[LatentObject]:
        oid = self.index.get(key)
        return self.objects.get(oid) if oid is not None else None

    def __len__(self):
        return len(self.objects)


class Trace:
    """One complete hypothesis about the latent database.

    An SMC particle *is* one of these. Because particles get cloned on resampling, this
    class has to be cheap to copy -- see `fork()`.
    """

    def __init__(self, model: PCleanModel):
        self.model = model
        self.tables: Dict[ClassID, Table] = {c: Table(c) for c in model.classes}
        self.log_weight: float = 0.0

    def table(self, cls: ClassID) -> Table:
        return self.tables[cls]

    def fork(self) -> "Trace":
        """Copy this hypothesis so it can be extended independently.

        NOTE (divergence): the Julia keeps particles in a shared persistent structure so
        cloning is O(1). We deep-copy the dicts, which is O(size of database) and the
        main reason this package is slow. Correct, obvious, and unusable at scale --
        exactly the trade this package is making everywhere.
        """
        t = Trace(self.model)
        t.log_weight = self.log_weight
        for cls, tab in self.tables.items():
            nt = t.tables[cls]
            nt._next_oid = tab._next_oid
            nt.index = dict(tab.index)
            for oid, obj in tab.objects.items():
                nt.objects[oid] = LatentObject(oid=oid, cls=cls,
                                               values=dict(obj.values),
                                               ref_count=obj.ref_count)
        return t

    def summary(self) -> str:
        return ", ".join(f"{c}={len(t)}" for c, t in self.tables.items())
