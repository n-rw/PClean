"""The non-parametric structure prior — the paper's contribution (1) [§2.2].

This file has no direct counterpart in the Julia, where the same logic is spread across
the proposal compiler and the trace. It is pulled out here because it is the single
most important *modelling* idea in the paper, and it is short.

The question it answers: **how many entities are there, and which mentions refer to the
same one?** The user's program never says. It declares a schema ("a Record has a
Physician; a Physician attended a School") and PClean supplies the prior over how many
Physicians and Schools exist and which rows share them.

That prior is a two-parameter Chinese restaurant process over the *reference set*: the
set of all places in the database that point at class C. The CRP partitions those
references into groups, and each group becomes one object [§2.2, Fig. 4].

The sequential form is the one actually used, because it is what makes SMC possible
[§3.1, Fig. 5]. When a new reference needs a target in class C:

    P(point at existing object r)  proportional to   n_r - d
    P(create a brand new object)   proportional to   s + d * (number of objects so far)

where `n_r` is how many references already point at r, `d` is the discount and `s` the
strength. Note `n_r - d`: **rich get richer.** A city already mentioned by 400 practices
is a far more likely target than one mentioned by two. That is not a heuristic bolted on
for entity resolution -- it falls out of the prior, and it is exactly the behaviour you
want when deciding whether two similar records denote the same real thing.

The paper notes this as an advantage over BLOG, which can only choose *uniformly* among
objects satisfying a predicate and so cannot express that some targets are more popular
than others [App. C.3, footnote].
"""

import math
from typing import List, Tuple

from .model import ForeignKeyNode, PitmanYorParams, strip_submodel as _strip
from .trace import Table


def crp_candidate_log_weights(table: Table, py: PitmanYorParams,
                              candidate_oids: List[int]) -> Tuple[List[int], List[float]]:
    """Log-weights for pointing at each candidate object, plus one for "a new object".

    Returns ``(oids, log_weights)`` where a trailing ``None`` in `oids` means "create a
    new object". Weights are unnormalised; the caller combines them with the likelihood
    of the data and normalises once, which is what makes the resulting proposal the
    *locally optimal* one of [§3.2] rather than a guess.
    """
    s, d = py.strength, py.discount
    n_objects = len(table)

    oids: List[int] = []
    weights: List[float] = []

    for oid in candidate_oids:
        n_r = table.objects[oid].ref_count
        w = n_r - d
        # A candidate with no references yet contributes nothing; skip rather than log(0).
        if w > 0:
            oids.append(oid)
            weights.append(math.log(w))

    # ...and the option of inventing a new entity. This is what makes the model
    # "open-universe": the number of hospitals is not fixed in advance, it is inferred.
    new_weight = s + d * n_objects
    oids.append(None)
    weights.append(math.log(new_weight) if new_weight > 0 else -math.inf)

    return oids, weights


def reachable_objects(model, trace) -> set:
    """Every ``(class, oid)`` reachable from the observation class by reference slots."""
    obs_cls = model.observation_class
    seen, frontier = set(), [(obs_cls, oid) for oid in trace.table(obs_cls).objects]
    while frontier:
        key = frontier.pop()
        if key in seen:
            continue
        seen.add(key)
        cname, oid = key
        cls = model.classes[cname]
        obj = trace.table(cname).objects.get(oid)
        if obj is None:
            continue
        for v, node in enumerate(cls.nodes):
            if "." in cls.node_name(v):
                continue                 # a flattened copy; the owner holds the real slot
            inner = _strip(node)
            if isinstance(inner, ForeignKeyNode) and v in obj.values:
                frontier.append((inner.target_class, obj.values[v]))
    return seen


def recount_references(model, trace) -> None:
    """Recompute every object's ``ref_count`` from the surviving reference slots.

    `ref_count` is the CRP's ``n_r``. The Julia maintains it incrementally, and so do we
    during a sweep, but after garbage collection it is worth restoring from scratch: it
    makes the invariant explicit, and a count that has drifted is otherwise invisible --
    inference keeps running, just against a prior that believes in objects nobody
    references.
    """
    for cname in model.classes:
        for obj in trace.table(cname).objects.values():
            obj.ref_count = 0
    for cname, cls in model.classes.items():
        for obj in trace.table(cname).objects.values():
            for v, node in enumerate(cls.nodes):
                if "." in cls.node_name(v):
                    continue
                inner = _strip(node)
                if isinstance(inner, ForeignKeyNode) and v in obj.values:
                    target = trace.table(inner.target_class).objects.get(obj.values[v])
                    if target is not None:
                        target.ref_count += 1


def collect_garbage(model, trace) -> int:
    """Delete objects no longer reachable from the observed data. Returns how many.

    ## Why this is not housekeeping

    It is tempting to read "garbage collection" as memory management. It is not. Go back
    to the structure prior [§2.2]:

        p(S; |D|, C) places mass only on relational skeletons in which there are exactly
        |D| objects in C_obs and **every other object is connected via some chain of
        reference slots to one of them**.

    An object nothing can reach is therefore not a slightly wasteful state -- it is a
    state the prior assigns **zero** probability. A latent database containing one is not
    a sample from the posterior at all, however good its other numbers look.

    ## When objects become unreachable

    Only when a reference slot moves, which is why this arrives together with slot
    revision in rejuvenation [§3.1]: "these moves may also lead to the garbage collection
    of objects that are no longer connected to the observed dataset". Suppose two
    practices are the only ones citing a particular City, and rejuvenation decides both
    actually belong to a differently-spelled City. The abandoned one still sits in the
    table, still counts toward `len(table)` in the CRP's `s + d*n_objects` term, and is
    still offered as a candidate target to every future row -- a phantom entity competing
    for references.

    Mark and sweep, from the observation class outward. The index is pruned too: a stale
    entry would let a hash-key lookup hand back an object that no longer exists.
    """
    keep = reachable_objects(model, trace)
    removed = 0
    for cname in model.classes:
        if cname == model.observation_class:
            continue           # one object per row, always, by construction [§2.2]
        table = trace.table(cname)
        for oid in list(table.objects):
            if (cname, oid) not in keep:
                del table.objects[oid]
                removed += 1
        table.index = {k: v for k, v in table.index.items() if v in table.objects}
    if removed:
        recount_references(model, trace)
    return removed
