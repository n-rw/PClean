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

from .model import PitmanYorParams
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
