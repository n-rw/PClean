"""Garbage collection, and why it is not housekeeping [§2.2, §3.1].

When rejuvenation moves a reference slot, the object it abandoned may have nothing else
pointing at it. The paper mentions this in half a sentence — these moves "may also lead to
the garbage collection of objects that are no longer connected to the observed dataset" —
which makes it sound like tidying up. It is not. Go back to the structure prior:

    p(S; |D|, C) places mass only on relational skeletons in which there are exactly |D|
    objects in C_obs and **every other object is connected via some chain of reference
    slots to one of them**.                                            [§2.2]

An unreachable object is not a slightly wasteful state. It is a state the prior assigns
**zero** probability, so a latent database containing one is not a sample from the
posterior at all, however healthy its accuracy looks.

Two parts below:

  1. Real benchmark data. Watch collection fire during rejuvenation, and watch the support
     invariant (every object reachable) hold afterwards.

  2. A phantom injected by hand. Inference here is good enough that orphans are *rare*,
     so rather than contrive data to produce one, we add them directly and measure what
     they cost. That isolates the effect from the luck of the sampler.

    python3 python/example_gc.py
"""

import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minipclean.builder import ModelBuilder
from minipclean.distributions import AddTypos, StringPrior, Unmodeled
from minipclean.smc import SMC, Rejuvenator, encode_rows
from minipclean.structure_prior import (collect_garbage, crp_candidate_log_weights,
                                        reachable_objects)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def build(candidates):
    b = ModelBuilder()
    city = b.add_class("City")
    city.attribute("name", StringPrior(3, 30), args=[candidates])
    prac = b.add_class("Practice")
    prac.attribute("pid", Unmodeled())
    prac.guaranteed("pid")
    prac.reference("city", "City")
    rec = b.add_class("Record")
    rec.reference("practice", "Practice")
    rec.attribute("obs_city", AddTypos(), args=[rec.via("practice", "city", "name"), 2])
    return b.finish("Record")


def totals(model, trace):
    return sum(len(trace.table(c)) for c in model.classes)


def p_new_entity(model, trace, cls_name="City"):
    """What the CRP tells the *next* reference slot: P(invent a brand-new entity).

    Three separate ways a phantom moves this, worth keeping straight:

      * it joins the candidate list, so future slots are offered a target that does not
        exist in any meaningful sense;
      * its stale `ref_count` enters the normaliser, which *suppresses* P(new) -- phantoms
        crowd out the option of inventing a genuine new entity;
      * with a non-zero discount it also feeds `s + d*n_objects` directly.

    The runs below use discount 0, so what you see is the second and third of these.
    """
    table = trace.table(cls_name)
    _, weights = crp_candidate_log_weights(table, model.classes[cls_name].py,
                                           list(table.objects))
    tot = sum(math.exp(w) for w in weights)
    return math.exp(weights[-1]) / tot, len(weights) - 1


def part1(n_rows=400, sweeps=4, seed=0):
    random.seed(seed)
    path = os.path.join(REPO, "datasets", "hospital_dirty.csv")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        dirty = list(csv.DictReader(f))[:n_rows]
    cands = sorted({r["City"].strip().lower() for r in dirty})
    rows = [{"practice.pid": r["ProviderNumber"].strip(),
             "obs_city": r["City"].strip().lower()} for r in dirty]

    model = build(cands)
    enc = encode_rows(model, "Record", rows)
    smc = SMC(model)
    trace = smc.run(enc)
    print(f"\n  1. REAL DATA ({n_rows} rows)")
    print(f"    after SMC: {trace.summary()}")

    rej = Rejuvenator(model, trace, enc, smc)
    for s in range(sweeps):
        changed = rej.sweep()
        print(f"      sweep {s+1}: {changed:3d} revised, {rej.collected} collected"
              f"  ->  {trace.summary()}")

    reach = len(reachable_objects(model, trace))
    tot = totals(model, trace)
    print(f"    support invariant: {reach} reachable / {tot} total"
          f"  -> {'HOLDS' if reach == tot else 'VIOLATED'}")
    print("    (Collection is rare here precisely because the enumerative proposals are")
    print("     good -- SMC rarely over-segments, so few objects are ever abandoned.)")
    return model, trace


def part2(model, trace, n_phantoms=25):
    print(f"\n  2. WHAT ONE PHANTOM COSTS")
    before_p, before_c = p_new_entity(model, trace)
    print(f"    before            : {len(trace.table('City'))} cities, "
          f"{before_c} candidate targets, P(next invents new) = {before_p:.4f}")

    # Inject unreachable objects: real rows in the City table that nothing points at.
    # This is exactly the state a slot move leaves behind.
    name_v = model.classes["City"].names["name"]
    table = trace.table("City")
    for i in range(n_phantoms):
        ghost = table.new_object()
        ghost.values[name_v] = f"ghosttown{i}"
        ghost.ref_count = 1          # stale count -- nothing actually references it
    after_p, after_c = p_new_entity(model, trace)
    reach, tot = len(reachable_objects(model, trace)), totals(model, trace)
    print(f"    +{n_phantoms} phantoms      : {len(table)} cities, "
          f"{after_c} candidate targets, P(next invents new) = {after_p:.4f}")
    print(f"                        support invariant: {reach}/{tot} "
          f"-> {'HOLDS' if reach == tot else 'VIOLATED'}")

    removed = collect_garbage(model, trace)
    end_p, end_c = p_new_entity(model, trace)
    reach, tot = len(reachable_objects(model, trace)), totals(model, trace)
    print(f"    after collection  : {len(table)} cities, "
          f"{end_c} candidate targets, P(next invents new) = {end_p:.4f}  "
          f"({removed} collected)")
    print(f"                        support invariant: {reach}/{tot} "
          f"-> {'HOLDS' if reach == tot else 'VIOLATED'}")
    return before_p, after_p


def main():
    model, trace = part1()
    before, after = part2(model, trace)
    shift = 100 * (after - before) / max(before, 1e-12)
    print("\n  ---")
    print(f"  The reconstructed dataset is byte-for-byte identical with those phantoms in")
    print(f"  place -- every accuracy number you could print looks the same. But the prior")
    print(f"  has moved: P(the next reference invents a fresh entity) shifted by {shift:+.1f}%,")
    print(f"  because the phantoms' stale reference counts crowd the normaliser -- and each")
    print(f"  of them is offered as a candidate target to every future slot.")
    print()
    print(f"  That is why this is not memory management. An unreachable object is outside")
    print(f"  the structure prior's support [§2.2], and leaving it there silently biases")
    print(f"  every entity-resolution decision that follows.")


if __name__ == "__main__":
    main()
