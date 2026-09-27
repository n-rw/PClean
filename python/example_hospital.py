"""A runnable end-to-end example — mirrors ``experiments/hospital/run.jl``.

Two models, one switch, because the contrast *is* the lesson.

Default: the hospital benchmark's real shape, with an independent typo per row. The
clean city name lives on a `City`, two reference hops away from the observation that
bears on it. Before flattening this package could not do this model at all -- the city
was chosen before the typo was scored, and every repair was wrong.

`--figure1`: the Physicians shape from the paper. The typo lives on the **Practice**, so
one practice has one spelling however many rows it appears in, and a misspelling repeated
across 152 rows costs the model *one* typo rather than 152 votes. That is what lets
'abington' (152 occurrences) lose to 'abingdon' (42).

Same machinery, same data, different placement of the error model -- and completely
different behaviour. Where the error model sits in the schema is a modelling claim with
teeth, not a detail.

    python3 python/example_hospital.py
    python3 python/example_hospital.py --figure1
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minipclean.builder import ModelBuilder
from minipclean.distributions import AddTypos, StringPrior, Unmodeled
from minipclean.smc import SMC, Rejuvenator, encode_rows

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def build_per_row(candidates):
    """The benchmark's shape: an independent typo on every row.

        @class City     begin name ~ StringPrior(3, 30, candidates) end
        @class Practice begin pid ~ Unmodeled(); @guaranteed pid; city ~ City end
        @class Record   begin
            practice ~ Practice
            obs_city ~ AddTypos(practice.city.name, 2)     # two hops
        end

    `practice.city.name` is two reference slots deep. After flattening it is simply a
    vertex of `Record`, so choosing the practice, choosing its city, and scoring the
    dirty spelling are one enumeration in one topological order.
    """
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


def build_systematic(candidates):
    """The paper's Figure 1 shape: one spelling per practice, systematically wrong."""
    b = ModelBuilder()
    city = b.add_class("City")
    city.attribute("name", StringPrior(3, 30), args=[candidates])

    prac = b.add_class("Practice")
    prac.attribute("pid", Unmodeled())
    prac.guaranteed("pid")
    prac.reference("city", "City")
    prac.attribute("bad_city", AddTypos(), args=[prac.via("city", "name"), 2])

    rec = b.add_class("Record")
    rec.reference("practice", "Practice")
    return b.finish("Record")


def load_benchmark(limit):
    """Load clean and dirty hospital data.  For each provider, align the clean and dirty city names."""
    d = os.path.join(REPO, "datasets")
    with open(os.path.join(d, "hospital_dirty.csv"), newline="",
              encoding="utf-8", errors="replace") as f:
        dirty = list(csv.DictReader(f))[:limit]
    with open(os.path.join(d, "hospital_clean.csv"), newline="",
              encoding="utf-8", errors="replace") as f:
        clean = list(csv.DictReader(f))[:limit]
    rows = [{"practice.pid": r["ProviderNumber"].strip(),
             "obs_city": r["City"].strip().lower()} for r in dirty]
    return (rows,
            [r["City"].strip().lower() for r in dirty],
            [c["City"].strip().lower() for c in clean])


def load_figure1():
    """152 rows of one practice spelling it wrong; 42 across seven spelling it right."""
    rows, observed, truth = [], [], []

    def add(pid, spelling, n):
        for _ in range(n):
            rows.append({"practice.pid": pid, "practice.bad_city": spelling})
            observed.append(spelling)
            truth.append("abingdon")

    add("PRACTICE-BIG", "abington", 152)
    for i in range(7):
        add(f"PRACTICE-{i}", "abingdon", 6)
    return rows, observed, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--figure1", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.figure1:
        rows, observed, truth = load_figure1()
        print("Figure 1: 'abington' appears 152x (one practice), 'abingdon' 42x (seven).")
        print("Majority vote over cells picks 'abington' -- and is wrong.\n")
        model = build_systematic(sorted(set(observed)))
    else:
        rows, observed, truth = load_benchmark(args.rows)
        print("Per-row typos; the clean name is two reference hops from the observation.\n")
        model = build_per_row(sorted(set(observed)))

    rec = model.classes["Record"]
    print(f"{len(rows)} rows | {len(set(observed))} distinct spellings observed")
    print("Record vertices after flattening: " +
          ", ".join(rec.node_name(v) for v in range(len(rec.nodes))))

    enc = encode_rows(model, "Record", rows)
    print("\n--- SMC [§3.1] ---")
    smc = SMC(model)
    trace = smc.run(enc, progress_every=max(1, len(rows) // 3))
    print(f"  latent database: {trace.summary()}")

    print("\n--- Rejuvenation [App. D.1] ---")
    rej = Rejuvenator(model, trace, enc, smc)
    for s in range(args.sweeps):
        n = rej.sweep(verbose=args.figure1)
        print(f"  sweep {s+1}: {n} attribute(s) revised")

    name_v = model.classes["City"].names["name"]
    city_v = model.classes["Practice"].names["city"]
    prac_v = rec.names["practice"]

    repairs, correct, wrong = Counter(), 0, 0
    for i, oid in enumerate(smc.row_assignments):
        r = trace.table("Record").objects.get(oid)
        p = trace.table("Practice").objects.get(r.values.get(prac_v)) if r else None
        c = trace.table("City").objects.get(p.values.get(city_v)) if p else None
        inferred = c.values.get(name_v) if c else None
        if inferred and inferred != observed[i]:
            repairs[(observed[i], inferred, truth[i])] += 1
            correct += inferred == truth[i]
            wrong += inferred != truth[i]

    print("\n--- Result ---")
    if repairs:
        for (o, inf, t), n in repairs.most_common(10):
            print(f"  {'OK ' if inf == t else 'BAD'}  {o!r} -> {inf!r}   "
                  f"(truth {t!r})  x{n}")
        print(f"\n  precision: {correct}/{correct + wrong}")
    else:
        print("  no repairs made")
    errs = sum(1 for i in range(len(rows)) if observed[i] != truth[i])
    if errs:
        print(f"  recall:    {correct}/{errs}")


if __name__ == "__main__":
    main()
