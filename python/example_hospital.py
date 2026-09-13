"""A runnable end-to-end example — mirrors ``experiments/hospital/run.jl``.

Deliberately smaller than the real hospital program (which models 15 columns across 7
classes). This one models two columns and three classes, because the point is to watch
the mechanism work, not to win a benchmark.

The mechanism in question is the paper's central claim, and you can see it here:

    A misspelling that appears MORE often than the correct spelling still gets
    repaired, because the model knows the error is systematic to one entity.

Run it:

    python3 python/example_hospital.py
    python3 python/example_hospital.py --rows 400 --verbose
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
from minipclean.proposal import ObsNode
from minipclean.smc import SMC, Rejuvenator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def build_model(city_candidates):
    """The model — the *Physicians* structure from the paper, which is the one that
    demonstrates Figure 1. Compare with appendix B.4.4.

        @class City begin
            name ~ StringPrior(3, 30, city_candidates)   # the CLEAN spelling, latent
        end
        @class Practice begin
            pid      ~ Unmodeled()                       # trusted key -> exact identity
            city     ~ City
            bad_city ~ AddTypos(city.name, 2)            # ONE spelling per practice
        end
        @class Record begin
            practice ~ Practice
        end

    Everything hinges on where `bad_city` lives. It is an attribute of **Practice**, so a
    practice has exactly one city spelling no matter how many rows it appears in. A
    misspelling repeated across 152 rows is therefore *one* typo draw, costing the model
    one typo penalty -- not 152 independent votes for the wrong spelling.

    That is the entire mechanism behind the paper's headline result, where "Abington, MD"
    (152 occurrences) loses to "Abingdon, MD" (42). Majority vote cannot do this. It is
    not a smarter search; it is a claim about how errors are *generated*, and the
    arithmetic follows from the claim.

    Run `--figure1` to watch exactly that happen on constructed data.

    A caveat you should see rather than have hidden from you: on the *hospital*
    benchmark this model is a deliberate mismatch. That dataset has independent
    per-cell typos, while this model asserts one spelling per practice. The result
    (400 rows) is recall 12/13 -- it finds nearly every typo -- against precision
    12/30, because rows of a practice that disagree get flattened onto the practice's
    single inferred spelling, overwriting cells that were already right.

    That is the lesson, not a bug to paper over: the error model's *position in the
    schema* is a modelling claim, and claiming the wrong one costs you precision even
    though the inference machinery is working perfectly. The real hospital program in
    run.jl puts AddTypos on the Record for exactly this reason.
    """
    b = ModelBuilder()

    city = b.add_class("City")
    city.attribute("name", StringPrior(3, 30), args=[city_candidates])

    prac = b.add_class("Practice")
    prac.attribute("pid", Unmodeled())
    prac.guaranteed("pid")
    prac.reference("city", "City")
    prac.attribute("bad_city", AddTypos(), args=[prac.via("city", "name"), 2])

    rec = b.add_class("Record")
    rec.reference("practice", "Practice")

    return b.finish(observation_class="Record")


def rows_from_hospital_csv(limit):
    """Real benchmark data. One Practice per ProviderNumber."""
    path = os.path.join(REPO, "datasets", "hospital_dirty.csv")
    clean_path = os.path.join(REPO, "datasets", "hospital_clean.csv")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        dirty = list(csv.DictReader(f))[:limit]
    with open(clean_path, newline="", encoding="utf-8", errors="replace") as f:
        clean = list(csv.DictReader(f))[:limit]
    rows = [ObsNode(children={"practice": ObsNode(
                attrs={"pid": r["ProviderNumber"].strip(),
                       "bad_city": r["City"].strip().lower()},
                children={"city": ObsNode()})})
            for r in dirty]
    truth = [c["City"].strip().lower() for c in clean]
    observed = [r["City"].strip().lower() for r in dirty]
    return rows, observed, truth


def rows_figure1():
    """Constructed data reproducing the paper's Figure 1 situation.

    One practice with 152 rows all spelling the city 'abington'; several other practices,
    42 rows in total, spelling it 'abingdon'. The wrong spelling outnumbers the right one
    almost four to one. A majority vote over cells picks 'abington' and is wrong.
    """
    rows, observed, truth = [], [], []

    def add(pid, spelling, n):
        for _ in range(n):
            rows.append(ObsNode(children={"practice": ObsNode(
                attrs={"pid": pid, "bad_city": spelling},
                children={"city": ObsNode()})}))
            observed.append(spelling)
            truth.append("abingdon")

    add("PRACTICE-BIG", "abington", 152)       # one practice, one systematic typo
    for i in range(7):
        add(f"PRACTICE-{i}", "abingdon", 6)    # 42 rows across 7 practices
    return rows, observed, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure1", action="store_true",
                    help="run the constructed Abington/Abingdon scenario")
    args = ap.parse_args()
    random.seed(args.seed)

    if args.figure1:
        rows, observed, truth = rows_figure1()
        print("Figure 1 scenario: 'abington' appears 152x (one practice), "
              "'abingdon' 42x (seven practices).")
        print("Majority vote over cells would pick 'abington' -- which is wrong.\n")
    else:
        rows, observed, truth = rows_from_hospital_csv(args.rows)
    city_candidates = sorted(set(observed))
    print(f"{len(rows)} rows | {len(city_candidates)} distinct city spellings observed")

    model = build_model(city_candidates)

    print("\n--- SMC [§3.1] ---")
    smc = SMC(model, n_particles=2, verbose=args.verbose)
    trace = smc.run(rows, progress_every=max(1, len(rows) // 4))
    print(f"  latent database: {trace.summary()}")

    # Which Record object did each row become? SMC recorded it; rejuvenation needs it
    # to walk evidence backwards from an entity to the rows that mention it.
    row_assignments = smc.row_assignments

    print(f"\n--- Rejuvenation [§3.1, App. D.1] ---")
    rej = Rejuvenator(model, trace, rows, row_assignments)
    for s in range(args.sweeps):
        n = rej.sweep(verbose=True)
        print(f"  sweep {s+1}: {n} attribute(s) revised")

    # ---- Did it clean anything? ----------------------------------------------
    print("\n--- Result ---")
    cn, pn, rn = model.classes["City"], model.classes["Practice"], model.classes["Record"]
    name_v, pcity_v, rprac_v = _v(cn, "name"), _v(pn, "city"), _v(rn, "practice")

    repairs, correct, wrong = Counter(), 0, 0
    for i, oid in enumerate(row_assignments):
        rec = trace.table("Record").objects.get(oid)
        if rec is None:
            continue
        pr = trace.table("Practice").objects.get(rec.values.get(rprac_v))
        if pr is None:
            continue
        c = trace.table("City").objects.get(pr.values.get(pcity_v))
        if c is None:
            continue
        inferred = c.values.get(name_v)
        if inferred and inferred != observed[i]:
            repairs[(observed[i], inferred, truth[i])] += 1
            if inferred == truth[i]:
                correct += 1
            else:
                wrong += 1

    if repairs:
        print(f"{sum(repairs.values())} cell(s) changed:")
        for (obs, inf, tr), n in repairs.most_common(12):
            mark = "OK  " if inf == tr else "BAD "
            print(f"  {mark} {obs!r} -> {inf!r}   (truth: {tr!r})  x{n}")
        print(f"\n  precision: {correct}/{correct+wrong} repairs correct")
    else:
        print("no repairs made")

    errors = sum(1 for i in range(len(rows)) if observed[i] != truth[i])
    if errors:
        print(f"  recall:    {correct}/{errors} of the city errors present were fixed")


def _v(cls, name):
    return [v for v in range(len(cls.nodes)) if cls.node_name(v) == name][0]


if __name__ == "__main__":
    main()
