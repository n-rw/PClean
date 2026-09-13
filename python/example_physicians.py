"""Learned parameters, on real data — the *other* half of the paper's Figure 1.

The typo repair (`example_hospital.py --figure1`) is the famous half. This is the other
one, and it shows a different capability: **the model learns, from the dirty data, a
fact nobody told it, and then uses that fact to overturn a prior.**

The setup, in the paper's words: a clinician's degree is missing. Most Family Practice
physicians are MDs, so the obvious guess is MD. But this clinician trained at the
Philadelphia College of Osteopathic Medicine, and PCOM awards overwhelmingly DOs. The
right answer is DO, and no one supplied that fact -- it was counted from the same messy
file we are trying to clean.

Measured on the live CMS file (see ../PHYSICIANS-DATA.md):

    FAMILY PRACTICE overall :  74.4% MD,  21.8% DO      <- the prior
    PCOM graduates          :   5.4% MD,  84.2% DO      <- what the model learns

We hold out known credentials, impute them, and check.

    python3 python/example_physicians.py
    python3 python/example_physicians.py --rows 8000 --holdout 0.4
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minipclean.builder import ModelBuilder
from minipclean.distributions import (ChooseProportionally, MaybeSwap, ProbParameter,
                                      ProportionsParameter, Unmodeled)
from minipclean.parameters import fit_parameters, show_distribution
from minipclean.smc import SMC, Rejuvenator, encode_rows

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "experiments", "physicians_data",
                    "DAC_NationalDownloadableFile.csv")


def build_model(degrees, concentration=0.2):
    """The model — appendix B.4.4's Physician class, trimmed to what this demo needs.

        @class School begin
            name ~ Unmodeled(); @guaranteed name
        end
        @class Physician begin
            @learned error_prob  :: ProbParameter{1.0, 1000.0}
            @learned degree_dist :: Dict{String, ProportionsParameter{3.0}}
            npi    ~ Unmodeled(); @guaranteed npi
            school ~ School
            degree     ~ ChooseProportionally(degrees, degree_dist[school.name])
            obs_degree ~ MaybeSwap(degree, degrees, error_prob)
        end
        @class Record begin
            physician ~ Physician
        end

    Two things to notice.

    `degree_dist` is **indexed by school**, so one line of program declares one learned
    distribution per medical school -- 396 of them in the paper, several hundred here.
    The user names none of them.

    `degree` is latent and `obs_degree` is what the file actually contains. Separating
    them is what lets the same machinery do two jobs at once: where `obs_degree` is
    present, MaybeSwap can decide it is an error and overrule it; where it is blank,
    there is simply no likelihood term and `degree` is drawn from what the school implies.
    Repair and imputation are not two features, they are one model read two ways.

    On `concentration`: the paper writes `dirichlet(3 * ones(num_degrees))`. With 15
    credentials that is 45 pseudo-counts of prior before any data is seen -- invisible
    against 2.2M rows, but overwhelming against the few thousand this demo reads, which
    flattens every school toward the global average. We default to 0.2 so the counts can
    actually speak at demo scale. It is a real dial, not a fudge: it says how strongly you
    believe a school's degree mix resembles everyone else's.
    """
    b = ModelBuilder()

    school = b.add_class("School")
    school.attribute("name", Unmodeled())
    school.guaranteed("name")

    phys = b.add_class("Physician")
    phys.learned("error_prob", ProbParameter(1.0, 1000.0))
    phys.learned_indexed("degree_dist", lambda: ProportionsParameter(concentration))
    phys.attribute("npi", Unmodeled())
    phys.guaranteed("npi")
    phys.reference("school", "School")
    phys.attribute("degree", ChooseProportionally(),
                   args=[degrees, phys.param("degree_dist", phys.via("school", "name"))])
    phys.attribute("obs_degree", MaybeSwap(),
                   args=[phys.ref("degree"), degrees, phys.param("error_prob")])

    rec = b.add_class("Record")
    rec.reference("physician", "Physician")

    return b.finish(observation_class="Record")


def load(limit, holdout, seed):
    """Real CMS rows with a known school and a known credential; hide some credentials."""
    if not os.path.exists(DATA):
        sys.exit(f"Missing {DATA}\n(see ../PHYSICIANS-DATA.md for how to fetch it)")
    rng = random.Random(seed)
    kept = []
    with open(DATA, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            school = (r.get("Med_sch") or "").strip()
            cred = (r.get("Cred") or "").strip()
            # "OTHER" is 62% of the column and is not a school; excluding it is a
            # modelling decision, documented in PHYSICIANS-DATA.md.
            if not school or school == "OTHER" or not cred:
                continue
            kept.append({"npi": (r.get("NPI") or "").strip(), "school": school,
                         "cred": cred, "spec": (r.get("pri_spec") or "").strip()})
            if len(kept) >= limit:
                break

    rows, truth, hidden = [], [], []
    for k in kept:
        hide = rng.random() < holdout
        # Flat, dotted observations -- exactly what `@query` expresses. Each key names a
        # vertex that flattening put into `Record`'s own graph.
        rows.append({"physician.npi": k["npi"],
                     "physician.school.name": k["school"],
                     "physician.obs_degree": None if hide else k["cred"]})
        truth.append(k["cred"])
        hidden.append(hide)
    return rows, truth, hidden, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concentration", type=float, default=0.2)
    args = ap.parse_args()
    random.seed(args.seed)

    rows, truth, hidden, raw = load(args.rows, args.holdout, args.seed)
    degrees = sorted({k["cred"] for k in raw})
    n_hidden = sum(hidden)
    print(f"{len(rows)} clinicians | {len(degrees)} credentials | "
          f"{len({k['school'] for k in raw})} schools | {n_hidden} credentials hidden")

    model = build_model(degrees, concentration=args.concentration)
    enc = encode_rows(model, "Record", rows)

    print("\n--- SMC [§3.1] ---")
    smc = SMC(model)
    trace = smc.run(enc, progress_every=max(1, len(rows) // 3))
    print(f"  latent database: {trace.summary()}")

    rej = Rejuvenator(model, trace, enc, smc)
    pn = model.classes["Physician"]
    deg_v = pn.names["degree"]
    rphys_v = model.classes["Record"].names["physician"]

    def predictions():
        """The current posterior sample's guess for every hidden credential."""
        out = {}
        for i, oid in enumerate(smc.row_assignments):
            if not hidden[i]:
                continue
            rec = trace.table("Record").objects.get(oid)
            ph = trace.table("Physician").objects.get(rec.values.get(rphys_v)) if rec else None
            if ph is not None:
                out[i] = ph.values.get(deg_v)
        return out

    def score(pred):
        ok = sum(1 for i, v in pred.items() if v == truth[i])
        return ok, len(pred)

    # Each sweep leaves one posterior *sample*, and a single sample is a noisy estimator.
    # [§4, Experiment 4] shows the fix: take the most common prediction across several
    # samples. That approximates the MAP clean dataset and, on the paper's Rents
    # benchmark, lifted F1 from 0.69 to 0.73. We accumulate votes here and report both.
    votes = {}

    # Inference and learning are the same loop: the degree distribution is estimated
    # from records whose degrees were themselves partly inferred. Repeat and it sharpens.
    print("\n--- Learning + rejuvenation [App. D.2] ---")
    for r in range(args.rounds):
        fit_parameters(model, trace)
        changed = rej.sweep()
        pred = predictions()
        for i, v in pred.items():
            votes.setdefault(i, Counter())[v] += 1
        ok, tot = score(pred)
        print(f"  round {r+1}: {changed:5d} degrees revised | "
              f"this sample {ok}/{tot} = {100*ok/max(tot,1):.1f}%")

    modal = {i: c.most_common(1)[0][0] for i, c in votes.items()}
    mok, mtot = score(modal)
    print(f"\n  single posterior sample : {100*ok/max(tot,1):.1f}%")
    print(f"  modal over {args.rounds} samples : {100*mok/max(mtot,1):.1f}%  "
          f"({mok}/{mtot})   <- the estimator the paper recommends [§4, Expt 4]")

    # Confidence = how many samples agreed. The paper finds this is well calibrated, and
    # that repairing only above a threshold trades recall for precision [Fig. 7].
    conf_hi = [i for i, c in votes.items()
               if c.most_common(1)[0][1] == args.rounds]
    if conf_hi:
        hok = sum(1 for i in conf_hi if modal[i] == truth[i])
        print(f"  where all {args.rounds} samples agreed ({len(conf_hi)} cells): "
              f"{100*hok/len(conf_hi):.1f}% correct   <- calibration")

    # ---- What did it learn, and did the school beat the prior? -----------------
    print("\n--- What was learned ---")
    dd = pn.nodes[pn.names["degree_dist"]].param
    glob = Counter(k["cred"] for k in raw)
    gtot = sum(glob.values())
    print("  global (the prior):        " +
          ", ".join(f"{d} {100*c/gtot:.1f}%" for d, c in glob.most_common(4)))

    for school in sorted(dd.table, key=lambda s: -sum(dd.table[s].counts))[:1]:
        print(f"  largest school learned:    {show_distribution(dd[school], degrees)}")
        print(f"                             ({school[:52]})")

    pcom = [s for s in dd.table if "OSTEOPATHIC" in s and "PHILADELPHIA" in s]
    if pcom:
        print(f"\n  PCOM learned:              {show_distribution(dd[pcom[0]], degrees)}")
        idx = [i for i, k in enumerate(raw) if k["school"] == pcom[0] and hidden[i]]
        if idx:
            ok = sum(1 for i in idx
                     if trace.table("Physician").objects[
                         trace.table("Record").objects[smc.row_assignments[i]]
                         .values[rphys_v]].values.get(deg_v) == truth[i])
            print(f"  held-out PCOM clinicians:  {ok}/{len(idx)} imputed correctly")
            print("  (the global prior would have said MD for every one of them)")


if __name__ == "__main__":
    main()
