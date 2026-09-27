"""Record a PClean run as a step-by-step event log for the visualizer in ../viz.

    python3 python/viz_record.py                 # all scenarios -> viz/public/data/
    python3 python/viz_record.py --scenario towns

Nothing here changes the algorithm. It drives `ParticleSMC` and `Rejuvenator` one step at
a time with their decision logs switched on, and after every step diffs each particle's
latent database against the previous snapshot. The web app replays those diffs, so the
file is the whole story: the model's schema, the rows, and for every step what was
decided, why (the candidates and their scores), and what changed in which particle's net.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import example_hospital as hospital
from minipclean.distributions import logsumexp
from minipclean.model import (ForeignKeyNode, NewObject, RandomChoiceNode, Ref,
                              strip_submodel)
from minipclean.particles import ParticleSMC
from minipclean.smc import encode_rows

OUT = os.path.join(REPO, "viz", "public", "data")


# ============================================================================ scenarios

def scenario_towns():
    """The default: Figure 1 in miniature, plus one genuinely ambiguous slip.

    * abingdon -- practice P01 misspells it 'abington' on 6 rows; three practices spell it
      right. P01 comes first, so SMC names the city 'abington'. Rejuvenation after row 8
      sees one practice each way, a coin flip -- some particles fix the name, some do not.
      The next 'abingdon' practice then costs the unfixed particles a typo, their weight
      drops, and resampling copies the fixed databases over them.
    * dothan / bethel -- two real towns. 'bethan' is exactly two edits from each, so which
      town P10 belongs to is decided by the CRP's rich-get-richer counts, and particles
      disagree. That disagreement is the posterior uncertainty, and it should survive.
    """
    order = [("P01", "abington"), ("P01", "abington"), ("P01", "abington"),
             ("P05", "dothan"), ("P02", "abingdon"), ("P08", "bethel"),
             ("P01", "abington"), ("P06", "dothan"),
             # -- rejuvenation sweep --
             ("P03", "abingdon"), ("P10", "bethan"), ("P09", "bethel"),
             ("P02", "abingdon"), ("P07", "dothan"), ("P04", "abingdon"),
             ("P10", "bethan"), ("P01", "abington"),
             # -- rejuvenation sweep --
             ("P03", "abingdon"), ("P05", "dothan"), ("P06", "dothan"),
             ("P08", "bethel"), ("P09", "bethel"), ("P10", "bethan"),
             ("P04", "abingdon"), ("P07", "dothan"), ("P01", "abington")]
    truth = {"abington": "abingdon", "abingdon": "abingdon", "dothan": "dothan",
             "bethel": "bethel", "bethan": None}      # None: genuinely ambiguous
    rows = [{"practice.pid": pid, "practice.bad_city": sp} for pid, sp in order]
    spellings = sorted({sp for _, sp in order})
    return dict(
        id="towns",
        title="Three towns: a systematic typo and an ambiguous slip",
        blurb=("25 rows, 10 practices, 3 real towns. Practice P01 misspells Abingdon as "
               "'abington' on every one of its rows, and 'bethan' is exactly two edits "
               "from both Bethel and Dothan. Watch SMC commit to the misspelling, "
               "rejuvenation question it, reweighting punish the particles that kept it, "
               "and resampling copy the fix."),
        model=hospital.build_systematic(spellings), rows=rows,
        observed=[sp for _, sp in order], truth=[truth[sp] for _, sp in order],
        n_particles=6, rejuv_after=[8, 16], final_sweeps=2, seed=None,
        seed_search=range(40))


def scenario_figure1():
    """The paper's Figure 1, scaled down: one practice's misspelling outnumbers the truth."""
    rows, observed, truth = [], [], []
    order = [("P-BIG", "abington")] * 16 + [(f"P-{i}", "abingdon") for i in range(7)] * 2
    rnd = random.Random(3)
    rnd.shuffle(order)
    # Put the big practice first so SMC commits to the misspelling.
    order.remove(("P-BIG", "abington"))
    order.insert(0, ("P-BIG", "abington"))
    for pid, sp in order:
        rows.append({"practice.pid": pid, "practice.bad_city": sp})
        observed.append(sp)
        truth.append("abingdon")
    return dict(
        id="figure1",
        title="Figure 1: the misspelling outnumbers the truth",
        blurb=("'abington' appears on 16 rows (one practice), 'abingdon' on 14 rows spread "
               "across seven practices. Majority vote over cells picks 'abington'. Because "
               "the typo lives on the Practice, P-BIG's sixteen rows are one error, not "
               "sixteen votes. Every row here is either a hash-key hit or a clear call, so "
               "the particles never disagree -- this one is about rejuvenation."),
        model=hospital.build_systematic(sorted(set(observed))), rows=rows,
        observed=observed, truth=truth, n_particles=3, rejuv_after=[], final_sweeps=2,
        seed=0, seed_search=None)


def scenario_benchmark():
    """Real hospital rows, with the benchmark's per-row typo model."""
    all_rows, observed, truth = hospital.load_benchmark(400)
    # A slice with some real typos in it, small enough to read.
    idx = [i for i in range(len(all_rows)) if observed[i] != truth[i]][:3]
    lo = max(0, idx[0] - 12) if idx else 0
    sl = list(range(lo, lo + 30))
    rows = [all_rows[i] for i in sl]
    obs = [observed[i] for i in sl]
    tru = [truth[i] for i in sl]
    return dict(
        id="benchmark",
        title="Hospital benchmark: independent typos per row",
        blurb=("30 real rows from the hospital benchmark, with its per-row error model: "
               "the typo is on each Record, and the clean name is two reference hops away "
               "on a City. Every row is scored, so particles' weights move on every row."),
        model=hospital.build_per_row(sorted(set(obs))), rows=rows, observed=obs,
        truth=tru, n_particles=4, rejuv_after=[15], final_sweeps=2, seed=0,
        seed_search=None)


SCENARIOS = {"towns": scenario_towns, "figure1": scenario_figure1,
             "benchmark": scenario_benchmark}


# ============================================================================= schema

def export_schema(model, rows):
    """Classes, their own fields, and which field depends on which.

    Only a class's *own* vertices are listed (no dotted names): a flattened copy lives on
    the object that owns it, so that is where the visualizer draws it. Parents are given
    as dotted paths from the owning object, e.g. Practice.bad_city <- "city.name", which
    the app resolves by following the object's reference slots.
    """
    observed = set()
    obs_cls = model.observation_class
    for r in rows:
        for k in r:
            path = k.split(".")
            cname = obs_cls
            for slot in path[:-1]:
                cls = model.classes[cname]
                cname = strip_submodel(cls.nodes[cls.names[slot]]).target_class
            observed.add((cname, path[-1]))

    classes = []
    for cname in model.topological_class_order():
        cls = model.classes[cname]
        fields = []
        for v in range(len(cls.nodes)):
            name = cls.node_name(v)
            if "." in name:
                continue
            node = strip_submodel(cls.nodes[v])
            f = {"name": name, "observed": (cname, name) in observed,
                 "key": v in cls.hash_keys}
            if isinstance(node, ForeignKeyNode):
                f["kind"] = "slot"
                f["target"] = node.target_class
                f["parents"] = []
            else:
                f["kind"] = "attr"
                f["dist"] = type(node.dist).__name__ if isinstance(
                    node, RandomChoiceNode) else type(node).__name__
                f["parents"] = [cls.node_name(a.v) for a in
                                getattr(node, "arg_node_ids", []) if isinstance(a, Ref)]
            fields.append(f)
        classes.append({"name": cname, "fields": fields,
                        "flattened": [cls.node_name(v) for v in range(len(cls.nodes))]})
    return {"classes": classes, "observation_class": obs_cls}


# ========================================================================== snapshots

def snapshot(model, trace) -> Dict[str, dict]:
    """A particle's database as plain JSON: 'City#0' -> {cls, oid, values, refs}."""
    out = {}
    for cname, cls in model.classes.items():
        for oid, obj in trace.table(cname).objects.items():
            vals = {}
            for v, val in obj.values.items():
                name = cls.node_name(v)
                if "." not in name:
                    vals[name] = val
            out[f"{cname}#{oid}"] = {"cls": cname, "oid": oid, "values": vals,
                                     "refs": obj.ref_count}
    return out


def diff(k, old: Dict[str, dict], new: Dict[str, dict]) -> List[dict]:
    ops = []
    for key, obj in new.items():
        if key not in old:
            ops.append({"op": "add", "p": k, "key": key, "obj": obj})
            continue
        o = old[key]
        for f, val in obj["values"].items():
            if o["values"].get(f, "__missing__") != val:
                ops.append({"op": "set", "p": k, "key": key, "field": f, "value": val,
                            "old": o["values"].get(f)})
        if o["refs"] != obj["refs"]:
            ops.append({"op": "refs", "p": k, "key": key, "value": obj["refs"]})
    for key in old:
        if key not in new:
            ops.append({"op": "del", "p": k, "key": key})
    return ops


# ============================================================================ helpers

def fin(x):
    """JSON has no -inf."""
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return None
    return round(x, 4)


def softmax(scores):
    finite = [s for s in scores if s is not None]
    if not finite:
        return [0.0] * len(scores)
    z = logsumexp(finite)
    return [round(math.exp(s - z), 4) if s is not None else 0.0 for s in scores]


def slot_label(val, target, snap):
    if isinstance(val, NewObject) or val is None:
        return "new"
    key = f"{target}#{val}"
    return key


def describe_value(val):
    if isinstance(val, NewObject):
        return "new"
    # StringPrior's DUMMY stand-in: all the prior mass outside the candidate shortlist.
    if isinstance(val, str) and val and set(val) == {"*"}:
        return "‹any other string›"
    return val


class Recorder:
    def __init__(self, sc):
        self.sc = sc
        self.model = sc["model"]
        self.steps: List[dict] = []
        self.prev: List[Dict[str, dict]] = []
        self.targets = {}
        for cname, cls in self.model.classes.items():
            for v, node in enumerate(cls.nodes):
                inner = strip_submodel(node)
                if isinstance(inner, ForeignKeyNode):
                    self.targets[(cname, cls.node_name(v))] = inner.target_class

    # ---------------------------------------------------------------- emitting

    def emit(self, smc, kind, phase, title, text, particle=None, detail=None,
             resample=None, row=None, sweep=None):
        ops = []
        if resample is not None:
            ops.append({"op": "resample", "ancestors": resample})
            self.prev = [copy.deepcopy(self.prev[a]) for a in resample]
        ks = range(smc.n) if particle is None else [particle]
        for k in ks:
            snap = snapshot(self.model, smc.particles[k].trace)
            ops.extend(diff(k, self.prev[k], snap))
            self.prev[k] = snap
        self.steps.append({
            "i": len(self.steps), "kind": kind, "phase": phase, "title": title,
            "text": text, "particle": particle, "row": row, "sweep": sweep,
            "ops": ops, "detail": detail or {},
            "logw": [fin(p.log_weight) for p in smc.particles],
            "ess": fin(smc.ess()),
        })

    # --------------------------------------------------- converting decision logs

    def convert_choices(self, entries, snap):
        out = []
        for e in entries:
            kind = e["kind"]
            v = e["vertex"]
            if kind in ("slot", "attribute"):
                scores = [fin(c["score"]) for c in e["candidates"]]
                probs = softmax(scores)
                cands = []
                target = None
                if kind == "slot":
                    target = self.slot_target(v)
                for c, s, pr in zip(e["candidates"], scores, probs):
                    val = c["value"]
                    if kind == "slot":
                        label = slot_label(val, target, snap)
                        name = self.object_summary(label, snap)
                    else:
                        label, name = describe_value(val), None
                    cands.append({"label": label, "summary": name,
                                  "prior": fin(c["prior"]), "score": s, "p": pr})
                out.append({"kind": kind, "vertex": v, "target": target,
                            "candidates": cands, "chosen": e["chosen"]})
            elif kind == "key_lookup":
                out.append({"kind": kind, "vertex": v, "target": e["target"],
                            "key": e["key"],
                            "hit": None if e["hit"] is None else f"{e['target']}#{e['hit']}"})
            elif kind == "observed":
                out.append({"kind": kind, "vertex": v, "value": e["value"],
                            "logp": fin(e["logp"]),
                            "args": [a if isinstance(a, (str, int, float)) else str(a)
                                     for a in e.get("args", [])]})
            elif kind == "determined":
                val = describe_value(e["value"])
                if self.is_slot(v) and isinstance(val, int):
                    val = f"{self.slot_target(v)}#{val}"
                out.append({"kind": kind, "vertex": v, "value": val})
            elif kind == "sampled":
                out.append({"kind": kind, "vertex": v, "value": e["value"],
                            "logp": fin(e["logp"])})
        return out

    def is_slot(self, vertex):
        cname = self.model.observation_class
        for part in vertex.split("."):
            if (cname, part) not in self.targets:
                return False
            cname = self.targets[(cname, part)]
        return True

    def slot_target(self, vertex):
        """Target class of a (possibly dotted) slot vertex of the observation class."""
        cname = self.model.observation_class
        for part in vertex.split("."):
            cname = self.targets[(cname, part)]
        return cname

    def object_summary(self, key, snap):
        if key == "new" or key not in snap:
            return None
        vals = snap[key]["values"]
        shown = {f: v for f, v in vals.items()
                 if (snap[key]["cls"], f) not in self.targets}
        return ", ".join(f"{f}={v}" for f, v in shown.items()) or None

    # ------------------------------------------------------------------ running

    def run(self):
        sc = self.sc
        model = self.model
        random.seed(sc["seed"])
        enc = encode_rows(model, model.observation_class, sc["rows"])
        smc = ParticleSMC(model, sc["n_particles"])
        self.prev = [{} for _ in range(smc.n)]
        stats = {"resamples": 0, "min_ess": float(smc.n), "resample_rows": []}

        self.emit(smc, "intro", "model", "The model",
                  "Before any data: a schema and nothing else. Each particle's latent "
                  "database is empty. Step forward to feed rows in one at a time.")

        rej_every = set(sc["rejuv_after"])
        n_rows = len(enc)
        sweep_no = 0
        for i, row in enumerate(enc):
            raw = sc["rows"][i]
            self.emit(smc, "row", "smc", f"Row {i + 1} arrives",
                      "Every particle must now explain this row, each against its own "
                      "database: reuse objects it already has, or create new ones.",
                      row=i, detail={"row": raw})
            before = copy.deepcopy(self.prev)
            for k in range(smc.n):
                log: list = []
                snap_before = before[k]
                inc = smc.extend(k, row, log=log)
                choices = self.convert_choices(log, snap_before)
                self.emit(smc, "propose", "smc",
                          f"Particle {k + 1} explains row {i + 1}",
                          self.narrate_proposal(choices), particle=k, row=i,
                          detail={"choices": choices, "increment": fin(inc)})
            ess = smc.ess()
            stats["min_ess"] = min(stats["min_ess"], ess)
            lw = smc.normalized_log_weights()
            self.emit(smc, "reweight", "smc", f"Reweight after row {i + 1}",
                      self.narrate_reweight(ess, smc), row=i,
                      detail={"weights": [round(math.exp(w), 4) for w in lw],
                              "ess": fin(ess), "threshold": smc.ess_threshold})
            anc = smc.maybe_resample()
            if anc is not None:
                stats["resamples"] += 1
                stats["resample_rows"].append(i + 1)
                self.emit(smc, "resample", "resample", f"Resample after row {i + 1}",
                          self.narrate_resample(anc), row=i, resample=anc,
                          detail={"ancestors": anc})
            if (i + 1) in rej_every:
                sweep_no += 1
                self.sweep(smc, enc[: i + 1], sweep_no, i, final=False)

        for s in range(sc["final_sweeps"]):
            sweep_no += 1
            self.sweep(smc, enc, sweep_no, n_rows - 1, final=True)

        self.emit(smc, "done", "done", "Done",
                  "Inference is over. Each particle is one sample from (approximately) "
                  "the posterior over latent databases. Where they agree, the model is "
                  "confident; where they disagree, that disagreement is the uncertainty.")
        stats["min_ess"] = round(stats["min_ess"], 3)
        return stats

    def sweep(self, smc, rows, sweep_no, row_idx, final):
        where = "final" if final else f"after row {row_idx + 1}"
        self.emit(smc, "sweep", "rejuv", f"Rejuvenation sweep {sweep_no} ({where})",
                  "SMC can only extend a database, never revise it. Rejuvenation goes "
                  "back over every object, in reverse topological order, and redraws each "
                  "one's free variables from their full conditional, given everything "
                  "else, including rows seen since. Weights do not change: this is an "
                  "MCMC move that leaves the target invariant.",
                  sweep=sweep_no, row=row_idx)
        for k in range(smc.n):
            rej = smc.rejuvenator(k, rows)
            hook = _HookList(lambda e, k=k: self.on_rejuv(smc, k, e, sweep_no, row_idx))
            rej.log = hook
            rej.sweep()
            if rej.collected:
                self.emit(smc, "gc", "rejuv",
                          f"Particle {k + 1}: garbage collection",
                          f"{rej.collected} object(s) no longer reachable from any row "
                          "were deleted. The structure prior gives such databases zero "
                          "probability [§2.2], so this is not housekeeping.",
                          particle=k, sweep=sweep_no, row=row_idx,
                          detail={"collected": rej.collected})

    def on_rejuv(self, smc, k, e, sweep_no, row_idx):
        scores = [fin(s["score"]) for s in e["settings"]]
        probs = softmax(scores)
        key = f"{e['cls']}#{e['oid']}"
        settings = []
        for s, sc_, pr in zip(e["settings"], scores, probs):
            vals = {}
            for f, v in s["values"].items():
                tgt = self.targets.get((e["cls"], f))
                vals[f] = (f"{tgt}#{v}" if tgt is not None and v is not None
                           else describe_value(v))
            settings.append({
                "values": vals, "prior": fin(s["prior"]), "score": sc_, "p": pr,
                "evidence": [{"key": f"{c}#{o}", "logp": fin(x)}
                             for c, o, x in s["evidence"]],
            })
        old = {}
        for f, v in e["old"].items():
            tgt = self.targets.get((e["cls"], f))
            old[f] = f"{tgt}#{v}" if tgt is not None and v is not None else v
        chosen = settings[e["chosen"]]["values"]
        changed = e["changed"] > 0
        detail = {"key": key, "targets": e["targets"], "old": old,
                  "settings": settings, "chosen": e["chosen"], "changed": changed}
        self.emit(smc, "rejuv", "rejuv",
                  f"Particle {k + 1}: revisit {key}",
                  self.narrate_rejuv(key, e["targets"], old, chosen, settings,
                                     e["chosen"], changed),
                  particle=k, sweep=sweep_no, row=row_idx, detail=detail)

    # ---------------------------------------------------------------- narration

    def narrate_proposal(self, choices):
        """One or two sentences; the detail panel carries the numbers."""
        parts = []
        for c in choices:
            if c["kind"] == "key_lookup":
                if c["hit"]:
                    parts.append(f"Key {c['key'][0]!r} is already {c['hit']}, so the row "
                                 "just points at it; nothing else to decide.")
                else:
                    parts.append(f"Key {c['key'][0]!r} is new: create a {c['target']}.")
            elif c["kind"] == "slot" and c["chosen"] is not None:
                ch = c["candidates"][c["chosen"]]
                what = (f"a new {c['target']}" if ch["label"] == "new" else
                        f"{ch['label']}" + (f" ({ch['summary']})" if ch["summary"] else ""))
                runner = sorted((x for x in c["candidates"] if x is not ch),
                                key=lambda x: -x["p"])
                alt = ""
                if runner and runner[0]["p"] >= 0.005:
                    r = runner[0]
                    alt = (f", over {'a new one' if r['label'] == 'new' else r['label']}"
                           f" at {r['p']:.2f}")
                parts.append(f"{c['vertex']} -> {what} with p = {ch['p']:.2f}{alt}.")
            elif c["kind"] == "attribute" and c["chosen"] is not None:
                ch = c["candidates"][c["chosen"]]
                parts.append(f"{c['vertex']} = {ch['label']!r} (p = {ch['p']:.2f}).")
        return " ".join(parts) if parts else "Nothing to decide."

    def narrate_reweight(self, ess, smc):
        verdict = ("below" if ess < smc.ess_threshold else "above")
        return (f"Each particle's weight grew by the probability its database gave this "
                f"row. Effective sample size is {ess:.2f} of {smc.n}, {verdict} the "
                f"resampling threshold of {smc.ess_threshold:g}.")

    def narrate_resample(self, anc):
        counts = {}
        for a in anc:
            counts[a] = counts.get(a, 0) + 1
        kept = ", ".join(f"P{a + 1}x{n}" for a, n in sorted(counts.items()))
        dead = [str(a + 1) for a in range(len(anc)) if a not in counts]
        s = (f"The weights got too uneven, so N new particles were drawn in proportion to "
             f"weight: {kept}.")
        if dead:
            s += (f" Particle(s) {', '.join(dead)} left no descendants -- their databases "
                  "are gone, replaced by copies of better ones.")
        return s + " All weights reset to equal."

    def narrate_rejuv(self, key, targets, old, chosen, settings, ci, changed):
        top = sorted(range(len(settings)), key=lambda j: -settings[j]["p"])[:3]
        opts = "; ".join(
            ", ".join(f"{f}={v}" for f, v in settings[j]["values"].items()) +
            f" p={settings[j]['p']:.2f}" for j in top)
        what = ", ".join(targets)
        if changed:
            moves = ", ".join(f"{f}: {old.get(f)} -> {chosen.get(f)}" for f in targets
                              if old.get(f) != chosen.get(f))
            return f"Redraw {what} of {key}. Options: {opts}. Changed {moves}."
        return f"Redraw {what} of {key}. Options: {opts}. Kept as is."


class _HookList(list):
    """A list that calls back on append -- lets the recorder snapshot after each blocked
    update without the Rejuvenator knowing it is being watched."""

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def append(self, x):
        super().append(x)
        self.cb(x)


# ========================================================================== recording

def interest(stats, sc):
    """How well a run tells the scenario's intended story -- used only to pick a seed.

    Wanted: SMC commits to the misspelling, rejuvenation questions it, and resampling
    *afterwards* propagates the fix. A resample before the first sweep means some particle
    made a lucky long-shot guess at row 1 -- real, but not the story, so it is penalised.
    Every run the search considers is an honest run of the algorithm; this only picks
    which one to show.
    """
    first = min(sc["rejuv_after"]) if sc["rejuv_after"] else 0
    late = sum(1 for r in stats["resample_rows"] if r > first)
    early = sum(1 for r in stats["resample_rows"] if r <= first)
    return min(late, 2) * 10 - early * 20 + stats["divergent"] * 5


def record(sc):
    rec = Recorder(sc)
    stats = rec.run()
    # Particle disagreement at the end (distinct databases, by city names per practice).
    finals = []
    for k in range(sc["n_particles"]):
        finals.append(json.dumps(sorted(
            (o["cls"], o["oid"], sorted(o["values"].items()))
            for o in rec.prev[k].values() if o["cls"] != sc["model"].observation_class),
            default=str))
    stats["final_distinct"] = len(set(finals))
    stats["divergent"] = stats["final_distinct"] > 1
    return rec, stats


def build(name):
    sc = SCENARIOS[name]()
    if sc["seed_search"] is not None:
        best = None
        for seed in sc["seed_search"]:
            sc["seed"] = seed
            _, stats = record(sc)
            score = interest(stats, sc)
            if best is None or score > best[0]:
                best = (score, seed, stats)
        sc["seed"] = best[1]
    rec, stats = record(sc)
    doc = {
        "id": sc["id"], "title": sc["title"], "blurb": sc["blurb"],
        "seed": sc["seed"], "n_particles": sc["n_particles"],
        "schema": export_schema(sc["model"], sc["rows"]),
        # Where each row's clean value lives, from its Record. Record #i is row i: every
        # row creates exactly one Record and Records are never collected.
        "clean_path": "practice.city.name",
        "rows": [{"data": r, "observed": o, "truth": t}
                 for r, o, t in zip(sc["rows"], sc["observed"], sc["truth"])],
        "stats": stats, "steps": rec.steps,
    }
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = [args.scenario] if args.scenario else list(SCENARIOS)
    index = []
    for name in names:
        doc = build(name)
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"), default=str)
        print(f"{name}: seed {doc['seed']}, {len(doc['steps'])} steps, "
              f"{os.path.getsize(path) // 1024} KB, stats {doc['stats']}")
        index.append({"id": name, "title": doc["title"]})
    if not args.scenario:
        with open(os.path.join(OUT, "index.json"), "w") as f:
            json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
