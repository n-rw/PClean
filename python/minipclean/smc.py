"""SMC and rejuvenation over a flattened model — mirrors
``src/inference/row_inference.jl``, ``inference.jl`` and [App. D.1].

Same algorithm as before; what changed is that proposing a row is now a single flat
enumeration (see proposal.py) rather than a recursion across classes.
"""

import math
import random as _random
from typing import Any, Dict, List, Optional

from .distributions import DUMMY, logsumexp
from .model import (ForeignKeyNode, PCleanModel, RandomChoiceNode, VertexID,
                    strip_submodel)
from .parameters import fit_parameters
from .proposal import FlatProposer, _sample
from .trace import Trace


def encode_rows(model: PCleanModel, cls_name: str,
                rows: List[Dict[str, Any]]) -> List[Dict[VertexID, Any]]:
    """Turn dotted-name observations into vertex-keyed ones.

    This is the whole of `@query` after flattening. A query line

        City   hosp.loc.city   city

    just says "column City is observed at the vertex named `hosp.loc.city`" -- and that
    vertex exists, locally, because flattening put it there.
    """
    names = model.classes[cls_name].names
    out = []
    for r in rows:
        enc = {}
        for k, val in r.items():
            if k not in names:
                raise KeyError(f"{k!r} is not a vertex of {cls_name}; "
                               f"known: {sorted(names)}")
            enc[names[k]] = val
        out.append(enc)
    return out


class SMC:
    """Per-observation SMC [§3.1].

    The model admits a sequential form -- build the latent database one row at a time,
    each row contributing the objects it refers to that no earlier row did (Fig. 5). SMC
    consumes exactly that. Rows are customers entering restaurants; to start a new table
    you must first send friends to the other restaurants, one per reference slot.

    SMC can only ever *extend* a hypothesis, never revise it, which is fatal on its own
    for cleaning: you cannot tell a misspelling from the truth until you have seen the
    other rows of the same entity. That is what rejuvenation is for.
    """

    def __init__(self, model: PCleanModel, n_particles: int = 1):
        self.model = model
        self.n_particles = n_particles
        self.row_assignments: List[int] = []
        self.owner_maps: List[Dict[VertexID, tuple]] = []

    def run(self, rows: List[Dict[VertexID, Any]], progress_every: int = 0) -> Trace:
        cls_name = self.model.observation_class
        cls = self.model.classes[cls_name]
        trace = Trace(self.model)
        proposer = FlatProposer(self.model, trace)

        self.row_assignments, self.owner_maps = [], []
        for i, row in enumerate(rows):
            res = proposer.propose(cls_name, row)
            trace.log_weight += res.log_marginal
            self.row_assignments.append(res.oid)
            self.owner_maps.append(proposer.owner_map(cls, res.assignment))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  SMC: row {i+1}/{len(rows)}  [{trace.summary()}]")
        return trace


class Rejuvenator:
    """Object-wise blocked Gibbs [§3.1, App. D.1].

    ## Per object, not per row

    The single subtlest point in the system. A Practice has **one** `bad_city` however
    many rows mention it, so each practice must contribute its spelling **once**.

    Score per row instead -- the obvious implementation -- and a practice appearing in 152
    rows contributes its single typo 152 times. You have then rebuilt majority voting
    inside a Bayesian model, and the paper's headline example comes out backwards:

        per row     'abington': 152 exact +  42 typos   ->  the misspelling wins
        per object  'abington':   1 exact +   7 typos   ->  the truth wins, by ~31 nats

    Identical data. Only the counting differs, and the counting follows from where the
    error model sits in the schema.

    ## Blocked, and in reverse topological order

    Updating one variable at a time gets stuck: a city's clean name and the spellings
    depending on it are tightly correlated, so no single-variable move escapes a bad mode.
    We sweep objects in reverse topological order so a Record's City is settled before the
    Record is revisited.
    """

    def __init__(self, model: PCleanModel, trace: Trace,
                 rows: List[Dict[VertexID, Any]], smc: SMC):
        self.model = model
        self.trace = trace
        # (cls, oid) -> {vertex in that class: observed value}. One entry per object,
        # which is exactly what makes the counting per-object.
        self.obj_obs: Dict[tuple, Dict[VertexID, Any]] = {}
        # (cls, oid) -> objects holding a slot pointing at it.
        self.referrers: Dict[tuple, set] = {}

        obs_cls = model.observation_class
        for row, omap, root_oid in zip(rows, smc.owner_maps, smc.row_assignments):
            for local_v, val in row.items():
                if val is None:
                    continue
                if local_v in omap:
                    # A flattened vertex: the observation belongs to the object further
                    # down the chain that actually owns that attribute.
                    cname, oid, tv = omap[local_v]
                    self.obj_obs.setdefault((cname, oid), {})[tv] = val
                elif root_oid is not None:
                    # An attribute of the observation class itself (a per-row typo, say),
                    # which belongs to this row's own object. Easy to forget, and if you
                    # do, every per-row observation becomes invisible to rejuvenation.
                    self.obj_obs.setdefault((obs_cls, root_oid), {})[local_v] = val

        for cname, cls in model.classes.items():
            for oid, obj in trace.table(cname).objects.items():
                for v, node in enumerate(cls.nodes):
                    inner = strip_submodel(node)
                    if isinstance(inner, ForeignKeyNode) and v in obj.values:
                        key = (inner.target_class, obj.values[v])
                        self.referrers.setdefault(key, set()).add((cname, oid))

    def _score_local(self, cls, cname, obj) -> float:
        """One object's own observed attributes, counted once."""
        obs = self.obj_obs.get((cname, obj.oid))
        if not obs:
            return 0.0
        pr = FlatProposer(self.model, self.trace)
        total = 0.0
        for v, val in obs.items():
            node = strip_submodel(cls.nodes[v])
            if not isinstance(node, RandomChoiceNode):
                continue
            args = pr._args(cls, node, obj.values)
            if args is None:
                continue
            total += node.dist.logdensity(val, *args)
        return total

    def sweep(self, verbose: bool = False) -> int:
        changed = 0
        for cname in reversed(self.model.topological_class_order()):
            cls = self.model.classes[cname]
            for oid, obj in list(self.trace.table(cname).objects.items()):
                for v in range(len(cls.nodes)):
                    node = strip_submodel(cls.nodes[v])
                    if not isinstance(node, RandomChoiceNode):
                        continue
                    if v in cls.hash_keys or "." in cls.node_name(v):
                        continue      # guaranteed fields are trusted; flattened copies
                                      # belong to another object, not this one
                    if not node.dist.has_discrete_proposal():
                        continue
                    if self._update(cls, cname, obj, v, node, verbose):
                        changed += 1
        return changed

    def _transitive_referrers(self, cname, oid) -> set:
        """Every object that transitively points at this one.

        One hop is not enough. In the hospital model the evidence about a City's clean
        name is the dirty `obs_city` spelling on the **Records** -- but a City's direct
        referrers are Practices, which observe nothing. Stop at one hop and the city has
        no evidence at all, every candidate scores identically on the prior, and
        rejuvenation quietly does nothing while looking like it ran.

        Counting stays per *object*: each Record contributes its own `obs_city` once,
        because that attribute really is one draw per record. Where the error model sits
        on the entity instead (the Figure 1 / Physicians shape), the referrers are the
        entities and each contributes once. The rule is the same either way -- score every
        object that owns a relevant observation, exactly once.
        """
        seen, frontier = set(), [(cname, oid)]
        while frontier:
            cur = frontier.pop()
            for r in self.referrers.get(cur, ()):
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
        return seen

    def _update(self, cls, cname, obj, v, node, verbose) -> bool:
        """Re-draw one attribute from its full conditional.

        The conditional needs this object's own observed cells plus those of every object
        holding a slot pointing at it -- a Practice's dirty `bad_city` depends on its
        City's clean `name`. Each is scored exactly once.

        NOTE (divergence): the Julia erases precisely the affected slice of the database
        (`R_minus_r`, [App. D.1]) and recomputes only the terms that changed. We recompute
        the local scores of the object and its direct referrers; untouched terms are
        constant across candidates and cancel in the normalisation.
        """
        pr = FlatProposer(self.model, self.trace)
        args = pr._args(cls, node, obj.values)
        if args is None:
            return False
        options, priors = node.dist.discrete_proposal(*args)
        if len(options) <= 1:
            return False

        refs = self._transitive_referrers(cname, obj.oid)
        old = obj.values.get(v)
        scores, vals = [], []
        for opt, lp in zip(options, priors):
            if opt is DUMMY:
                opt = node.dist.discrete_proposal_dummy_value(*args)
            obj.values[v] = opt
            s = lp + self._score_local(cls, cname, obj)
            for (rc, ro) in refs:
                robj = self.trace.table(rc).objects.get(ro)
                if robj is not None:
                    s += self._score_local(self.model.classes[rc], rc, robj)
            scores.append(s)
            vals.append(opt)

        obj.values[v] = old
        marginal = logsumexp(scores)
        if marginal == -math.inf:
            return False
        i = _sample(scores, marginal)
        obj.values[v] = vals[i]
        if verbose and vals[i] != old:
            print(f"    rejuv {cname}#{obj.oid}.{cls.node_name(v)}: {old!r} -> {vals[i]!r}")
        return vals[i] != old
