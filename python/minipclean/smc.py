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
from .structure_prior import collect_garbage, crp_candidate_log_weights
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
                 rows: List[Dict[VertexID, Any]], smc: SMC, gc: bool = True):
        self.model = model
        # Collect unreachable objects after each sweep. Off only to demonstrate what
        # goes wrong without it -- see example_gc.py.
        self.gc = gc
        self.trace = trace
        # (cls, oid) -> {vertex in that class: observed value}. One entry per object,
        # which is exactly what makes the counting per-object.
        self.obj_obs: Dict[tuple, Dict[VertexID, Any]] = {}
        # (cls, oid) -> objects holding a slot pointing at it.
        self.referrers: Dict[tuple, set] = {}
        # Joint settings scored across all blocked updates -- the cost subproblem hints
        # are there to reduce.
        self.scored = 0
        # Objects removed by the most recent sweep.
        self.collected = 0
        # Optional decision log for the visualizer (../viz), as in FlatProposer: None is
        # off; a list receives one dict per blocked update that had a choice to make.
        self.log: Optional[list] = None

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

        self._build_referrers()

    def _build_referrers(self) -> None:
        """Recompute who points at whom from the database as it stands.

        This has to be kept current, not built once. Rejuvenation moves reference slots,
        and a City's evidence is read *through* its referrers -- so a stale map scores a
        city against practices that have left it and ignores ones that have joined.
        """
        self.referrers = {}
        for cname, cls in self.model.classes.items():
            for oid, obj in self.trace.table(cname).objects.items():
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

    def _transitive_referrers(self, cname, oid) -> set:
        """Every object that transitively points at this one.

        One hop is not enough. In the hospital model the evidence about a City's clean
        name is the dirty spelling on the **Records** -- but a City's direct referrers are
        Practices, which observe nothing. Stop at one hop and the city has no evidence,
        every candidate scores the same on the prior, and rejuvenation quietly does
        nothing while appearing to run.

        Counting stays per *object*: each object contributes its own observations once.
        """
        seen, frontier = set(), [(cname, oid)]
        while frontier:
            cur = frontier.pop()
            for r in self.referrers.get(cur, ()):
                if r not in seen:
                    seen.add(r)
                    frontier.append(r)
        return seen

    def sweep(self, verbose: bool = False) -> int:
        """One pass over every object, re-enumerating each of its subproblems.

        Reverse topological order, so a Record's City is settled before the Record is
        revisited [App. D.1].
        """
        changed = 0
        for cname in reversed(self.model.topological_class_order()):
            cls = self.model.classes[cname]
            for oid, obj in list(self.trace.table(cname).objects.items()):
                for block in cls.blocks:
                    changed += self._update_block(cls, cname, obj, block, verbose)

        # Slots moved, so some objects may now be unreachable from the data -- states the
        # structure prior gives zero mass [§2.2]. Collect them before anyone reads the
        # database again.
        self.collected = collect_garbage(self.model, self.trace) if self.gc else 0
        if self.collected:
            self._build_referrers()
        if verbose and self.collected:
            print(f"    collected {self.collected} unreachable object(s)")
        return changed

    def _targets(self, cls, cname, obj, block) -> List[VertexID]:
        """Which vertices of this block are free to move.

        Reference slots are included. The paper is explicit that a rejuvenation move
        updates "all r's attributes **and reference slots**" [§3.1] -- and it is the slot
        moves that make garbage collection necessary at all, because the object a slot
        abandons may have nothing else pointing at it.
        """
        obs = self.obj_obs.get((cname, obj.oid), {})
        out = []
        for v in block:
            if "." in cls.node_name(v):
                continue                      # owned by another object
            node = strip_submodel(cls.nodes[v])
            if isinstance(node, ForeignKeyNode):
                # A slot into a class with guaranteed keys is *pinned*, not free.
                # `@guaranteed` says the key field is trusted and always observed, so
                # identity is settled by exact lookup -- App. D.4's index reduces the
                # legal targets to a singleton. Revising such a slot lets a Record drift
                # onto a Practice with a different provider number, which is not a
                # hypothesis the model entertains.
                if self.model.classes[node.target_class].hash_keys:
                    continue
                out.append(v)
                continue
            if not isinstance(node, RandomChoiceNode):
                continue
            if v in cls.hash_keys or v in obs:
                continue      # trusted key / directly observed
            if not node.dist.has_discrete_proposal():
                continue
            out.append(v)
        return out

    def _settings(self, cls, obj, targets, i, cur, lp, out):
        """Enumerate every joint setting of a block's free vertices, with its log prior.

        Recursion in topological order, so each vertex's arguments are resolved against
        the partial setting chosen so far -- which is what makes `specialty ~ ...(degree)`
        work: specialty's candidate list is asked for *given* the degree under test.

        This is the combinatorial term subproblem hints control. k dependent vertices with
        n candidates each cost n**k settings in one block, and n*k across k blocks.
        """
        if i == len(targets):
            out.append((dict(cur), lp))
            return
        v = targets[i]
        node = strip_submodel(cls.nodes[v])
        obj.values.update(cur)

        if isinstance(node, ForeignKeyNode):
            # Candidate targets for a reference slot, weighted by the CRP.
            #
            # Leave-one-out: this object's own reference is removed from the counts
            # before weighing candidates. That is what makes the move a correct Gibbs
            # step -- you are asking "given every *other* reference, where should this
            # one go?", not letting the slot vote for its own current target.
            table = self.trace.table(node.target_class)
            cur_target = obj.values.get(v)
            if cur_target is not None and cur_target in table.objects:
                table.objects[cur_target].ref_count -= 1
            py = self.model.classes[node.target_class].py
            oids, weights = crp_candidate_log_weights(table, py, list(table.objects))
            if cur_target is not None and cur_target in table.objects:
                table.objects[cur_target].ref_count += 1

            # Staying put must remain an option. After the leave-one-out decrement, an
            # object this slot is the *sole* referrer of has count 0, and a zero-weight
            # candidate is dropped -- so without this the slot would be *forced* to move,
            # and every singleton entity would be destroyed on the first sweep.
            #
            # In CRP terms, re-seating at a table you just vacated is the same event as
            # starting a fresh one, so it takes the new-table weight.
            new_w = math.log(py.strength + py.discount * len(table)) \
                if py.strength + py.discount * len(table) > 0 else -math.inf
            if cur_target is not None and cur_target not in oids:
                oids = list(oids) + [cur_target]
                weights = list(weights) + [new_w]

            for oid, w in zip(oids, weights):
                if oid is None:
                    continue      # see NOTE in _update_block: we do not insert new
                                  # objects during rejuvenation
                cur[v] = oid
                self._settings(cls, obj, targets, i + 1, cur, lp + w, out)
            cur.pop(v, None)
            return

        args = FlatProposer(self.model, self.trace)._args(cls, node, obj.values)
        if args is None:
            self._settings(cls, obj, targets, i + 1, cur, lp, out)
            return
        options, priors = node.dist.discrete_proposal(*args)
        for opt, p in zip(options, priors):
            if opt is DUMMY:
                opt = node.dist.discrete_proposal_dummy_value(*args)
            cur[v] = opt
            self._settings(cls, obj, targets, i + 1, cur, lp + p, out)
        cur.pop(v, None)

    def _update_block(self, cls, cname, obj, block, verbose) -> int:
        """Blocked Gibbs: re-draw a whole subproblem at once, from its full conditional.

        **Blocked, not single-site.** Updating one variable at a time gets stuck, because
        a city's clean name and the spellings that depend on it are tightly correlated and
        no single-variable move can escape a bad mode. Redrawing an entire subproblem
        together can [App. D.1].

        The conditional needs this object's own observed cells plus those of every object
        that transitively points at it, each counted **once**.

        NOTE (divergence): the Julia scores incrementally, exploiting the Plan forest so
        that conditionally independent vertices add rather than multiply, and erasing only
        the affected slice of the database (`R_minus_r`). We enumerate complete settings
        and rescore the object and its referrers for each. Same distribution -- shared
        terms are constant across settings and cancel in the normalisation -- but we do
        not get the forest's savings on this path, only on the SMC path in proposal.py.
        """
        targets = self._targets(cls, cname, obj, block)
        if not targets:
            return 0

        settings = []
        old = {v: obj.values.get(v) for v in targets}
        self._settings(cls, obj, targets, 0, {}, 0.0, settings)
        obj.values.update(old)
        if len(settings) <= 1:
            return 0

        refs = self._transitive_referrers(cname, obj.oid)
        scores, evidence = [], []
        for vals, lp in settings:
            obj.values.update(vals)
            own = self._score_local(cls, cname, obj)
            s = lp + own
            ev = [(cname, obj.oid, own)] if own else []
            for (rc, ro) in sorted(refs):
                robj = self.trace.table(rc).objects.get(ro)
                if robj is not None:
                    c = self._score_local(self.model.classes[rc], rc, robj)
                    s += c
                    if c:
                        ev.append((rc, ro, c))
            scores.append(s)
            evidence.append(ev)
        self.scored += len(settings)
        obj.values.update(old)

        marginal = logsumexp(scores)
        if marginal == -math.inf:
            return 0
        ci = _sample(scores, marginal)
        chosen, _ = settings[ci]

        # Keep the CRP's counts honest. `ref_count` is the n_r that drives rich-get-richer
        # weighting in structure_prior.py, so a slot that moves without decrementing its
        # old target leaves the prior permanently pointing at a phantom.
        for v in targets:
            if not isinstance(strip_submodel(cls.nodes[v]), ForeignKeyNode):
                continue
            before, after = old.get(v), chosen.get(v)
            if before == after:
                continue
            tname = strip_submodel(cls.nodes[v]).target_class
            tbl = self.trace.table(tname)
            if before is not None and before in tbl.objects:
                tbl.objects[before].ref_count -= 1
            if after is not None and after in tbl.objects:
                tbl.objects[after].ref_count += 1
            self.referrers.get((tname, before), set()).discard((cname, obj.oid))
            self.referrers.setdefault((tname, after), set()).add((cname, obj.oid))

        obj.values.update(chosen)
        n = sum(1 for v in targets if chosen.get(v) != old.get(v))
        if self.log is not None:
            names = [cls.node_name(v) for v in targets]
            self.log.append({
                "cls": cname, "oid": obj.oid, "targets": names,
                "old": {cls.node_name(v): old.get(v) for v in targets},
                "settings": [{"values": {cls.node_name(v): vals.get(v) for v in targets},
                              "prior": lp, "score": sc, "evidence": ev}
                             for (vals, lp), sc, ev in zip(settings, scores, evidence)],
                "chosen": ci,
                "changed": n,
            })
        if verbose and n:
            for v in targets:
                if chosen.get(v) != old.get(v):
                    print(f"    rejuv {cname}#{obj.oid}.{cls.node_name(v)}: "
                          f"{old.get(v)!r} -> {chosen.get(v)!r}")
        return n
