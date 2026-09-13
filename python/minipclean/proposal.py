"""Data-driven enumerative proposals — mirrors ``src/inference/proposal_compiler.jl``
and ``block_proposal.jl``. **This is the file to read.**

## What the Julia does, and why you cannot read it

`proposal_compiler.jl` is a compiler. For each combination of (class, subproblem, pattern
of missing values) it builds a Julia syntax tree, `eval`s it into a natively-compiled
function, and caches it. Measured on the hospital benchmark: 8 such functions, 4,207
lines of generated Julia, then called thousands of times. The code that does the work
does not exist on disk.

## What this file does instead

It walks the `Plan` directly, performing at runtime the steps the Julia would have
emitted as code. Identical semantics, same output distribution, roughly 50x slower.
The compiler/interpreter distinction is the *only* deep difference between this package
and the original.

## The idea being implemented [§3.2]

A generic PPL proposes latent values from the prior and reweights. For a model like this
it never gets anywhere: guess a hospital name from a character-level prior and you will
guess wrong forever. PClean instead **enumerates**.

For each unobserved discrete variable, ask its distribution for a finite candidate list.
Score every candidate by the prior times the likelihood of everything downstream of it,
including the observed data. Normalise. Sample. The result is the *locally optimal*
proposal: of all possible proposals it minimises

    KL( pi_{i-1}(R) Q(delta) || pi_i(R union delta) )

and when every variable in the subproblem is finite-discrete, it is exactly the
posterior over that subproblem. It is not an approximation you hope is good; it is
the best proposal that exists, computed exactly.

## Why it is not exponentially expensive

Because of the `Plan` forest. Sibling trees are conditionally independent, so their
marginals **add** instead of multiplying out into a cross-product. See `_walk_plan`,
where that addition is a single `+=` -- three lines that are the whole optimisation.
"""

import math
import random as _random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .distributions import DUMMY, logsumexp


def enumerate_first(vs):
    """The target attributes we enumerate jointly with the reference choice."""
    return vs[:1]
from .model import (ForeignKeyNode, JuliaNode, ParameterNode, PCleanClass,
                    PCleanModel, Plan, RandomChoiceNode, Ref, Via, VertexID)
from .structure_prior import crp_candidate_log_weights
from .trace import LatentObject, Trace


@dataclass
class ObsNode:
    """Observations for one object, arranged as a tree mirroring the reference slots.

    `attrs` maps an attribute *name* in this class to its observed value (None means
    the cell was blank -- an imputation target). `children` maps a reference-slot name
    to the observations belonging to the object on the other end.

    This is what the Julia's `@query` block compiles to. A query line like

        HospitalName   hosp.name   name

    says: column `HospitalName` is observed at `Record.name`, and the *clean* value
    lives at `Record.hosp.name`. So the dirty value lands in this node's `attrs`, and
    `hosp` becomes a child ObsNode.
    """
    attrs: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, "ObsNode"] = field(default_factory=dict)


class ProposalResult:
    """What a proposal returns: how good, what it chose, and how likely it was to."""

    def __init__(self, log_marginal: float, oid: Optional[int], log_q: float):
        self.log_marginal = log_marginal   # log P(observations under this object)
        self.oid = oid                     # the object we ended up pointing at
        self.log_q = log_q                 # log Q(this choice) -- proposal density


class Proposer:
    """Runs enumerative proposals against a trace.

    One instance per SMC step. Holds the trace being extended plus scratch state for
    the recursion.
    """

    def __init__(self, model: PCleanModel, trace: Trace, verbose: bool = False):
        self.model = model
        self.trace = trace
        self.verbose = verbose
        self._depth = 0

    def _log(self, msg):
        if self.verbose:
            print("  " * self._depth + msg)

    # ---------------------------------------------------------------- arguments

    def _resolve_args(self, cls: PCleanClass, node, assignment: Dict[VertexID, Any],
                      slots: Dict[VertexID, int]) -> Optional[List[Any]]:
        """Look up a node's argument values.

        An argument is either a vertex id in this class, or a ``(slot_vertex, attr_name)``
        pair meaning "the attribute `attr_name` of whatever object this reference slot
        points at". The second form is how `degree ~ ChooseProportionally(degrees,
        degree_dist[school.name])` reaches across a reference.

        Returns None if some argument is not yet available, which tells the caller to
        skip this node for now -- the Julia does the same via `any_unavailable`.
        """
        out = []
        for a in node.arg_node_ids:
            if isinstance(a, Via):
                # Walk the reference chain hop by hop: hosp -> city -> name.
                oid = slots.get(a.slot)
                if oid is None:
                    return None
                cur_cls = cls.nodes[a.slot].target_class
                cur_obj = self.trace.table(cur_cls).objects.get(oid)
                if cur_obj is None:
                    return None
                for hop in a.path[:-1]:
                    cm = self.model.classes[cur_cls]
                    sv = _vertex_by_name(cm, hop)
                    if sv is None or sv not in cur_obj.values:
                        return None
                    cur_cls = cm.nodes[sv].target_class
                    cur_obj = self.trace.table(cur_cls).objects.get(cur_obj.values[sv])
                    if cur_obj is None:
                        return None
                idx = _vertex_by_name(self.model.classes[cur_cls], a.path[-1])
                if idx is None or idx not in cur_obj.values:
                    return None
                out.append(cur_obj.values[idx])
            elif isinstance(a, Ref):
                if a.v not in assignment:
                    return None
                out.append(assignment[a.v])
            else:
                out.append(a)          # a literal baked into the model
        return out

    # ---------------------------------------------------------------- the walk

    def propose_object(self, cls_name: str, obs: ObsNode,
                       force_new: bool = False) -> ProposalResult:
        """Propose (or find) an object of `cls_name` that explains `obs`.

        This is the recursive `GenerateDbIncr` of [§3.1, Fig. 5], run *conditioned on
        data* instead of forward. The recursion is genuinely the same shape: to make an
        object you first resolve its reference slots (visiting the "parent restaurants"),
        then draw its own attributes.

        NOTE (divergence): the Julia flattens every referenced class's nodes into the
        referring class's graph, so this recursion becomes a single flat Bayes net (see
        model.py). Recursing is much easier to read and gives the same answer; flattening
        is an optimisation, not a semantic difference.
        """
        cls = self.model.classes[cls_name]
        table = self.trace.table(cls_name)

        # --- The observation class is exempt from the CRP -----------------------
        # [§2.2]: "we set S_Cobs = {1, ..., |D|}" -- with probability 1, there is exactly
        # one object of the observation class per observed row. Rows are never merged
        # with each other; only the *entities they refer to* are. Forgetting this makes
        # the model quietly absurd (two different rows becoming "the same record"), so it
        # is worth stating in code rather than leaving implicit.
        if cls_name == self.model.observation_class:
            force_new = True

        # --- Fast path: exact-match lookup on the hash keys [App. D.4] ----------
        # If this class declares guaranteed (always-observed, trusted) fields and we
        # have them, identity is settled -- no enumeration, no CRP. This is the single
        # optimisation that separates "millions of rows" from "hopeless", because it
        # replaces a scan over every object with one dict lookup.
        if cls.hash_keys and not force_new:
            key = []
            ok = True
            for v in cls.hash_keys:
                name = cls.node_name(v)
                if name in obs.attrs and obs.attrs[name] is not None:
                    key.append(obs.attrs[name])
                else:
                    ok = False
                    break
            if ok:
                existing = table.lookup(tuple(key))
                if existing is not None:
                    self._log(f"{cls_name}: matched existing #{existing.oid} by key {tuple(key)}")
                    existing.ref_count += 1
                    # The object is settled, but its *unobserved* attributes may still
                    # need scoring against this row's data.
                    m = self._score_existing(cls, existing, obs)
                    return ProposalResult(m, existing.oid, 0.0)

        # --- Otherwise: enumerate candidate objects under the CRP --------------
        candidates: List[Optional[int]] = []
        cand_logw: List[float] = []
        if not force_new and not cls.hash_keys:
            oids, logw = crp_candidate_log_weights(table, cls.py, list(table.objects))
            candidates, cand_logw = oids, logw
        else:
            candidates, cand_logw = [None], [0.0]

        scores, payloads = [], []
        for oid, lw in zip(candidates, cand_logw):
            if oid is None:
                m, values, slots, q = self._propose_fresh(cls, obs)
                scores.append(lw + m)
                payloads.append(("new", values, slots, q))
            else:
                m = self._score_existing(cls, table.objects[oid], obs)
                scores.append(lw + m)
                payloads.append(("old", oid, None, 0.0))

        marginal = logsumexp(scores)
        i = _sample_index(scores, marginal)
        kind = payloads[i][0]
        log_q = scores[i] - marginal

        if kind == "old":
            oid = payloads[i][1]
            table.objects[oid].ref_count += 1
            return ProposalResult(marginal, oid, log_q)

        _, values, slots, inner_q = payloads[i]
        obj = table.new_object()
        obj.values.update(values)
        obj.values.update(slots)
        obj.ref_count = 1
        table.register(obj, cls.hash_keys)
        self._log(f"{cls_name}: created #{obj.oid}")
        return ProposalResult(marginal, obj.oid, log_q + inner_q)

    def _propose_fresh(self, cls: PCleanClass, obs: ObsNode):
        """Draw a brand-new object's reference slots and attributes, given the data.

        Reference slots are *not* resolved up front. They are vertices in the plan like
        any other, so that each candidate target is scored against the attributes that
        depend on it -- see `_enumerate_reference`.
        """
        assignment: Dict[VertexID, Any] = {}
        slots: Dict[VertexID, int] = {}
        plan = cls.plans[0] if cls.plans else _default_plan(cls)
        m, q = self._walk_plan(cls, plan, obs, assignment, slots)
        return m, assignment, slots, q

    def _walk_plan(self, cls, plan: Plan, obs: ObsNode,
                   assignment: Dict[VertexID, Any], slots) -> Tuple[float, float]:
        """Walk one level of the conditional-independence forest.

        **The three lines that matter are the `+=` below.** Sibling steps are, by the
        Plan's construction invariant, conditionally independent given their common
        ancestors. So their log-marginals *add*. Nothing here ever forms the joint
        cross-product of two independent variables -- that is the entire "exponential
        savings over naive enumeration" of [§3.2], and it is why the Julia's generated
        code has sequential rather than nested loops.
        """
        total_m, total_q = 0.0, 0.0
        for step in plan.steps:
            m, q = self._walk_step(cls, step, obs, assignment, slots)
            total_m += m
            total_q += q
        return total_m, total_q

    def _walk_step(self, cls, step, obs, assignment, slots) -> Tuple[float, float]:
        v = step.idx
        node = cls.nodes[v]
        name = cls.node_name(v)

        if isinstance(node, ParameterNode):
            return 0.0, 0.0     # hyperparameters are not sampled here

        if isinstance(node, ForeignKeyNode):
            return self._enumerate_reference(cls, step, node, obs, assignment, slots)

        args = self._resolve_args(cls, node, assignment, slots)
        if args is None:
            return 0.0, 0.0     # not computable yet; the Julia skips these too

        if isinstance(node, JuliaNode):
            assignment[v] = node.f(*args)
            return self._walk_plan(cls, step.rest, obs, assignment, slots)

        assert isinstance(node, RandomChoiceNode)
        observed = obs.attrs.get(name, None)
        is_observed = name in obs.attrs and observed is not None

        # ---- Case 1: the value is right there in the data. Just score it. --------
        if is_observed:
            assignment[v] = observed
            lp = node.dist.logdensity(observed, *args)
            m, q = self._walk_plan(cls, step.rest, obs, assignment, slots)
            return lp + m, q

        # ---- Case 2: latent and enumerable. This is the whole point. -------------
        if node.dist.has_discrete_proposal():
            options, log_priors = node.dist.discrete_proposal(*args)
            scores, chosen_vals, sub_qs, sub_assigns = [], [], [], []
            for opt, lp in zip(options, log_priors):
                if opt is DUMMY:
                    # Nothing in the shortlist; fall back to the prior [§3.3]. The DUMMY
                    # branch is what keeps the `preferring` hint a *proposal* hint and
                    # not a change to the model.
                    opt = node.dist.discrete_proposal_dummy_value(*args)
                assignment[v] = opt
                scratch = dict(assignment)
                m, q = self._walk_plan(cls, step.rest, obs, scratch, slots)
                scores.append(lp + m)
                chosen_vals.append(opt)
                sub_qs.append(q)
                sub_assigns.append(scratch)

            marginal = logsumexp(scores)
            if marginal == -math.inf:
                assignment[v] = chosen_vals[0]
                return -math.inf, 0.0
            i = _sample_index(scores, marginal)
            assignment.clear()
            assignment.update(sub_assigns[i])
            assignment[v] = chosen_vals[i]
            self._log(f"{name}: enumerated {len(options)} -> {chosen_vals[i]!r}")
            # Marginal over all candidates is the evidence this subproblem contributes;
            # log_q is how likely we were to pick the one we did. Their difference is
            # what makes the SMC weight correct.
            return marginal, (scores[i] - marginal) + sub_qs[i]

        # ---- Case 3: latent and not enumerable. Sample the prior. ----------------
        # Continuous attributes land here [App. D.2]. The Julia calls this
        # `propose_non_enumerable!` and notes the proposals are poor, which is why it
        # offers Particle Gibbs rejuvenation to compensate.
        val = node.dist.random(*args)
        assignment[v] = val
        lp = node.dist.logdensity(val, *args)
        m, q = self._walk_plan(cls, step.rest, obs, assignment, slots)
        return lp + m, q + lp

    def _enumerate_reference(self, cls, step, node: ForeignKeyNode, obs, assignment,
                             slots) -> Tuple[float, float]:
        """Choose what a reference slot points at, scored by the data that depends on it.

        **This is the method that earns the Julia's flattening.** The naive thing --
        which this package did at first, and got badly wrong -- is to resolve a reference
        slot first and enumerate the dependent attributes afterwards. But then the choice
        of target is made *blind*: when a Record picks its City, the only evidence about
        which city it is (`obs_city ~ AddTypos(hosp.city.name)`) has not been looked at
        yet, so the choice collapses onto whatever the CRP likes best -- the most popular
        city, every time.

        The fix is to enumerate the slot *jointly* with everything downstream of it. For
        each candidate target we bind the slot, walk the rest of the plan (which scores
        the dirty spelling against that candidate's clean name), and only then normalise.

        That is exactly what the Julia buys by copying every reachable node into the
        referring class's graph: once `City.name` is a vertex of `Record`, choosing the
        city and scoring the typo are the same enumeration, automatically.

        NOTE (limitation): for the "create a new object" branch we enumerate the target's
        own enumerable attributes jointly with the remainder of this plan, but we do not
        recurse further into *its* reference slots -- those are resolved afterwards from
        the prior. The Julia's flattening has no such cutoff. For models one level deep
        (the example here) the two agree.
        """
        target_cls_name = node.target_class
        target_cls = self.model.classes[target_cls_name]
        table = self.trace.table(target_cls_name)
        child_obs = obs.children.get(node.name, ObsNode())

        # -- exact-match key lookup first [App. D.4] ----------------------------
        if target_cls.hash_keys:
            key, ok = [], True
            for v in target_cls.hash_keys:
                nm = target_cls.node_name(v)
                if nm in child_obs.attrs and child_obs.attrs[nm] is not None:
                    key.append(child_obs.attrs[nm])
                else:
                    ok = False
                    break
            if ok:
                existing = table.lookup(tuple(key))
                if existing is not None:
                    existing.ref_count += 1
                    slots[step.idx] = existing.oid
                    m, q = self._walk_plan(cls, step.rest, obs, assignment, slots)
                    return m, q
                self._depth += 1
                res = self.propose_object(target_cls_name, child_obs, force_new=True)
                self._depth -= 1
                slots[step.idx] = res.oid
                m, q = self._walk_plan(cls, step.rest, obs, assignment, slots)
                return res.log_marginal + m, res.log_q + q

        # -- otherwise enumerate: every existing object, plus "a new one" --------
        oids, logw = crp_candidate_log_weights(table, target_cls.py, list(table.objects))

        scores, payloads = [], []
        for oid, lw in zip(oids, logw):
            if oid is not None:
                slots[step.idx] = oid
                scratch = dict(assignment)
                m, q = self._walk_plan(cls, step.rest, obs, scratch, slots)
                scores.append(lw + m)
                payloads.append((oid, None, scratch, q))
            else:
                # The "new object" branch, expanded over the new object's own enumerable
                # attributes so the downstream likelihood can see them.
                for values, vlp, scratch, q, m in self._new_target_options(
                        cls, step, target_cls, target_cls_name, child_obs,
                        obs, assignment, slots):
                    scores.append(lw + vlp + m)
                    payloads.append((None, values, scratch, q))

        if not scores:
            return 0.0, 0.0
        marginal = logsumexp(scores)
        if marginal == -math.inf:
            return -math.inf, 0.0
        i = _sample_index(scores, marginal)
        oid, values, scratch, sub_q = payloads[i]

        if oid is None:
            obj = table.new_object()
            obj.values.update(values)
            obj.ref_count = 1
            table.register(obj, target_cls.hash_keys)
            oid = obj.oid
        else:
            table.objects[oid].ref_count += 1

        slots[step.idx] = oid
        assignment.clear()
        assignment.update(scratch)
        return marginal, (scores[i] - marginal) + sub_q

    def _new_target_options(self, cls, step, target_cls, target_cls_name, child_obs,
                            obs, assignment, slots):
        """Candidate fresh target objects, each paired with the rest-of-plan score.

        Enumerates the target's own enumerable attributes (for a City, its `name`) and,
        for each setting, walks the remainder of the *referring* class's plan so the
        dirty observation gets scored against that hypothetical clean value.
        """
        enumerable = [v for v, n in enumerate(target_cls.nodes)
                      if isinstance(n, RandomChoiceNode) and n.dist.has_discrete_proposal()]
        base_values: Dict[VertexID, Any] = {}
        # Anything directly observed on the target is pinned, not enumerated.
        for v, n in enumerate(target_cls.nodes):
            nm = target_cls.node_name(v)
            if nm in child_obs.attrs and child_obs.attrs[nm] is not None:
                base_values[v] = child_obs.attrs[nm]

        todo = [v for v in enumerate_first(enumerable) if v not in base_values]
        if not todo:
            tmp = self.trace.table(target_cls_name).new_object()
            tmp.values.update(base_values)
            slots[step.idx] = tmp.oid
            scratch = dict(assignment)
            m, q = self._walk_plan(cls, step.rest, obs, scratch, slots)
            del self.trace.table(target_cls_name).objects[tmp.oid]
            yield base_values, 0.0, scratch, q, m
            return

        v = todo[0]
        tnode = target_cls.nodes[v]
        targs = self._resolve_args(target_cls, tnode, base_values, base_values)
        if targs is None:
            targs = []
        options, log_priors = tnode.dist.discrete_proposal(*targs)
        table = self.trace.table(target_cls_name)
        for opt, lp in zip(options, log_priors):
            if opt is DUMMY:
                opt = tnode.dist.discrete_proposal_dummy_value(*targs)
            vals = dict(base_values)
            vals[v] = opt
            tmp = table.new_object()
            tmp.values.update(vals)
            slots[step.idx] = tmp.oid
            scratch = dict(assignment)
            m, q = self._walk_plan(cls, step.rest, obs, scratch, slots)
            del table.objects[tmp.oid]
            yield vals, lp, scratch, q, m

    def _score_existing(self, cls, obj: LatentObject, obs: ObsNode) -> float:
        """How well does an object we already hypothesized explain this row?

        Only observed cells are scored -- an existing object's latent attributes are
        already fixed, and revising them is rejuvenation's job, not the proposal's.
        """
        total = 0.0
        for v, node in enumerate(cls.nodes):
            if not isinstance(node, RandomChoiceNode):
                continue
            name = cls.node_name(v)
            if name not in obs.attrs or obs.attrs[name] is None:
                continue
            args = self._resolve_args(cls, node, obj.values, obj.values)
            if args is None:
                continue
            total += node.dist.logdensity(obs.attrs[name], *args)

        # ...and recursively, the objects this one points at.
        for v, node in enumerate(cls.nodes):
            if isinstance(node, ForeignKeyNode) and v in obj.values:
                child_obs = obs.children.get(node.name)
                if child_obs is None:
                    continue
                target = self.trace.table(node.target_class).objects.get(obj.values[v])
                if target is not None:
                    total += self._score_existing(self.model.classes[node.target_class],
                                                  target, child_obs)
        return total


# ------------------------------------------------------------------- helpers

def _vertex_by_name(cls: PCleanClass, name: str) -> Optional[VertexID]:
    for v in range(len(cls.nodes)):
        if cls.node_name(v) == name:
            return v
    return None


def _default_plan(cls: PCleanClass) -> Plan:
    from .model import make_plan
    vertices = [v for v, n in enumerate(cls.nodes) if not isinstance(n, ParameterNode)]
    return make_plan(cls, vertices)


def _sample_index(scores: List[float], marginal: float) -> int:
    """Sample proportionally to exp(score - marginal). Mirrors the Categorical draw at
    the end of every enumeration block in the generated Julia."""
    r = _random.random()
    acc = 0.0
    for i, s in enumerate(scores):
        acc += math.exp(s - marginal)
        if r <= acc:
            return i
    return len(scores) - 1
