"""Flat enumeration over a flattened model — the faithful version of
``src/inference/proposal_compiler.jl`` [§3.2, Algorithm 1].

## What changed, and why it matters

The earlier `proposal.py` walked classes *recursively*: to propose a Record it proposed a
Practice, which proposed a City. That is easy to read and subtly wrong, because a slot
gets chosen before the data bearing on it is scored. It needed a special case
(`_enumerate_reference`) to paper over, and even then nested new objects fell back to the
prior.

With flattening there is nothing to paper over. `Record`'s graph already contains
`practice`, `practice.city`, `practice.city.name` and `practice.bad_city` as ordinary
vertices in one topological order. So the walk is a single pass over one Bayes net, and
choosing the city is scored against the typo *by construction* -- they are adjacent
vertices, not separate recursions.

## Algorithm 1, in three cases

Every flattened vertex `K.X` has a conditional distribution that switches on what its
slot `K` turned out to be. The paper writes it as:

    φ(v_u | ...) =  1[v_u = v_{K.X}]     if v_K is an existing object
                    φ_prior(...)          if v_K is a brand new object
                    1[v_u = v_{u'.X}]     if v_K is the same new object as slot u'

In code, `_step`:

  * **slot points at an existing object** -- the attribute is *determined*. Look it up in
    the database and score nothing: no freedom, no likelihood term.
  * **slot creates a new object** -- enumerate the attribute normally, from its own prior
    and against its own observations.
  * **slot shares another slot's new object** -- copy that slot's value.

A reference slot's own domain is every existing object of the target class (weighted by
the CRP) plus `NewObject`. Because slot and attributes sit in one enumeration, the
marginal that comes out is the locally optimal proposal of [§3.2], not an approximation
of it.
"""

import math
import random as _random
from typing import Any, Dict, List, Optional, Tuple

from .distributions import DUMMY, logsumexp
from .model import (ForeignKeyNode, JuliaNode, NewObject, ParameterNode, ParamLookup,
                    PCleanClass, PCleanModel, Plan, RandomChoiceNode, Ref,
                    SubmodelNode, VertexID, strip_submodel)
from .structure_prior import crp_candidate_log_weights
from .trace import LatentObject, Trace


class RowResult:
    """What proposing one row yields."""

    def __init__(self, log_marginal: float, oid: Optional[int], log_q: float,
                 assignment: Dict[VertexID, Any]):
        self.log_marginal = log_marginal
        self.oid = oid
        self.log_q = log_q
        self.assignment = assignment


class FlatProposer:
    """Enumerative proposal over a flattened class."""

    def __init__(self, model: PCleanModel, trace: Trace):
        self.model = model
        self.trace = trace

    # ------------------------------------------------------------------ helpers

    def _args(self, cls: PCleanClass, node, assignment) -> Optional[List[Any]]:
        """Resolve a node's arguments. After flattening every reference is local."""
        out = []
        for a in node.arg_node_ids:
            if isinstance(a, ParamLookup):
                # Flattening wraps the ParameterNode in a SubmodelNode, but `shift_node`
                # deliberately does NOT copy the parameter itself -- every class that
                # absorbs it shares one object, which is the point of a learned
                # parameter. So peel the wrapper and use the shared instance.
                param = strip_submodel(cls.nodes[a.param]).param
                if a.key is None:
                    out.append(param)
                else:
                    sub = self._args(cls, _One([a.key]), assignment)
                    if sub is None or sub[0] is None:
                        return None
                    out.append(param[sub[0]])
            elif isinstance(a, Ref):
                val = assignment.get(a.v)
                if val is None and "." in cls.node_name(a.v):
                    # A flattened vertex we are not carrying in this assignment -- it
                    # belongs to another object, so go and read it there. This is the
                    # single-source-of-truth rule: an attribute lives on the object that
                    # owns it, and nowhere else. Cache a copy on every referring object
                    # and rejuvenation will silently score against a stale value.
                    val, ok = self._determined(cls, a.v, assignment)
                    if not ok:
                        return None
                if val is None:
                    return None
                out.append(val)
            else:
                out.append(a)
        return out

    def _owner_slot(self, cls: PCleanClass, v: VertexID):
        """The slot that immediately owns a flattened vertex, and its id over there.

        Peels the `SubmodelNode` wrappers down to the innermost one. Its
        `foreign_key_node_id` is the *nearest* enclosing reference slot, and `subnode_id`
        is this attribute's vertex in that slot's target class.

        Nearest matters. `practice.city.name` is wrapped twice: the outer wrapper points
        at the `practice` slot, the inner at `practice.city`. The value is determined as
        soon as **`practice.city`** resolves to an existing City -- it does not matter
        whether the Practice itself is new. Keying off the outer wrapper instead makes
        every city look undetermined whenever the practice is fresh, and the CRP then
        collapses every practice onto one city with nothing to discriminate them.
        """
        node = cls.nodes[v]
        last = None
        while isinstance(node, SubmodelNode):
            last = node
            node = node.subnode
        if last is None:
            return None
        return last.foreign_key_node_id, last.subnode_id

    def _determined(self, cls: PCleanClass, v: VertexID, values):
        """Algorithm 1's first branch: the slot points at an object that already exists,
        so this attribute has no freedom -- read it off that object.

        Returns ``(value, True)`` when determined, ``(None, False)`` when the owning slot
        is creating a new object and the attribute must actually be enumerated.

        Reading it off the owner (rather than caching a copy on every referring object) is
        deliberate: one source of truth. Cache copies and rejuvenation will happily score
        candidates against a stale value, every candidate will score the same, and the
        prior will decide -- which looks exactly like inference working, and is not.
        """
        own = self._owner_slot(cls, v)
        if own is None:
            return None, False
        fkv, sub_id = own
        val = values.get(fkv)
        if val is None:
            # The owning slot is itself a flattened vertex (`practice.city` inside
            # `practice.city.name`), so resolve it first. The recursion walks the chain
            # one slot at a time, which is what makes arbitrary depth work:
            # Record -> practice -> city -> name, with only `practice` stored locally.
            val, ok = self._determined(cls, fkv, values)
            if not ok:
                return None, False
        if val is None or isinstance(val, NewObject):
            return None, False
        tcls_name = strip_submodel(cls.nodes[fkv]).target_class
        obj = self.trace.table(tcls_name).objects.get(val)
        if obj is None or sub_id not in obj.values:
            return None, False
        return obj.values[sub_id], True

    # --------------------------------------------------------------------- walk

    def propose(self, cls_name: str, obs: Dict[VertexID, Any]) -> RowResult:
        cls = self.model.classes[cls_name]
        plan = cls.plans[0]
        assignment: Dict[VertexID, Any] = {}
        m, q = self._walk(cls, plan, obs, assignment)
        oid = self._materialize(cls, cls_name, assignment, obs)
        return RowResult(m, oid, q, assignment)

    def _walk(self, cls, plan: Plan, obs, assignment) -> Tuple[float, float]:
        """Sibling steps are conditionally independent, so their marginals add."""
        tm = tq = 0.0
        for step in plan.steps:
            m, q = self._step(cls, step, obs, assignment)
            tm += m
            tq += q
        return tm, tq

    def _step(self, cls, step, obs, assignment) -> Tuple[float, float]:
        v = step.idx
        node = cls.nodes[v]
        inner = strip_submodel(node)

        if isinstance(inner, ParameterNode):
            return 0.0, 0.0

        # --- Case: this vertex belongs to a referenced object ---------------------
        if isinstance(node, SubmodelNode):
            determined, ok = self._determined(cls, v, assignment)
            if ok:
                # Algorithm 1, first branch: the slot points at an object that already
                # exists, so this attribute has no freedom. Bind and move on -- crucially,
                # score nothing, because nothing was chosen.
                assignment[v] = determined
                return self._walk(cls, step.rest, obs, assignment)
            # Otherwise the owning slot is creating a new object, so fall through and
            # enumerate this attribute for real (Algorithm 1, second branch).

        if isinstance(inner, ForeignKeyNode):
            return self._enumerate_slot(cls, step, inner, obs, assignment)

        args = self._args(cls, inner, assignment)
        if args is None:
            return 0.0, 0.0

        if isinstance(inner, JuliaNode):
            assignment[v] = inner.f(*args)
            return self._walk(cls, step.rest, obs, assignment)

        observed = obs.get(v)
        if observed is not None:
            assignment[v] = observed
            lp = inner.dist.logdensity(observed, *args)
            m, q = self._walk(cls, step.rest, obs, assignment)
            return lp + m, q

        if inner.dist.has_discrete_proposal():
            options, priors = inner.dist.discrete_proposal(*args)
            scores, vals, subs, qs = [], [], [], []
            for opt, lp in zip(options, priors):
                if opt is DUMMY:
                    opt = inner.dist.discrete_proposal_dummy_value(*args)
                assignment[v] = opt
                scratch = dict(assignment)
                m, q = self._walk(cls, step.rest, obs, scratch)
                scores.append(lp + m)
                vals.append(opt)
                subs.append(scratch)
                qs.append(q)
            marginal = logsumexp(scores)
            if marginal == -math.inf:
                assignment[v] = vals[0]
                return -math.inf, 0.0
            i = _sample(scores, marginal)
            assignment.clear()
            assignment.update(subs[i])
            assignment[v] = vals[i]
            return marginal, (scores[i] - marginal) + qs[i]

        # Not enumerable (continuous attributes) -- sample the prior [App. D.2].
        val = inner.dist.random(*args)
        assignment[v] = val
        lp = inner.dist.logdensity(val, *args)
        m, q = self._walk(cls, step.rest, obs, assignment)
        return lp + m, q + lp

    def _enumerate_slot(self, cls, step, fk: ForeignKeyNode, obs, assignment):
        """Choose what a reference slot points at, scored by everything downstream.

        Domain: every existing object of the target class, weighted by the CRP, plus
        `NewObject`. For each candidate we bind the slot and walk the *rest of the plan*
        -- which, thanks to flattening, contains the attributes and the observations that
        depend on this choice. No special case, no recursion into another class: the
        dependent vertices are simply later in the same topological order.
        """
        v = step.idx
        table = self.trace.table(fk.target_class)
        target_cls = self.model.classes[fk.target_class]

        # Exact-match key lookup first [App. D.4] -- identity settled, no enumeration.
        key = self._hash_key(cls, v, target_cls, obs)
        if key is not None:
            hit = table.lookup(key)
            if hit is not None:
                assignment[v] = hit.oid
                m, q = self._walk(cls, step.rest, obs, assignment)
                return m, q
            assignment[v] = NewObject(v)
            m, q = self._walk(cls, step.rest, obs, assignment)
            return m, q

        oids, logw = crp_candidate_log_weights(table, target_cls.py, list(table.objects))
        scores, cands, subs, qs = [], [], [], []
        for oid, lw in zip(oids, logw):
            assignment[v] = NewObject(v) if oid is None else oid
            scratch = dict(assignment)
            m, q = self._walk(cls, step.rest, obs, scratch)
            scores.append(lw + m)
            cands.append(assignment[v])
            subs.append(scratch)
            qs.append(q)

        marginal = logsumexp(scores)
        if marginal == -math.inf:
            assignment[v] = cands[-1]
            return -math.inf, 0.0
        i = _sample(scores, marginal)
        assignment.clear()
        assignment.update(subs[i])
        assignment[v] = cands[i]
        return marginal, (scores[i] - marginal) + qs[i]

    def _hash_key(self, cls, slot_v, target_cls, obs):
        """The guaranteed-field key for the object a slot points at, if fully observed."""
        if not target_cls.hash_keys:
            return None
        prefix = cls.node_name(slot_v)
        key = []
        for hv in target_cls.hash_keys:
            local = cls.names.get(f"{prefix}.{target_cls.node_name(hv)}")
            if local is None or obs.get(local) is None:
                return None
            key.append(obs[local])
        return tuple(key)

    # ------------------------------------------------------------- materialize

    def _materialize(self, cls, cls_name, assignment, obs) -> Optional[int]:
        """Turn the chosen assignment into real objects, deepest slot first.

        Everything up to here was arithmetic over candidate values. This is where the
        latent database actually grows: each slot that chose `NewObject` becomes a row in
        its class's table, filled from the flattened vertices that belong to it.
        """
        created: Dict[VertexID, int] = {}

        slots = [v for v in range(len(cls.nodes))
                 if isinstance(strip_submodel(cls.nodes[v]), ForeignKeyNode)]
        # Deepest first, so a parent's slot value is available when we build it.
        slots.sort(key=lambda v: -cls.node_name(v).count("."))

        for v in slots:
            val = assignment.get(v)
            if not isinstance(val, NewObject):
                if val is not None:
                    obj = self.trace.table(
                        strip_submodel(cls.nodes[v]).target_class).objects.get(val)
                    if obj is not None:
                        obj.ref_count += 1
                continue
            fk = strip_submodel(cls.nodes[v])
            tcls = self.model.classes[fk.target_class]
            table = self.trace.table(fk.target_class)
            obj = table.new_object()
            obj.ref_count = 1
            prefix = cls.node_name(v)
            for tv in range(len(tcls.nodes)):
                # Only this object's OWN vertices. A dotted name belongs to some object
                # further down the chain, which stores it itself.
                if "." in tcls.node_name(tv):
                    continue
                local = cls.names.get(f"{prefix}.{tcls.node_name(tv)}")
                if local is None:
                    continue
                lv = assignment.get(local)
                if isinstance(lv, NewObject):
                    lv = created.get(local)
                if lv is not None:
                    obj.values[tv] = lv
            table.register(obj, tcls.hash_keys)
            created[v] = obj.oid
            assignment[v] = obj.oid

        # Finally the observation object itself.
        table = self.trace.table(cls_name)
        root = table.new_object()
        root.ref_count = 1
        for v in range(len(cls.nodes)):
            if "." in cls.node_name(v):
                continue
            val = assignment.get(v)
            if isinstance(val, NewObject):
                val = created.get(v)
            if val is not None:
                root.values[v] = val
        table.register(root, cls.hash_keys)
        return root.oid


    def owner_map(self, cls, assignment) -> Dict[VertexID, tuple]:
        """Map each flattened vertex to the object that actually owns it.

        After materialization `practice.bad_city` lives on a Practice row, not on the
        Record. Rejuvenation needs that correspondence to count each object's evidence
        **once** rather than once per row mentioning it -- which is the difference between
        Bayesian cleaning and majority voting.
        """
        out: Dict[VertexID, tuple] = {}
        for v in range(len(cls.nodes)):
            own = self._owner_slot(cls, v)
            if own is None:
                continue
            fkv, sub_id = own
            oid = assignment.get(fkv)
            if oid is None or isinstance(oid, NewObject):
                continue
            tcls_name = strip_submodel(cls.nodes[fkv]).target_class
            if self.trace.table(tcls_name).objects.get(oid) is None:
                continue
            out[v] = (tcls_name, oid, sub_id)
        return out


class _One:
    def __init__(self, args):
        self.arg_node_ids = args


def _sample(scores: List[float], marginal: float) -> int:
    r = _random.random()
    acc = 0.0
    for i, s in enumerate(scores):
        acc += math.exp(s - marginal)
        if r <= acc:
            return i
    return len(scores) - 1
