"""Sequential Monte Carlo with object-wise rejuvenation — mirrors
``src/inference/row_inference.jl`` and ``inference.jl`` [§3.1].

## Why SMC at all

PClean's model has a **sequential representation**: the latent database can be built one
observed row at a time. Row *i* contributes a "database increment" -- the objects it
refers to that no previous row did [§3.1, Fig. 5]. That turns a static posterior over a
whole database into a sequence of targets, and a sequence of targets is exactly what
SMC consumes.

The paper's restaurant metaphor is worth holding onto: each class is a restaurant, each
table a distinct entity, and a customer wanting a *new* table must first send friends to
the other restaurants -- one per reference slot -- before choosing their dish. Rows
arrive, customers sit down, and the seating chart is the latent database.

## Why rejuvenation is not optional

SMC can only ever *extend* a hypothesis. It commits to what a row means when it sees it,
with no way to revise that in light of row 400. For data cleaning this is fatal on its
own: the whole point of the Abington example is that you cannot tell the misspelling from
the truth until you have seen the practice's other rows.

Rejuvenation fixes it. After (or during) the sweep, revisit each object and re-enumerate
its attributes against **all** the data that references it [§3.1]. That is when
"Abingdon" wins: at that moment the City object is scored against all 152 mentions at
once, and the model's claim that typos are systematic per-practice finally has the
evidence it needs to act on.

Rejuvenation can also delete objects nothing points at any more ("garbage collection")
and create new ones. Here we implement the attribute update, which is the part that does
the cleaning.
"""

import math
import random as _random
from typing import Any, Dict, List, Optional

from .distributions import DUMMY, ProbParameter, ProportionsParameter, logsumexp
from .model import (ForeignKeyNode, JuliaNode, ParameterNode, PCleanModel,
                    RandomChoiceNode, Ref, Via)
from .proposal import ObsNode, Proposer, _sample_index, _vertex_by_name
from .trace import Trace


class SMC:
    """Per-observation SMC over a PClean model."""

    def __init__(self, model: PCleanModel, n_particles: int = 2, verbose: bool = False):
        self.model = model
        self.n_particles = n_particles
        self.verbose = verbose
        self.particles: List[Trace] = []

    # -------------------------------------------------------------- the sweep

    def run(self, rows: List[ObsNode], progress_every: int = 100) -> Trace:
        """Incorporate every row, then return the surviving particle.

        Also records, per particle, which object each row became -- rejuvenation needs
        it to walk evidence backwards from an entity to the rows that mention it.
        """
        obs_cls = self.model.observation_class
        self.particles = [Trace(self.model) for _ in range(self.n_particles)]
        assignments: List[List[int]] = [[] for _ in self.particles]

        for i, row in enumerate(rows):
            for pi, p in enumerate(self.particles):
                proposer = Proposer(self.model, p, verbose=self.verbose)
                res = proposer.propose_object(obs_cls, row)
                assignments[pi].append(res.oid)
                # Weight update. When every variable in the subproblem is finite-discrete
                # and fully enumerated, the locally-optimal proposal's normalising
                # constant *is* the incremental evidence, and the q terms cancel exactly
                # [§3.2]. With DUMMY fallbacks or continuous attributes present this
                # becomes approximate; the Julia tracks p and q separately to stay exact
                # in the general case.
                p.log_weight += res.log_marginal

            # Resample when the particle cloud has collapsed onto one hypothesis.
            if self.n_particles > 1 and i < len(rows) - 1:
                before = list(range(len(self.particles)))
                self._resample_map = before
                self._maybe_resample()
                # Row->object bookkeeping has to follow the particles through resampling.
                assignments = [list(assignments[i]) for i in self._resample_map]

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  SMC: row {i+1}/{len(rows)}  [{self.particles[0].summary()}]")

        best = max(range(len(self.particles)), key=lambda i: self.particles[i].log_weight)
        self.row_assignments = assignments[best]
        return self.particles[best]

    def _effective_sample_size(self, logws: List[float]) -> float:
        m = logsumexp(logws)
        if m == -math.inf:
            return 1.0
        ws = [math.exp(w - m) for w in logws]
        return (sum(ws) ** 2) / sum(w * w for w in ws)

    def _maybe_resample(self) -> None:
        logws = [p.log_weight for p in self.particles]
        if self._effective_sample_size(logws) >= self.n_particles / 2:
            return
        m = logsumexp(logws)
        probs = [math.exp(w - m) for w in logws]
        chosen = _random.choices(range(len(self.particles)), weights=probs,
                                 k=self.n_particles)
        # Cull the low-weight hypotheses and clone the promising ones; reset weights,
        # since after resampling every particle is equally plausible.
        self.particles = [self.particles[i].fork() for i in chosen]
        self._resample_map = chosen
        for p in self.particles:
            p.log_weight = 0.0


# ------------------------------------------------------------------ rejuvenation

class Rejuvenator:
    """Object-wise blocked Gibbs — mirrors the rejuvenation half of
    ``block_proposal.jl`` and [App. D.1].

    "Blocked" matters. Updating one variable at a time gets stuck: a city's clean name
    and the spellings that depend on it are tightly correlated, so no single-variable
    move can escape a bad mode. Updating all of one object's attributes together, against
    all of its evidence at once, can.

    ## Scoring per *object*, not per *row* — the thing that makes Figure 1 work

    This is the subtlest point in the whole system, and the easiest to get backwards.

    When we rescore a City's candidate names, the evidence is the `bad_city` spelling on
    each **Practice** that points at it. A Practice has exactly one `bad_city`, however
    many rows mention that practice. So each practice must contribute its spelling
    **once**.

    Score per row instead -- the obvious implementation -- and a practice with 152 rows
    contributes its one typo 152 times. You have then rebuilt majority voting inside a
    Bayesian model and will confidently conclude that the misspelling is correct, which
    is precisely the failure the paper is about.

    The asymmetry is worth seeing in numbers. With one practice spelling it 'abington'
    across 152 rows and seven practices spelling it 'abingdon' across 42:

        per row     'abington': 152 exact + 42 typos    -> wrong spelling wins easily
        per object  'abington':   1 exact +  7 typos    -> right spelling wins by ~31 nats

    The data is identical. Only the counting differs, and the counting follows from where
    the error model sits in the schema.
    """

    def __init__(self, model: PCleanModel, trace: Trace, rows: List[ObsNode],
                 row_assignments: List[int]):
        self.model = model
        self.trace = trace
        self.rows = rows
        self.row_assignments = row_assignments
        # A representative observation slice per object. The model asserts an object's
        # attributes are a single draw, so any row that reaches it reports the same
        # values; we keep the first.
        self._obj_obs: Dict[tuple, ObsNode] = {}
        # (cls, oid) -> the objects that hold a reference slot pointing at it.
        self._referrers: Dict[tuple, set] = {}
        self._build_backpointers()

    def _build_backpointers(self):
        obs_cls = self.model.observation_class

        def walk(cls_name, oid, obs: ObsNode, parent):
            key = (cls_name, oid)
            self._obj_obs.setdefault(key, obs)
            if parent is not None:
                self._referrers.setdefault(key, set()).add(parent)
            cls = self.model.classes[cls_name]
            obj = self.trace.table(cls_name).objects.get(oid)
            if obj is None:
                return
            for v, node in enumerate(cls.nodes):
                if isinstance(node, ForeignKeyNode) and v in obj.values:
                    child = obs.children.get(node.name)
                    if child is not None:
                        walk(node.target_class, obj.values[v], child, key)

        for row, oid in zip(self.rows, self.row_assignments):
            if oid is not None:
                walk(obs_cls, oid, row, None)

    def sweep(self, verbose: bool = False) -> int:
        """One pass over every object, re-enumerating its latent attributes.

        Reverse topological order, so a Record's City is settled before the Record is
        revisited [App. D.1]. Returns how many values actually changed -- watch it fall
        to zero, which is the chain telling you it has stopped finding improvements.
        """
        changed = 0
        for cls_name in reversed(self.model.topological_class_order()):
            cls = self.model.classes[cls_name]
            for oid, obj in list(self.trace.table(cls_name).objects.items()):
                if (cls_name, oid) not in self._obj_obs:
                    continue
                for v, node in enumerate(cls.nodes):
                    if not isinstance(node, RandomChoiceNode):
                        continue
                    if v in cls.hash_keys:
                        continue          # guaranteed fields are trusted, never revised
                    if not node.dist.has_discrete_proposal():
                        continue
                    if self._update_attribute(cls, cls_name, obj, v, node, verbose):
                        changed += 1
        return changed

    def _score_local(self, cls, cls_name, obj) -> float:
        """Likelihood of one object's own observed attributes. Counted once."""
        obs = self._obj_obs.get((cls_name, obj.oid))
        if obs is None:
            return 0.0
        proposer = Proposer(self.model, self.trace)
        total = 0.0
        for v, node in enumerate(cls.nodes):
            if not isinstance(node, RandomChoiceNode):
                continue
            name = cls.node_name(v)
            if name not in obs.attrs or obs.attrs[name] is None:
                continue
            args = proposer._resolve_args(cls, node, obj.values, obj.values)
            if args is None:
                continue
            total += node.dist.logdensity(obs.attrs[name], *args)
        return total

    def _update_attribute(self, cls, cls_name, obj, v, node, verbose) -> bool:
        """Re-draw one attribute from its full conditional.

        The conditional needs everything that depends on this attribute: the object's own
        observed cells, plus the observed cells of every object holding a reference slot
        pointing at it (a Practice's dirty `bad_city` depends on its City's clean `name`).
        Each such object is scored exactly once -- see the class docstring.

        NOTE (divergence): the Julia erases exactly the affected slice of the database
        (`R_minus_r` in [App. D.1]) and recomputes only the likelihood terms that changed.
        We recompute the local scores of the object and its direct referrers. Same
        distribution, since untouched terms are constant across candidates and cancel in
        the normalisation.
        """
        proposer = Proposer(self.model, self.trace)
        args = proposer._resolve_args(cls, node, obj.values, obj.values)
        if args is None:
            return False
        options, log_priors = node.dist.discrete_proposal(*args)
        if len(options) <= 1:
            return False

        referrers = self._referrers.get((cls_name, obj.oid), set())
        old = obj.values.get(v)

        scores, cand_values = [], []
        for opt, lp in zip(options, log_priors):
            if opt is DUMMY:
                opt = node.dist.discrete_proposal_dummy_value(*args)
            obj.values[v] = opt
            score = lp + self._score_local(cls, cls_name, obj)
            for (rcls_name, roid) in referrers:
                robj = self.trace.table(rcls_name).objects.get(roid)
                if robj is not None:
                    score += self._score_local(self.model.classes[rcls_name],
                                               rcls_name, robj)
            scores.append(score)
            cand_values.append(opt)

        obj.values[v] = old
        marginal = logsumexp(scores)
        if marginal == -math.inf:
            return False
        i = _sample_index(scores, marginal)
        new = cand_values[i]
        obj.values[v] = new
        if verbose and new != old:
            print(f"    rejuv {cls_name}#{obj.oid}.{cls.node_name(v)}: "
                  f"{old!r} -> {new!r}")
        return new != old
