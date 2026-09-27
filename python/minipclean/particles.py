"""Multi-particle SMC over latent databases — the textbook form of [§3.1].

`smc.SMC` runs one particle: a single latent database, extended row by row, with
rejuvenation doing all the correcting afterwards. This module runs **N** of them, which is
what the sequential form of the model [§3.1, Fig. 5] invites:

    target_i  =  p(latent database | first i rows)

Each particle is a complete partial database -- its own unrolled Bayes net. Row i extends
every particle with the locally optimal proposal of [§3.2] (`FlatProposer`), and the
incremental importance weight is exactly the marginal that proposal already computes: the
probability, under that particle's database, of the row it was just asked to explain.
Particles that explained the data badly lose weight; when the weights get too uneven
(effective sample size below a threshold) we resample, cloning the good databases over the
bad ones.

NOTE (divergence): the Julia does not do this. `row_inference.jl` keeps **one** database
and spends its particles *inside* a row -- N copies of the row's block-by-block proposal,
resampled between subproblem blocks, one committed at the end -- and rejuvenates with
conditional SMC (particle Gibbs) per row. That is cheaper at scale because nothing but a
row is ever copied. The form here copies whole databases (see `Trace.fork`), which is
hopeless for big data and exactly right for watching the algorithm: every particle is a
full hypothesis you can look at, and resampling visibly replaces one net with another.

Rejuvenation is `smc.Rejuvenator`, applied to each particle independently. It is an MCMC
move that leaves the target invariant, so it changes a particle's database but not its
weight.
"""

import math
import random as _random
from typing import Any, Dict, List, Optional

from .distributions import logsumexp
from .model import PCleanModel, VertexID
from .proposal import FlatProposer
from .smc import Rejuvenator
from .trace import Trace


class Particle:
    """One hypothesis: a latent database plus the bookkeeping rejuvenation needs.

    Quacks like `smc.SMC` (`row_assignments`, `owner_maps`) so a `Rejuvenator` can be
    built on it directly.
    """

    def __init__(self, model: PCleanModel):
        self.trace = Trace(model)
        self.row_assignments: List[Optional[int]] = []
        self.owner_maps: List[Dict[VertexID, tuple]] = []
        self.log_weight = 0.0

    def clone(self) -> "Particle":
        p = Particle.__new__(Particle)
        p.trace = self.trace.fork()
        p.row_assignments = list(self.row_assignments)
        p.owner_maps = [dict(m) for m in self.owner_maps]
        p.log_weight = self.log_weight
        return p


class ParticleSMC:
    """N latent databases, extended row by row, reweighted and resampled.

    Driven one step at a time (`extend`, then `maybe_resample`) rather than by a single
    `run`, so a caller -- the visualizer's recorder -- can look at the particles between
    steps. `run` does the whole thing for everyone else.
    """

    def __init__(self, model: PCleanModel, n_particles: int = 4,
                 ess_fraction: float = 0.5):
        self.model = model
        self.n = n_particles
        # Resample when ESS drops below this fraction of N. 1/2 is the Julia's default
        # (`maybe_resample`) and the usual textbook choice.
        self.ess_threshold = ess_fraction * n_particles
        self.particles = [Particle(model) for _ in range(n_particles)]
        # Running estimate of log p(rows so far): each resampling banks the average
        # weight before resetting, and `log_ml` adds whatever is still unbanked.
        self._banked_log_ml = 0.0

    # --------------------------------------------------------------- weights

    def normalized_log_weights(self) -> List[float]:
        ws = [p.log_weight for p in self.particles]
        tot = logsumexp(ws)
        return [w - tot for w in ws]

    def ess(self) -> float:
        """Effective sample size, 1 / sum(w_i^2) over normalized weights.

        N when the weights are even, 1 when one particle has all of it. It measures how
        many particles are actually pulling their weight.
        """
        return math.exp(-logsumexp([2 * w for w in self.normalized_log_weights()]))

    def log_ml(self) -> float:
        return self._banked_log_ml + logsumexp(
            [p.log_weight for p in self.particles]) - math.log(self.n)

    # ----------------------------------------------------------------- steps

    def extend(self, k: int, row: Dict[VertexID, Any],
               log: Optional[list] = None) -> float:
        """Extend particle k by one row. Returns the incremental log weight.

        The proposal enumerates candidate settings for the row -- existing objects versus
        new ones, candidate attribute values -- scores each against the particle's own
        database, and samples one. The normalizer of that enumeration, `log_marginal`, is
        the incremental weight: with the locally optimal proposal, target/proposal
        collapses to it [§3.2].
        """
        p = self.particles[k]
        cls_name = self.model.observation_class
        proposer = FlatProposer(self.model, p.trace)
        proposer.log = log
        res = proposer.propose(cls_name, row)
        p.log_weight += res.log_marginal
        p.row_assignments.append(res.oid)
        p.owner_maps.append(proposer.owner_map(self.model.classes[cls_name],
                                               res.assignment))
        return res.log_marginal

    def maybe_resample(self) -> Optional[List[int]]:
        """Multinomial resampling when ESS < threshold. Returns the ancestor of each new
        particle, or None if the weights were still healthy enough to leave alone.

        Resampling does not change what the particle cloud represents, only how: instead
        of a few heavy particles and many near-weightless ones, N equally weighted
        particles concentrated on the good hypotheses. It is also the step that makes the
        cloud collapse -- after enough of it every particle descends from one ancestor --
        which is the other reason rejuvenation matters: it is how cloned databases diverge
        again.
        """
        if self.ess() >= self.ess_threshold:
            return None
        ws = [math.exp(w) for w in self.normalized_log_weights()]
        self._banked_log_ml += logsumexp(
            [p.log_weight for p in self.particles]) - math.log(self.n)
        ancestors = _random.choices(range(self.n), weights=ws, k=self.n)
        self.particles = [self.particles[a].clone() for a in ancestors]
        for p in self.particles:
            p.log_weight = 0.0
        return ancestors

    def rejuvenator(self, k: int, rows: List[Dict[VertexID, Any]]) -> Rejuvenator:
        p = self.particles[k]
        return Rejuvenator(self.model, p.trace, rows, p)

    def run(self, rows: List[Dict[VertexID, Any]], sweeps: int = 0) -> None:
        for row in rows:
            for k in range(self.n):
                self.extend(k, row)
            self.maybe_resample()
        for k in range(self.n):
            rej = self.rejuvenator(k, rows)
            for _ in range(sweeps):
                rej.sweep()
