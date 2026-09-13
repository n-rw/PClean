"""Learning the parameters — mirrors the conjugate-update machinery scattered through
``src/distributions/*.jl`` and ``[App. D.2]``.

## What this is for

A PClean program can declare quantities it does *not* know and wants inferred from the
dirty data itself:

    @learned degree_dist :: Dict{String, ProportionsParameter}
    degree ~ ChooseProportionally(degrees, degree_dist[school.name])

Nobody tells this model that the Philadelphia College of Osteopathic Medicine awards
mostly DOs. It counts. And that counted distribution is what lets it overturn a prior:
a Family Practice physician is 74% likely to be an MD, but a Family Practice physician
*who went to PCOM* is overwhelmingly a DO. The paper's Figure 1 turns on exactly this,
and the caption is explicit that "all parameters enabling this reasoning are learned from
the dirty data".

Note the circularity, which is the interesting part: the degree distribution is learned
from records whose degrees are themselves partly inferred. Inference and learning are the
same loop. Run it twice and the estimates sharpen.

## How the Julia does it, and how this differs

The Julia recognises conjugate pairs (Dirichlet/Categorical, Beta/Bernoulli,
Normal/Normal) and maintains **sufficient statistics incrementally**: every time
inference commits to a choice it calls `incorporate_choice!` (a counter increment), and
every time it revises one it calls `unincorporate_choice!` (a decrement). The parameter
is then redrawn from its posterior in O(1). As [App. D.2] puts it, "the inference engine
tracks the relevant sufficient statistics as inference progresses, so these updates need
not perform costly counts or summations."

NOTE (divergence): we do the costly count. `fit_parameters` walks the entire latent
database and recounts from scratch. It is O(database) instead of O(1) per update, and it
is the reason you can read this file and be sure it is right. The distributions still
expose `incorporate`/`unincorporate` so you can see the shape of the fast path.
"""

from typing import Any, Dict, List

from .distributions import (ChooseProportionally, IndexedParameter, MaybeSwap,
                            ProbParameter, ProportionsParameter)
from .model import ParameterNode, ParamLookup, PCleanModel, RandomChoiceNode
from .trace import Trace


def _all_parameters(model: PCleanModel):
    for cls in model.classes.values():
        for node in cls.nodes:
            if isinstance(node, ParameterNode):
                yield node.param


def fit_parameters(model: PCleanModel, trace: Trace, verbose: bool = False) -> None:
    """Recount every learned parameter from the current latent database.

    Three passes, deliberately kept separate so each is obvious:

      1. zero the sufficient statistics
      2. walk every object and incorporate its committed choices
      3. redraw each parameter from its posterior

    Call this after SMC and after each rejuvenation sweep. Because inference and learning
    feed each other, repeating the cycle is what sharpens both.
    """
    from .proposal import Proposer

    for p in _all_parameters(model):
        p.reset()

    proposer = Proposer(model, trace)

    for cls_name, cls in model.classes.items():
        for obj in trace.table(cls_name).objects.values():
            for v, node in enumerate(cls.nodes):
                if not isinstance(node, RandomChoiceNode) or v not in obj.values:
                    continue
                if not any(isinstance(a, ParamLookup) for a in node.arg_node_ids):
                    continue
                args = proposer._resolve_args(cls, node, obj.values, obj.values)
                if args is None:
                    continue
                value = obj.values[v]

                # ChooseProportionally(options, probs): count which option was chosen.
                if isinstance(node.dist, ChooseProportionally) and len(args) >= 2:
                    options, probs = args[0], args[1]
                    if isinstance(probs, ProportionsParameter):
                        probs.value(len(list(options)))     # size the counts lazily
                        probs.incorporate(options, value)

                # MaybeSwap(val, options, p): count whether this was an error.
                elif isinstance(node.dist, MaybeSwap) and len(args) >= 3:
                    clean, prob = args[0], args[2]
                    if isinstance(prob, ProbParameter):
                        prob.incorporate(value != clean)

    for p in _all_parameters(model):
        p.resample()

    if verbose:
        report(model)


def report(model: PCleanModel, max_keys: int = 4) -> None:
    """Print what was learned. Useful for convincing yourself it counted the right thing."""
    for cls_name, cls in model.classes.items():
        for v, node in enumerate(cls.nodes):
            if not isinstance(node, ParameterNode):
                continue
            name = cls.node_name(v)
            p = node.param
            if isinstance(p, IndexedParameter):
                print(f"  {cls_name}.{name}: {len(p.table)} keys learned")
            elif isinstance(p, ProbParameter):
                print(f"  {cls_name}.{name}: {p.value():.5f} "
                      f"({p.heads} errors / {p.heads + p.tails} observations)")


def show_distribution(param, options: List[Any], key: str = "", top: int = 4) -> str:
    """Render one learned categorical as 'DO 84.2%, MD 5.4%, ...'."""
    vals = param.value(len(options))
    pairs = sorted(zip(options, vals), key=lambda t: -t[1])[:top]
    body = ", ".join(f"{o} {100*p:.1f}%" for o, p in pairs)
    return f"{key}: {body}" if key else body
