"""The model IR — mirrors ``src/model/model.jl``.

A PClean model is a set of **classes**. Each class is a little Bayesian network over its
own attributes, plus **reference slots** pointing at other classes. `Physician.school`
points at a `School`; `Record.physician` points at a `Physician`.

Two structural ideas do most of the work, and neither is obvious from the paper:

1. **Flattening.** A class's graph does not merely *reference* its neighbours, it
   *absorbs* them. When `Record` declares `physician ~ Physician`, every node of
   `Physician` (and transitively of `School`) is copied into `Record`'s graph as a
   `SubmodelNode`. So `hosp.loc.county.state` is a single vertex in `Record`'s DAG, not
   a three-hop pointer chase. Inference over one record is then inference over one
   ordinary Bayes net — which is why the proposal machinery can be so simple.

2. **The Plan.** Given that flattened DAG, a `Plan` is a *forest*: sibling trees are
   conditionally independent given their common ancestors. That single fact is what
   turns enumeration from a cross-product into a sequence of small independent loops.
   See the `Plan` docstring below — it is the most important comment in this package.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

VertexID = int      # index of a node within one class's node list
ClassID = str       # a class name; Julia uses a Symbol


@dataclass(frozen=True)
class Ref:
    """An argument that means "the value of vertex `v` in this class".

    Needed because a bare int is ambiguous: is `2` vertex 2, or the literal number 2?
    The Julia never faces this -- its macro resolves names against the set of declared
    variables at expansion time, so by the time a node is built it already knows which
    arguments are references. Having no macro, we make the distinction explicit.
    """
    v: VertexID


@dataclass(frozen=True)
class ParamLookup:
    """An argument that means "the learned parameter at `param`, for key `key`".

    `@learned degree_dist::Dict{String, ProportionsParameter}` in the Julia DSL, written
    `degree_dist[school.name]` at the use site. `key=None` means a plain, unindexed
    parameter (`@learned error_prob::ProbParameter`).

    Indexing is what lets one declaration stand for hundreds of separate learned
    distributions -- one per medical school, in the paper's Physicians model -- without
    the user naming any of them.
    """
    param: VertexID
    key: object = None


@dataclass(frozen=True)
class Via:
    """An argument reaching across one or more reference slots: `hosp.loc.county.state`.

    `slot` is the first hop (a vertex id in *this* class); `path` is the remaining hops
    by name, ending in the attribute to read. So `hosp.city.name` is
    ``Via(slot=<hosp>, path=("city", "name"))``.

    Multi-hop matters: the real hospital program reads `hosp.loc.county.state`, three
    slots deep. The Julia handles this by flattening every reachable node into the
    referring class's graph (see model.py header), which makes the path disappear
    entirely -- it becomes one local vertex. We keep the path and walk it, which is
    slower and much easier to follow.
    """
    slot: VertexID
    path: tuple


@dataclass(frozen=True)
class AbsoluteVertexID:
    """A (class, vertex) pair — needed once nodes are flattened across classes."""
    cls: ClassID
    idx: VertexID


# --------------------------------------------------------------------------------
# Node types — mirrors the `PCleanNode` subtypes in model.jl
# --------------------------------------------------------------------------------

class Node:
    """Base for all vertices in a class's dependency graph."""


@dataclass
class RandomChoiceNode(Node):
    """`x ~ SomeDistribution(args...)` — a latent or observed random variable."""
    dist: Any                       # a distributions.Distribution
    arg_node_ids: List[VertexID] = field(default_factory=list)
    name: str = ""


@dataclass
class JuliaNode(Node):
    """A deterministic function of other nodes: `stateavg = f(state, code)`.

    Named `JuliaNode` in the original because it holds a literal Julia closure. Kept
    here under the same name so the mapping is obvious; in Python it holds a lambda.

    NOTE (divergence): the Julia `@model` macro *infers* the argument list by walking
    the expression's AST. Having no macro, we ask the caller to list the arguments
    explicitly. Same node, less magic.
    """
    f: Callable
    arg_node_ids: List[VertexID] = field(default_factory=list)
    name: str = ""


@dataclass
class ForeignKeyNode(Node):
    """A reference slot: `school ~ School`.

    Its *value* in a trace is the identity of the target object. Which object it can
    point to is decided by the structure prior (the CRP) plus, in practice, the
    observation index [App. D.4].
    """
    target_class: ClassID
    vmap: Dict[VertexID, VertexID] = field(default_factory=dict)
    name: str = ""


@dataclass
class SubmodelNode(Node):
    """A node belonging to a *referenced* class, flattened into this one.

    `subnode` is the original node with its argument indices rewritten to this class's
    numbering. `foreign_key_node_id` says which reference slot we travelled to get here,
    which matters because the same class can be reached by two different slots.
    """
    foreign_key_node_id: VertexID
    subnode_id: VertexID
    subnode: Node
    name: str = ""


@dataclass
class ParameterNode(Node):
    """A `@learned` hyperparameter shared by every object of the class."""
    param: Any
    name: str = ""


def strip_submodel(node: Node) -> Node:
    """Peel SubmodelNode wrappers to get at the real node. Mirrors `strip_subnodes`."""
    while isinstance(node, SubmodelNode):
        node = node.subnode
    return node


# --------------------------------------------------------------------------------
# Plan — mirrors `Step` and `Plan` in model.jl
# --------------------------------------------------------------------------------

@dataclass
class Step:
    """One node of a Plan tree, plus the sub-plan that depends on it."""
    idx: VertexID
    rest: "Plan"


@dataclass
class Plan:
    """A forest of trees covering the vertices of one subproblem.

    **This is the data structure that makes PClean fast, and the one the Julia turns
    into generated code.** Read the invariant carefully:

        Any two nodes in the forest are conditionally independent
        given their common ancestors.

    Why that matters. Suppose a class has three unobserved discrete attributes, each
    with 100 candidate values. Naive enumeration considers 100^3 = 1,000,000 joint
    settings. But if the three are conditionally independent given what precedes them,
    they become three *sibling trees* in the forest, and you can enumerate each in turn:
    100 + 100 + 100 = 300 settings, combining the results by logsumexp.

    That is the "potentially exponential savings over naive enumeration" of [§3.2], and
    it is why the Julia's generated code has loops that sit *beside* each other rather
    than *inside* each other. (Dump the generated code and measure: maximum `for`-nesting
    depth is 1.) Dependent variables nest, via `Step.rest`; independent ones do not.

    `steps` is the forest's roots. Each root carries its dependants in `rest`.
    """
    steps: List[Step] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.steps


def make_plan(class_model: "PCleanClass", vertices: List[VertexID]) -> Plan:
    """Build the conditional-independence forest over `vertices`.

    The algorithm is just a topological walk that nests a node under the *last* of its
    parents that is also in this subproblem. Nodes with no parent inside the subproblem
    become new roots — and roots are exactly the conditionally-independent siblings.

    NOTE (divergence): the Julia computes this inside `dependency_tracking.jl` with more
    bookkeeping (it also prunes nodes that cannot be reached from any observation, see
    `prune_plan`). This is the same idea at its simplest.
    """
    in_subproblem = set(vertices)
    ordered = [v for v in class_model.topological_order() if v in in_subproblem]

    plan_for: Dict[VertexID, Plan] = {}
    root = Plan()

    for v in ordered:
        parents = [p for p in class_model.parents(v) if p in in_subproblem]
        # Attach under the latest parent in topological order, so every ancestor this
        # node depends on has already been enumerated by the time we reach it.
        host = root
        if parents:
            latest = max(parents, key=lambda p: ordered.index(p))
            host = plan_for[latest]
        sub = Plan()
        host.steps.append(Step(idx=v, rest=sub))
        plan_for[v] = sub

    return root


# --------------------------------------------------------------------------------
# Classes and models — mirrors `PCleanClass` and `PCleanModel`
# --------------------------------------------------------------------------------

@dataclass
class PitmanYorParams:
    """Strength and discount of the CRP governing how many objects of a class exist.

    Both are learned. See structure_prior.py — this is the paper's contribution (1),
    the non-parametric prior over the latent database's shape [§2.2].
    """
    strength: float = 1.0
    discount: float = 0.0


@dataclass
class PCleanClass:
    """One class: a DAG of nodes, plus indexing and blocking metadata."""
    name: ClassID
    nodes: List[Node] = field(default_factory=list)
    edges: Dict[VertexID, List[VertexID]] = field(default_factory=dict)   # child -> parents

    # Vertices whose observed value is *trusted and always present*, used as an exact
    # lookup key for this class [App. D.4]. This is the difference between resolving a
    # reference slot in O(1) and scanning every object hypothesized so far -- i.e. the
    # difference between millions of rows being feasible and not.
    # The paper writes `index by x`; the Julia DSL spells it `@guaranteed x`.
    hash_keys: List[VertexID] = field(default_factory=list)

    # User-declared subproblems [§3.3]. Each block is enumerated as its own SMC step,
    # so smaller blocks mean cheaper but more myopic proposals -- rejuvenation is what
    # buys back the myopia.
    blocks: List[List[VertexID]] = field(default_factory=list)
    plans: List[Plan] = field(default_factory=list)

    py: PitmanYorParams = field(default_factory=PitmanYorParams)

    def add_node(self, node: Node, parents: List[VertexID]) -> VertexID:
        idx = len(self.nodes)
        self.nodes.append(node)
        self.edges[idx] = list(parents)
        return idx

    def parents(self, v: VertexID) -> List[VertexID]:
        return self.edges.get(v, [])

    def children(self, v: VertexID) -> List[VertexID]:
        return [c for c, ps in self.edges.items() if v in ps]

    def topological_order(self) -> List[VertexID]:
        """Nodes ordered so every node follows its parents. The graph is acyclic by
        construction — PClean *requires* an acyclic class dependency graph [App. C.3],
        which is why `Person -> Person` models (genealogy) need a workaround."""
        seen, order = set(), []

        def visit(v):
            if v in seen:
                return
            seen.add(v)
            for p in self.parents(v):
                visit(p)
            order.append(v)

        for v in range(len(self.nodes)):
            visit(v)
        return order

    def node_name(self, v: VertexID) -> str:
        n = self.nodes[v]
        return getattr(n, "name", "") or f"v{v}"


@dataclass
class PCleanModel:
    """The whole program: classes, plus which class the observed rows correspond to."""
    classes: Dict[ClassID, PCleanClass] = field(default_factory=dict)
    class_order: List[ClassID] = field(default_factory=list)
    observation_class: Optional[ClassID] = None

    def topological_class_order(self) -> List[ClassID]:
        """Classes ordered so a class follows everything it refers to."""
        seen, order = set(), []

        def visit(c):
            if c in seen:
                return
            seen.add(c)
            for node in self.classes[c].nodes:
                if isinstance(node, ForeignKeyNode):
                    visit(node.target_class)
            order.append(c)

        for c in self.class_order:
            visit(c)
        return order
