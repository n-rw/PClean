"""Building a model — mirrors ``src/dsl/builder.jl``.

The Julia has a `@model` macro on top of this (`src/dsl/syntax.jl`). It is worth knowing
what that macro actually is, because it looks like the clever part and is not: 162 lines
that walk the Julia AST at expansion time and emit a flat sequence of `add_new_class!`,
`add_choice_node!`, `add_foreign_key!` calls against a builder object. Every bit of
meaning lives in `builder.jl`, which is this file.

So there is no macro here. You call the builder. It is wordier and entirely transparent,
and it is exactly what `@model` compiles to.

    b = ModelBuilder()
    city = b.add_class("City")
    city.attribute("name", StringPrior(3, 30), args=[city_candidates])

One thing the macro does that we must do by hand: it *infers* each node's dependencies by
looking at which names appear in the expression. Here you pass `args` explicitly. An
argument is a literal, a vertex id (use `cls.ref("other_attr")`), or a
`(slot_vertex, "attr_name")` pair reaching across a reference slot.
"""

from typing import Any, Dict, List, Optional

from .distributions import IndexedParameter
from .model import (ForeignKeyNode, JuliaNode, ParameterNode, ParamLookup,
                    PCleanClass, PCleanModel, PitmanYorParams, RandomChoiceNode,
                    Ref, Via, make_plan)


class ClassBuilder:
    """Fluent builder for one class. Mirrors the `add_*!(builder, class, ...)` family."""

    def __init__(self, model: PCleanModel, name: str):
        self.model = model
        self.name = name
        self.cls = PCleanClass(name=name)
        model.classes[name] = self.cls
        model.class_order.append(name)
        self._by_name: Dict[str, int] = {}

    # -- lookups ----------------------------------------------------------------

    def ref(self, attr_name: str) -> Ref:
        """Reference a previously declared node, for use in an `args` list."""
        return Ref(self._by_name[attr_name])

    def via(self, slot_name: str, *path: str) -> Via:
        """An argument reaching across reference slots: `via("hosp", "city", "name")`."""
        return Via(self._by_name[slot_name], tuple(path))

    # -- declarations -----------------------------------------------------------

    def attribute(self, name: str, dist, args: Optional[List[Any]] = None) -> int:
        """`name ~ dist(args...)` — a random variable owned by this class."""
        args = args or []
        # A Via argument depends on the reference slot it travels through, so that slot
        # must be ordered before this node in the plan.
        # A Via argument depends on the reference slot it travels through, and a
        # ParamLookup on whatever keys it. Both must be ordered earlier in the plan.
        parents = ([a.v for a in args if isinstance(a, Ref)]
                   + [a.slot for a in args if isinstance(a, Via)]
                   + [a.key.slot for a in args
                      if isinstance(a, ParamLookup) and isinstance(a.key, Via)]
                   + [a.key.v for a in args
                      if isinstance(a, ParamLookup) and isinstance(a.key, Ref)])
        v = self.cls.add_node(RandomChoiceNode(dist=dist, arg_node_ids=args, name=name),
                              parents)
        self._by_name[name] = v
        return v

    def reference(self, name: str, target_class: str) -> int:
        """`name ~ TargetClass` — a reference slot.

        The structure prior decides how many objects of `target_class` exist and which
        of them this slot points at; see structure_prior.py [§2.2].
        """
        v = self.cls.add_node(ForeignKeyNode(target_class=target_class, name=name), [])
        self._by_name[name] = v
        return v

    def deterministic(self, name: str, f, args: List[Any]) -> int:
        """`name = f(args...)` — a deterministic node (the Julia's `JuliaNode`)."""
        parents = [a.v for a in args if isinstance(a, Ref)]
        v = self.cls.add_node(JuliaNode(f=f, arg_node_ids=args, name=name), parents)
        self._by_name[name] = v
        return v

    def learned(self, name: str, param) -> int:
        """`@learned name::Param` — a quantity inferred from the dirty data itself."""
        v = self.cls.add_node(ParameterNode(param=param, name=name), [])
        self._by_name[name] = v
        return v

    def learned_indexed(self, name: str, factory) -> int:
        """`@learned name::Dict{String, Param}` — one learned parameter per key.

        `factory` makes a fresh parameter the first time a key is seen. The paper's
        Physicians model uses this for `degree_proportions[school]`: one distribution per
        medical school, none of them named by the user [App. B.4.4].
        """
        return self.learned(name, IndexedParameter(factory))

    def param(self, name: str, key=None) -> ParamLookup:
        """Use a learned parameter as an argument: `param("degree_dist", via(...))`."""
        return ParamLookup(self._by_name[name], key)

    def guaranteed(self, *attr_names: str) -> None:
        """`@guaranteed x` (the paper writes `index by x`) [App. D.4].

        Declares that these fields are always observed and trusted, so they can serve as
        an exact-match key for this class. This is the difference between resolving a
        reference slot with a dict lookup and scanning every object hypothesized so far.
        Without it, millions of rows are out of reach; the paper's Physicians experiment
        leans on it heavily.
        """
        for n in attr_names:
            self.cls.hash_keys.append(self._by_name[n])

    def pitman_yor(self, strength: float = 1.0, discount: float = 0.0) -> None:
        self.cls.py = PitmanYorParams(strength=strength, discount=discount)

    def finish(self) -> None:
        """Compute the conditional-independence plan. Mirrors `finish_class!`.

        NOTE (divergence): the Julia supports user-declared `subproblem` blocks that
        split a class into several sequential SMC steps [§3.3]. We use one block per
        class, which is the simplest thing that is still correct -- bigger enumerations,
        better proposals, slower.
        """
        # Reference slots ARE plan vertices: a slot must be enumerated jointly with the
        # attributes that depend on it, or the target is chosen without looking at the
        # data. See proposal._enumerate_reference.
        vertices = [v for v, n in enumerate(self.cls.nodes)
                    if not isinstance(n, ParameterNode)]
        self.cls.blocks = [vertices]
        self.cls.plans = [make_plan(self.cls, vertices)]


class ModelBuilder:
    """Mirrors `PCleanModelBuilder` + `finish_model!`."""

    def __init__(self):
        self.model = PCleanModel()
        self._classes: List[ClassBuilder] = []

    def add_class(self, name: str) -> ClassBuilder:
        cb = ClassBuilder(self.model, name)
        self._classes.append(cb)
        return cb

    def finish(self, observation_class: str) -> PCleanModel:
        for cb in self._classes:
            cb.finish()
        self.model.observation_class = observation_class
        return self.model
