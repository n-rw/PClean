# minipclean — an annotated Python reading of PClean

This is **not** a port. It is a translation written to be *read alongside* the Julia,
to answer "what is this code actually doing?". It runs, slowly, on small data, so you
can put a `print` anywhere and watch the algorithm work.

Design rules, so you know what to trust:

- **Same shape as the Julia.** Files, types and function names mirror `../src/` so you
  can read them side by side. Where a name differs, the docstring says what it was.
- **Annotated for the *why*.** Comments explain intent and the paper's reasoning, not
  Python syntax. Paper sections are cited as `[§3.2]`, appendices as `[App. D.4]`.
- **Clarity beats speed, every time.** Where the Julia does something fast and opaque,
  this does the slow obvious thing and says so in a `NOTE (divergence)` comment.
- **No NumPy in the core.** Plain lists and floats, so every number is inspectable.

## The one real design difference

The Julia's `proposal_compiler.jl` is a *compiler*: it builds a Julia AST per
(class, block, missingness-pattern), `eval`s it into a natively-compiled function, and
memoizes it (the hospital run compiles 8 such functions, 4,207 lines of generated Julia,
then calls them thousands of times).

That is load-bearing for speed and useless for understanding — you cannot read a function
that does not exist until runtime. So `proposal.py` **interprets the same plan** instead
of compiling it. Identical semantics, identical output distribution, ~50x slower, and
you can step through it in a debugger.

This is the single most important thing to understand about this translation: everything
the Julia achieves by generating code, this achieves by walking a data structure.

## Flattening is implemented, and it is not an optimisation

`builder.reference()` does what `add_foreign_key!` does: declaring `hosp ~ Hospital`
copies *every node of Hospital* into Record's graph, shifting the argument indices. Since
Hospital was flattened when it was declared, this is transitive — `hosp.loc.county.state`
ends up a single vertex of `Record`, three slots deep.

It is tempting to read that as a speed trick. It is not. Without it, a reference slot gets
chosen *before* the data bearing on it is scored, and the choice collapses onto whatever
the CRP likes best. An earlier draft of this package recursed across classes instead and
scored **0/13** on the benchmark's real two-hop model — every repair wrong. With
flattening the same model scores **13/13**, because the slot and the typo are now adjacent
vertices in one topological order and get enumerated together.

## File mapping

| This file | Mirrors | What it is |
|---|---|---|
| `distributions.py` | `src/distributions/*.jl` | The 5-method distribution protocol + 6 concrete distributions |
| `model.py` | `src/model/model.jl`, `trace.jl` | The model IR: classes, nodes, the per-class DAG |
| `builder.py` | `src/dsl/builder.jl` | Constructing a model. Replaces the `@model` macro (see below) |
| `trace.py` | `src/model/trace.jl` | The latent database being inferred |
| `structure_prior.py` | *(implicit in Julia)* | The CRP over reference slots, `[§2.2]`. Explicit here because it is the paper's main modeling contribution and the Julia scatters it |
| `proposal.py` | `src/inference/proposal_compiler.jl`, `block_proposal.jl` | **The core.** Enumerative data-driven proposals, `[§3.2]` |
| `smc.py` | `src/inference/row_inference.jl`, `inference.jl` | Per-row SMC + object-wise rejuvenation, `[§3.1]` |
| `parameters.py` | conjugate updates in `distributions/*.jl` | Learning `@learned` quantities from the dirty data, `[App. D.2]` |
| `example_hospital.py` | `experiments/hospital/run.jl` | Systematic-typo repair, end to end |
| `example_physicians.py` | `experiments/physicians/run.jl` | Learned parameters on real CMS data |
| `example_subproblems.py` | Figure 6's comparison | Subproblem hints measured: cost vs accuracy |
| `example_gc.py` | the GC half-sentence in `[§3.1]` | Garbage collection, and why it is not housekeeping |
| `particles.py` | the sequential form of `[§3.1]`, Fig. 5 | Multi-particle SMC over whole databases: reweighting, ESS, resampling |
| `viz_record.py` | *(none)* | Records a run as a step-by-step event log for the visualizer in `../viz` |

## On the missing `@model` macro

There isn't one, deliberately. The Julia `@model` macro (`src/dsl/syntax.jl`, 162 lines)
looks like the clever part but is only a *parser*: it walks the Julia AST at expansion
time and emits a flat sequence of `add_class!` / `add_choice_node!` / `add_foreign_key!`
calls against a builder. All the semantics live in `builder.jl`.

So here you call the builder directly. It is more verbose and completely transparent —
and it is exactly what the macro compiles down to, which is the point.

## Start here

**[WALKTHROUGH.md](WALKTHROUGH.md)** — a reading order in six sittings, each pairing a
file here with the Julia to read beside it and a question to answer before moving on.

## Running it

```
python3 python/example_hospital.py --figure1      # the systematic-typo repair
python3 python/example_hospital.py --rows 400     # the real benchmark
python3 python/example_physicians.py --rows 3000  # learned parameters, real CMS data
python3 python/example_subproblems.py             # subproblem hints, measured
python3 python/example_gc.py                      # garbage collection and the prior's support
```

To *watch* the algorithm instead, step through a run in the browser:

```
python3 python/viz_record.py         # record the event logs (already committed)
cd viz && npm install && npm run dev # http://localhost:5178
```

See [../viz/README.md](../viz/README.md).

No dependencies beyond the standard library. The physicians example needs the CMS file;
see [../PHYSICIANS-DATA.md](../PHYSICIANS-DATA.md).

## What it actually does

```
$ python3 python/example_hospital.py --figure1
Figure 1 scenario: 'abington' appears 152x (one practice), 'abingdon' 42x (seven practices).
Majority vote over cells would pick 'abington' -- which is wrong.
...
    rejuv City#3.name: 'abington' -> 'abingdon'
  precision: 152/152 repairs correct
  recall:    152/152 of the city errors present were fixed
```

The misspelling outnumbers the truth almost four to one and still loses. That is the
paper's central claim, reproduced in readable Python.

On the real benchmark (`--rows 400`), using the faithful two-hop model where the clean
name sits two reference slots from the observation:

```
$ python3 python/example_hospital.py --rows 400
Record vertices after flattening: practice, practice.pid, practice.city,
                                  practice.city.name, obs_city
  OK   'bxrmxngham' -> 'birmingham'   ...
  precision: 13/13
  recall:    13/13
```

And `example_physicians.py`, on 3,000 real CMS clinicians with 30% of credentials hidden,
lands three separate paper results at once:

```
round 1: this sample 66.8%  ...  round 5: this sample 75.9%    <- learning loop tightening
  modal over 5 samples : 82.6%           <- §4, Experiment 4
  where all 5 agreed   : 98.6% correct   <- Figure 7, calibration
  PCOM learned : DO 77.5%, MD 12.9%      <- Figure 1 (ground truth: 84.2% DO)
  global prior : MD 75.7%, DO 12.9%
  held-out PCOM clinicians: 11/14 correct (the prior says MD for all 14)
```

Nobody told the model about PCOM. It counted, from the same dirty file it is cleaning,
and then used the count to overturn a prior.

## Subproblem hints, measured

`subproblem begin ... end` partitions a class's variables into ordered blocks [§3.3]. SMC
treats each as an intermediate target: enumerate jointly, commit, move on. Smaller blocks
are cheaper and more myopic.

`example_subproblems.py` runs the same model both ways — `specialty` depends on `degree`,
so in one block they nest (15 x 50 settings) and split they do not (15 + 50):

```
cost     : 11.5x fewer settings scored (177,645 vs 2,049,750), 3.8x faster
accuracy : 36.3% vs 37.0% (mean of 3 seeds, +0.7 points)
```

That is the paper's Figure 6 finding: an order of magnitude cheaper, landing in the same
place. The joint enumeration sees more per step, but rejuvenation revisits every object
anyway, so the extra sight buys little a second sweep would not. The hints change only the
*proposal* — the model, and therefore the posterior, is identical either way.

Rejuvenation is genuinely **blocked** Gibbs now, redrawing a whole subproblem at once
rather than one variable at a time. That is not a detail: the paper is explicit that
single-site moves get stuck, because a city's clean name and the spellings depending on it
are too correlated for any one-variable move to escape a bad mode.

## Garbage collection is not housekeeping

Rejuvenation revises reference slots, not just attributes [§3.1], so an object can lose
its last referrer. The paper disposes of this in half a sentence, which undersells it. The
structure prior places mass **only** on skeletons where every object is reachable from the
observed data [§2.2] — so an unreachable object is not untidy, it is a state of
probability zero, and a database containing one is not a sample from the posterior.

Nor is it inert. `example_gc.py` injects 25 phantoms into a settled database:

```
before            : 17 cities, 17 candidate targets, P(next invents new) = 0.0333
+25 phantoms      : 42 cities, 42 candidate targets, P(next invents new) = 0.0182   VIOLATED
after collection  : 17 cities, 17 candidate targets, P(next invents new) = 0.0333   HOLDS
```

The reconstructed dataset is byte-for-byte identical throughout — every accuracy number
looks the same — while the prior has moved 45%. That is the failure mode: not a leak, a
silent bias on every entity-resolution decision that follows.

## Seven bugs worth knowing about

All of them were live in this code. Each is one of the paper's ideas showing up in negative,
so each is documented where it happened rather than quietly fixed.

1. **Choosing a reference slot before scoring its dependents.** Picked blind, the slot
   collapses onto whatever the CRP likes best. Flattening removes the possibility.

2. **Scoring evidence per row instead of per object.** A Practice has one `bad_city`
   however many rows mention it. Score per row and a practice with 152 rows contributes
   its single typo 152 times — majority voting, rebuilt inside a Bayesian model, and
   Figure 1 comes out backwards. See `smc.Rejuvenator`.

3. **Keying a flattened attribute off the wrong slot.** `practice.city.name` is
   determined as soon as **`practice.city`** resolves — not `practice`. Key off the outer
   slot and every city looks undetermined whenever the practice is new, and all 29
   practices collapse onto one city. See `proposal._owner_slot`.

4. **Leave-one-out that deletes the option of staying put.** Gibbs over a CRP removes
   the item before weighing candidates — but an object this slot is the *sole* referrer
   of then has count 0, and a zero-weight candidate gets dropped. The slot is forced to
   move, and every singleton entity is destroyed on the first sweep. Re-seating at a table
   you just vacated is the same event as starting a fresh one, so it takes the new-table
   weight. See `smc.Rejuvenator._settings`.

5. **Caching flattened copies on every referring object.** Then rejuvenation scores
   candidates against a stale value, every candidate scores identically, and the prior
   decides — which looks exactly like inference working. One source of truth: an
   attribute lives on the object that owns it. See `proposal._determined`.

6. **Counting a flattened slot as a new reference.** A row whose Practice already
   exists still carries `practice.city` in its flattened graph, and `_materialize` used to
   bump that City's `ref_count` for it. So the CRP's `n_r` counted *rows* rather than
   referring *objects*: a practice with 152 rows made its city look 152 times as popular.
   That is number 2 again, this time in the structure prior rather than the likelihood --
   and invisible in the repair scores, because a hash key pins the practice anyway. It
   showed up in the visualizer as a City reading `n_r = 5` with two practices pointing
   at it. A flattened slot is a new reference only when the object owning it is being
   created by this row. See `proposal._materialize`.

7. **A referrer map built once.** `Rejuvenator` reads a City's evidence through the
   objects that point at it, but built that map at construction and never updated it. After
   the first slot move a City was scored against practices that had left it and ignored
   ones that had joined. It is now updated on every move and rebuilt after garbage
   collection. See `smc.Rejuvenator._build_referrers`.

All five examples reproduce their documented numbers with 6 and 7 fixed; both
bite hardest on datasets with many rows per entity and on repeated sweeps.

Number 2 is the whole paper in miniature. The data is identical either way; only the
counting differs, and the counting follows from where the error model sits in the schema.
