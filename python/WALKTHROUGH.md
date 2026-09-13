# Reading PClean in six sittings

A suggested order, with the Julia to read alongside each step and a question to answer
before moving on. Everything referenced is in `../src/` unless stated.

Run this first, so you have output to refer back to:

```
python3 python/example_hospital.py --figure1
python3 python/example_physicians.py --rows 3000
```

---

## 1. What is the model, actually?

**Read:** `minipclean/model.py` → `../src/model/model.jl`

PClean models a **database of entities**, not a table. `Record`, `Hospital`, `City` are
classes; `Record.hosp` is a *reference slot*. The clean table is read off at the end by
following slots.

Two structural ideas to take away:

- **Flattening.** When `Record` declares `hosp ~ Hospital`, the Julia copies every node
  of `Hospital` (and transitively `City`) into `Record`'s graph as a `SubmodelNode`. So
  `hosp.loc.county.state` is *one vertex*, not a pointer chase. Inference over a row is
  then inference over one ordinary Bayes net.
- **The Plan.** Read that docstring twice. A `Plan` is a **forest**, and sibling trees
  are conditionally independent given common ancestors.

> **Question:** Three unobserved attributes with 100 candidates each. Why is enumeration
> 300 settings and not 1,000,000?

*Check yourself:* dump the Julia's generated code and measure maximum `for`-nesting depth.
It is 1. Independent variables produce loops that sit *beside* each other, never inside.

---

## 2. How does it decide two rows mean the same thing?

**Read:** `minipclean/structure_prior.py` → the paper, §2.2 and Figure 4

The user's program never says how many hospitals exist. The prior does: a two-parameter
CRP over the **reference set** — every place in the database that points at a class.

    P(point at existing r)  ∝  n_r - d
    P(create a new object)  ∝  s + d·(objects so far)

`n_r - d` is **rich get richer**. A city mentioned by 400 practices is a far likelier
target than one mentioned by two. That is not an entity-resolution heuristic bolted on;
it falls out of the prior.

> **Question:** The paper says BLOG cannot express this (App. C.3, footnote). What
> exactly can't it say?

---

## 3. Why doesn't inference fall over?

**Read:** `minipclean/proposal.py` → `../src/inference/proposal_compiler.jl`

**This is the file.** A generic PPL proposes latent values from the prior and reweights;
for a string-valued attribute that never gets anywhere. PClean **enumerates**: it asks
each distribution for a finite candidate list, scores every candidate against the data
exactly, normalises, samples. The result is not a good guess, it is the *locally optimal*
proposal — for finite-discrete subproblems, exactly the posterior.

The Julia does this by **generating and `eval`ing code** (line 421). We interpret the same
Plan. That is the only deep difference between this package and the original.

Then read `_step` and `_determined`. Every flattened vertex has a CPD that switches on
what its slot turned out to be (Algorithm 1): points at an existing object → the attribute
is *determined*, look it up and score nothing; creates a new object → enumerate it for
real. Because slot and attributes are adjacent vertices in one topological order, choosing
the city and scoring the typo happen in the same enumeration, with no special case.

An earlier draft of this package recursed across classes instead, resolving a slot before
scoring its dependents. It scored **0/13** on the two-hop benchmark model — every repair
wrong, everything collapsing onto `'opp'`, the shortest city name. Flattening took it to
**13/13**.

> **Question:** Why does flattening make that failure structurally impossible, rather than
> just less likely?

---

## 4. How does it see the whole dataset one row at a time?

**Read:** `minipclean/smc.py` → `../src/inference/row_inference.jl`

The model has a sequential form: build the database one row at a time, each contributing
a "database increment" (§3.1, Fig. 5). Hold onto the restaurant metaphor — to start a new
table you must first send friends to the other restaurants, one per reference slot.

SMC can only ever *extend* a hypothesis. It commits to what row 1 means before seeing row
400. For cleaning that is fatal alone, which is why **rejuvenation** exists: revisit each
object and re-enumerate it against *all* the data referencing it.

> **Question:** Why can't SMC alone ever fix "Abington"?

---

## 5. The per-object/per-row distinction — the paper in miniature

**Read:** `minipclean.smc.Rejuvenator`, class docstring

The other bug this code had, and the one worth the most.

A `Practice` has **one** `bad_city`, however many rows mention it. Score it once per row
and a practice with 152 rows contributes its single typo 152 times — you have rebuilt
majority voting inside a Bayesian model, and Figure 1 comes out backwards. It did:
`'abington'` beat `'abingdon'` by 585 nats. Scoring per object flipped it to a ~31-nat win
for the truth.

    per row     'abington': 152 exact +  42 typos  →  wrong spelling wins
    per object  'abington':   1 exact +   7 typos  →  right spelling wins

Identical data. Only the counting differs — and the counting follows from *where the error
model sits in the schema*.

> **Question:** `example_hospital.py` gets recall 12/13 but precision 12/30 on the real
> benchmark. Given the above, why? (The docstring answers it, but work it out first.)

---

## 6. Learning what nobody told it

**Read:** `minipclean/parameters.py` → the paper, App. D.2 and Figure 1's caption

`@learned degree_dist :: Dict{String, ProportionsParameter}` — one learned distribution
per medical school, from one line, none of them named by the user.

Run `example_physicians.py` and watch three separate paper claims land at once:

```
round 1: this sample 55.4%   ...   round 5: this sample 69.5%
  modal over 5 samples : 79.8%          <- §4, Experiment 4
  where all 5 agreed   : 96.9% correct  <- Figure 7, calibration
  PCOM learned: DO 77.5%, MD 8.8%       <- Figure 1 (real: 84.2% DO)
  global prior: MD 75.7%, DO 12.9%
```

The model was told nothing about PCOM. It counted — from the same dirty file it is
cleaning. Then it used that count to overturn the prior: most Family Practice clinicians
are MDs, but a Family Practice clinician *from PCOM* is a DO.

Note the circularity, which is the real idea: the degree distribution is estimated from
records whose degrees were themselves partly inferred. Inference and learning are one
loop, and the accuracy climbing 55% → 69.5% across rounds is that loop tightening.

> **Question:** A single posterior sample gets 69.5%; the mode over five gets 79.8%; cells
> where all five agreed are 96.9% correct. What does that last number buy you in
> production, and what does it cost? (Figure 7.)

---

## Where this translation is weaker than the Julia

Stated plainly so you don't mistake a simplification for the design:

| | Julia | here |
|---|---|---|
| Proposals | generated + JIT-compiled, memoized | interpreted, ~50× slower |
| Cross-class deps | flattened into one Bayes net | **flattened too** — `builder.reference()` |
| Subproblem blocking `[§3.3]` | user-declared, splits SMC steps | one block per class |
| Parameters | incremental sufficient statistics | full recount each round |
| Particle cloning | persistent structure, O(1) | deep copy, O(database) |
| Continuous variables `[App. D.2]` | Particle Gibbs rejuvenation | prior sampling only |
| Garbage collection | unreferenced objects deleted | not implemented |
| String prior | English character-bigram LM | flat per-character cost |

Only the first changes *answers* now — the rest are honest shortcuts in speed or in
scope. Everything is flagged at its site with a `NOTE (divergence)` comment.
