# Getting PClean running (2026)

The repo is a 2021 research prototype, frozen since March 2021 (23 commits, last real
change 2022-05). It does run on modern Julia with two small fixes, both applied here.

## Install

```
brew install juliaup
juliaup add 1.10 && juliaup default 1.10    # native arm64, no Rosetta
cd <this dir>
JULIA_PROJECT=. julia -e 'using Pkg; Pkg.instantiate()'
```

## Run

```
JULIA_PROJECT=. julia experiments/hospital/run.jl
JULIA_PROJECT=. julia experiments/flights/run.jl
JULIA_PROJECT=. julia experiments/rents/run.jl
```

## Fixes applied (vs upstream master)

1. **`Project.toml`: removed 9 unused deps.** BenchmarkTools, Interpolations, Plots,
   Polynomials, PyPlot, Revise, StatsBase, Memoize, DelimitedFiles are listed but
   referenced nowhere in `src/` or `experiments/`. PyPlot's build step fails (it wants a
   Python matplotlib via PyCall), which kills `Pkg.instantiate()` for the whole project.
   Also dropped the `CSV = "0.8.5"` compat pin — see (2).

   Real runtime deps are only: CSV, DataFrames, Dates, Distributions, JSON,
   LightGraphs, MacroTools, StringDistances.

   LightGraphs is deprecated (superseded by Graphs.jl) but still resolves and
   precompiles fine on Julia 1.10. It is used for exactly four things — `DiGraph()`,
   `nv()`, `add_edge!()`, `edges()` — so swapping it is a ~10-line change if it ever
   breaks.

2. **`experiments/*/load_data.jl`: `CSV.File(...; stringtype=String)`.** Modern CSV.jl
   returns `InlineStrings` (`String3`, `String7`, `String15`) for short string columns.
   PClean's distribution methods are typed to `::String`, so inference dies with
   `MethodError: no method matching logdensity(::AddTypos, ::String3, ::String3)`.
   This is the breaking change the upstream `CSV = "0.8.5"` pin was avoiding; forcing
   `stringtype=String` is the cleaner fix and leaves `src/` untouched.

`src/` is unmodified from upstream.

## Results reproduced (M-series Mac, Julia 1.10.12)

| Benchmark | F1 here | F1 in paper | Time here | Time in paper |
|-----------|---------|-------------|-----------|---------------|
| Hospital  | 0.904   | 0.91        | 4.7s      | 4.5s          |
| Flights   | 0.888   | 0.90        | 3.0s      | 3.1s          |
| Rents     | 0.695   | 0.69        | 15.2s     | 1m 20s        |

Accuracy matches within run-to-run noise. Rents is ~5x faster than the paper, which is
hardware plus four Julia releases.

The Medicare Physicians experiment (2.2M rows, 7h36m in the paper) is not in the repo —
the dataset was excluded for size and is hosted by Medicare.

## Notes for reading the code

- `src/dsl/syntax.jl` — the `@model` macro. It is only a parser: it walks the Julia AST
  and emits a flat sequence of `add_*!(builder, ...)` calls. All semantics live in
  `src/dsl/builder.jl`.
- `src/inference/proposal_compiler.jl:421` — the heart of the system. Builds a Julia
  `Expr` per (class, block, missingness-pattern), `eval`s it into a native function,
  memoized in `class_model.compiled_proposals`. The hospital run compiles **8** such
  functions total (4,207 lines of generated Julia), then calls them ~2,000+ times.
- Generated code shape: flat (max `for`-nesting depth 1) loops over candidate option
  lists, accumulating log-probs, then `logsumexp` + `Categorical` sample. The
  conditional-independence exploitation shows up as *sequential* loops rather than a
  nested cross-product.
- `src/distributions/` — 12 distributions behind a 5-method protocol (`random`,
  `logdensity`, `has_discrete_proposal`, `discrete_proposal`,
  `discrete_proposal_dummy_value`).
