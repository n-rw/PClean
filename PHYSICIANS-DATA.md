# The Medicare dataset (Figure 1 / Experiment 3)

The repo ships `datasets/` for hospital, flights and rents, but **not** the 2.2M-row
Medicare file behind Figure 1 and the scalability experiment — it was excluded for size,
and there is no `experiments/physicians/` runner either. The PClean program for it is in
the paper, appendix B.4.4.

## What the paper used

"Physician Compare National" — the public CMS file as of ~2020. **Physician Compare was
retired in December 2020** and folded into Medicare Care Compare. The successor file is:

- Landing page: https://data.cms.gov/provider-data/dataset/mj5m-pzi6
- Direct CSV (resolve fresh, the hash path rotates on each release):
  ```
  curl -s "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/mj5m-pzi6" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['distribution'][0]['data']['downloadURL'])"
  ```

Downloaded here to `experiments/physicians_data/` (gitignored via `experiments/*_data`).

| | Paper (2020) | This file (released 2026-09-10) |
|---|---|---|
| Rows | 2,183,992 | 3,388,628 |
| Size | — | 839 MB |
| Columns | — | 31 |

## Column mapping

CMS renamed everything between the Physician Compare era and the current file. The
paper's model (appendix B.4.4) maps as:

| Paper / model field | Current column |
|---|---|
| `npi` | `NPI` |
| `degree`, `degree_obs` (`:Credential`) | `Cred` |
| `specialty` (`Primary specialty`) | `pri_spec` |
| `School.name` | `Med_sch` |
| `Practice.addr` / `addr2` | `adr_ln_1` / `adr_ln_2` |
| `Practice.zip` | `ZIP Code` |
| `Practice.legal_name` | `Facility Name` |
| `City.name` / `city_name` | `City/Town` |
| `state` | `State` |
| `c2z3` (blocking key) | derived: first 2 of city + first 3 of zip |

## Data profile (measured on the 3,388,628-row file)

| Column | Distinct | Blank | Blank % |
|---|---|---|---|
| `Cred` | 22 | 563,189 | 16.6% |
| `Med_sch` | 468 | 55 | 0.0% |
| `pri_spec` | 100 | 5 | 0.0% |
| `City/Town` | 16,633 | 0 | 0.0% |
| `State` | 57 | 0 | 0.0% |
| `ZIP Code` | 332,379 | 0 | 0.0% |
| `adr_ln_1` | 304,498 | 21,419 | 0.6% |
| `Facility Name` | 82,768 | 356,034 | 10.5% |
| `NPI` | 1,627,468 | 0 | 0.0% |

## What still reproduces, and what doesn't

**The showcase inference is intact.** The paper's example is that K. Ryan's missing degree
should be inferred as DO, because although Family Medicine skews MD, the school PCOM
awards mostly DOs. Both halves still hold, and more strongly than the paper describes:

- PCOM graduates: **84.2% DO**, 5.4% MD (and 4.2% blank — real imputation targets)
- FAMILY PRACTICE overall: **74.4% MD**, 21.8% DO

So the prior fights the school evidence exactly as in Figure 1.

**The typo showcase is gone.** Figure 1's headline repair is "Abington, MD" (a systematic
misspelling appearing 152 times) being corrected to "Abingdon, MD" (42 times). In the
current file there is no `ABINGTON` in Maryland at all — only `ABINGDON`, 95 rows. CMS
has normalized city names since 2020. Likewise `City/Town`, `State` and `ZIP Code` now
have **zero** blanks, so those imputation targets are gone too.

Net: the *entity-resolution and degree-imputation* story reproduces; the *systematic-typo*
story does not, because the upstream data got cleaner. `Cred` at 16.6% blank is still
563K genuine imputation targets, which is the bulk of what's interesting.

**Caveat on `Med_sch`:** 62% of rows are the literal value `OTHER` (2,092,798 rows).
The remaining 468 real school names cover ~1.3M rows, which is plenty, but any model
should treat `OTHER` as its own category rather than a school entity.

## Getting an exact 2020 vintage

CMS keeps snapshots at https://data.cms.gov/provider-data/archived-data/doctors-clinicians
but the page is JavaScript-rendered and exposes no static link pattern I could resolve
(probed `archive/...DOC_YYYY_MM.zip` and Physician_Compare variants — all 404). If the
original vintage matters, that page in a browser, or the Internet Archive's capture of
`data.medicare.gov`, are the two routes.

## To actually run it

Needs `experiments/physicians/{load_data.jl,run.jl}` written from appendix B.4.4. All the
required DSL constructs exist in this implementation:

- `unmodeled()` → `Unmodeled()`
- `index by x` → `@guaranteed x` (the observation hashing of appendix D.4 — this is what
  makes millions of rows tractable; without it reference-slot resolution is O(#objects))
- `x ~ d(...) preferring E` → `x ~ d(..., E)`
- `parameter p[_] ~ dirichlet(...)` → `@learned p::Dict{String, ProportionsParameter}`

Expect hours, not minutes: the paper reports 7h36m for 2.2M rows, and this file is 1.55x
larger (though rents ran ~5x faster here than in the paper, so the real figure is
uncertain).
