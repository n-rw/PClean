# PClean, step by step

An interactive replay of PClean inference. Step through a run and watch each particle's
**latent database** — its unrolled Bayes net — grow row by row under SMC, get reweighted,
resampled, and revised by rejuvenation.

```
cd viz
npm install
npm run dev            # http://localhost:5178
```

The event logs in `public/data/` are committed, so the app runs without Python. To
regenerate them (after changing `minipclean` or a scenario):

```
npm run record         # = python3 ../python/viz_record.py
```

## What you are looking at

- **The net** is one particle's hypothesis: every City, Practice and Record it currently
  believes in. Lanes run parents-first, so every arrow points down. Circles are attributes
  (filled = observed, hollow = latent, ringed = a guaranteed key), squares are reference
  slots. Grey curves are "refers to"; orange arrows are Bayes-net dependencies
  (`bad_city ~ AddTypos(city.name)`). `n_r` on each card is the CRP count driving
  rich-get-richer.
- **Colour is the city name** an object ultimately hangs off, so two particles that
  disagree look different even as thumbnails.
- **The particle strip** shows every particle's net in miniature, its weight, and — after
  resampling — which particle it was copied from (`← P6`).
- **The side panel** explains the current step with the numbers behind it: the candidate
  distribution a proposal sampled from, the evidence table behind a rejuvenation move
  (counted once per *object*), the weights and ESS at a reweight, the ancestry at a
  resample.
- **The table** is the dirty data, cleaned by the particle you are looking at, with repairs
  marked as they happen.

Keys: `←`/`→` step, `Shift+←`/`→` jump a phase, `Space` play, `1`–`9` focus a particle.
The URL tracks the step (`#s=towns&step=138`), so a moment can be shared.

## Scenarios

| | |
|---|---|
| **towns** (default) | 25 rows, 10 practices, 3 towns from the hospital data. P01 misspells Abingdon as `abington` on all its rows, and `bethan` is exactly two edits from both Bethel and Dothan. SMC commits to the misspelling; the first rejuvenation sweep finds a genuine 50/50 and fixes it in half the particles; the next `abingdon` rows cost the rest a typo each; resampling (after row 11) keeps exactly the fixed ones. `bethan` ends split 53/47 between Dothan and Bethel — genuinely ambiguous, and the posterior says so. |
| **figure1** | The paper's Figure 1, scaled down: the misspelling outnumbers the truth 16 rows to 14, but on one practice against seven. The particles never disagree — every row is a key hit or a clear call — so this one is about rejuvenation. |
| **benchmark** | 30 real hospital rows with the per-row typo model, two reference hops from the clean name. |

The towns seed is chosen by a search over 40 seeds for the run that best shows the
designed story (resampling *after* rejuvenation, particles disagreeing at the end). Every
candidate is an honest run of the algorithm; the search only picks which one to show.

## How it works

`python/viz_record.py` drives `minipclean.particles.ParticleSMC` and `smc.Rejuvenator` one
step at a time with their decision logs switched on (`FlatProposer.log`,
`Rejuvenator.log` — off by default, and they never touch the RNG, so a logged run makes
exactly the choices an unlogged one would). After every step it diffs each particle's
database against the previous snapshot and writes the diffs plus the decision detail to
JSON. The app (`src/replay.ts`) replays the diffs once into an array of immutable states,
so scrubbing is an array index.

The multi-particle SMC here is the textbook form of [§3.1]: each particle is a whole
database. The Julia instead keeps one database and spends its particles within a row,
resampling between subproblem blocks — see the note at the top of `particles.py`.
