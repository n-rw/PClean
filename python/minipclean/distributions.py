"""Distributions — mirrors ``src/distributions/*.jl``.

Every distribution in PClean answers five questions. Four are ordinary; the fifth is
the one that makes PClean fast, and it is worth understanding before anything else.

    random(*args)                  sample from the prior
    logdensity(observed, *args)    score an observation
    has_discrete_proposal()        can you enumerate my plausible values?
    discrete_proposal(*args)       ...if so, give me (options, log_priors)
    discrete_proposal_dummy_value  ...and a stand-in for "none of the above"

`discrete_proposal` is the fifth. A generic PPL proposes latent values by sampling the
prior and hoping — for a string-valued attribute that is hopeless. PClean instead asks
the distribution to hand over a *finite candidate list*, then scores every candidate
against the data exactly [§3.2]. That turns a blind guess into an enumeration.

When the true support is infinite (any string of length 3..30), the candidate list is a
`preferring` hint plus a `DUMMY` token carrying the leftover prior mass [§3.3]. If no
candidate explains the data, DUMMY wins and we fall back to sampling the prior. That is
the "adaptive mixture proposal" of the paper, and it is only a few lines below.
"""

import math
import random as _random
from functools import lru_cache


class ProposalDummyValue:
    """The "none of the above" token — mirrors Julia's ``ProposalDummyValue``.

    Stands for the entire tail of an infinite support that the preferred-value list
    does not cover. Carries whatever prior mass the named candidates left over.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<DUMMY>"


DUMMY = ProposalDummyValue()

IMPOSSIBLE = -1e5  # Julia's `const IMPOSSIBLE` in add_typos.jl. Not -inf: we want a
                   # very bad explanation to remain *an* explanation, so inference can
                   # back out of it later rather than hitting a zero-probability wall.


def logsumexp(xs):
    """Numerically stable log(sum(exp(x))). Used everywhere to normalise enumerations."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Distribution:
    """Base class — mirrors Julia's ``abstract type PCleanDistribution``.

    Julia dispatches on the distribution's *type* via multiple dispatch; Python uses
    ordinary method calls on an instance. Same effect, less magic.
    """

    def random(self, *args):
        raise NotImplementedError

    def logdensity(self, observed, *args):
        raise NotImplementedError

    def has_discrete_proposal(self):
        return False

    def discrete_proposal(self, *args):
        """Return ``(options, log_priors)``. Only called when has_discrete_proposal()."""
        raise NotImplementedError

    def discrete_proposal_dummy_value(self, *args):
        raise NotImplementedError

    def supports_explicitly_missing_observations(self):
        """If True, this distribution can score `observed=None` itself.

        If False, a missing observation means the node is simply unconstrained and the
        proposal machinery skips scoring it. The distinction matters: `MaybeSwap` wants
        to *insist* that an imputed value be one of the legal options even when nothing
        was observed, which is a real constraint, not an absence of one.
        """
        return False


# --------------------------------------------------------------------------------
# Unmodeled — mirrors src/distributions/unmodeled.jl
# --------------------------------------------------------------------------------

class Unmodeled(Distribution):
    """A value we take at face value and never try to correct.

    Used for fields that are trusted and always observed (a street address, an ID).
    Its logdensity is a flat 0.0 — every value is equally fine — so it contributes
    nothing to inference *except* identity: two records with different `addr` cannot
    be the same Practice. That is exactly what you want a key to do.
    """

    def random(self, *args):
        raise RuntimeError("Sampling an unmodeled value — the model is asking for a "
                           "value it was told it would always observe.")

    def logdensity(self, observed, *args):
        return 0.0

    def supports_explicitly_missing_observations(self):
        return True


# --------------------------------------------------------------------------------
# ChooseUniformly / ChooseProportionally — mirrors choose_uniformly.jl,
# choose_proportionally.jl
# --------------------------------------------------------------------------------

class ChooseUniformly(Distribution):
    """Pick one of `options` uniformly. The simplest enumerable distribution."""

    def random(self, options):
        return _random.choice(list(options))

    def logdensity(self, observed, options):
        options = list(options)
        n = sum(1 for o in options if o == observed)
        if n == 0:
            return -math.inf
        return math.log(n) - math.log(len(options))

    def has_discrete_proposal(self):
        return True

    def discrete_proposal(self, options):
        options = list(options)
        lp = -math.log(len(options))
        return options, [lp] * len(options)


class ChooseProportionally(Distribution):
    """Pick from `options` with probabilities `probs`.

    `probs` may be a plain list, or a `ProportionsParameter` whose values are *learned
    from the dirty data* as inference proceeds. That second case is how the paper's
    headline example works: nobody tells the model that PCOM awards mostly DOs, it
    counts them [Fig. 1].
    """

    def _values(self, options, probs):
        if isinstance(probs, ProportionsParameter):
            return probs.value(len(options))
        return list(probs)

    def random(self, options, probs):
        p = self._values(options, probs)
        return _random.choices(list(options), weights=p, k=1)[0]

    def logdensity(self, observed, options, probs):
        options, p = list(options), self._values(options, probs)
        total = sum(p)
        hits = [math.log(pi / total) for o, pi in zip(options, p) if o == observed]
        return logsumexp(hits) if hits else -math.inf

    def has_discrete_proposal(self):
        return True

    def discrete_proposal(self, options, probs):
        options, p = list(options), self._values(options, probs)
        total = sum(p)
        return options, [math.log(pi / total) if pi > 0 else -math.inf for pi in p]


class ProportionsParameter:
    """A Dirichlet-categorical whose counts are learned — mirrors
    ``ProportionsParameter`` in choose_proportionally.jl.

    This is the `@learned` / `parameter` construct [§2.1]. It holds sufficient
    statistics (`counts`) rather than the data itself, so incorporating a choice is
    `counts[i] += 1` and unincorporating is `counts[i] -= 1`. Inference calls those
    constantly as it revises hypotheses, which is why conjugacy matters so much here
    [App. D.2]: without it every revision would be a re-scan of the dataset.
    """

    def __init__(self, concentration=1.0):
        self.concentration = concentration
        self.counts = []
        self._value = []

    def value(self, n_options):
        """Current proportions, lazily sized to the option list on first use."""
        if len(self._value) != n_options:
            self.counts = [0] * n_options
            self._value = [1.0 / n_options] * n_options
        return self._value

    def incorporate(self, options, observed):
        options = list(options)
        if observed in options:
            self.counts[options.index(observed)] += 1

    def unincorporate(self, options, observed):
        options = list(options)
        if observed in options:
            self.counts[options.index(observed)] -= 1

    def reset(self):
        """Zero the sufficient statistics, ready to be recounted."""
        self.counts = [0] * len(self.counts)

    def resample(self):
        """Redraw proportions from the Dirichlet posterior given current counts.

        NOTE (divergence): the Julia draws from a true Dirichlet. We use the posterior
        *mean*, which is deterministic and makes runs reproducible while you are reading.
        For understanding the algorithm this is the right trade; for matching the Julia's
        sampling behaviour exactly, it is not.
        """
        a = [self.concentration + c for c in self.counts]
        total = sum(a)
        self._value = [ai / total for ai in a]


# --------------------------------------------------------------------------------
# MaybeSwap — mirrors maybe_swap.jl
# --------------------------------------------------------------------------------

class MaybeSwap(Distribution):
    """Report `val` faithfully with probability 1-p, else report a random option.

    The simplest possible error model, and the one used for categorical fields like a
    medical credential. `p` is typically a *learned* error rate, so the model works out
    for itself how noisy the column is.
    """

    def random(self, val, options, p):
        p = p.value() if isinstance(p, ProbParameter) else p
        if _random.random() < p:
            return _random.choice(list(options))
        return val

    def logdensity(self, observed, val, options, p):
        p = p.value() if isinstance(p, ProbParameter) else p
        # Nothing observed: this is an *imputation*. We do not get to say anything about
        # which value is right, but we do insist it be a legal one. The Julia uses a
        # hard -1000.0 for illegal values; same idea, kept verbatim.
        if observed is None:
            return 0.0 if val in list(options) else -1000.0
        if val == observed:
            return math.log1p(-p)
        return math.log(p) - math.log(len(list(options)))

    def has_discrete_proposal(self):
        return False

    def supports_explicitly_missing_observations(self):
        return True


class ProbParameter:
    """A learned Beta-Bernoulli rate — mirrors ``ProbParameter`` in maybe_swap.jl.

    Same sufficient-statistic trick as ProportionsParameter: keep (heads, tails), not
    the observations.
    """

    def __init__(self, a=1.0, b=1000.0):
        self.a, self.b = a, b       # beta(1, 1000): errors are rare a priori
        self.heads = self.tails = 0

    def reset(self):
        self.heads = self.tails = 0

    def resample(self):
        pass    # `value()` already returns the posterior mean; nothing to redraw

    def value(self):
        return (self.a + self.heads) / (self.a + self.heads + self.b + self.tails)

    def incorporate(self, was_error):
        if was_error:
            self.heads += 1
        else:
            self.tails += 1

    def unincorporate(self, was_error):
        if was_error:
            self.heads -= 1
        else:
            self.tails -= 1


# --------------------------------------------------------------------------------
# AddTypos — mirrors add_typos.jl
# --------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance allowing insert, delete, substitute and *transpose*.

    Transposition matters: `teh` for `the` is one slip of the fingers, and a model that
    charges it as two edits will under-rate the commonest typo there is.

    NOTE (divergence): the Julia calls out to StringDistances.jl. This is the textbook
    DP, written out so you can see the four edit operations as four branches. It is
    `lru_cache`d because inference scores the same (observed, candidate) pair over and
    over — the Julia does the same thing with an explicit `add_typos_density_dict`.
    """
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # delete
                          d[i][j - 1] + 1,          # insert
                          d[i - 1][j - 1] + cost)   # substitute
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)   # transpose
    return d[la][lb]


class AddTypos(Distribution):
    """`observed` is `word` with some typos in it.

    This is the distribution that does the real work in the paper's Figure 1, and the
    reason it beats majority vote. Read the generative story carefully:

        num_typos ~ NegativeBinomial(len(word)/5, 0.9)
        repeat num_typos times: insert / delete / substitute / transpose at random

    The key is *where this lives in the model*. In the hospital and physician programs,
    the typo'd city name is an attribute of the **Practice**, not of the Record. So all
    152 rows of one practice share one `bad_city` draw. A misspelling repeated across
    those 152 rows is therefore **one error**, costing one `num_typos` penalty — not 152
    independent votes for the wrong spelling. That is how "Abington" (152 occurrences)
    loses to "Abingdon" (42) [§4, Fig. 1].

    Put the same distribution on the Record instead and the model would confidently
    conclude the misspelling is correct. The error model's *position in the schema*
    carries as much information as its shape.
    """

    def __init__(self, letters_per_typo=5.0, typo_prob=0.1):
        self.letters_per_typo = letters_per_typo
        self.typo_prob = typo_prob

    def random(self, word, max_typos=None):
        n = min(_random.randint(0, 3), max_typos if max_typos is not None else 3)
        out = word
        for _ in range(n):
            op = _random.choice(["insert", "delete", "substitute", "transpose"])
            i = _random.randrange(max(1, len(out)))
            ch = _random.choice("abcdefghijklmnopqrstuvwxyz")
            if op == "insert":
                out = out[:i] + ch + out[i:]
            elif op == "delete" and out:
                out = out[:i] + out[i + 1:]
            elif op == "substitute" and out:
                out = out[:i] + ch + out[i + 1:]
            elif op == "transpose" and i + 1 < len(out):
                out = out[:i] + out[i + 1] + out[i] + out[i + 2:]
        return out

    def logdensity(self, observed, word, max_typos=None):
        if observed is None:
            return 0.0
        if not word:
            return IMPOSSIBLE
        n = damerau_levenshtein(str(observed), str(word))
        # `max_typos` is a modelling choice with teeth. Set it too high and the model
        # cheerfully "corrects" AGUADILLA to AGUADA -- two different real towns three
        # edits apart. Set it to 1-2 and it catches genuine slips only.
        if max_typos is not None and n > max_typos:
            return IMPOSSIBLE

        # Negative binomial on the typo count: longer words earn more typos.
        r = max(1.0, math.ceil(len(word) / self.letters_per_typo))
        p = 0.9
        ll = (math.lgamma(n + r) - math.lgamma(r) - math.lgamma(n + 1)
              + r * math.log(p) + n * math.log1p(-p))
        # Each typo also had to choose a position and a letter. Dividing that cost out
        # is what stops the model preferring long words as explanations for everything.
        ll -= math.log(len(word)) * n
        ll -= math.log(26) * n / 2.0
        return ll

    def has_discrete_proposal(self):
        return False

    def supports_explicitly_missing_observations(self):
        return True


# --------------------------------------------------------------------------------
# StringPrior — mirrors string_prior.jl
# --------------------------------------------------------------------------------

class StringPrior(Distribution):
    """A prior over strings, with a `preferring` candidate list [§3.3].

    The support is every string of length `min_len..max_len` — astronomically large and
    utterly unenumerable. `preferring` is how PClean makes it tractable: the user hands
    over a shortlist of values the posterior is expected to concentrate on (in practice,
    "the spellings actually seen somewhere in this column"), and enumeration considers
    only those, plus DUMMY for everything else.

    That yields the **adaptive mixture proposal**: if any candidate explains the data
    well it will be proposed; if none do, DUMMY dominates and we sample the prior. So
    the hint changes the *proposal*, never the model — a wrong shortlist costs you speed
    and accuracy, not correctness.

    NOTE (divergence): the real one scores unlisted strings with a character-bigram
    language model fitted to English. We use a flat per-character cost, which preserves
    the shape that matters (longer strings are less likely) without the CSV of bigram
    frequencies.
    """

    def __init__(self, min_len, max_len):
        self.min_len, self.max_len = min_len, max_len

    def _log_prior(self, s):
        if not (self.min_len <= len(s) <= self.max_len):
            return -math.inf
        n_lengths = self.max_len - self.min_len + 1
        return -math.log(n_lengths) - len(s) * math.log(28)   # 26 letters + space + dot

    def random(self, candidates=()):
        if candidates:
            return _random.choice(list(candidates))
        n = _random.randint(self.min_len, self.max_len)
        return "".join(_random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))

    def logdensity(self, observed, candidates=()):
        if observed is None:
            return 0.0
        return self._log_prior(str(observed))

    def has_discrete_proposal(self):
        return True

    def discrete_proposal(self, candidates=()):
        """Return the shortlist plus DUMMY holding all remaining prior mass."""
        candidates = [c for c in dict.fromkeys(candidates)]   # de-dup, keep order
        lps = [self._log_prior(c) for c in candidates]
        named = logsumexp(lps)
        # Leftover mass = 1 - sum(named). Guard the log when the shortlist already
        # accounts for essentially everything.
        leftover = math.log1p(-math.exp(named)) if named < -1e-9 else IMPOSSIBLE
        return candidates + [DUMMY], lps + [leftover]

    def discrete_proposal_dummy_value(self, candidates=()):
        """A stand-in value if DUMMY is chosen — the Julia returns '***...'."""
        return "*" * ((self.min_len + self.max_len) // 2)


class IndexedParameter:
    """One learned parameter per key — mirrors ``IndexedParameter`` in distributions.jl.

    `degree_dist[school.name]` gives a separate ProportionsParameter for every school,
    created on first use. The paper's Physicians model has one of these per school (396
    of them) and one per degree, all from two lines of the program [App. B.4.4].
    """

    def __init__(self, factory):
        self.factory = factory
        self.table = {}

    def __getitem__(self, key):
        if key not in self.table:
            self.table[key] = self.factory()
        return self.table[key]

    def reset(self):
        for p in self.table.values():
            p.reset()

    def resample(self):
        for p in self.table.values():
            p.resample()
