# Ranker

`rank.py` consumes `pool.json`, `draft.json`, normalized provider boards, and
`data_source_matches.json`, then publishes `rankings.json`.

```bash
uv run rank.py
uv run rank.py --report
uv run rank.py --no-draft
uv run rank.py --draft other.json
uv run rank.py --selftest
```

A live board is the simulation's starting state, not a post-processing filter. Made picks
remain on their rosters, only pending picks are played, and output ranking rows contain
undrafted players only.

## Modules

- `league.py`: league shape, the guillotine season structure, and strategy constants
- `pool.py`: pool document to `Player` objects with per-week point vectors
- `board.py`: live `draft.json` to the immutable starting state
- `opponents.py`: inferred provider boards to complete opponent strategies
- `value.py`: per-week expected lineup value, guillotine weighting, and wire measurement
- `guillotine.py`: elimination bars, week weights, and the waiver escalation
- `simulation.py`: one deterministic draft state and pick policies
- `convergence.py`: league-level fixed point
- `planning.py`: Monte Carlo availability, candidate survival, lookahead, and rollouts
- `rankings.py`: ranking rows and serialized next-pick recommendations
- `output.py`: top-level `rankings.json` payload
- `report.py`: human-readable stderr diagnostics
- `validate.py`: every-run output and league invariants
- `selftest.py`: solver, opponent-separation, planning, and malformed-board checks

## Value model

The board is ranked by `lineup_gain`, each player's marginal guillotine-weighted weekly
lineup value on my current roster at the converged levels.

A roster is valued week by week over the league's 17 weeks. The value input is
`weekly_points` from pool.json — DraftSharks' native per-week projections for weeks
1–17 in this league's scoring, so byes and known absences are explicit zero weeks. Each week is solved
under that week's actual starting shape from the expansion schedule (+1 WR wk 7, +1 RB
wk 9, +1 flex wk 12, +1 superflex wk 14, modeled as a second dedicated QB slot). The
17 weekly values are combined by the guillotine week weights.

`guillotine.py` measures those weights, per fixed-point iteration, from the simulated
rosters: an elimination Monte Carlo gives every team a persistent projection-error bias
plus weekly score noise, cuts the bottom two opponents each week, and reads off the
elimination bar (the second-lowest surviving opponent's score — beat it and I survive).
A week's weight is the marginal effect of a weekly point on log P(title), averaged
under the P(title)-weighted posterior over draws — seasons where my roster busts
dominate exactly the weeks they die in, so early safety is priced, while weeks cleared
in every draw get ~no weight (30th place survives week 1 exactly like 1st place). The
championship weeks carry the marginal effect on winning the week 16–17 total-points
final. Weights are normalized to sum to 1, so roster value reads as guillotine-weighted
expected weekly lineup points; scaling changes no decision.

The waiver wire is per-week and tiered: each roster holds WEEK_WIRE_BODIES
always-available bodies per position (more as rosters expand to half-waiver late
shapes), the j-th at the better of the j-th best undrafted player that week and the
(j × WIRE_DROP_RANK)-th best fresh eliminated-roster drop — the best cut players cost
real FAAB against ~30 competing survivors, and anything good dropped earlier was
claimed long ago. This is why drafted depth matters most early, why late expanded
slots are cheap to fill (the draft-time face of delaying FAAB spending), and why
drafted stars keep their value all season.

The levels affect my choices, and those choices affect who remains undrafted and which
rosters get eliminated. This creates a fixed point:

```text
levels (weights + weekly wire) -> roster value -> simulated draft -> guillotine MC -> levels
```

The map is discrete and can alternate between neighboring league shapes, so convergence
detects a repeated draft outcome and averages levels over that cycle. The single-week
solver values the weekly re-optimized lineup: the starter composition is re-chosen per
availability draw, so a flex seat vacated at one position is refilled by the best
remaining body at any flex position. It is exact and closed-form — dedicated slots via a
per-position Bernoulli cascade ordered by points when active, the week's FLEX seats via
layer-cake integrals of the pooled RB/WR/TE marginal count (a QB can never take a flex
seat) — and `--selftest` checks it against brute-force enumeration of every
availability subset for every in-season shape, tiered wire bodies included. A waiver
body can fill one job; it is not an unlimited scalar. Expectation-of-max keeps value
monotone when a projection improves or a player is added.

## Opponents and planning

Personal and opponent strategies are intentionally separate. My slot alone uses
projections and expected-lineup roster value. Each opponent uses the provider board
closest to its completed picks, with a soft boost for unfilled dedicated starters and a
compounding source-rank penalty for adding players beyond comfortable positional depth
(2nd QB/TE, 4th RB/WR). These are preferences, not draft limits: a large enough
source-rank gap can still justify another player at a deep position. `MAX_POSITIONS`
stays the hard-limit mechanism, but this league sets no per-position caps, so it is
pinned at the roster size and never binds.
Observed `mean_log2_loss` calibrates randomness around that preference, after shrinking
toward the cold-start prior as if that prior had been observed on two extra picks — one
or two on-board picks otherwise fit a near-deterministic policy.
`OPPONENT_POSITION_TILT` is empty until this league's own draft supplies replay
evidence (`evaluate_opponents.py`); the previous league's RB tilt was fitted to its
draft, not this one. Missing provider
players are appended in consensus-average order (the investigator's `consensus`
source, which also models opponents with no observed picks); opponents never fall
back to my board.

My simulated pick policy does not use those targets or any other positional roster-size
heuristic beyond the caps. Positional depth is priced only by projected expected-lineup
value, so a roster shape that differs from the opponents' conventional behavior can be a
source of value.

The bulk deterministic policy scores value now plus the expected best option at its next
pick. The live shortlist starts from the current board before intervening opponents pick,
then removes candidates below 5% survival to my turn. Candidate branches are evaluated
conditional on reaching that turn. Four-pick planning applies the same 5% floor to each
later target's conditional survival before playing finalists to the end of the draft.
The first `take` is the EV recommendation if available; when the noiseless example has
already removed it, `deterministic_fallback` identifies the example draft's legal choice.
Worker processes receive immutable inputs once, and seeded task ids keep results
deterministic across scheduling. Planning uses at most eight workers so a rerank does not
saturate every host CPU; this changes elapsed time, not the simulations or their output.

## Output contract

`rankings.json.rankings` is ranked by the roster-aware `lineup_gain` decision metric and
contains projected and simulated pick fields, opponent consensus deltas, and
availability estimates for each undrafted player.
`my_next_picks` is the full recommendation and can intentionally disagree with the
static board order. `example_draft` contains structured final-roster records for dashboard
lineup comparison. `validation.problems` is empty on success; any problem makes the CLI
exit nonzero.
