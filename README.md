# 2026 guillotine

A draft and in-season toolkit for a 32-team 0.5 PPR guillotine redraft league on
Sleeper ("Gnosis Guillotine", league 1397662420398247936). The code is three packages:

- `shared/`: what both halves agree on. The league's shape (`league.py`), the weekly
  score noise model (`noise.py`), the Sleeper client (`sleeper.py`), name resolution
  across data providers (`names.py`), the projection fetch and pool build, the six-worker
  cap (`workers.py`), every artifact path (`paths.py`) and the dashboard server.
- `draft/`: the live draft board. Sleeper draft fetch and geometry, the pool of
  valued players, the guillotine-weighted roster valuation, the opponent model, the
  fixed-point level convergence, the pick planner, and `sources/` (provider boards and
  the per-drafter source investigator).
- `season/`: the in-season desk. Sleeper league fetch, the season state and weekly
  lineup, the FAAB bidding model, the agent-based elimination race, and this week's
  claims priced by replayed title odds.

Generated artifacts live in `out/` (`pool.json`, `draft.json`, `rankings.json`,
`data_source_matches.json`, `league.json`, `season.json`, `bidding_evaluation.json`)
next to the three dashboards that read them (`index.html` is the season desk,
`draft.html` the draft board, `sources.html` the source investigator); `shared/serve.py`
serves that directory. Provider snapshots live with the code that reads them
(`shared/data/`, `draft/sources/data/`).

## League assumptions

- 0.5 PPR with a +1.0/rec tight end premium, 1 QB (superflex arrives in week 14)
- Guillotine: the two lowest weekly scores are eliminated each of weeks 1–15 and
  their players go to FAAB waivers; the last two teams play a week 16–17
  total-points championship
- Opening starters: 1 QB, 1 RB, 2 WR, 1 TE, 2 W/R/T flex; no D/ST or kicker slot;
  lineups expand in-season (+1 WR wk 7, +1 RB wk 9, +1 flex wk 12, +1 superflex
  wk 14, bench grows from 1 to 5, assumed one spot at each expansion)
- $1000 FAAB, claims processed once a week, unclaimed players free afterwards
- 2 reserve spots holding Out/IR/PUP players (reserve is not drafted into), no
  per-position roster caps
- 32 teams and 8 drafted players per team (256 picks, all offense)
- Snake draft with a third-round reversal: round 1 forward, rounds 2–3 reversed,
  alternating from there; picks can be traded
- My slot is 20 (johnor): 1.20, 2.13, 3.13, 4.20, 5.13, 6.20, 7.13, 8.20 before trades

The league's shape is `shared/league.py`, the draft's geometry and strategy knobs are
`draft/settings.py`; neither is runtime configuration. The draft board loader and the
season state loader complain loudly when Sleeper disagrees with them.

## Setup

[uv](https://docs.astral.sh/uv/) pins Python 3.12 and installs the three packages into
the environment, so every process runs as a module from anywhere in the repository:

```bash
uv sync
uv run -m season.refresh
```

Dependencies are numpy and numba: the bidding model's lineup arithmetic is compiled
(`season/waivers.py`), which is what makes the season model fast. The first run
compiles it (about twenty seconds); the compiled code is cached under
`season/__pycache__` after that.

## Workflows

Refresh projections and league information, compute this week's FAAB bids, and
optimize the lineup. Run before the week's waiver deadline, and again before games
start to update the lineup. No manual input:

```bash
uv run -m season.refresh
```

It refetches projections (DraftSharks weekly for the weeks still to play, Sleeper
season), fetches the league state (including Sleeper weekly projections), and runs the
season model. The NFL week comes from Sleeper. Each stage must succeed before the next
starts; a failure exits nonzero without printing recommendations from an older run.
The season model takes about two minutes at 8 workers; `--sims 256` runs it in
under half a minute for a quick check, with correspondingly noisier odds.

The terminal summary shows the remaining budget, any reserve body to activate and the
forced cut that makes room for him, recommended bids with their drops and reserve
moves, and the optimal lineup with start/sit changes. Bids are evaluated individually,
so treat them as alternatives. After this week's waivers process, players dropped
since are still on waivers: their recommendations are claims with a bid, the
approximate time they clear, win chance, and break-even. Everyone else is a free
pickup to add now. Enter the recommended claims and lineup on Sleeper yourself. Add
`--report` for detailed model diagnostics and league odds. The season model alone is
`uv run -m season.run`, with the same flags.

Results are saved to `out/league.json` and `out/season.json`; `uv run -m shared.serve`
displays them at http://127.0.0.1:8123/. The season model reads refreshed
weekly projections directly and uses the existing `pool.json` only to join player
IDs, so this workflow does not need a pool rebuild or a new hand-saved DraftSharks
page.

Refresh the live board and recommendations between picks (Sleeper's draft API is
real-time, so this is the whole live loop):

```bash
uv run -m draft.refresh --report
```

It runs three steps: fetch the draft, re-run source inference against the existing
boards snapshot, then rank. It never rebuilds the pool or fetches provider boards.

Rebuild the pool after refetching projections or saving a new DraftSharks page to
`shared/data/projections.html`:

```bash
uv run -m shared.fetch_projections      # Sleeper season + DraftSharks weekly, manual
uv run -m shared.build_pool --report
```

Refresh the provider boards (rebuild the pool first if it changed, since every board
resolves onto it):

```bash
uv run -m draft.sources.fetch_rankings
uv run -m draft.sources.build_rankings --report
```

Follow a different draft, such as a league mock at `sleeper.com/draft/nfl/<id>`:

```bash
uv run -m draft.fetch --draft-id <id>
uv run -m draft.sources.investigate
uv run -m draft.rank --report
```

Offline checks and evaluations:

```bash
uv run -m unittest discover -p tests.py   # draft/tests.py and season/tests.py
uv run -m draft.evaluate_opponents         # replays every completed opponent pick through the model
uv run -m season.evaluate_waivers          # -> bidding_evaluation.json
```

Before and after changing the draft opponent model, compare
`draft.evaluate_opponents`'s replay accuracy. `season.evaluate_waivers` holds out every
bid on a player before predicting that player's positive submitted bids, and predicts
the latest auction from the earlier ones alone, comparing the guide prior with the
fitted price curve by mean absolute log error. It also compares my future bid values
(`race.PRICES`, what a point of season gain is worth) on paired opponent seasons with
a separate seed, each against the best, then
repeats with every opponent active, and reconstructs the current week's pre-auction
rosters to exercise paid claim pricing with the learned model (a diagnostic using
observed bids, not an ex-ante backtest). A few observed auctions cannot establish
future activity or saving habits, or validate absolute championship probabilities.

## CPU limits

Draft and season simulation pools use at most **six worker processes**, limited
further by CPU affinity when available (`shared/workers.py`). Simulation counts,
candidate searches, and seeded results are unchanged; runs take longer instead of
using every CPU. This is a limit per process pool, so run one pipeline at a time.

On this Windows/WSL desktop, `C:\Users\johnm\.wslconfig` also sets a limit for the
whole WSL 2 VM, shared across its distributions:

```ini
[wsl2]
processors=6
```

That file lives outside the repository. The limit takes effect after WSL shuts down
and restarts. Save work in WSL, run `wsl --shutdown` from **Windows PowerShell**, then
reopen the terminal; `nproc` should report `6`. This does not limit native Windows
programs. See [Microsoft's WSL configuration reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

These limits reduce load; they do not repair a machine that abruptly powers off
under load. The September 15 shutdown logged Kernel-Power event 41 with no bugcheck
code, which does not establish the cause. Check CPU cooling, power supply, and any
overclock/undervolt settings before trying sustained full-load tests; see
[AMD's stability troubleshooting](https://www.amd.com/en/resources/support-articles/faqs/PIBRMATS2.html).

## Data flow

```text
shared/  data/projections.html + data/sleeper_projections.json + data/weekly_projections.json
         └─ build_pool ──────────────────────────────────> pool.json ─────────────────┐
draft/   sources.fetch_rankings -> sources/data/raw/ -> sources.build_rankings         │
                                                        -> sources/data/boards.json ───┤
draft/   fetch ──────────────────────────────────────────> draft.json ────────────────┤
draft/   sources.investigate (boards + draft) ───────────> data_source_matches.json ──┤
draft/   rank ───────────────────────────────────────────> rankings.json <────────────┘

season/  fetch_league (Sleeper rosters, FAAB, moves, weekly projections) -> league.json ─┐
shared/data/weekly_projections.json + pool.json (id join; name fallback via league.json) ─┤
season/  run ────────────────────────────────────────────> season.json <───────────────────┘
```

- `pool.json`: ~370 QB/RB/WR/TE players. DraftSharks supplies identity, age, bye,
  rookie flag and 1QB ADP from a hand-saved rankings page; a Sleeper season projection
  gates membership and supplies `sleeper_id`; `weekly_points` is DraftSharks' per-week
  projection for weeks 1–17 in the league's scoring, with byes and known absences as
  zero weeks. It is the draft's value input, after `draft/pool.py` blends each season
  total 2:1 with Sleeper's league-scored projection (`points`).
- `draft.json`: all 256 made and pending picks from Sleeper's public, real-time draft
  API, with pending picks derived from the draft settings and traded picks applied.
- `draft/sources/data/boards.json`: provider boards in two families, ordinary 1QB
  redraft (FantasyCalc, KeepTradeCut, FF Calculator ADP, FantasyPros ECR, DraftSharks
  ADP, Sleeper ADP) and the superflex/2QB variant of each provider for this room's QB
  scarcity, plus Sleeper's league-scored points order and a value-over-replacement
  board on that projection under the opening lineup across 32 teams (the two boards
  that price the TE premium; the VORP board is what an LLM handed the league id and
  Sleeper's API arrives at), a consensus average, and the `cold_start` room prior:
  50% Sleeper's half-PPR ADP, which is what the draft room displays and autopick
  drafts from, 30% the VORP board for the LLM-assisted minority this office is
  expected to hold, and 20% the format-adjusted boards. Each row is resolved to the
  pool's `sleeper_id` by `shared/names.py`.
- `data_source_matches.json`: for each drafter, the board closest to its picks so far,
  with fit scores and pick-level evidence.
- `rankings.json`: undrafted-player rankings, next-pick recommendations, the example
  draft, and validation.
- `league.json`: the in-season state from Sleeper: the current NFL week, every roster
  with its players, set starters, reserve, FAAB used and points, every waiver claim and
  free-agent move so far with its bid, a directory of every referenced player (name,
  position, team, injury status), and Sleeper's weekly projections in league scoring for
  the current week through week 17. A roster the commissioner has emptied after week 1
  is an eliminated team (Sleeper has no guillotine flag). It also records
  `waiver_clear_days`, which the season model checks against `shared/league.py`.
  Transactions retain Sleeper's submission `leg`, ID and processing timestamp; `week`
  follows processing time relative to Sleeper's season-start date. Wednesday claims may
  remain under the previous submission leg. Both successful and failed bids are
  retained; pending claims do not establish that waivers have processed.
- `season.json`: this week's optimal lineup and the moves it implies, the optimal FAAB
  bid on each free agent worth a look with the drop and any move onto reserve it needs
  (after the weekly run, the players still on waivers carry `waiver_clears`, and the
  rest are free adds), the cut that must come first when a reserve body lost
  Out/IR/PUP status and the roster no longer fits (claims are evaluated on the roster
  after that cut), which objective my bidding uses (points alike every week or
  title-weighted, by replayed title odds) with the per-week title weights, every team's
  chance of being cut this week, of reaching the final and of the title, its expected
  spend and budget path, the elimination bar by week, and the observed versus simulated
  waiver market. Budget paths show cash entering each week, before that week's claims,
  conditional on surviving to that week. The current week's opening cash adds back
  completed claims to the live balance. Post-claim cash is recorded separately. Already
  eliminated teams have zero survival odds and no future budget estimates, and are
  excluded from league averages.

`sleeper_id` is the cross-process player key; `roster_id` and `draft_slot` connect
opponent source matches to the live board.

## Draft method

The guillotine is the objective. A roster is valued week by week as the expected
optimal lineup under that week's starting shape and position-wide availability, using
per-week projections, and the 17 weekly values are combined by guillotine week weights:
each week's weight is the marginal effect of a weekly point on log P(surviving that
week's cut), measured by simulating the elimination race over the opponents' simulated
rosters, with the championship weeks entering through log P(winning the final). The
waiver wire is per week and tiered: one free-agent pool, the undrafted tail plus every
roster eliminated so far, split equally among the survivors, so in week 1 a lone
undrafted starter is worth a thirty-second of himself and by the final the wire is
other teams' first-round picks; drafted depth is worth nothing by then while drafted
stars still clear it. That equal split prices the opponents and so the elimination
bars; my own roster is priced under my FAAB policy, since the draft should not assume
free claims I do not intend to make: the undrafted tail alone through week 8 while I
hold the budget, the equal split through the week 9-12 lineup expansion, and the top
half of every tier from week 13 once the saved budget is spent. Levels and the
simulated draft are a fixed point that converges to a limit cycle.
My slot alone uses this objective, on DraftSharks projections blended 2:1 with
Sleeper's so the draft does not build around one model's outliers; each opponent
follows the external board most associated with its picks (the `cold_start` blend
until it has any), with roster-balance adjustments, fitted choice noise, a residual
TE tilt for the share of the room that half-notices the premium without a board
that prices it, and QB scarcity sense (a team without a quarterback never takes one
who does not start week 1, and takes a starter once the run leaves none likely to
last to its next pick), and never sees my projections. A drafter named in
`OPPONENT_INTEL` (`draft/settings.py`) follows its stated plan for its next picks
before any of that applies. The first pending decision searches target plans across
my next four held picks and plays each plan out to the end of the draft. See
`draft/rank.py` and the `draft/` module docstrings for the details.

## In-season method

The season desk runs an agent-based race from the live state (`season/race.py`). The
value input is per week: DraftSharks' weekly projection blended 2:1 with Sleeper's for
the same week, Sleeper alone for anyone DraftSharks does not project
(`season/state.py`). DraftSharks rows join Sleeper ids through `pool.json`; a row
outside the draft pool, such as a backup who became a starter, joins `league.json`'s
directory by normalized name and position when that names exactly one player.
Each simulated week every alive team fields its optimal lineup under the draft model's
noise, the two lowest are cut, and their players hit the wire. `season/waivers.py`
values a claim by its projected net lineup points through Week 17, per remaining week,
and gives the room's bids a reference price from [Paul Charchian's guillotine FAAB guide](https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2):
early elite players about 15–20% of the starting budget, ordinary starters 2.5–5%,
and depth 0.1–1%. This is an 18-team guide, not a fitted rule for our 32-team format,
and it prices nothing on my side (my bids are below). The room's reference uses
league-scored positional ranks by points per game played for the rest of the season
(elite anchors QB4/RB6/WR6/TE3, then an inverse-square price curve), the claim's net
lineup points, remaining cash and weeks, and projected cut risk. Ranking by games
played keeps a player returning from an absence from being charged twice: the net
lineup points already count only the weeks he plays. Opponents weight the week they
are bidding for 32 times each later week, in
both the ranks and the net lineup points (`ROOM_CURRENT_WEEK_WEIGHT`): the room pays for
this week's fill-ins and passes on injured stashes. In an ex-ante backtest of the
week-3 auction, fit on week-2 bids from the Tuesday state, bid sizes, who got claimed
and clearing prices all improved as that weight rose to 32-64; 32 predicted who got
claimed best, and valuing that week alone did worse. Each candidate's drop is chosen
with him: the body whose loss leaves the best remaining-season roster with the
candidate on it, including distant byes and superflex, so a backup QB goes when a
better QB arrives rather than a bench RB. Opponents charge the drop's whole remaining
season, so a returning starter is not a free placeholder for a short-term fill-in. The
two reserve slots hold Out/IR/PUP bodies while their projection is zero; once a body's
projection resumes he needs a regular spot, and every simulated team cuts its least
valuable body to make room before that week's claims; a cut is charged its whole
loss, since no spot is left to refill.

My own bidding is chosen for title odds (`season/claims.py` `title_objective`). A
pickup may be held only for its useful weeks: afterwards the vacated spot is refilled
from the wire the room leaves untaken (free agents taken, by claim or free pickup, in
fewer than half of the opponent seasons at the next auction, assumed to stay
available), and the drop is charged only what the best such body cannot restore. A
one-week starter can then displace depth the wire replaces for free, while a
returning starter nothing on the wire replaces is still charged in full. Each run
also derives per-week title weights, d log P(title) / d(points), the draft's week
weights: the standing roster's replay through the recorded opponent races, the
per-week survival hazard and championship term weighted by each season's title
probability, normalized to average 1. My bidding weights weeks by them only if that
replays the standing roster to better title odds than weighting every week alike.
In the week-3 state it did not (16.2% against 17.1%): the weights are a first-order
fit computed once, so a policy on them gives up points in weeks that look safe until
they are not. The chosen objective screens this week's candidates, ranks each one's
drops, values my future claims, and drives my roster cuts in the replays. Opponents
keep the points behavior above, with their current-week weight. My future bids
(below) take nothing from the guide: a claim is worth a price per point of its gain
that the replays choose, and the bid is shaded against the recorded market, with no
separate saving plan.

This week's claims are decided by replay, not by that heuristic. Each candidate's
three best drops by the heuristic are replayed and the best by title odds is kept: the
heuristic prices a fixed roster, where a future lineup expansion is an empty seat
worth a body's full points, so in the week-3 state it would have dropped Michael
Penix, the week's starting QB, for a receiver the wire could supply by week 7. The bid
is then whatever maximizes replayed title odds anywhere in the budget, with no
ceiling from the future policy: each candidate bid is scored per recorded season, won
or lost at that season's price. The break-even ("worth up to") is the bid at which
winning no longer beats standing pat.

Opponents' bids follow one room-wide price curve fit to every submitted bid, including
losses: log bid = a + b log(guide reference), the reference being the guide ceiling for
that manager's roster, cash and cut risk. This room is flatter than the guide (b about
0.69 after two auctions): depth and fill-ins sell for several times their guide price,
stars for less. The curve's residual spread is each bid's noise. There are no
per-manager bid multipliers: managers' levels around the curve did not persist from
the week-2 auction to week 3 (correlation -0.24), and multipliers fit on week 2 predicted
week 3 worse than the curve alone. Participation does persist (15 of 16 week-2 bidders
bid again), so each manager keeps his own participation probability under a
Beta(2/3, 1/3) prior worth one auction, rising toward the end of the season; that
reactivation curve is an assumption. Duplicate team/player/week claims use the latest
submission; a winner's roster and budget are rolled back before measuring his need.
Earlier bids use current projections as a proxy for historical player value. The curve,
participation estimates and held-out bid-size errors (per player, and the latest auction
predicted from the earlier ones alone) are published in `season.json` under `market`.

Opponents' saving habits are a uniform prior over three persistent season-long plans:
no reserve, balanced saving, and patient saving (`waivers.SAVING_PLANS`; my own agent
does not use them). The balanced plan targets 75% of
cash entering Week 9, 25% entering Week 13, and 20% entering Week 14, informed by
[Charchian's month-by-month guidance](https://www.fantasylife.com/articles/guillotine-leagues/how-to-manage-your-faab-in-guillotine-league-fantasy-football).
The patient plan targets 85%, 55%, and 50%, respectively, to preserve buying power
for late chopped rosters and superflex. Targets interpolate between milestones and
rescale to the manager's live remaining cash; unused allowances carry forward.
Reserves relax as projected cut risk rises from 25% to 50%, and reach zero after
Week 17. Bids and their noise cannot exceed the resulting auction allowance.
These habits and their equal prior weights are assumptions, not inferred from one
auction or optimized for opponents. Participation, target noise and observed bid
tendencies still distinguish managers.

Our future bidding, inside the replays and the full race, submits an offer for every
candidate improving the chosen objective, including early bargains, with no saving
allowance and no price guide (`race.my_bid`). A claim is worth `race.PRICES` weeks of
remaining cash per point per week of its gain, so a permanent upgrade is worth a fixed
share of cash and a rental more as the weeks run out, and anything at the final
auction, after which cash is worthless. The bid is what the simulated market makes
that worth paying: the race excluding us records what every free agent cleared for in
every week of every season, and `race.Market` turns those prices into the bid that
maximizes the claim's expected surplus, value minus bid times the share of seasons the
bid wins in (a paid bid beats every lower price and any free pickup; a $0 claim only
lands a player nobody wanted). The table pools every recorded season, so a season's
own price is one draw in thousands rather than a peek. A claim's value also counts
what it takes from the team I would meet in the final: each recorded season scores
every acquisition's lineup with and without the player, week by week while its buyer
holds him, and for the survivor's final lineup that loss, capped by his margin over
the last teams cut (a survivor a star made can be replaced by the next team up), is
what `Market.denial` averages over the seasons where the player was in play. The
price per point is chosen by replayed title odds
(below), so holding cash back happens only when the simulated seasons reward it
rather than by a chosen tactic. Total
paid spending in an auction is also limited to its largest individual bid. Claims
naming the same drop are alternatives; open spots, remaining cash and redundant
upgrades are checked as claims resolve (a claim that no longer improves the roster
after an earlier win is passed over, which Sleeper itself cannot do for you: keep
your live claim list short). Opponents choose their best few targets with preference
noise. Active managers can also make a free pickup after claims. Week 1 is free agency.

After the weekly run, a player dropped since (including drops on winning claims) is on
waivers until he clears, about 23 hours after the drop on Sleeper's next 20-minute
processing tick (`WAIVER_CLEAR_HOURS`, observed in week 2), and claims on him process
then. The race auctions these players off-cycle before the week's games. Opponents
participate at their usual rate times the observed off-cycle share: opponents who
claimed off-cycle per opponent in that week's run, pooled over completed weeks (week 2:
5 of 15). Bids follow the same price curve, and every other free agent stays free. The
windows clear at different times, but they are modeled as one auction.

Every simulated week's auction is followed by a cascade (`race.CASCADE_ROUNDS`, two
rounds): the players its winners and free pickups dropped sit on waivers until they
clear, the room bids on them in a further round at the off-cycle share of its
participation, and that round's drops feed one more. The week-2 and week-3 logs each
show a round on the run's drops, and week 3 a second round on that round's drops (Tyjae
Spears and Romeo Doubs at $45 each); a third round has not drawn a bid. Simulated from
the reconstructed pre-run week-3 rosters, the room's cascade matches the log in volume,
about four paid claims on fifteen players in play in the first round and two on five in
the second, against four on about twenty and two on six observed, at prices between the
two observed weeks ($55 and $27 a round against week 3's $129 and $90 and week 2's $24
and $0); it makes fewer free pickups of the drops than the log (one or two a round
against four to eight), since only the attentive share of the room picks up at all. The
record carries every round, so my replayed agent bids in each, and the market table
pools a player's clearing prices across rounds. The off-cycle share is pooled over
completed weeks only, so week 3's busier cascade (12 opponents bidding off-cycle
against 20 in the run) enters it next week. In the week-3 state (after the run) the
cascade lowers the standing roster's replayed title odds from 27.5% to 26.7% (one round
26.2%) and the full race's from 23.5% to 22.0%: the room gets a little more out of the
wire mid-week and the early bars rise by up to a point and a half. The chosen price,
the objective and this week's free adds are unchanged, and a run takes about a fifth
longer.

The race excluding us records opponent markets and cut bars. Our roster/budget
variants are replayed through those same seasons to choose this week's best modeled
bid (`season/claims.py`). The standing roster is replayed at several budgets at every
future bid value in `race.PRICES`; that grid is the value of cash, and the value that
replays best at each budget is what every variant at that budget is priced under. A
value is chosen by its average outcome across seasons, never separately using a
record's future prices or scores. This searches a one-dimensional family of
continuation strategies (what a point of gain is worth from next week on), not every
possible sequence of future auction decisions; the shape of my bids across players
comes from the recorded market, but valuing gain in weeks of remaining cash is still a
modeling assumption. In the week-3 state the replays chose 0.7 weeks of cash per
point per week. On the same recorded seasons, under the earlier accounting of what my
claims take from the room (below), this policy replayed a little better than the best
multiple of the guide ceiling it replaced (26.6% against 25.1% with the survivor's
denial, 17.3% against 17.1% without); valuing gain at a constant price per
point, or in cash alone, or in weeks alone, replayed worse, and bidding the value
itself with no shading against the market replayed far worse, since this room prices
depth above its gain, so paying full value for it wins only depth that earns nothing.
The full race disagrees: with my agent bidding live against the simulated room it
reports 24% for this policy against 28–29% for the guide multiple, a gap well outside
its noise. The guide's positional curve (money for the elite by rank, almost none for
anyone else) and its unshaded bids each account for about half of it; the cut-risk
kicker, the value's time profile, the objective's week weights, in-sample fitting of
the shaded bids and measuring gain beyond the free pickup were each tested and do not
explain it. The replay's account of what a claim of mine does to the room is first
order (below), so the race is the check on any change to this family, and the
discrepancy is open. Winning and losing outcomes are evaluated per recorded season,
preserving their connection to future opportunity. These are individual alternatives,
not an optimized simultaneous claim portfolio. Championship weeks are scored with the
roster held in each week; a Week 17 pickup cannot improve Week 16 retroactively. The
full race uses our selected bid value when reporting league odds.

A player my replay takes is one the recorded room never got. Each recorded season
carries every opponent acquisition's weekly lineup loss while he holds the player
with the best body nobody took in his place, and every week's opponent scores.
Whoever bought the player after my agent did plays without him while I hold him: the
week's bar is the second-lowest of the lowered scores, a lowered team that falls into
the bottom two is cut then and lowered no further (the recorded field stands in for
the team cut in his place), and the survivor's championship total loses the same
way, no more than his margin over the last teams cut. What the buyer would have done
with his cash, and the drop he keeps, are not replayed, and neither is the room's
response to a stronger rival, so replay levels stay approximate and optimistic. In
the week-3 state (after the run, before the cascade above) the standing roster replays to 27.5% for the title
and 42.4% to reach the final against 23.5% and 38.6% in the full race. The earlier
accounting, under which every recorded buyer kept the player and only the survivor's
final lineup was charged, replayed to 26.6% and 34.7%: an optimistic final masking a
pessimistic survival profile. Charging each buyer the player's whole lineup value
replayed to 39.0% and 53.2%, so the substitute is first order too. On the pre-auction
week-3 decision, paired full races (8,192 seasons, score noise shared across roster
variants, 22.2% for the standing roster against 27.2% replayed) valued the free
pickups of Luther Burden, Saquon Barkley, Alec Pierce and Carnell Tate at +6.5%,
+14.0%, +5.5% and +5.2% of title odds (±2.2), where this accounting replays +5.7%,
+9.7%, +2.4% and +3.9% and the earlier one +5.6%, +10.2%, +2.3% and +3.9%: the
accounting moves the level, not the relative values, and the race's higher marks,
about two of its standard errors on average, are part of the open discrepancy.

Sleeper documents its [suggested bid ranges](https://support.sleeper.com/en/articles/12111984-suggested-faab-bids),
but its [public API](https://docs.sleeper.com/) does not document an endpoint for them.
The model does not depend on those suggestions or on future-week guide publications.

## Dashboards

`uv run -m shared.serve` serves `out/` at http://127.0.0.1:8123 (direct `file://`
access cannot fetch the JSON). `/` renders `season.json`: this week's lineup, the
waiver claims with their optimal bids, the elimination bar by week, every team's odds
and budget outlook, and the waiver market; `/draft.html` renders `rankings.json`,
including the live board state embedded in it and when that snapshot was taken;
`/sources.html` renders `data_source_matches.json` as a team-by-source fit heatmap
with pick-level evidence. Re-run `season.refresh` or `draft.refresh` and
reload to advance.
