"""Elimination bars, week weights, and the waiver escalation behind the levels.

Once per fixed-point iteration, the deterministic draft's rosters become the league
levels valuation runs on (value.Levels):

1. Waiver escalation. Each week the two lowest teams are cut and their players hit
   waivers, so the in-season replacement level rises as the season runs. Opponents are
   ranked by projected strength and assumed to be eliminated weakest-first (two per
   week); their players join the undrafted tail in one free-agent pool that the
   survivors split, so a roster's j-th waiver body (WEEK_WIRE_BODIES — more as rosters
   expand) is the j-th tier of S pooled values, S the teams still alive
   (value.combine_wire). Early the pool is the undrafted tail shared thirty ways and
   the wire is close to nothing. Late it is everything: four finalists share
   twenty-eight eliminated rosters, so cheap drafted depth washes out while drafted
   stars keep clearing the tiers all season.

2. Elimination bars. GUILLOTINE_SIMS seasons are simulated over the 31 opponents'
   weekly expected lineup values: each opponent gets a persistent projection-error
   bias (TEAM_SEASON_SIGMA) plus weekly score noise (WEEKLY_SIGMA, scaled by lineup
   size, with the catastrophic tail floored at SCORE_FLOOR_Z — a full lineup's worst
   week bottoms out); each week the bottom two are cut. The bar is the second-lowest
   surviving opponent's score — beat it and I survive the cut, whoever I am. The lone
   opponent reaching the final supplies the week 16-17 championship bar.

3. Week weights. My weekly score is my expected lineup value plus the same persistent
   projection bias and weekly noise, so each draw yields P(title | draw) =
   prod_w Phi((mu_w + b - bar_w)/sigma_w) * Phi(championship). A week's weight is
   d log E[P(title)] / d mu_w — the per-week hazard averaged under the
   P(title)-weighted posterior over draws. Seasons where my roster busts (b low)
   dominate exactly the weeks they die in, so early safety margins are priced even
   when the projected margin looks comfortable, while weeks cleared in every draw get
   ~no weight (30th place survives week 1 exactly like 1st). Weights are normalized
   to sum to 1; scaling changes no argmax.

The map from rosters to levels is deterministic (one seeded MC per iteration), so the
convergence loop's cycle detection still sees exact repeats.
"""

from __future__ import annotations

import math
import random

from .league import (
    GUILLOTINE_SIMS,
    POSITIONS,
    REGULAR_WEEKS,
    SCORE_FLOOR_Z,
    TEAM_SEASON_SIGMA,
    WEEK_STARTERS,
    WEEKLY_SIGMA,
    WEEKS,
)
from .pool import Player
from .value import Levels, combine_wire, top_values, weekly_team_values, weekly_undrafted

SIGMA_WEEK = tuple(
    WEEKLY_SIGMA * math.sqrt(WEEK_STARTERS[w] / WEEK_STARTERS[0]) for w in range(WEEKS)
)
# My championship score is two independent weeks of noise on top of two expected values.
SIGMA_CHAMP = math.hypot(SIGMA_WEEK[REGULAR_WEEKS], SIGMA_WEEK[REGULAR_WEEKS + 1])

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) * _INV_SQRT2PI


def _cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _dropped(
    opponent_rosters: list[list[Player]],
    strength_order: list[int],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Per position, per week: the best weekly values among every roster eliminated
    so far, weakest opponents cut first, two per week — the eliminated half of the
    free-agent pool the survivors split (value.tier_bodies).

    Early the crowd is large and the drops few, so a typical survivor gets nothing
    here without spending real FAAB. Late, a handful of survivors split twenty-plus
    eliminated rosters and every finalist fills up with other teams' early-round
    picks, which is what makes drafted depth worthless by then and drafted stars the
    only draft-day asset that still matters.
    """
    out = []
    for pos in POSITIONS:
        col = []
        for w in range(WEEKS):
            # Going into week w+1, cut rounds 1..min(w, 15) have run.
            cut = strength_order[: 2 * min(w, REGULAR_WEEKS)]
            col.append(
                top_values(
                    (
                        player.weekly[w]
                        for team in cut
                        for player in opponent_rosters[team]
                        if player.position == pos
                    ),
                    pos,
                    w,
                )
            )
        out.append(tuple(col))
    return tuple(out)


def solve(
    my_roster: list[Player],
    opponent_rosters: list[list[Player]],
    taken: set[int],
    pos: dict[str, list[Player]],
    prev: Levels,
    seed: int,
) -> tuple[Levels, dict]:
    """One draft outcome -> the levels to value the next iteration with, plus
    diagnostics for my roster at those levels. Deterministic given its inputs."""
    # Strength order for the deterministic waiver escalation uses the previous
    # iteration's levels; the escalated wire then prices everyone's weekly values.
    prev_mus = [weekly_team_values(r, prev) for r in opponent_rosters]
    strength_order = sorted(
        range(len(opponent_rosters)),
        key=lambda i: sum(prev_mus[i][:REGULAR_WEEKS]),
    )
    dropped = _dropped(opponent_rosters, strength_order)
    wire = combine_wire(weekly_undrafted(taken, pos), dropped)
    levels = Levels(weights=prev.weights, wire=wire, dropped=dropped)

    opp_mus = [weekly_team_values(r, levels) for r in opponent_rosters]
    my_mu = weekly_team_values(my_roster, levels)
    champ_mu = my_mu[REGULAR_WEEKS] + my_mu[REGULAR_WEEKS + 1]

    # Elimination MC over the opponents: conditioned on my surviving, the two cuts
    # fall on them, and I survive a week exactly when I beat the second-lowest
    # surviving opponent. Each draw also gives MY team a persistent projection-error
    # bias, so a season where my roster is genuinely weaker than projected — the
    # season most likely to kill me early — carries its full elimination risk. One
    # seeded run per iteration keeps the level map exact.
    rng = random.Random(seed)
    bar_sums = [0.0] * REGULAR_WEEKS
    champ_bar_sum = 0.0
    # Per-draw accumulators: O = P(title | this draw's biases and bars), and
    # O * (phi/Phi)/sigma per week — the numerator of the posterior-weighted hazard.
    o_total = 0.0
    surv_total = 0.0
    surv_by_week = [0.0] * REGULAR_WEEKS
    hazard_num = [0.0] * (REGULAR_WEEKS + 1)
    win_num = 0.0
    for _ in range(GUILLOTINE_SIMS):
        bias = [rng.gauss(0.0, TEAM_SEASON_SIGMA) for _ in opp_mus]
        my_bias = rng.gauss(0.0, TEAM_SEASON_SIGMA)
        alive = list(range(len(opp_mus)))
        o = 1.0
        hazards = [0.0] * (REGULAR_WEEKS + 1)
        for w in range(REGULAR_WEEKS):
            sigma = SIGMA_WEEK[w]
            deviation_floor = SCORE_FLOOR_Z * math.hypot(TEAM_SEASON_SIGMA, sigma)
            scored = sorted(
                (
                    opp_mus[i][w]
                    + max(bias[i] + rng.gauss(0.0, sigma), deviation_floor),
                    i,
                )
                for i in alive
            )
            bar = scored[1][0]
            bar_sums[w] += bar
            alive = [i for _, i in scored[2:]]
            z = (my_mu[w] + my_bias - bar) / sigma
            mass = max(_cdf(z), 1e-12)
            o *= mass
            surv_by_week[w] += mass
            hazards[w] = _phi(z) / (sigma * mass)
        surv = o
        finalist = alive[0]
        champ_bar = (
            opp_mus[finalist][REGULAR_WEEKS]
            + opp_mus[finalist][REGULAR_WEEKS + 1]
            + 2.0 * bias[finalist]
            + rng.gauss(0.0, SIGMA_CHAMP)
        )
        champ_bar_sum += champ_bar
        z = (champ_mu + 2.0 * my_bias - champ_bar) / SIGMA_CHAMP
        win = max(_cdf(z), 1e-12)
        o *= win
        hazards[REGULAR_WEEKS] = _phi(z) / (SIGMA_CHAMP * win)
        o_total += o
        surv_total += surv
        win_num += surv * win
        for w in range(REGULAR_WEEKS + 1):
            hazard_num[w] += o * hazards[w]

    # d log E[P(title)] / d mu per week: hazards averaged under the P(title)-weighted
    # posterior over draws. Draws where my roster busts dominate exactly the weeks
    # they die in, so safety in the early weeks is priced even when the projected
    # margin looks comfortable; weeks my roster clears in every draw get ~no weight.
    if o_total > 1e-300:
        raw = [hazard_num[w] / o_total for w in range(REGULAR_WEEKS)]
        champ_weight = hazard_num[REGULAR_WEEKS] / o_total
        raw.extend([champ_weight, champ_weight])
        total = sum(raw)
        weights = tuple(value / total for value in raw)
    else:
        # Every draw died outright — no gradient signal; keep the previous emphasis.
        weights = prev.weights

    draws = GUILLOTINE_SIMS
    p_reach = surv_total / draws
    p_win = win_num / surv_total if surv_total > 1e-300 else 0.0
    diagnostics = {
        "p_survive_by_week": [round(s / draws, 4) for s in surv_by_week],
        "p_reach_final": round(p_reach, 4),
        "p_win_final": round(p_win, 4),
        "p_title": round(o_total / draws, 4),
        "my_weekly_mu": [round(m, 1) for m in my_mu],
        "bar_mean_by_week": [round(s / draws, 1) for s in bar_sums],
        "championship_bar_mean": round(champ_bar_sum / draws, 1),
    }
    return Levels(weights=weights, wire=wire, dropped=dropped), diagnostics
