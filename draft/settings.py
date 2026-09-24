"""The draft's geometry and the draft model's strategy constants.

Snake with a third-round reversal, 8 rounds x 32 teams = 256 picks, all offense; my slot
is 20. Everything below the geometry is a modeling choice. The league's shape itself
(positions, weekly lineups, roster sizes) lives in shared.league.
"""

from __future__ import annotations

from shared.league import STARTING_SLOTS, TEAMS

# --- geometry -----------------------------------------------------------------------
DRAFT_ID = "1397662421937565696"  # this league's Sleeper draft
MY_SLOT = 20
BENCH_SLOTS = 1
ROUNDS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS  # 8
TOTAL_PICKS = TEAMS * ROUNDS  # 256
# Round the snake stops alternating: round 3 repeats round 2's direction, inverting
# parity from there on (forward, reverse, reverse, forward, reverse, forward, ...).
REVERSAL_ROUND = 3


def draft_order() -> list[int]:
    """Slot (1-based) picking at each overall pick, honoring the reversal round."""
    order: list[int] = []
    for rnd in range(1, ROUNDS + 1):
        forward = rnd % 2 == 1
        if rnd >= REVERSAL_ROUND:
            forward = not forward
        order.extend(range(1, TEAMS + 1) if forward else range(TEAMS, 0, -1))
    return order


def pick_label(pick_no: int) -> str:
    """1-based overall pick number -> 'round.slot-in-round' as the draft room shows it."""
    rnd, idx = divmod(pick_no - 1, TEAMS)
    return f"{rnd + 1}.{idx + 1:02d}"


def picks_for_slot(slot: int, order: list[int]) -> list[int]:
    return [i + 1 for i, s in enumerate(order) if s == slot]


# --- valuation ----------------------------------------------------------------------
# Waiver-tier bodies a surviving roster holds by week, per position. The roster grows
# from 8 spots (week 1) to 16 (week 14) while only 8 players are ever drafted, so
# in-season adds accumulate on every surviving team; each is modeled as an
# always-available body at that week's wire level. The allocation leans RB/WR, where
# injury churn drives adds, with the second QB arriving for the week-14 superflex.
# This is what stops drafted depth from being credited with the whole late-season
# lineup: by week 15 half of everyone's roster came off the wire.
WEEK_WIRE_BODIES = {
    "QB": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    "RB": (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4),
    "WR": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4),
    "TE": (1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3),
}
# Chance a player is unavailable when a lineup job must be filled. Byes and known
# absences are explicit zero weeks in weekly_points, so these rates price only the
# surprise in-week unavailability (injury, inactive, benching).
UNAVAILABLE_RATE = {"QB": 0.05, "RB": 0.15, "WR": 0.08, "TE": 0.06}
# My FAAB policy as the draft priced it: the undrafted tail alone while I hold the
# budget through the bye gauntlet, the survivors' equal split through the week 9-12
# lineup expansion, the top half of every tier from week 13 once the saved budget is
# spent. Only my roster's valuation sees this; opponents keep the equal split.
FAAB_HOLD_WEEKS = 8
FAAB_SPEND_WEEK = 13

# --- guillotine measurement ---------------------------------------------------------
GUILLOTINE_SIMS = 512  # simulated seasons per fixed-point iteration for elimination bars
MAX_ITERS = 40  # cap on fixed-point iterations before a cycle must have closed

# --- my pick policy -----------------------------------------------------------------
SURVIVAL_SIGMA = 3.5  # softness of "will he last until my next pick"
LOOKAHEAD_PER_POS = 2  # candidates per position for the bulk two-pick policy
# The live decision gets a broader pool: the top three at each position, which retains
# useful interior tradeoffs behind each position's head.
FIRST_PICK_PER_POS = 3
# A live-board candidate or later target must survive to that decision in at least one
# redraw out of twenty. Rarer paths are noise, not useful draft choices.
CANDIDATE_SURVIVAL_FLOOR = 0.05
# Survivors of that floor go to the branch redraws and full-draft rollouts in two-pick
# score order, up to this many.
ROLLOUT_CANDIDATES = 8
# The live decision plans targets across this many of my held picks before the ordinary
# two-pick policy resumes. Four reaches across both sides of the next snake turn here.
LOOKAHEAD_PICKS = 4
SIMS = 200  # noisy opponent redraws
ROLLOUT_SIMS = 100  # full-draft playouts per candidate at my next pick
NOISE = 1.0  # multiplier on each opponent's fitted choice noise; 0 removes variation

# --- opponent model -----------------------------------------------------------------
# An entirely unfilled dedicated starter group receives a 3x source-rank boost; the
# boost fades linearly as that position's dedicated starters are filled. QB gets a
# scarcity rule on top (simulation.Draft.opponent_candidates).
OPPONENT_BALANCE_STRENGTH = 2.0
# Opponents grow reluctant to add players beyond these comfortable depths: the penalty
# starts at the 2nd QB/TE and the 4th RB/WR and compounds per extra body.
OPPONENT_DEPTH_TARGETS = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
OPPONENT_DEPTH_PENALTY = 2.0
# Flat source-rank multiplier per position; < 1 pulls the position up an opponent's
# board. QB scarcity is priced by the superflex boards in the cold-start blend, not a
# tilt (a QB tilt on top double-counted). The TE premium is priced by two boards; a
# residual 0.8 says the rest of the room half-notices. A prior, not a fit.
OPPONENT_POSITION_TILT: dict[str, float] = {"TE": 0.8}
# Intel on named drafters: each one's plan for its next picks, in order, applied over
# its board and the QB scarcity rule. A step is a tuple of player names (the first
# still available is taken) or a position (the board's best there, with the usual
# noise). Indexed by how many picks the drafter has made, so it holds up live.
OPPONENT_INTEL: dict[str, list[tuple[str, ...] | str]] = {
    "MyFatherLamar": [("Brock Bowers", "Trey McBride"), "TE", "QB"],
}
