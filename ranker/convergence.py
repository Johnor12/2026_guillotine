"""Solve the league levels from the league shape produced by a simulated draft."""

from __future__ import annotations

import sys

from . import guillotine
from .board import Board
from .league import MAX_ITERS, POSITIONS, REGULAR_WEEKS, SEED, WEEKS
from .opponents import OpponentStrategy
from .pool import Player
from .simulation import Draft
from .value import Levels, pos_sorted, seed_levels, wire_replacement

# Guillotine weight mass in each report band: the early no-bench weeks, the bye
# gauntlet, the expanded-roster run-in, and the two championship weeks.
_BANDS = ((0, 4), (4, 9), (9, 15), (15, WEEKS))


def _weight_bands(weights: tuple[float, ...]) -> str:
    return " ".join(
        f"wk{lo + 1}-{hi}:{sum(weights[lo:hi]):.2f}" for lo, hi in _BANDS
    )


def _average_levels(cycle: list[Levels]) -> Levels:
    n = len(cycle)
    weights = tuple(
        sum(levels.weights[w] for levels in cycle) / n for w in range(WEEKS)
    )
    total = sum(weights)
    weights = tuple(w / total for w in weights)  # keep the sum exactly 1

    def mean_bodies(pick) -> tuple:
        # Body and pool sizes per (position, week) are league constants, so they zip.
        return tuple(
            tuple(
                tuple(sum(values) / n for values in zip(*bodies_across_cycle))
                for bodies_across_cycle in zip(*(pick(levels)[i] for levels in cycle))
            )
            for i in range(len(POSITIONS))
        )

    return Levels(
        weights=weights,
        wire=mean_bodies(lambda levels: levels.wire),
        dropped=mean_bodies(lambda levels: levels.dropped),
    )


def converge(
    players: list[Player],
    board: Board,
    report: bool,
    opponents: dict[int, OpponentStrategy],
) -> tuple[Levels, Draft, dict, dict]:
    """Fixed point: levels -> draft -> guillotine measurement -> levels.

    The map is piecewise-constant: the draft is a discrete function of the levels, and
    the levels (weekly wire, drop floor, week weights) are a deterministic seeded
    function of the draft's rosters. Gradient-style damping cannot settle such a map —
    it just orbits the discontinuity — while plain undamped iteration is eventually
    periodic. So iterate undamped, watch for a repeated draft outcome, and average the
    levels over the cycle once one closes. The state key is the full set of final
    rosters: levels are a function of exactly that, so a repeated outcome means a
    repeated map state. A cycle of length 1 is an exact fixed point; a longer cycle
    means the league genuinely alternates between neighbouring shapes and the average
    across it is the honest answer. The cycle is reported rather than hidden.

    Every draft here starts from `board`, so on a live board the fixed point is over
    the rosters this league will actually finish with, and the guillotine bars are
    measured against the opponents this league actually fields.

    Returns (levels, final draft, history, guillotine diagnostics at the final state).
    """
    pos = pos_sorted(players)
    levels = seed_levels(players)
    trace = [
        {
            "iteration": 0,
            "source": "slot_assignment",
            "weight_bands": _weight_bands(levels.weights),
        }
    ]
    seen: dict[tuple, int] = {}
    observations: list[Levels] = []
    cycle_start: int | None = None

    for it in range(1, MAX_ITERS + 1):
        draft = Draft(players, levels, board, opponents=opponents)
        draft.run()
        my_roster = draft.rosters[board.my_slot - 1]
        opponent_rosters = [
            roster
            for slot, roster in enumerate(draft.rosters, start=1)
            if slot != board.my_slot
        ]
        new_levels, diag = guillotine.solve(
            my_roster, opponent_rosters, draft.taken, pos, levels, SEED
        )
        observations.append(new_levels)
        key = tuple(tuple(sorted(p.player_id for p in r)) for r in draft.rosters)
        trace.append(
            {
                "iteration": it,
                "source": "draft_simulation",
                "p_reach_final": diag["p_reach_final"],
                "p_title": diag["p_title"],
                "weight_bands": _weight_bands(new_levels.weights),
                "wire_week1": {
                    k: round(new_levels.wire[i][0][0], 1)
                    for i, k in enumerate(POSITIONS)
                },
                "wire_week15": {
                    k: [round(v, 1) for v in new_levels.wire[i][REGULAR_WEEKS - 1]]
                    for i, k in enumerate(POSITIONS)
                },
            }
        )
        if report:
            print(
                f"  iter {it}: p_reach {diag['p_reach_final']:.3f} "
                f"p_title {diag['p_title']:.3f}  weights {_weight_bands(new_levels.weights)}",
                file=sys.stderr,
            )
        if key in seen:
            cycle_start = seen[key]
            break
        seen[key] = it
        levels = new_levels

    assert cycle_start is not None, f"no cycle within {MAX_ITERS} iterations"
    cycle = observations[cycle_start - 1 : -1] or observations[cycle_start - 1 :]
    levels = _average_levels(cycle)
    if report:
        print(
            f"  cycle of length {len(cycle)} closed at iteration {cycle_start}; "
            "averaging levels across it",
            file=sys.stderr,
        )

    # Final deterministic draft at the settled levels, so sim_pick, my_decisions and
    # the reported levels describe one and the same draft — then one more guillotine
    # measurement of that draft for the reported diagnostics (levels stay averaged).
    draft = Draft(players, levels, board, opponents=opponents)
    draft.run()
    my_roster = draft.rosters[board.my_slot - 1]
    opponent_rosters = [
        roster
        for slot, roster in enumerate(draft.rosters, start=1)
        if slot != board.my_slot
    ]
    _, diagnostics = guillotine.solve(
        my_roster, opponent_rosters, draft.taken, pos, levels, SEED
    )
    diagnostics["post_draft_wire_season_points"] = {
        k: round(v, 1) for k, v in wire_replacement(draft.taken, pos).items()
    }
    history = {
        "method": (
            "undamped iteration to a limit cycle over draft outcomes, levels averaged "
            "across the cycle"
        ),
        "cycle_length": len(cycle),
        "cycle_first_seen_at_iteration": cycle_start,
        "iterations_run": len(observations),
        "trace": trace,
    }
    return levels, draft, history, diagnostics
