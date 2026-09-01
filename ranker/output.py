"""Assemble the rankings.json payload. The method is documented in the module
docstrings and README.md; the payload carries the numbers plus one-line pointers."""

from __future__ import annotations

from . import league
from .board import Board
from .league import (
    CANDIDATE_SURVIVAL_FLOOR,
    FIRST_PICK_PER_POS,
    GUILLOTINE_SIMS,
    LOOKAHEAD_PICKS,
    NOISE,
    OPPONENT_DEPTH_PENALTY,
    OPPONENT_DEPTH_TARGETS,
    OPPONENT_POSITION_TILT,
    POSITIONS,
    REGULAR_WEEKS,
    SEED,
    SIMS,
    STARTING_SLOTS,
    SURVIVAL_SIGMA,
    TEAM_SEASON_SIGMA,
    UNAVAILABLE_RATE,
    WEEKLY_SHAPES,
    WEEKLY_SIGMA,
    WEEKS,
    draft_order,
    pick_label,
    picks_for_slot,
)
from .opponents import OpponentStrategy
from .pool import Player
from .rankings import my_next_picks
from .simulation import Draft
from .value import Levels


def team_names(board: Board) -> dict[int, str | None]:
    return {
        s["draft_slot"]: (s.get("team_name") or s.get("username"))
        for s in ((board.live or {}).get("slots") or [])
    }


def _position_counts(roster: list[Player], off: list[dict]) -> dict[str, int]:
    counts = {pos: 0 for pos in POSITIONS}
    for p in roster:
        counts[p.position] += 1
    for o in off:
        if o.get("position") in counts:
            counts[o["position"]] += 1
    return counts


def draft_block(board: Board) -> dict | None:
    """The live board the simulation started from: made picks are already on their
    rosters and out of the pool, only pending picks are played (a traded pick by the
    roster that acquired it), and `rankings` covers the undrafted players only."""
    if not board.live:
        return None
    names = team_names(board)
    block = {k: v for k, v in board.live.items() if k != "slots"}
    block["my_remaining_picks"] = [pick_label(n) for n in board.my_picks]
    block["off_pool_note"] = (
        "Made picks with no match in pool.json by sleeper_id: they fill a roster spot "
        "and satisfy a mandatory position but are never started or valued."
    )
    block["rosters"] = []
    for slot in range(1, league.TEAMS + 1):
        made, off = board.rosters[slot - 1], board.off_pool[slot - 1]
        block["rosters"].append(
            {
                "draft_slot": slot,
                "team": names.get(slot),
                "is_mine": slot == board.my_slot,
                "picks_made": len(made) + len(off),
                "picks_left": board.picks_left[slot - 1],
                "positions": {pos: n for pos, n in _position_counts(made, off).items() if n},
                "players": [f"{p.name} ({p.position})" for p in made],
                "off_pool": [f"{o['name']} ({o['position']})" for o in off],
            }
        )
    return block


def example_rosters(draft: Draft, board: Board) -> list[dict]:
    """Every team's final roster from the deterministic draft: the live board's made
    picks first (the board does not keep their pick numbers), then the simulated picks
    with the pick they were taken at. Each entry carries what the dashboard needs, since
    `rankings` only lists undrafted players."""
    names = team_names(board)
    out: list[dict] = []
    for slot in range(1, league.TEAMS + 1):
        roster, off = draft.rosters[slot - 1], draft.off_pool[slot - 1]
        picks: list[dict] = []
        for p in roster:
            pick_no = draft.pick_of.get(p.player_id)
            picks.append(
                {
                    "pick": pick_label(pick_no) if pick_no else None,
                    "is_made": pick_no is None,
                    "player_id": p.player_id,
                    "name": p.name,
                    "position": p.position,
                    "nfl_team": p.team,
                    "age": p.age,
                    "bye_week": p.bye_week,
                    "is_rookie": p.is_rookie,
                    "points": p.points,
                    "off_pool": False,
                }
            )
        picks += [
            {
                "pick": o["pick"],
                "is_made": True,
                "player_id": None,
                "name": o["name"],
                "position": o["position"],
                "nfl_team": None,
                "age": None,
                "bye_week": None,
                "is_rookie": None,
                "points": None,
                "off_pool": True,
            }
            for o in off
        ]
        out.append(
            {
                "draft_slot": slot,
                "team": names.get(slot),
                "is_mine": slot == board.my_slot,
                "players": len(roster) + len(off),
                "positions": _position_counts(roster, off),
                "picks": picks,
            }
        )
    return out


def build_payload(
    players: list[Player],
    pool_meta: dict,
    board: Board,
    levels: Levels,
    draft: Draft,
    history: dict,
    rows: list[dict],
    problems: list[str],
    opponents: dict[int, OpponentStrategy],
    guillotine: dict,
    option_redraw: dict | None,
    rollout: dict | None,
    survival: dict[int, dict[int, float]] | None,
) -> dict:
    return {
        "value_input": (
            "pool.json weekly_points (DraftSharks per-week, league scoring), blended "
            "equally with Sleeper's season projection and the consensus board's "
            "rank-matched level (ranker/market.py)"
        ),
        "method": (
            "Roster value is each week's expected optimal legal lineup under that week's "
            "starting shape and position-wide availability, combined across weeks 1-17 by "
            "the converged guillotine week weights; see rank.py and ranker/*.py."
        ),
        "league": {
            "teams": league.TEAMS,
            "format": (
                "guillotine: the two lowest weekly scores are eliminated each of weeks "
                f"1-{REGULAR_WEEKS} and their players hit waivers; the last two teams "
                "play a week 16-17 total-points championship"
            ),
            "regular_weeks": REGULAR_WEEKS,
            "starting_slots": STARTING_SLOTS,
            "bench_slots": league.BENCH_SLOTS,
            "rounds": league.ROUNDS,
            "total_picks": league.TOTAL_PICKS,
            "draft_type": "snake",
            "reversal_round": league.REVERSAL_ROUND,
            "my_slot": board.my_slot,
            "my_picks": [pick_label(p) for p in picks_for_slot(board.my_slot, draft_order())],
        },
        "pool": pool_meta,
        "draft": draft_block(board),
        "example_draft": {
            "note": (
                "Full final rosters from the single deterministic draft at the converged "
                "levels, the same draft sim_pick reports. Made picks are facts from the "
                "live board; every pending pick is the model drafting."
            ),
            "rosters": example_rosters(draft, board),
        },
        "my_next_picks": {
            "note": (
                "value_now is the candidate's marginal guillotine-weighted lineup value; "
                "next_pick_ev is E[best option at my following pick] if I take him. At my "
                "first pending pick next_pick_ev comes from conditional opponent redraws, "
                "and rollout_ev/rollout_edge/rollout_se score each candidate's best "
                "four-pick target plan over the whole remaining draft; `take` is the "
                "rollout's recommendation conditional on availability. "
                "p_available_if_i_pass is measured from redraws where my slot never takes "
                "him. See ranker/rankings.py and ranker/planning.py."
            ),
            "option_sims": option_redraw["sims"] if option_redraw else None,
            "rollout_sims": rollout["sims"] if rollout else None,
            "lookahead_picks": LOOKAHEAD_PICKS,
            "first_pick_candidates_per_position": FIRST_PICK_PER_POS,
            "candidate_survival_floor": CANDIDATE_SURVIVAL_FLOOR,
            "picks": my_next_picks(draft, board, rollout, survival),
        },
        "rankings_note": (
            "Undrafted players only, ranked by lineup_gain: the player's marginal "
            "guillotine-weighted weekly lineup value on my current roster at the converged "
            "levels. Rank columns are renumbered over the rows emitted here."
        ),
        "opponent_model": {
            "how": (
                "Each opponent orders legal available players by the provider board most "
                "associated with its completed picks in data_source_matches.json, with a "
                "soft boost for unfilled dedicated starters, a compounding penalty beyond "
                "comfortable depth, and a per-position tilt; Monte Carlo draws around that "
                "preference with noise calibrated to the investigator's mean_log2_loss. "
                "Opponents never read my projections or board."
            ),
            "depth_targets": OPPONENT_DEPTH_TARGETS,
            "depth_penalty_per_extra_player": OPPONENT_DEPTH_PENALTY,
            "position_tilt": OPPONENT_POSITION_TILT,
            "divergence": {
                "distinct_sources": len({s.source_id for s in opponents.values()}),
                "mean_absolute_rank_delta": round(
                    sum(abs(row["opponent_rank_delta"]) for row in rows) / len(rows), 1
                ),
                "max_absolute_rank_delta": max(abs(row["opponent_rank_delta"]) for row in rows),
            },
            "strategies": [opponents[slot].public() for slot in sorted(opponents)],
        },
        "guillotine": {
            "note": (
                "Elimination race simulated over the opponents' rosters each fixed-point "
                "iteration; a week's weight is d log P(title) / d(weekly point). "
                "See ranker/guillotine.py."
            ),
            "weekly_shapes": [{"week": w + 1, **WEEKLY_SHAPES[w]} for w in range(WEEKS)],
            "weekly_sigma": WEEKLY_SIGMA,
            "team_season_sigma": TEAM_SEASON_SIGMA,
            "sims": GUILLOTINE_SIMS,
            "week_weights": [round(w, 4) for w in levels.weights],
            "diagnostics": guillotine,
        },
        "wire": {
            "note": (
                "Per position per week, the waiver bodies a surviving roster holds: the "
                "better of the best undrafted players and the survivors' equal split of "
                "every eliminated roster's players so far."
            ),
            "weekly_levels": {
                k: [[round(v, 1) for v in bodies] for bodies in levels.wire[i]]
                for i, k in enumerate(POSITIONS)
            },
            "drop_floor": {
                k: [[round(v, 1) for v in bodies] for bodies in levels.drop_floor[i]]
                for i, k in enumerate(POSITIONS)
            },
            "convergence": history,
        },
        "strategy": {
            "unavailable_rate": UNAVAILABLE_RATE,
            "survival_sigma": SURVIVAL_SIGMA,
            "lookahead": (
                "first pending decision: four held picks with survival-aware target plans; "
                "bulk policy: value now + E[best value at the following pick]"
            ),
        },
        "monte_carlo": {
            "sims": SIMS,
            "noise": NOISE,
            "seed": SEED,
            "note": (
                "sim_adp, p_drafted and p_available_at_my_picks are over the opponents' "
                "takes in these redraws (Kaplan-Meier: an opponent take is the event, my "
                "own take censors the redraw); sim_pick is from the noiseless draft."
            ),
        },
        "validation": {"problems": problems, "ok": not problems},
        "count": len(rows),
        "rankings": rows,
    }
