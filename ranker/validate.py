"""Every-run invariants over the board, the simulation and the emitted rows.

These run on every invocation and land in `validation.problems`; a non-empty list is a
non-zero exit. `--selftest` (selftest.py) covers the states a real draft.json cannot
currently reach.
"""

from __future__ import annotations

from . import league
from .board import Board
from .league import (
    DEDICATED_SLOTS,
    MAX_ITERS,
    MAX_POSITIONS,
    POSITIONS,
    STARTING_SLOTS,
    draft_order,
    pick_label,
    picks_for_slot,
)
from .pool import Player
from .simulation import Draft
from .value import pos_sorted, starting_positions


def validate(
    rows: list[dict],
    players: list[Player],
    stream: dict[str, float],
    draft: Draft,
    board: Board,
    history: dict,
) -> list[str]:
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    # The static snake is the yardstick even on a live board. The pinned README labels
    # only apply to the README's own geometry; a reconfigured test draft has no external
    # pin to check the snake against.
    full = picks_for_slot(board.my_slot, draft_order())
    if (league.TEAMS, league.ROUNDS, board.my_slot) == (10, 12, 2):
        readme = ["1.02", "2.09", "3.02", "4.09", "5.02", "6.09", "11.02", "12.09"]
        labels = [pick_label(p) for p in full]
        check(labels[:6] == readme[:6], f"draft order head {labels[:6]} != README {readme[:6]}")
        check(
            labels[-2:] == readme[-2:], f"draft order tail {labels[-2:]} != README {readme[-2:]}"
        )
    check(
        len(full) == league.ROUNDS,
        f"{len(full)} picks for slot {board.my_slot}, want {league.ROUNDS}",
    )
    check(
        len(board.order) == len(board.pick_nos),
        f"board has {len(board.order)} owners for {len(board.pick_nos)} pending picks",
    )
    accounted = board.picks_made + len(board.order)
    check(
        accounted == league.TOTAL_PICKS,
        f"board accounts for {accounted} picks, want {league.TOTAL_PICKS}",
    )
    check(
        sum(board.owed_size(s) for s in range(1, league.TEAMS + 1)) == league.TOTAL_PICKS,
        "the picks each team owns do not sum to the board",
    )

    if board.live:
        first = board.pick_nos[0] if board.pick_nos else league.TOTAL_PICKS + 1
        check(
            board.pick_nos == list(range(first, league.TOTAL_PICKS + 1)),
            "pending picks are not the contiguous tail of the board",
        )
        clock = (board.live.get("on_the_clock") or {}).get("pick_no")
        check(
            clock is None or clock == first,
            f"on the clock is pick {clock}, board resumes at {first}",
        )
        mine = (board.live.get("next_pick_of_mine") or {}).get("pick_no")
        check(
            mine is None or board.my_picks[:1] == [mine],
            f"draft.json says my next pick is {mine}, board says {board.my_picks[:1]}",
        )
        if not board.live.get("traded_picks"):
            want = [n for n in full if n >= first]
            check(
                board.my_picks == want,
                f"my {len(board.my_picks)} remaining picks are not the snake's tail "
                f"({len(want)} picks) even though no picks were traded",
            )

    want_taken = sum(len(r) for r in board.rosters) + len(board.order)
    check(
        len(draft.taken) == want_taken,
        f"{len(draft.taken)} unique pool players drafted, want {want_taken}",
    )
    for i, roster in enumerate(draft.rosters, start=1):
        made = board.rosters[i - 1]
        off = board.off_pool[i - 1]
        # Made picks are facts: they must come through the simulation untouched, in order.
        check(
            roster[: len(made)] == made,
            f"slot {i} lost or reordered one of its {len(made)} made picks",
        )
        want = len(made) + board.picks_left[i - 1]
        check(len(roster) == want, f"slot {i} ends with {len(roster)} players, want {want}")
        starters = starting_positions(roster)
        check(
            len(starters) == sum(STARTING_SLOTS.values()),
            f"slot {i} cannot field a full lineup "
            f"({len(starters)}/{sum(STARTING_SLOTS.values())})",
        )
        for pos, need in DEDICATED_SLOTS.items():
            have = sum(1 for p in roster if p.position == pos)
            have += sum(1 for o in off if o.get("position") == pos)
            check(have >= need, f"slot {i} has {have} {pos}, needs {need}")
            # Off-pool picks count against the cap too: Sleeper enforced it live.
            check(
                have <= MAX_POSITIONS[pos],
                f"slot {i} has {have} {pos}, over the {MAX_POSITIONS[pos]} cap",
            )
    gains = [r["lineup_gain"] for r in rows]
    check(gains == sorted(gains, reverse=True), "rows are not sorted by lineup gain descending")
    source_ids = {strategy.source_id for strategy in draft.opponents.values()}
    # Before any opponent has picked, all of them share the consensus cold-start board.
    if any(board.rosters[slot - 1] for slot in draft.opponents):
        check(len(source_ids) >= 2, f"opponents use only {len(source_ids)} distinct source board(s)")
    check(
        any(row["opponent_rank_delta"] != 0 for row in rows),
        "opponent consensus order does not diverge from my board order",
    )
    want_rows = len(players) - len(board.taken)
    check(
        len(rows) == want_rows,
        f"{len(rows)} rows for {want_rows} undrafted players in a {len(players)}-player pool",
    )
    check(
        not (board.taken & {r["player_id"] for r in rows}),
        "a player already drafted in draft.json is on the emitted board",
    )
    # The reported wire level is a mean over the limit cycle, so the invariant that
    # actually holds is that it lies inside the range the cycle spanned — not that it
    # coincides with any single draft's level, which a long cycle can genuinely straddle.
    pos = pos_sorted(players)
    for k in POSITIONS:
        # The stored range is rounded to 0.1, so allow half a rounding step.
        lo, hi = history["cycle_wire_range"][k]
        check(
            lo - 0.05 - 1e-6 <= stream[k] <= hi + 0.05 + 1e-6,
            f"{k} wire {stream[k]:.1f} outside its cycle range [{lo}, {hi}]",
        )
        check(
            pos[k][-1].points <= stream[k] <= pos[k][0].points,
            f"{k} wire {stream[k]:.1f} outside the {k} pool range",
        )
    check(
        history["iterations_run"] < MAX_ITERS,
        f"no limit cycle found within {MAX_ITERS} iterations",
    )
    return problems
