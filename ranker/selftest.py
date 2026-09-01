"""Offline checks for the states the live files cannot currently reach.

Four suites: the weekly lineup solver against brute force over every in-season
starting shape, the guillotine level map (bars, week weights, waiver escalation),
source-based opponent behavior, and the board loader against synthetic draft.json
boards — a traded pick, a selection outside the pool, a resumed partial board, and six
malformed boards that must be rejected.
"""

from __future__ import annotations

import dataclasses
import sys

from . import guillotine, league
from .board import fresh_board, load_board
from .league import (
    DEDICATED_SLOTS,
    FIRST_PICK_PER_POS,
    MAX_POSITIONS,
    MY_SLOT,
    OPPONENT_DEPTH_TARGETS,
    OPPONENT_POSITION_TILT,
    POSITIONS,
    REGULAR_WEEKS,
    ROUNDS,
    SEED,
    TEAMS,
    UNAVAILABLE_RATE,
    TOTAL_PICKS,
    WEEKLY_SHAPES,
    WEEKS,
    draft_order,
    pick_label,
)
from .opponents import OpponentStrategy, expected_log2_rank, rank_power
from .planning import (
    apply_survival_floor,
    broaden_first_pick,
    conditional_survival,
    rollout_decision,
)
from .pool import Player
from .rankings import my_next_picks
from .simulation import Draft
from .value import (
    Levels,
    pos_sorted,
    position_expected_value,
    seed_levels,
    sorted_roster,
    team_value,
    week_value,
    weekly_team_values,
)


def synthetic_opponents(
    players: list[Player], board, first_order: list[Player] | None = None
) -> dict[int, OpponentStrategy]:
    """Complete external boards for simulation tests; no personal value is stored."""
    default = first_order or players
    order = tuple(p.player_id for p in default)
    ranks = {player_id: rank for rank, player_id in enumerate(order, start=1)}
    return {
        slot: OpponentStrategy(
            slot=slot,
            roster_id=slot,
            username=f"team{slot}",
            source_id=f"test_source_{slot}",
            source_name=f"Test source {slot}",
            source_format="selftest",
            fit_score=100.0,
            confidence="strong",
            mean_log2_loss=0.5,
            rank_power=rank_power(0.5, len(players)),
            primary_players=len(players),
            ranks=ranks,
            order=order,
        )
        for slot in range(1, TEAMS + 1)
        if slot != board.my_slot
    }


def opponent_selftest(players: list[Player]) -> list[str]:
    """Opponent source order stays personal-value-independent and bends toward balance."""
    fails: list[str] = []
    board = fresh_board()
    levels = seed_levels(players)
    # Put the lowest-projected player first on every external board. An opponent must take
    # him; my optimizer must not, demonstrating that candidate generation is separated too.
    external = sorted(players, key=lambda p: (p.points, p.player_id))
    opponents = synthetic_opponents(players, board, external)
    draft = Draft(players, levels, board, opponents=opponents)
    opponent_take = draft.choose_opponent(0, 1)
    my_take = draft.choose(1, board.my_slot)
    if opponent_take != external[0]:
        fails.append("opponent ignored the top player on its inferred source board")
    if my_take == external[0]:
        fails.append("my optimizer followed the opponent source board")

    # A filled WR group versus an empty TE group gives the best TE a soft 3x rank boost:
    # close source values bend toward TE, while a large enough source gap still wins.
    wrs = [p for p in players if p.position == "WR"]
    te = next(p for p in players if p.position == "TE")
    board = fresh_board()
    board.rosters[0] = wrs[: DEDICATED_SLOTS["WR"]]
    board.picks_left[0] -= len(board.rosters[0])

    def complete(prefix: list[Player]) -> list[Player]:
        ids = {p.player_id for p in prefix}
        return prefix + [p for p in players if p.player_id not in ids]

    filled = DEDICATED_SLOTS["WR"]
    close_order = complete([wrs[filled], te])
    close = Draft(
        players,
        levels,
        board,
        opponents=synthetic_opponents(players, board, close_order),
    )
    if close.choose_opponent(0, 1) != te:
        fails.append("opponent did not prefer a close TE with the TE starter unfilled")

    value_order = complete([wrs[filled], wrs[filled + 1], wrs[filled + 2], te])
    value = Draft(
        players,
        levels,
        board,
        opponents=synthetic_opponents(players, board, value_order),
    )
    if value.choose_opponent(0, 1) != wrs[filled]:
        fails.append("opponent balance preference overrode too large a source-rank gap")

    adjustments = value.opponent_position_adjustments(1)
    tilt = OPPONENT_POSITION_TILT.get("RB", 1.0)
    if abs(adjustments["RB"] - tilt * adjustments["TE"]) > 1e-12:
        fails.append("RB/TE opponent adjustments diverged beyond the configured tilt")

    # Once a roster reaches comfortable WR depth, another WR's adjusted source rank is
    # doubled against other positions. This is a preference, not a positional limit.
    board = fresh_board()
    board.rosters[0] = wrs[: OPPONENT_DEPTH_TARGETS["WR"]]
    board.picks_left[0] -= len(board.rosters[0])
    start = OPPONENT_DEPTH_TARGETS["WR"]
    depth_order = complete([wrs[start], te])
    depth = Draft(
        players,
        levels,
        board,
        opponents=synthetic_opponents(players, board, depth_order),
    )
    if depth.choose_opponent(0, 1) != te:
        fails.append("opponent did not prefer close TE over excessive WR depth")

    depth_value_order = complete(wrs[start : start + 6] + [te])
    depth_value = Draft(
        players,
        levels,
        board,
        opponents=synthetic_opponents(players, board, depth_value_order),
    )
    if depth_value.choose_opponent(0, 1) != wrs[start]:
        fails.append("opponent depth preference acted like a hard positional limit")
    if depth.opponent_depth_penalty(wrs[: start - 1], (), "WR") != 1.0:
        fails.append("opponent depth preference started before its target")
    if depth.opponent_depth_penalty(wrs[:start], (), "WR") != 2.0:
        fails.append("opponent depth preference did not start at its target")
    if depth.opponent_depth_penalty(wrs[: start + 1], (), "WR") != 4.0:
        fails.append("opponent depth preference did not compound with excess depth")

    # The Sleeper position caps are hard: an opponent at the WR cap cannot take a WR
    # however high its source board puts one, and my candidate set drops WR too.
    # picks_left=None isolates the cap from the mandatory-position narrowing, so the
    # check exercises the cap alone whatever the board's geometry.
    board = fresh_board()
    board.rosters[0] = wrs[: MAX_POSITIONS["WR"]]
    board.picks_left[0] -= len(board.rosters[0])
    capped = Draft(
        players,
        levels,
        board,
        opponents=synthetic_opponents(players, board, complete([wrs[MAX_POSITIONS["WR"]]])),
    )
    if capped.choose_opponent(0, 1).position == "WR":
        fails.append("an opponent at the WR cap drafted another WR")
    capped_cands = capped.candidates(wrs[: MAX_POSITIONS["WR"]], per_pos=1, picks_left=None)
    if any(c.position == "WR" for c in capped_cands):
        fails.append("my candidate set offered a WR past the WR cap")

    for loss in (0.3, 1.5, 2.7):
        power = rank_power(loss, len(players))
        if abs(expected_log2_rank(power, len(players)) - loss) > 1e-6:
            fails.append(f"source-adherence calibration missed mean log2 loss {loss}")

    print(
        "  opponent strategies: provider order stays personal-value-independent, starter "
        "needs and excessive depth softly adjust it, the position caps bind, and fitted "
        "rank noise reproduces source adherence",
        file=sys.stderr,
    )
    return fails


def planning_selftest(players: list[Player]) -> list[str]:
    """My policy uses lineup value, survival-gates live choices, and falls back safely."""
    fails: list[str] = []
    board = fresh_board()
    levels = seed_levels(players)
    opponents = synthetic_opponents(players, board)
    first_index = board.pick_nos.index(board.my_picks[0])
    state = Draft(players, levels, board, opponents=opponents)
    narrow = state.score_my_candidates(first_index)
    broad = state.score_my_candidates(first_index, per_pos=FIRST_PICK_PER_POS)
    if len(broad) <= len(narrow):
        fails.append("planning: the first-pick candidate pool did not broaden")

    # The live pool is built before an intervening deterministic opponent can erase a
    # plausible option. The survival floor, rather than that one path, removes long shots.
    contested = broad[0][2]
    external = [contested] + [p for p in players if p.player_id != contested.player_id]
    opponents = synthetic_opponents(players, board, external)
    deterministic = Draft(players, levels, board, opponents=opponents)
    deterministic.run()
    if deterministic.pick_of.get(contested.player_id) != 1:
        fails.append("planning: synthetic opponent did not take the contested candidate")
    broaden_first_pick(deterministic, players, board, levels, opponents)
    detail = deterministic.my_decisions[board.my_picks[0]]
    if contested.player_id not in {candidate.player_id for _, _, candidate in detail}:
        fails.append("planning: deterministic pre-pick path erased a live candidate")
    low = detail[-1][2]
    survival = {
        candidate.player_id: {
            board.my_picks[0]: 0.04 if candidate.player_id == low.player_id else 0.50
        }
        for _, _, candidate in detail
    }
    apply_survival_floor(deterministic, board, survival)
    kept = {
        candidate.player_id
        for _, _, candidate in deterministic.my_decisions[board.my_picks[0]]
    }
    if low.player_id in kept or contested.player_id not in kept:
        fails.append("planning: the 5% first-pick survival floor kept the wrong candidates")
    rolled = {
        "pick_no": board.my_picks[0],
        "take_id": contested.player_id,
        "stats": {
            candidate.player_id: {"ev": 1.0, "edge": 0.0, "se": 0.0}
            for _, _, candidate in deterministic.my_decisions[board.my_picks[0]]
        },
        "plans": {},
    }
    recommendation = my_next_picks(
        deterministic, board, rolled, survival, limit=1
    )[0]
    if recommendation["take_id"] != contested.player_id:
        fails.append("planning: deterministic availability vetoed the conditional take")
    if recommendation.get("deterministic_fallback_id") in (None, contested.player_id):
        fails.append("planning: conditional take did not retain a distinct legal fallback")
    later = board.my_picks[1]
    conditional = {contested.player_id: {board.my_picks[0]: 0.50, later: 0.02}}
    if conditional_survival(
        conditional, contested.player_id, board.my_picks[0], later
    ) >= 0.05:
        fails.append("planning: later survival was not conditioned on reaching the first pick")

    # Base 1's own plan is worth +60 over the ordinary policy, so it keeps the take. A base
    # pinned at zero instead makes candidate 2's +20 look like the only improvement, and
    # candidate 3's larger mean margin is all playout noise.
    rollout_baselines = {pid: [100.0] * 4 for pid in (1, 2, 3)}
    rollout_stats, rollout_take = rollout_decision(
        [1, 2, 3],
        {1: [160.0] * 4, 2: [120.0] * 4, 3: [300.0, 60.0, 300.0, 60.0]},
        rollout_baselines,
    )
    if rollout_stats[1]["edge"] != 60.0:
        fails.append("planning: the rollout base's own plan edge was not measured")
    if rollout_take != 1:
        fails.append("planning: a rollout candidate worth less than the base took the pick")
    # A smaller margin that is consistent across the shared draws does override it.
    if rollout_decision(
        [1, 2, 3],
        {1: [160.0] * 4, 2: [120.0] * 4, 3: [165.0, 166.0, 164.0, 165.0]},
        rollout_baselines,
    )[1] != 3:
        fails.append("planning: the rollout take did not beat the base's plan through the noise")

    # Even past an opponent's comfortable depth, my reported value_now is the raw
    # marginal expected-lineup value. The opponent heuristic must not leak into my policy.
    deep_board = fresh_board()
    wrs = [p for p in players if p.position == "WR"]
    deep_roster = wrs[: OPPONENT_DEPTH_TARGETS["WR"]]
    deep_board.rosters[deep_board.my_slot - 1] = deep_roster
    deep_board.picks_left[deep_board.my_slot - 1] -= len(deep_roster)
    deep_state = Draft(
        players,
        levels,
        deep_board,
        opponents=synthetic_opponents(players, deep_board),
    )
    deep_index = deep_board.pick_nos.index(deep_board.my_picks[0])
    deep_sorted = sorted_roster(deep_roster)
    deep_base = team_value(deep_sorted, deep_state.levels)
    for now, _, candidate in deep_state.score_my_candidates(deep_index):
        raw_marginal = team_value(deep_sorted, deep_state.levels, candidate) - deep_base
        if abs(now - raw_marginal) > 1e-9:
            fails.append("planning: opponent depth heuristic changed my marginal value")
            break

    # A target can be an interior player the shortlist would omit. The opponents ahead
    # of my first pick take the top of the external order, so index 30 guarantees he
    # is still on the board when the plan reaches my turn.
    first_target = players[30]
    second_target = next(p for p in players if p.player_id != first_target.player_id)
    external = [second_target] + [p for p in players if p.player_id != second_target.player_id]
    opponents = synthetic_opponents(players, board, external)
    my_indices = [i for i, slot in enumerate(board.order) if slot == board.my_slot]
    planned = Draft(
        players,
        levels,
        board,
        opponents=opponents,
        targets={my_indices[0]: first_target, my_indices[1]: second_target},
    )
    planned.run(stop_before=my_indices[1] + 1)
    if planned.pick_of.get(first_target.player_id) != board.my_picks[0]:
        fails.append("planning: an available first target was not exercised")
    if planned.pick_of.get(second_target.player_id) != 1:
        fails.append("planning: the opponent did not take the later target first")
    if board.my_picks[1] not in planned.pick_of.values():
        fails.append("planning: an unavailable later target did not fall back")

    print(
        f"  planning: first-pick pool widened from {len(narrow)} to {len(broad)}; "
        "live candidates survived the deterministic prefix, the 5% floor removed long "
        "shots, the conditional take kept a distinct fallback, the rollout measured the "
        "base's own plan and only lost the take to a margin above the playout noise, my "
        "values stayed projection-only, and an unavailable later target fell back",
        file=sys.stderr,
    )
    return fails


def lineup_selftest() -> list[str]:
    """The weekly lineup solver is exact on small cases and monotone by construction."""
    fails: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            fails.append(f"lineup: {message}")

    def player(
        player_id: int, name: str, position: str, weekly: list[float]
    ) -> Player:
        weeks = tuple(weekly) + (0.0,) * (WEEKS - len(weekly))
        return Player(
            player_id=player_id,
            name=name,
            position=position,
            team="TEST",
            age=25.0,
            bye_week=None,
            is_rookie=False,
            points=round(sum(weeks), 2),
            provider_adp=None,
            weekly=weeks,
        )

    def flat(points_per_week: float) -> list[float]:
        return [points_per_week] * WEEKS

    def levels_with(wire: dict[str, float], weights=None) -> Levels:
        if weights is None:
            weights = tuple(1.0 / WEEKS for _ in range(WEEKS))
        cols = tuple(tuple((wire[pos],) for _ in range(WEEKS)) for pos in POSITIONS)
        zero = tuple(tuple((0.0,) for _ in range(WEEKS)) for _ in POSITIONS)
        return Levels(weights=tuple(weights), wire=cols, drop_floor=zero)

    # With one QB job: QB1 always supplies his unconditional projection, QB2 is used
    # when QB1 is unavailable, and the unique wire body is used only when both are out.
    u = UNAVAILABLE_RATE["QB"]
    one_qb = position_expected_value([(1, 100.0), (2, 80.0)], (50.0,), u, 1)
    check(
        abs(one_qb - (100.0 + u * 80.0 + u**2 * 50.0)) < 1e-9,
        f"one-QB expectation is {one_qb:.6f}",
    )
    two_qb = position_expected_value([(1, 100.0), (2, 80.0)], (50.0,), u, 2)
    check(
        abs(two_qb - (180.0 + (1.0 - (1.0 - u) ** 2) * 50.0)) < 1e-9,
        f"two-QB expectation is {two_qb:.6f}",
    )

    # One wire QB cannot fill the dedicated QB slot and a flex seat at once, and the
    # QB wire never enters the flex pool at all.
    empty = week_value(
        {}, {"QB": (100.0,), "RB": (), "WR": (), "TE": ()}, WEEKLY_SHAPES[0]
    )
    check(abs(empty - 100.0) < 1e-9, "one wire QB filled more than one lineup slot")

    # The closed-form weekly value against literal brute force, for every distinct
    # in-season starting shape (base, +WR, +RB, +FLEX, and the superflex-as-2QB
    # weeks): every availability subset, weighted exactly, each solved by
    # enumerating all legal lineup compositions.
    def legal_compositions(shape: dict[str, int]) -> list[dict[str, int]]:
        total = sum(shape.values())
        flex_seats = shape["FLEX"]
        out = []
        for rb in range(shape["RB"], shape["RB"] + flex_seats + 1):
            for wr in range(shape["WR"], shape["WR"] + flex_seats + 1):
                for te in range(shape["TE"], shape["TE"] + flex_seats + 1):
                    counts = {"QB": shape["QB"], "RB": rb, "WR": wr, "TE": te}
                    if sum(counts.values()) == total:
                        out.append(counts)
        return out

    def brute_week_value(
        projections: dict[str, list[tuple[int, float]]],
        wires: dict[str, tuple[float, ...]],
        shape: dict[str, int],
    ) -> float:
        comps = legal_compositions(shape)
        bodies = []
        certain: dict[str, list[float]] = {pos: [] for pos in POSITIONS}
        for pos in POSITIONS:
            for _, points in projections.get(pos, []):
                if points > 0:
                    rate = 1.0 - UNAVAILABLE_RATE[pos]
                    bodies.append((pos, points / rate, rate))
            certain[pos].extend(v for v in wires[pos] if v > 0)
        total = 0.0
        for mask in range(1 << len(bodies)):
            prob = 1.0
            avail = {pos: list(certain[pos]) for pos in POSITIONS}
            for i, (pos, w, rate) in enumerate(bodies):
                if mask >> i & 1:
                    prob *= rate
                    avail[pos].append(w)
                else:
                    prob *= 1.0 - rate
            for pos in POSITIONS:
                avail[pos].sort(reverse=True)
            total += prob * max(
                sum(sum(avail[pos][: c[pos]]) for pos in POSITIONS) for c in comps
            )
        return total

    brute_projections = {
        "QB": [(9001, 21.0), (9002, 17.5)],
        "RB": [(9010, 15.0), (9011, 13.5), (9012, 11.0), (9013, 10.5)],
        "WR": [(9020, 14.5), (9021, 12.0), (9022, 11.5)],
        "TE": [(9030, 12.5), (9031, 11.8)],
    }
    brute_wires = {"QB": (9.0,), "RB": (6.5,), "WR": (7.0,), "TE": (6.0,)}
    for shape in {tuple(sorted(s.items())) for s in WEEKLY_SHAPES}:
        shape = dict(shape)
        brute = brute_week_value(brute_projections, brute_wires, shape)
        closed = week_value(brute_projections, brute_wires, shape)
        check(
            abs(brute - closed) < 1e-9,
            f"shape {shape}: closed form {closed:.6f} != brute force {brute:.6f}",
        )
    # Late-season rosters hold several waiver bodies per position at decaying
    # tiers; the closed form must price the multiplicity exactly too.
    late_wires = {
        "QB": (9.0, 7.5),
        "RB": (6.5, 5.5, 4.5, 3.5),
        "WR": (7.0, 6.0, 5.0, 4.0),
        "TE": (6.0, 5.0, 4.0),
    }
    late_shape = WEEKLY_SHAPES[-1]
    brute = brute_week_value(brute_projections, late_wires, late_shape)
    closed = week_value(brute_projections, late_wires, late_shape)
    check(
        abs(brute - closed) < 1e-9,
        f"tiered wire bodies: closed form {closed:.6f} != brute force {brute:.6f}",
    )

    # Weekly decomposition: a zero week contributes nothing that week and leaves the
    # other weeks untouched.
    wires = {"QB": 5.0, "RB": 5.0, "WR": 5.0, "TE": 5.0}
    lv = levels_with(wires)
    active = player(900040, "Full WR", "WR", flat(12.0))
    missing_weeks = flat(12.0)
    missing_weeks[2] = 0.0
    absent = player(900040, "Absent WR", "WR", missing_weeks)
    full = weekly_team_values([active], lv)
    gap = weekly_team_values([absent], lv)
    check(
        all(abs(a - b) < 1e-9 for w, (a, b) in enumerate(zip(full, gap)) if w != 2),
        "a zero week changed another week's value",
    )
    check(gap[2] < full[2] - 1e-9, "a zero week did not lower that week's value")

    # Guillotine weights direct value toward the weeks that matter: with all weight
    # on week 3, the absent player is worth strictly less than the active one of
    # identical season total; with no weight there, they tie.
    week3_only = tuple(1.0 if w == 2 else 0.0 for w in range(WEEKS))
    lv3 = levels_with(wires, week3_only)
    check(
        team_value([active], lv3) > team_value([absent], lv3) + 1e-9,
        "a week-3 absence was not punished under week-3 weight",
    )
    no_week3 = tuple(0.0 if w == 2 else 1.0 / (WEEKS - 1) for w in range(WEEKS))
    lv_not3 = levels_with(wires, no_week3)
    check(
        abs(team_value([active], lv_not3) - team_value([absent], lv_not3)) < 1e-9,
        "a week-3 absence leaked into other weeks' value",
    )

    base = [
        player(900010, "Base QB", "QB", flat(19.0)),
        player(900011, "RB 1", "RB", flat(13.0)),
        player(900012, "RB 2", "RB", flat(12.0)),
        player(900013, "WR 1", "WR", flat(13.5)),
        player(900014, "WR 2", "WR", flat(12.5)),
        player(900015, "TE 1", "TE", flat(10.5)),
    ]
    lv = levels_with({"QB": 8.0, "RB": 6.0, "WR": 5.5, "TE": 5.7})
    low = player(900020, "Low WR", "WR", flat(9.0))
    high = player(900021, "High WR", "WR", flat(11.0))
    check(
        team_value(sorted_roster(base + [high]), lv)
        > team_value(sorted_roster(base + [low]), lv),
        "a higher same-position projection lost value",
    )
    before = team_value(sorted_roster(base), lv)
    after = team_value(sorted_roster(base + [low]), lv)
    check(after >= before, "adding a player lowered roster value")
    # Crossing the wire level must not create a discontinuous loss.
    at_wire = player(low.player_id, low.name, low.position, flat(5.5))
    above_wire = player(low.player_id, low.name, low.position, flat(5.6))
    check(
        team_value(sorted_roster(base + [above_wire]), lv)
        > team_value(sorted_roster(base + [at_wire]), lv),
        "crossing the wire threshold lowered value",
    )

    print(
        "  weekly lineup: exact QB depth probabilities, unique wire capacity, "
        f"brute-force agreement on {len({tuple(sorted(s.items())) for s in WEEKLY_SHAPES})} "
        "starting shapes, zero-week isolation, week-weight response, and "
        "projection/addition monotonicity",
        file=sys.stderr,
    )
    return fails



def synthetic_draft(
    players: list[Player],
    made: int = 0,
    unrankable: dict[int, str] | None = None,
    trades: dict[int, int] | None = None,
) -> dict:
    """A `draft.json`-shaped board built offline, for the states the live file cannot reach.

    Today's live file has no traded picks and no selection outside the pool, so the two
    branches that handle them would go unexercised until the night they matter. Made picks
    take the pool in points order, which is a legal board and enough to check bookkeeping.
    `unrankable` maps a pick number to a position for a selection the pool does not carry;
    `trades` maps a pick number to the roster id that acquired it.
    """
    order = draft_order()
    slots = [
        {
            "draft_slot": s,
            "roster_id": 20 + s,  # deliberately not equal to the slot, as Sleeper's are not
            "user_id": None,
            "username": f"team{s}",
            "team_name": None,
            "is_mine": s == MY_SLOT,
        }
        for s in range(1, TEAMS + 1)
    ]
    roster_of_slot = {s["draft_slot"]: s["roster_id"] for s in slots}
    take = iter(players)
    picks: list[dict] = []
    for n, slot in enumerate(order, start=1):
        owner = (trades or {}).get(n, roster_of_slot[slot])
        pick = {
            "pick_no": n,
            "round": (n - 1) // TEAMS + 1,
            "pick_in_round": (n - 1) % TEAMS + 1,
            "draft_slot": slot,
            "roster_id": owner,
            "user_id": None,
            "username": None,
            "is_mine": owner == roster_of_slot[MY_SLOT],
            "status": "made" if n <= made else "pending",
            "sleeper_id": None,
            "name": None,
            "position": None,
            "team": None,
            "is_keeper": None,
        }
        if n <= made and (unrankable or {}).get(n):
            pick |= {
                "sleeper_id": f"not-in-pool-{n}",
                "name": f"Unrankable {n}",
                "position": unrankable[n],
                "team": "FA",
            }
        elif n <= made:
            p = next(take)
            pick |= {
                "sleeper_id": p.sleeper_id,
                "name": p.name,
                "position": p.position,
                "team": p.team,
            }
        picks.append(pick)

    pending = [p for p in picks if p["status"] == "pending"]
    mine = next((p for p in pending if p["is_mine"]), None)

    def summary(pick: dict | None) -> dict | None:
        if pick is None:
            return None
        return {
            "pick_no": pick["pick_no"],
            "round": pick["round"],
            "pick_in_round": pick["pick_in_round"],
            "draft_slot": pick["draft_slot"],
            "username": pick["username"],
            "slot": pick_label(pick["pick_no"]),
        }

    return {
        "source": "synthetic",
        "fetched_at": "2026-08-04T00:00:00+00:00",
        "draft_id": "synthetic",
        "league_name": "selftest",
        "status": "drafting",
        "format": {
            "type": "snake",
            "teams": TEAMS,
            "rounds": ROUNDS,
            "reversal_round": league.REVERSAL_ROUND or None,
        },
        "pick_count": TOTAL_PICKS,
        "picks_made": made,
        "picks_pending": TOTAL_PICKS - made,
        "on_the_clock": summary(pending[0] if pending else None),
        "me": {"username": "me", "draft_slot": MY_SLOT, "roster_id": roster_of_slot[MY_SLOT]},
        "my_next_pick": summary(mine),
        "slots": slots,
        "traded_picks": [{"round": (n - 1) // TEAMS + 1} for n in (trades or {})],
        "picks": picks,
    }


def board_selftest(players: list[Player]) -> list[str]:
    """The live-board loader, on states the real draft.json does not currently contain."""
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            fails.append(f"board: {msg}")

    # An untouched live board must be the static snake, or the live path and the offline
    # path disagree about the league before a single pick is made.
    board, problems = load_board(synthetic_draft(players), players, "synthetic")
    fresh = fresh_board()
    check(not problems, f"empty synthetic board complained: {problems}")
    check(board.order == fresh.order, "empty live board's order != the static snake")
    check(board.pick_nos == fresh.pick_nos, "empty live board's pick numbers != 1..256")
    check(board.my_picks == fresh.my_picks, "empty live board's picks for me != the snake's")
    check(board.picks_left == fresh.picks_left, f"picks left {board.picks_left} != all {ROUNDS}")
    check(not board.taken and board.picks_made == 0, "empty live board has players drafted")

    # Made picks leave the pool and land on the team that made them.
    board, problems = load_board(synthetic_draft(players, made=33), players, "synthetic")
    check(not problems, f"33-pick board complained: {problems}")
    check(board.picks_made == 33 and len(board.taken) == 33, "33 made picks did not come through")
    check(board.pick_nos[:1] == [34], f"simulation resumes at {board.pick_nos[:1]}, want 34")
    check(board.my_picks[:1] == [45], f"my next pick is {board.my_picks[:1]}, want 45 (2.13)")
    check(
        [p.name for p in board.rosters[MY_SLOT - 1]] == [players[19].name],
        "pick 1.20 did not land on my roster",
    )
    check(board.picks_left[MY_SLOT - 1] == ROUNDS - 1, "my remaining picks did not drop by one")
    check(sum(board.picks_left) == TOTAL_PICKS - 33, "remaining picks do not sum to the board")
    # Picks 32 and 33 are both slot 32 — the turn at the end of round 1 into round 2.
    check(len(board.rosters[31]) == 2, "slot 32 did not get both sides of its turn")

    # A traded pick is exercised by the roster that acquired it, not by its column.
    board, problems = load_board(
        synthetic_draft(players, trades={5: 20 + MY_SLOT}), players, "synthetic"
    )
    check(board.order[4] == MY_SLOT, f"traded pick 5 is exercised by slot {board.order[4]}")
    check(
        board.my_picks[:3] == [5, 20, 45],
        f"my picks start {board.my_picks[:3]}, want the traded 5, my own 1.20, then 2.13",
    )
    check(
        board.picks_left[MY_SLOT - 1] == ROUNDS + 1 and board.picks_left[4] == ROUNDS - 1,
        "a traded pick did not move between the two teams' pick counts",
    )
    check(board.owed_size(MY_SLOT) == ROUNDS + 1, "the acquiring team's roster size did not grow")

    # A selection the pool cannot value fills a spot and answers its mandatory position.
    board, problems = load_board(
        synthetic_draft(players, made=1, unrankable={1: "QB"}), players, "synthetic"
    )
    check(not board.taken, "an unrankable pick took a pool player off the board")
    check(len(board.off_pool[0]) == 1, "an unrankable pick was not held as a roster spot")
    check(board.picks_left[0] == ROUNDS - 1, "an unrankable pick did not cost its team a pick")
    levels = seed_levels(players)
    draft = Draft(players, levels, board, opponents=synthetic_opponents(players, board))
    owed = sum(DEDICATED_SLOTS.values()) - 1  # the QB is answered, the rest still owed
    with_qb = draft.candidates([], picks_left=owed, off=board.off_pool[0])
    without = draft.candidates([], picks_left=owed, off=[])
    check(
        {c.position for c in with_qb} == {"RB", "WR", "TE"},
        "an unrankable QB did not satisfy the QB requirement",
    )
    check(
        {c.position for c in without} == set(POSITIONS),
        "the same team without him should still owe a QB",
    )

    # Resuming: every made pick survives, every pending pick is played exactly once.
    board, _ = load_board(synthetic_draft(players, made=57), players, "synthetic")
    opponents = synthetic_opponents(players, board)
    partial = Draft(players, levels, board, opponents=opponents)
    partial.run(stop_before=5)
    check(
        set(partial.pick_of.values()) == set(board.pick_nos[:5]),
        "a short redraw did not stop immediately before its requested pick index",
    )
    draft = Draft(players, levels, board, opponents=opponents)
    draft.run()
    check(
        len(draft.taken) == len(board.taken) + len(board.order),
        "the resumed draft did not take one new player per pending pick",
    )
    check(
        set(draft.pick_of.values()) == set(board.pick_nos),
        "the simulated picks are not exactly the board's pending picks",
    )
    check(not (set(draft.pick_of) & board.taken), "a player already drafted was drafted again")
    for slot in range(1, TEAMS + 1):
        made = board.rosters[slot - 1]
        got = draft.rosters[slot - 1]
        check(got[: len(made)] == made, f"slot {slot} lost one of its made picks")
        check(
            len(got) == len(made) + board.picks_left[slot - 1],
            f"slot {slot} finished with {len(got)} players, not what it owns",
        )

    # A board that disagrees with this script must say so, not be quietly absorbed. Every
    # one of these is a way a wrong draft.json could otherwise produce a plausible board.
    def complains(raw: dict, about: str, pool: list[Player] = players) -> None:
        _, problems = load_board(raw, pool, "synthetic")
        check(bool(problems), f"a board with {about} was accepted without complaint")

    raw = synthetic_draft(players, made=2)
    raw["format"]["teams"] = 12
    complains(raw, "12 teams")
    raw = synthetic_draft(players, made=3)
    raw["picks"][2] |= {
        "sleeper_id": raw["picks"][0]["sleeper_id"],
        "name": raw["picks"][0]["name"],
    }
    complains(raw, "the same player drafted twice")
    raw = synthetic_draft(players, made=0)
    raw["picks"][4] |= {"roster_id": None, "draft_slot": None}
    complains(raw, "a pick nobody owns")
    raw = synthetic_draft(players, made=3)
    raw["picks_made"] = 5
    complains(raw, "a header contradicting its own picks")
    raw = synthetic_draft(players, made=0)
    raw["me"]["draft_slot"] = 7
    complains(raw, "a different draft slot for me")
    complains(
        synthetic_draft(players, made=6),
        "a pool carrying no sleeper ids to join on",
        [dataclasses.replace(p, sleeper_id=None) for p in players],
    )

    print(
        "  live board: static snake reproduced, made picks retained, traded pick and "
        "unvalued pick handled, 63 pending picks resumed, 6 bad boards rejected",
        file=sys.stderr,
    )
    return fails


def guillotine_selftest(players: list[Player]) -> list[str]:
    """Bars, week weights, and the waiver escalation respond to rosters sanely."""
    fails: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            fails.append(f"guillotine: {message}")

    # A full league drafted round-robin off the top of the pool: slot 0 is mine.
    top = players[: TEAMS * ROUNDS]
    rosters = [top[i::TEAMS] for i in range(TEAMS)]
    mine, opponent_rosters = rosters[0], rosters[1:]
    taken = {p.player_id for p in top}
    pos = pos_sorted(players)
    seeded = seed_levels(players)
    solved, diag = guillotine.solve(mine, opponent_rosters, taken, pos, seeded, SEED)

    check(
        len(solved.weights) == WEEKS and abs(sum(solved.weights) - 1.0) < 1e-9,
        f"weights are not a normalized length-{WEEKS} distribution",
    )
    check(all(w >= 0.0 for w in solved.weights), "a week weight went negative")
    check(
        0.0 <= diag["p_reach_final"] <= 1.0 and 0.0 <= diag["p_win_final"] <= 1.0,
        "survival probabilities left [0, 1]",
    )
    for i, k in enumerate(POSITIONS):
        check(
            all(v == 0.0 for v in solved.drop_floor[i][0]),
            f"{k} drop floor is nonzero in week 1, before any elimination",
        )
        check(
            all(
                v + 1e-9 >= f
                for bodies, floor_bodies in zip(solved.wire[i], solved.drop_floor[i])
                for v, f in zip(bodies, floor_bodies)
            ),
            f"{k} wire dips below its drop floor",
        )
        check(
            all(
                all(hi + 1e-9 >= lo for hi, lo in zip(bodies, bodies[1:]))
                for bodies in solved.wire[i]
            ),
            f"{k} wire tiers are not decaying (a later add beat an earlier one)",
        )
    check(
        any(
            solved.drop_floor[i][REGULAR_WEEKS - 1][0] > 0.0
            for i in range(len(POSITIONS))
        ),
        "28 eliminated rosters raised no position's week-15 replacement level",
    )

    # Deterministic: the level map must repeat exactly for the cycle detector.
    again, _ = guillotine.solve(mine, opponent_rosters, taken, pos, seeded, SEED)
    check(again == solved, "the same rosters produced different levels")

    # A roster of late-round leftovers must be likelier to fall to the bar than the
    # slot-1 round-robin roster facing the same field.
    weak = top[-ROUNDS:]
    _, weak_diag = guillotine.solve(weak, opponent_rosters, taken, pos, seeded, SEED)
    check(
        weak_diag["p_reach_final"] < diag["p_reach_final"],
        "a leftover roster out-survived the top round-robin roster",
    )

    print(
        "  guillotine: normalized week weights, elimination floor only after cuts, "
        "deterministic level map, and weaker rosters dying sooner "
        f"(p_reach {weak_diag['p_reach_final']:.3f} < {diag['p_reach_final']:.3f})",
        file=sys.stderr,
    )
    return fails


def selftest(players: list[Player]) -> int:
    print("selftest:", file=sys.stderr)
    # The synthetic scenarios verify the balance/depth/cap machinery around a known
    # external order; a configured league tilt (e.g. the QB scarcity prior) would
    # reorder their hand-built boards, so it is neutralized for the duration.
    saved_tilt = dict(OPPONENT_POSITION_TILT)
    OPPONENT_POSITION_TILT.clear()
    try:
        fails = (
            lineup_selftest()
            + guillotine_selftest(players)
            + opponent_selftest(players)
            + planning_selftest(players)
            + board_selftest(players)
        )
    finally:
        OPPONENT_POSITION_TILT.update(saved_tilt)
    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    verdict = f"{len(fails)} failure(s)" if fails else "all checks passed"
    print(f"selftest: {verdict}", file=sys.stderr)
    return 1 if fails else 0
