"""The draft pool: pool.json rows as Player objects.

The value input is `weekly_points` — DraftSharks' native per-week projections in this
league's scoring for weeks 1-17, attached by the pool pipeline's weekly stage, so byes
and known absences are explicit zero weeks. `points` is kept as the sum of those
weeks, one currency throughout. Draftsharks' 3D value is ignored entirely
and is not even carried into the pool: it is a provider-scaled ordinal that already
bakes in someone else's roster assumptions, and it is not in points, so it cannot
enter a points-denominated lineup objective. Kickers and IDP are already dropped
upstream because the roster has no slot for them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .league import POINTS_FIELD, POSITIONS, WEEKS


@dataclass(slots=True)
class Player:
    player_id: int
    name: str
    position: str
    team: str
    age: float | None
    bye_week: int | None
    is_rookie: bool
    points: float
    provider_adp: float | None
    sleeper_id: str | None = None  # the only key draft.json shares with the pool
    availability_index: int = 0  # rank on the opponents' consensus board, 0-based
    # Projected points per league week (index 0 = week 1); 0.0 = bye or known absence.
    weekly: tuple[float, ...] = field(default=())


def load_pool(path: Path) -> tuple[list[Player], dict]:
    """Read pool.json: already filtered to QB/RB/WR/TE with a usable one-season projection.

    build_pool.py does the filtering — positions with no roster slot, the source's zeros
    where a null belongs — so this only re-checks the
    invariants it promises rather than re-deriving them. The guards stay because a
    hand-edited or stale pool is the likeliest way this ever gets bad input.
    """
    raw = json.loads(path.read_text())
    players: list[Player] = []
    dropped = {"non_offense": 0, "zero_projection": []}
    for rec in raw["players"]:
        if rec["position"] not in POSITIONS:
            dropped["non_offense"] += 1
            continue
        # Direct key access: a pool.json without the value columns predates this league's
        # scoring or the weekly-shape stage and must be rebuilt, not silently valued
        # at zero. Guillotine valuation is per-week, so `points` becomes the sum of
        # the weeks actually playable here (weeks 1-17; the shape stage already
        # dropped any week-18 share) — one currency for sorting, display, and value.
        season = rec[POINTS_FIELD]
        weekly = rec["weekly_points"]
        if len(weekly) != WEEKS:
            raise ValueError(
                f"{rec['name']} carries {len(weekly)} weekly points, want {WEEKS} — "
                "rebuild pool.json (pool_pipeline/apply_weekly_shape.py)"
            )
        if not season or season <= 0:
            dropped["zero_projection"].append(rec["name"])
            continue
        players.append(
            Player(
                player_id=rec["player_id"],
                name=rec["name"],
                position=rec["position"],
                team=rec["team"],
                age=rec.get("age"),
                bye_week=rec.get("bye_week"),
                is_rookie=bool(rec.get("is_rookie")),
                points=round(sum(weekly), 2),
                provider_adp=rec.get("adp"),
                sleeper_id=rec.get("sleeper_id"),
                weekly=tuple(weekly),
            )
        )

    players.sort(key=lambda p: (-p.points, p.player_id))
    counts = {pos: 0 for pos in POSITIONS}
    for p in players:
        counts[p.position] += 1

    meta = {
        "source_file": str(path),
        "source_player_count": raw.get("player_count", len(raw["players"])),
        "source_of_pool": raw.get("source_file"),
        "points_source": (raw.get("points_source") or {}).get("file"),
        "pool_size": len(players),
        "by_position": counts,
        # The join key to draft.json. A pool player without one can never be recognised
        # as drafted, so a shortfall here is a silent way for the live board to go wrong.
        "with_sleeper_id": sum(1 for p in players if p.sleeper_id),
        "dropped_non_offense": dropped["non_offense"],
        "dropped_zero_projection": sorted(dropped["zero_projection"]),
    }
    return players, meta


def by_position(players: list[Player]) -> dict[str, list[Player]]:
    out: dict[str, list[Player]] = {pos: [] for pos in POSITIONS}
    for p in players:
        out[p.position].append(p)
    return out
