"""The draft pool: pool.json rows as Player objects.

The value input is `weekly_points`: DraftSharks' native per-week projections in this
league's scoring for weeks 1-17, attached by pool/build_pool.py, so byes and known
absences are explicit zero weeks. `points` here is the sum of those weeks, one currency
throughout (pool.json's own `points` column is Sleeper's season projection, the pool's
membership gate). The pool build already restricts membership to QB/RB/WR/TE with a
Sleeper id, so this only checks the shape it promises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .league import POSITIONS, WEEKS


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
    raw = json.loads(path.read_text())
    players: list[Player] = []
    for rec in raw["players"]:
        if rec["position"] not in POSITIONS:
            raise ValueError(f"{rec['name']} is a {rec['position']}; rebuild pool.json")
        weekly = rec["weekly_points"]
        if len(weekly) != WEEKS:
            raise ValueError(
                f"{rec['name']} carries {len(weekly)} weekly points, want {WEEKS}; "
                "rebuild pool.json (pool/build_pool.py)"
            )
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
        "source_file": path.name,
        "sources": raw.get("sources"),
        "pool_size": len(players),
        "by_position": counts,
        # The join key to draft.json. A pool player without one can never be recognised
        # as drafted, so a shortfall here is a silent way for the live board to go wrong.
        "with_sleeper_id": sum(1 for p in players if p.sleeper_id),
    }
    return players, meta


def by_position(players: list[Player]) -> dict[str, list[Player]]:
    out: dict[str, list[Player]] = {pos: [] for pos in POSITIONS}
    for p in players:
        out[p.position].append(p)
    return out
