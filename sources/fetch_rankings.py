#!/usr/bin/env python3
"""Download the public ranking pages the investigator compares drafters against.

    uv run sources/fetch_rankings.py

The raw responses land in data/raw/ (gitignored) and are parsed by build_rankings.py,
so a provider parser can be fixed without downloading a newer board and losing the
snapshot that existed during the draft. All must succeed; a failed fetch leaves the
existing snapshots intact.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"
FETCH_META = RAW / "fetch_meta.json"
TIMEOUT_SECONDS = 30
USER_AGENT = "redraft-data-source-investigator/0.1"

# The league is a 32-team, 1 QB, 0.5 PPR TE-premium guillotine, a format no provider
# publishes. Ordinary 1QB redraft boards stand in for drafters who carry their usual
# rankings into any format; the superflex/2QB variant of each provider stands in for
# drafters pricing this room's QB scarcity (32 starters wanted from ~32 NFL jobs, plus
# the week-14 superflex). The exact format requested from each source is carried into
# boards.json.
SOURCES = {
    "fantasycalc": {
        "name": "FantasyCalc",
        "url": (
            "https://api.fantasycalc.com/values/current?"
            "isDynasty=false&numQbs=1&numTeams=10&ppr=0.5&includeAdp=true"
        ),
        "file": "fantasycalc.json",
        "format": "10-team redraft, 1 QB, 0.5 PPR",
    },
    "fantasycalc_sf": {
        "name": "FantasyCalc SF",
        # 14 teams is the deepest room FantasyCalc's calculator offers.
        "url": (
            "https://api.fantasycalc.com/values/current?"
            "isDynasty=false&numQbs=2&numTeams=14&ppr=0.5&includeAdp=true"
        ),
        "file": "fantasycalc_sf.json",
        "format": "14-team redraft, superflex, 0.5 PPR",
    },
    "keeptradecut": {
        "name": "KeepTradeCut",
        "url": "https://keeptradecut.com/fantasy-rankings?filters=QB%7CWR%7CRB%7CTE&format=1",
        "file": "keeptradecut.html",
        "format": "redraft 1QB (KTC fantasy rankings); no TE premium",
    },
    "ffcalculator": {
        "name": "FF Calculator ADP",
        "url": "https://fantasyfootballcalculator.com/api/v1/adp/half-ppr?teams=10&year=2026",
        "file": "ffcalculator.json",
        "format": "10-team half-PPR mock-draft ADP",
    },
    "ffcalculator_2qb": {
        "name": "FF Calculator 2QB ADP",
        # 14 teams is the deepest room FF Calculator runs 2QB mocks for.
        "url": "https://fantasyfootballcalculator.com/api/v1/adp/2qb?teams=14&year=2026",
        "file": "ffcalculator_2qb.json",
        "format": "14-team 2QB mock-draft ADP",
    },
    "fantasypros": {
        "name": "FantasyPros ECR",
        "url": "https://www.fantasypros.com/nfl/rankings/half-point-ppr-cheatsheets.php",
        "file": "fantasypros.html",
        "format": "redraft half-PPR expert consensus",
    },
    "fantasypros_sf": {
        "name": "FantasyPros SF ECR",
        "url": "https://www.fantasypros.com/nfl/rankings/half-point-ppr-superflex-cheatsheets.php",
        "file": "fantasypros_sf.html",
        "format": "redraft half-PPR superflex expert consensus",
    },
}

# Boards parsed out of another source's snapshot: one download, a second metadata entry
# so build_rankings.py can label the board.
DERIVED_FROM_SNAPSHOT = {
    "keeptradecut_sf": {
        "base": "keeptradecut",
        "name": "KeepTradeCut SF",
        "format": "redraft superflex (KTC fantasy rankings); no TE premium",
    },
}


def fetch(url: str) -> tuple[bytes, str | None]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json,text/html", "User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read(), response.headers.get("Content-Type")


def main() -> int:
    fetched_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    metadata: dict = {"fetched_at": fetched_at, "sources": {}}
    failures: list[str] = []
    downloads: dict[str, bytes] = {}

    for source_id, source in SOURCES.items():
        try:
            body, content_type = fetch(source["url"])
            if not body:
                raise ValueError("empty response")
        except (OSError, ValueError, urllib.error.URLError) as exc:
            failures.append(f"{source_id}: {exc}")
            continue
        downloads[source_id] = body
        metadata["sources"][source_id] = {
            **source,
            "fetched_at": fetched_at,
            "content_type": content_type,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    if failures:
        print("ranking fetch failed; existing snapshots were left intact:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    for source_id, spec in DERIVED_FROM_SNAPSHOT.items():
        metadata["sources"][source_id] = {
            **metadata["sources"][spec["base"]],
            "name": spec["name"],
            "format": spec["format"],
        }

    RAW.mkdir(parents=True, exist_ok=True)
    for source_id, body in downloads.items():
        (RAW / SOURCES[source_id]["file"]).write_bytes(body)
        print(f"  {source_id}: {len(body):,} bytes", file=sys.stderr)
    FETCH_META.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"fetched {len(SOURCES)} ranking sources -> {RAW.relative_to(RAW.parents[2])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
