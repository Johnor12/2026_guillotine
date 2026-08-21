# Draft pipeline

`fetch_draft.py` is the network-only entry point that publishes `draft.json` on demand.
It does not cache responses because a stale live board is worse than another small fetch.

```bash
uv run draft_pipeline/fetch_draft.py
uv run draft_pipeline/fetch_draft.py --report
uv run draft_pipeline/fetch_draft.py --selftest
```

## Internal boundaries

- `fetch_draft.py`: the ESPN request, ESPN-to-board adaptation, and CLI orchestration
- `draft_history.py`: parse a hand-pasted draft-room pick history and match it to Sleeper
- `draft_board.py`: draft geometry, ownership, pick rows, and JSON document construction
- `report.py`: integrity diagnostics and the optional `pool.json` join report
- `selftest.py`: offline formats, reversal, trade, autopick, and malformed-board checks
- `paths.py`: this process's input/output locations

## Inputs and output

One request to ESPN's undocumented v3 league API answers everything: the league
endpoint with the `mDraftDetail` (picks), `mSettings` (draft order and roster shape),
and `mTeam` (teams and members) views. The league id, season, and my SWID/espn_s2
auth cookies are hardcoded at the top of `fetch_draft.py`; refresh the cookies from a
logged-in browser session when ESPN answers 401/403.

The only on-disk input is the pool pipeline's cached Sleeper player dump, which carries
`espn_id` and translates ESPN player ids into `sleeper_id` (D/ST uses ESPN's fixed
`-16000 - proTeamId` form and Sleeper's team-abbreviation DEF ids). Picks with no
Sleeper match keep a null `sleeper_id` with a warning and join nothing in the pool.

## Live-draft lag and the pick-history overlay

ESPN's read API does not keep up with a running draft: in a live test, a pick made in
the draft room stayed `playerId` -1 on `mDraftDetail` for many minutes (the origin
itself, not an edge cache — the picks may only sync once the draft completes). The
board layout, settings, and members are all still served correctly mid-draft; only the
selections lag.

So on draft day the room's pick history is copied by hand into `draft_history.txt` at
the repo root and every fetch overlays it: `draft_history.py` parses the paste
(records anchor on the position line, so the trailing fantasy-team/points/rank columns
are ignored and cannot break it, and the `Pick` column is read as round-relative),
matches players to the Sleeper dump by normalized name with position required and team
as tiebreaker (D/ST by team abbreviation), and `fetch_draft.py` merges the result into
the fetched board — any pick ESPN does report wins over the paste, and ownership comes
from ESPN's laid-out board rather than the paste's team-name column. Delete or empty
the file when it is stale; a malformed paste fails the fetch loudly.

Check a paste by eye before a refresh consumes it:

```bash
uv run draft_pipeline/draft_history.py --teams 10
uv run draft_pipeline/draft_history.py --selftest
```

`draft.json.picks` always contains every pick, indexed by gap-free `pick_no`. Made rows
carry the matched Sleeper player fields. Pending rows carry null player fields but
already identify the current owner, so consumers can answer who picks next and when my
next pick occurs. `sleeper_id` joins made picks to `pool.json`; name, position, and NFL
team are informational. ESPN team ids flow through `roster_id` and member SWIDs through
`user_id`.

## Derived pending picks

ESPN lays out the whole board up front (unmade picks carry `playerId` -1, which this
pipeline treats as pending) but the board is still derived from the draft settings, as
it was for Sleeper:

- normal snake rounds alternate direction;
- a reversal round repeats the prior round's direction and flips parity thereafter;
- traded ownership is applied by round and the pick's original roster.

This league is a plain snake with no reversal round: odd rounds run forward, even
rounds reverse. Slot 2 therefore owns 1.02, 2.09, 3.02, 4.09, …, 12.09, 13.02 before
trades.
ESPN live drafts cannot trade picks, so `traded_picks` is always empty. (The reversal
and trade logic stay in the code and self-test because `draft_board.py` still supports
them.)

Every fetch compares the derived roster against the `teamId` ESPN reports on every made
pick. Disagreements are warnings and are retained in `board_derivation`. The offline
self-test covers later rounds and trade cases the current live board may not yet
exercise.
