"""Small economic and auction regressions, independent of live league files."""
import math
import random
import statistics
import datetime as dt
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .claims import _record_values, best_replays, claims, title_objective
from .league import WEEKS
from .race import RaceInputs, _auction, claim_plan, fit_roster, race_inputs, replay
from .season import draftsharks_by_sleeper, lineup_points
from .waivers import Bidding, Manager, fit_managers, spending_allowance, submitted_bids


class WaiverTests(unittest.TestCase):
    def setUp(self):
        # Full opening roster: one QB, one RB, three WRs, three TEs.
        self.positions = [0, 1, 2, 2, 2, 3, 3, 3, 1, 2]
        self.points = [20., 15., 12., 11., 5., 14., 10., 9., 8., 13.]
        self.bidding = Bidding(self.positions, [self.points[:] for _ in range(WEEKS)],
                               [self.points[:] for _ in range(WEEKS)], [0] * 10)
        self.roster = list(range(8))

    def inputs(self):
        return RaceInputs(1, self.positions, self.bidding.weekly, self.bidding.ros,
                          [self.roster[:]], [1000], [True], 0, [8, 9], False,
                          self.bidding, [Manager()], {}, [1000])

    def test_buy_before_expansion(self):
        early = self.bidding.offers(self.roster, [8], 1000, 1)
        preparing = self.bidding.offers(self.roster, [8], 1000, 6)
        self.assertTrue(early, "Value after a distant expansion must count from week 2")
        self.assertTrue(preparing, "A cheap RB should count before the second RB slot opens")
        self.assertGreater(preparing[0].ceiling, 0)

    def test_superflex_coverage_beyond_four_weeks(self):
        self.positions[8] = 0
        for points in self.bidding.weekly:
            points[8] = 18.
        offer = self.bidding.offers(self.roster, [8], 1000, 1)[0]
        self.assertGreater(offer.gain, 0)
        self.assertEqual(lineup_points(self.roster + [8], self.bidding.weekly[1], self.positions, 1),
                         lineup_points(self.roster, self.bidding.weekly[1], self.positions, 1))

    def test_saving_release_and_emergency(self):
        patient = spending_allowance(1000, 1000, 1, 8, "patient", 0)
        balanced = spending_allowance(1000, 1000, 1, 8, "balanced", 0)
        self.assertLess(patient, balanced)
        self.assertLess(balanced, 1000)
        self.assertEqual(spending_allowance(317, 1000, 1, 16, "patient", 0), 317)
        self.assertEqual(spending_allowance(1000, 1000, 1, 8, "patient", .5), 1000)

    def test_opponent_noise_cannot_spend_protected_cash(self):
        inputs = self.inputs()
        inputs.me = -1
        rosters, budgets = [self.roster[:4]], [1000]
        _auction(inputs, 1, rosters, budgets, [True], {8, 9}, random.Random(7), None,
                 [1e6], -100, ["patient"])
        self.assertGreaterEqual(budgets[0], 1000 - spending_allowance(1000, 1000, 1, 1, "patient", 0))

    def test_byes_and_legal_drops(self):
        self.bidding.weekly[2][1] = 0.0
        offer = self.bidding.offers(self.roster, [8], 1000, 1)[0]
        self.assertGreater(offer.ceiling, 0)
        self.assertNotIn(offer.drop, (0, 1, 2, 3), "Keep scarce starters instead of dropping by raw points")

    def test_owned_players_do_not_crowd_out_bargains(self):
        inputs = self.inputs()
        plan, allowance = claim_plan(inputs, inputs.bidding, self.roster, list(range(10)), 1000, 1, 0, random.Random(1))
        self.assertTrue(any(o.player == 9 for _, o in plan))
        self.assertTrue(all(o.player not in self.roster for _, o in plan))
        self.assertEqual(allowance, max(b for b, _ in plan))

    def test_drop_is_chosen_with_the_pickup(self):
        # Week 7's ten-man roster. QB0 misses weeks 7-9 and QB1 covers them; RB8 covers
        # RB2's five zero weeks for a point a week, so alone he is the cheapest cut.
        positions = [0, 0, 1, 1, 2, 2, 2, 3, 1, 3, 0]
        weekly = [[20., 12., 15., 14., 12., 11., 10., 14., 1., 5., 15.] for _ in range(WEEKS)]
        for w in (6, 7, 8):
            weekly[w][0] = 0.
        for w in range(9, 14):
            weekly[w][2] = 0.
        bidding = Bidding(positions, weekly, weekly, [0] * 11)
        roster = tuple(range(10))
        self.assertEqual(bidding.context(roster, 6).options[0][0], 8, "Alone, the point-a-week RB is the cheapest cut")
        offer = bidding.offers(list(roster), [10], 1000, 6)[0]
        self.assertEqual(offer.drop, 1, "With a better QB arriving, the backup QB is the drop")

    def test_one_week_fix_pays_for_the_drop(self):
        # QB0 sits out week 2 only; the streamer scores once. RB7 starts every week (RB1's
        # ten zero weeks, then the expanded lineup), so the streamer is worth a claim only
        # while that cover is worth less than one week.
        def offers(cover):
            positions = [0, 1, 1, 2, 2, 2, 3, 1, 0, 0]
            weekly = [[20., 15., 14., 12., 11., 10., 14., cover, 0., 0.] for _ in range(WEEKS)]
            weekly[1][0] = 0.
            weekly[1][8] = 15.
            for w in range(2, 12):
                weekly[w][1] = 0.
            return Bidding(positions, weekly, weekly, [0] * 10).offers(list(range(8)), [8], 1000, 1)
        cheap = offers(0.5)
        self.assertEqual(cheap[0].drop, 7)
        self.assertAlmostEqual(cheap[0].gain, (15. - 0.5 * (WEEKS - 2)) / (WEEKS - 1))
        self.assertFalse(offers(2.), "Thirty points of cover outweigh one fifteen-point week")

    def test_untaken_wire_refills_the_cover(self):
        # The same thirty points of cover, but an equal RB sits untaken on the wire: hold
        # the streamer his one week, then refill the spot.
        positions = [0, 1, 1, 2, 2, 2, 3, 1, 0, 1]
        weekly = [[20., 15., 14., 12., 11., 10., 14., 2., 0., 2.] for _ in range(WEEKS)]
        weekly[1][0] = 0.
        weekly[1][8] = 15.
        for w in range(2, 12):
            weekly[w][1] = 0.
        bidding = Bidding(positions, weekly, weekly, [0] * 10)
        refilled = bidding.objective(bidding.weights, [9])
        offer = refilled.offers(list(range(8)), [8], 1000, 1)[0]
        self.assertEqual(offer.drop, 7)
        self.assertAlmostEqual(offer.gain, 15. / (WEEKS - 1))
        self.assertEqual(refilled.crunch(list(range(8)) + [8], 1), bidding.crunch(list(range(8)) + [8], 1),
                         "A cut leaves no spot to refill")

    def test_returning_starter_is_not_a_placeholder(self):
        # RB7 is out through week index 3, then starts; RB8 fills in until then.
        positions = [0, 1, 1, 2, 2, 2, 3, 1, 1]
        weekly = [[20., 15., 14., 12., 11., 10., 14., 12., 0.] for _ in range(WEEKS)]
        for w in range(1, 4):
            weekly[w][7], weekly[w][8] = 0., 13.
        self.assertFalse(Bidding(positions, weekly, weekly, [0] * 9).offers(list(range(8)), [8], 1000, 1))
        # A one-point RB on the wire does not stand in for him.
        positions.append(1)
        for points in weekly:
            points.append(1.)
        bidding = Bidding(positions, weekly, weekly, [0] * 10)
        self.assertFalse(bidding.objective(bidding.weights, [9]).offers(list(range(8)), [8], 1000, 1))

    def test_guide_rank_ignores_missed_weeks(self):
        # RB0 misses half the season but outscores the 12- and 13-point RBs whenever he plays.
        positions = [1] * 8
        weekly = [[0. if w % 2 else 15., 12.] + [13.] * 6 for w in range(WEEKS)]
        bidding = Bidding(positions, weekly, weekly, [0] * 8)
        self.assertGreater(bidding.shares[1][0], bidding.shares[1][1])

    def test_reserve_bodies_hold_no_spot_until_they_return(self):
        bidding = Bidding(self.positions, self.bidding.weekly, self.bidding.ros, [2] + [0] * 9)
        roster = list(range(8))  # eight bodies, the QB on reserve through week index 1
        self.assertIsNone(bidding.offers(roster, [9], 1000, 0)[0].drop, "Reserve leaves an open spot")
        self.assertFalse(bidding.fits(roster + [9], 8, 0))
        self.assertNotEqual(bidding.crunch(roster + [8, 9], 0), 0, "Cutting a reserve body frees nothing")
        returned = roster + [9]
        fit_roster(bidding, returned, 2)
        self.assertEqual(len(returned), 8, "The returning QB needs a regular spot")
        self.assertIn(0, returned)

    def test_draftsharks_rows_outside_the_pool_join_by_name(self):
        pool = {"players": [{"player_id": 1, "sleeper_id": "11"}]}
        weekly = {"players": [{"player_id": 1, "name": "Josh Allen", "position": "QB", "weeks": {"3": {"points": 20.}}},
                              {"player_id": 2, "name": "Marcus Mariota", "position": "QB", "weeks": {"3": {"points": 15.}}},
                              {"player_id": 3, "name": "Mike Williams", "position": "WR", "weeks": {"3": {"points": 9.}}}]}
        directory = {"11": {"name": "Josh Allen", "position": "QB"},
                     "22": {"name": "Marcus Mariota", "position": "QB"},
                     "33": {"name": "Mike Williams", "position": "WR"},
                     "34": {"name": "Mike Williams", "position": "WR"}}
        joined = draftsharks_by_sleeper(pool, weekly, directory)
        self.assertEqual(joined, {"11": {"3": 20.}, "22": {"3": 15.}}, "Ambiguous names stay unjoined")

    def test_title_leverage_ignores_safe_weeks(self):
        inputs = self.inputs()
        bars = [-1000.] * WEEKS
        bars[2] = lineup_points(self.roster, self.points, self.positions, 2)
        record = {"seed": 1, "my_bias": 0., "bars": bars, "forecast_bars": [0.] * WEEKS,
                  "alive": [2] * WEEKS, "champ_bar": 0., "auctions": [None] * WEEKS}
        leverage = replay([record], inputs, self.roster, 1000, "value")["leverage"]
        self.assertEqual(leverage[0], 0., "A week cleared in every season carries no weight")
        self.assertGreater(leverage[1], 0.)

    def test_title_objective_keeps_what_replays_better(self):
        inputs = self.inputs()
        auctions = [None] * WEEKS
        auctions[1] = ([8, 9], [5, -1])
        for weighted_title, chosen in ((.2, "title_weighted"), (.05, "points")):
            def runs(inputs, records, variants):
                flat = set(inputs.my_bidding.weights) == {1.}
                return [{"p_title": .1 if flat else weighted_title, "leverage": [2.] + [1.] * (WEEKS - 2)}] * 3
            with patch("ranker.claims.run_replays", side_effect=runs):
                mine, summary = title_objective(inputs, [{"auctions": auctions}])
            self.assertEqual(summary["chosen"], chosen)
            self.assertEqual({r for col in mine.my_bidding.refills[1] for r in col}, {9})
            self.assertIs(mine.bidding, inputs.bidding, "The room keeps its own objective")
        weights = [row["weight"] for row in summary["title_weights"]]
        self.assertAlmostEqual(statistics.fmean(weights), 1., places=2)
        self.assertGreater(weights[0], weights[1])
        self.assertEqual(set(mine.my_bidding.weights), {1.}, "A weighting that replays worse is not used")

    def test_final_dollars_have_no_salvage_value(self):
        offer = self.bidding.offers(self.roster, [9], 317, WEEKS - 1)[0]
        self.assertEqual(offer.ceiling, 317)

    def test_capacity_and_same_drop_claims(self):
        for roster in (self.roster[:], self.roster[:-1]):
            inputs = self.inputs()
            rosters, budgets, free = [roster], [1000], {8, 9}
            _auction(inputs, 1, rosters, budgets, [True], free, random.Random(7), None, [1.0], 40, ["value"])
            self.assertLessEqual(len(rosters[0]), 8)
            self.assertEqual(len(rosters[0]), len(set(rosters[0])))
            self.assertFalse(set(rosters[0]) & free)
            self.assertGreaterEqual(budgets[0], 0)

    def test_losing_bid_is_activity_and_no_permanent_sleepers(self):
        state = SimpleNamespace(teams=[SimpleNamespace(roster_id=1), SimpleNamespace(roster_id=2)])
        obs = [{"team": 1, "player": 9, "week": 2, "bid": 100, "reference": 50.}]
        active, quiet = fit_managers(state, obs)
        self.assertGreater(active.activity, quiet.activity)
        self.assertGreater(quiet.activity, 0)
        self.assertGreater(quiet.participation(13, 1), quiet.activity)
        self.assertLess(active.log_scale, math.log(2), "One bid must stay shrunk toward the prior")

    def test_latest_duplicate_bid_and_pending_not_training_data(self):
        def tx(bid, created, status="failed"):
            return {"type": "waiver", "status": status, "week": 2, "roster_id": 1,
                    "adds": {"p": 1}, "bid": bid, "created": created}
        state = SimpleNamespace(transactions=[tx(10, "a"), tx(20, "b"), tx(30, "c", "pending")])
        self.assertEqual([t["bid"] for t in submitted_bids(state)], [20])

    def test_paired_interpolation_keeps_record_identity(self):
        self.assertEqual(_record_values([(0, [0., 1.]), (100, [1., 0.])], 25), [0.25, 0.75])

    def test_policy_selection_cannot_see_individual_futures(self):
        runs = [{"p_title": .5, "title_by_record": [1., 0.]},
                {"p_title": .5, "title_by_record": [0., 1.]},
                {"p_title": .6, "title_by_record": [.6, .6]}]
        with patch("ranker.claims.run_replays", return_value=runs):
            result = best_replays(self.inputs(), [{}, {}], [(tuple(self.roster), 1000)])
        self.assertEqual(result[0]["title_by_record"], [.6, .6])

    def test_holdout_does_not_learn_its_own_player(self):
        state = SimpleNamespace(teams=[SimpleNamespace(roster_id=1)])
        obs = [{"team": 1, "player": 9, "week": 2, "bid": 999, "reference": 1.}]
        self.assertEqual(fit_managers(state, obs, exclude_player=9)[0].log_scale, 0)

    def test_processing_date_overrides_submission_leg(self):
        fetch = runpy.run_path(str(Path(__file__).resolve().parents[1] / "season/fetch_league.py"))
        week = fetch["processing_week"](dt.datetime.fromisoformat("2026-09-16T07:05:00+00:00"), dt.date(2026, 9, 9))
        self.assertEqual(week, 2)

    def test_final_week_add_cannot_improve_previous_week(self):
        inputs = self.inputs()
        auctions = [None] * WEEKS
        auctions[-1] = ([9], [-1])
        expected = lineup_points(self.roster, self.points, self.positions, 15)
        expected += lineup_points(self.roster + [9], self.points, self.positions, 16)
        record = {"seed": 1, "my_bias": 0., "bars": [0.] * WEEKS, "forecast_bars": [0.] * WEEKS,
                  "alive": [2] * WEEKS, "champ_bar": expected, "auctions": auctions}
        result = replay([record], inputs, self.roster, 1000, "value")
        self.assertAlmostEqual(result["p_title"], 0.5)

    def test_pending_claim_prices_only_legal_budget_range(self):
        inputs = self.inputs()
        for points in inputs.weekly:
            points[9] = 25.
        inputs.free_agents = [9]
        state = SimpleNamespace(me=0, my_team=SimpleNamespace(roster=self.roster, faab_left=1000, reserve=[]),
                                players=[SimpleNamespace(sleeper_id=str(j), name=str(j), position="WR",
                                                         team="T", injury_status=None, source="test") for j in range(10)])
        records = []
        for price in (100, 200):
            auctions = [None] * WEEKS
            auctions[1] = ([9], [price])
            records.append({"auctions": auctions, "forecast_bars": [0.] * WEEKS})
        candidate_budgets = []

        def values(inputs, records, variants):
            out = []
            for roster, budget, policy in variants:
                if 9 in roster:
                    candidate_budgets.append(budget)
                title = budget * 0.00001 + (0.004 if 9 in roster else 0.)
                out.append({"p_title": title, "title_by_record": [title, title], "p_reach_final": title,
                            "p_cut_now": 0., "p_alive_by_week": [1.] * 14, "budget_by_week": [budget] * 16,
                            "budget_after_claims": [budget] * 16})
            return out

        with patch("ranker.claims.run_replays", side_effect=values):
            result = claims(state, inputs, records, 0.)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["optimal_bid"], 201)
        self.assertLessEqual(candidate["optimal_bid"], candidate["bid_ceiling"])
        self.assertNotIn(0, candidate_budgets)
        self.assertEqual(min(candidate_budgets), 600, "Keep the lower interpolation bracket")

    def test_reporting_excludes_eliminated_teams_and_uses_opening_cash(self):
        from season import league_odds, market

        teams = [SimpleNamespace(roster_id=j + 1, name=str(j), username=None, is_mine=j == 0,
                                 alive=j < 2, faab_left=cash, points_for=0, points_this_week=0,
                                 roster=[], unknown=[])
                 for j, cash in enumerate((400, 200, 1000))]
        state = SimpleNamespace(teams=teams, players=[], transactions=[])
        inputs = SimpleNamespace(week0=8, budgets=[400, 200, 1000], positions=[], weekly=[[]] * WEEKS,
                                 managers=[Manager()] * 3, market_fit={})
        opening = [[400, 200, 1000]] + [[300 - k * 10, 150 - k * 5, 1000] for k in range(8)]
        after = opening[1:] + [[220, 110, 1000]]
        record = {"cut_week": [None, 12, None], "budget_path": opening, "budget_after_claims": after,
                  "champion": 0, "claims": [[] for _ in range(WEEKS)], "bars": [0.] * WEEKS, "champ_bar": 0.}
        rows = league_odds(state, inputs, [record])
        self.assertEqual(rows[0]["budget_entering_week9"], 400)
        self.assertEqual(rows[0]["spend_now"], 100)
        self.assertEqual(rows[2]["p_reach_final"], 0)
        self.assertTrue(all(p == 0 for p in rows[2]["p_alive_by_week"]))
        self.assertIsNone(rows[2]["budget_entering_week13"])
        budgets = market(state, inputs, [record], [record])["mean_budget_by_week"]
        self.assertEqual(budgets[0]["budget"], 300)
        self.assertEqual(budgets[4]["budget"], round((270 + 135) / 2))
        self.assertEqual(budgets[5]["budget"], 260, "The team cut in week 13 is absent in week 14")

    def test_opening_cash_restores_only_completed_current_week_bids(self):
        players = [SimpleNamespace(index=j, sleeper_id=str(j), name=str(j), ir_until=0,
                                   position=("QB", "RB", "WR", "TE")[pos], weekly=(points,) * WEEKS)
                   for j, (pos, points) in enumerate(zip(self.positions, self.points))]
        team = SimpleNamespace(roster_id=1, faab_left=850, roster=self.roster[:], reserve=[],
                               alive=True, is_mine=True)
        def tx(week, bid, status, created):
            return {"week": week, "bid": bid, "status": status, "created": created,
                    "type": "waiver", "roster_id": 1, "adds": {"9": 1}, "drops": {}}
        state = SimpleNamespace(week=2, players=players, teams=[team], me=0, free_agents=[8, 9],
                                 waivers_ran=True, transactions=[tx(1, 50, "complete", "a"),
                                                               tx(2, 100, "complete", "b"),
                                                               tx(2, 300, "failed", "c")])
        inputs = race_inputs(state)
        self.assertEqual(inputs.budgets, [850])
        self.assertEqual(inputs.opening_budgets, [950])

    def test_replay_budget_is_before_claims(self):
        inputs = self.inputs()
        inputs.week0 = 15
        for points in inputs.weekly:
            points[9] = 25.
        auctions = [None] * WEEKS
        auctions[16] = ([9], [1])
        record = {"seed": 1, "my_bias": 0., "bars": [0.] * WEEKS, "forecast_bars": [0.] * WEEKS,
                  "alive": [2] * WEEKS, "champ_bar": 0., "auctions": auctions}
        result = replay([record], inputs, self.roster, 317, "patient")
        self.assertEqual(result["budget_by_week"], [1000, 317])
        self.assertEqual(result["budget_after_claims"], [317, 0])

    def test_replay_budget_conditions_on_survival(self):
        inputs = self.inputs()
        inputs.week0 = 14
        for points in inputs.weekly:
            points[9] = 100.
        records = []
        for survives in (False, True):
            auctions = [None] * WEEKS
            if survives:
                auctions[15] = ([9], [1])
            records.append({"seed": 1, "my_bias": 0., "bars": [-1000. if survives else 1000.] * WEEKS,
                            "forecast_bars": [0.] * WEEKS, "alive": [2] * WEEKS,
                            "champ_bar": 0., "auctions": auctions})
        result = replay(records, inputs, self.roster, 1000, "value")
        self.assertEqual(result["p_reach_final"], .5)
        self.assertEqual(result["budget_by_week"][-1], 0,
                         "Cash retained in a season where we were cut is not a survivor's budget")


if __name__ == "__main__":
    unittest.main()
