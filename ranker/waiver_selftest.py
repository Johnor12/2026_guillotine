"""Small economic and auction regressions, independent of live league files."""
import math
import random
import datetime as dt
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .claims import _record_values, best_replays, claims
from .league import WEEKS
from .race import RaceInputs, _auction, claim_plan, race_inputs, replay
from .season import lineup_points
from .waivers import Bidding, Manager, fit_managers, spending_allowance, submitted_bids


class WaiverTests(unittest.TestCase):
    def setUp(self):
        # Full opening roster: one QB, one RB, three WRs, three TEs.
        self.positions = [0, 1, 2, 2, 2, 3, 3, 3, 1, 2]
        self.points = [20., 15., 12., 11., 5., 14., 10., 9., 8., 13.]
        self.bidding = Bidding(self.positions, [self.points[:] for _ in range(WEEKS)],
                               [self.points[:] for _ in range(WEEKS)])
        self.roster = list(range(8))

    def inputs(self):
        return RaceInputs(1, self.positions, self.bidding.weekly, self.bidding.ros,
                          [self.roster[:]], [1000], [True], [0], 0, [8, 9], False,
                          self.bidding, [Manager()], {}, [1000])

    def test_buy_before_expansion(self):
        early = self.bidding.offers(self.roster, [8], 1000, 1, 0)
        preparing = self.bidding.offers(self.roster, [8], 1000, 6, 0)
        self.assertTrue(early, "Value after a distant expansion must count from week 2")
        self.assertTrue(preparing, "A cheap RB should count before the second RB slot opens")
        self.assertGreater(preparing[0].ceiling, 0)

    def test_superflex_coverage_beyond_four_weeks(self):
        self.positions[8] = 0
        for points in self.bidding.weekly:
            points[8] = 18.
        offer = self.bidding.offers(self.roster, [8], 1000, 1, 0)[0]
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
        offer = self.bidding.offers(self.roster, [8], 1000, 1, 0)[0]
        self.assertGreater(offer.ceiling, 0)
        self.assertNotIn(offer.drop, (0, 1, 2, 3), "Keep scarce starters instead of dropping by raw points")

    def test_owned_players_do_not_crowd_out_bargains(self):
        inputs = self.inputs()
        plan, allowance = claim_plan(inputs, self.roster, list(range(10)), 1000, 1, 0, 0, random.Random(1))
        self.assertTrue(any(o.player == 9 for _, o in plan))
        self.assertTrue(all(o.player not in self.roster for _, o in plan))
        self.assertEqual(allowance, max(b for b, _ in plan))

    def test_final_dollars_have_no_salvage_value(self):
        offer = self.bidding.offers(self.roster, [9], 317, WEEKS - 1, 0)[0]
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
        state = SimpleNamespace(me=0, my_team=SimpleNamespace(roster=self.roster, faab_left=1000),
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
        players = [SimpleNamespace(index=j, sleeper_id=str(j), name=str(j),
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
