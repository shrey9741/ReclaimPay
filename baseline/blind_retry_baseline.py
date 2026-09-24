"""
Baseline: the "dumb" system most merchants actually run today -- just
retry the same gateway/method blindly, regardless of why it failed.

This evaluates that policy against the SAME ground-truth ­ ­model the
synthetic data was sampled from, so it's an apples-to-apples number the
real optimizer must beat later.

Run: python3 blind_retry_baseline.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from ground_truth import SUCCESS_PROB, ACTION_COST  # noqa: E402

import csv
import random

random.seed(7)

TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions_test.csv")

BLIND_ACTION = "RETRY_SAME"  # what a naive system does for every failure


def evaluate_policy(policy_fn, test_path=TEST_PATH):
    """
    policy_fn(row) -> action string
    Re-samples a fresh outcome from ground truth for the chosen action
    (not the outcome logged in the CSV, since that was for a different,
    randomly-exploring action).
    """
    total = 0
    recovered = 0
    total_cost = 0.0
    revenue_recovered = 0.0
    revenue_at_stake = 0.0

    with open(test_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            amount = float(row["amount_inr"])
            revenue_at_stake += amount

            action = policy_fn(row)
            p_success = SUCCESS_PROB[row["decline_reason"]][action]
            success = 1 if random.random() < p_success else 0
            recovered += success
            total_cost += ACTION_COST[action]
            if success:
                revenue_recovered += amount

    recovery_rate = recovered / total
    return {
        "total_failed_transactions": total,
        "revenue_at_stake_inr": round(revenue_at_stake, 2),
        "recovered": recovered,
        "recovery_rate": round(recovery_rate, 4),
        "revenue_recovered_inr": round(revenue_recovered, 2),
        "total_action_cost_units": round(total_cost, 1),
        "avg_cost_per_recovered": round(total_cost / recovered, 3) if recovered else None,
    }


def blind_policy(row):
    return BLIND_ACTION


def oracle_policy(row):
    """Cheats by knowing the ground truth -- picks the action with the
    highest P(success) for the true reason. This is the ceiling: no agent
    built on this ground truth can beat it. Useful to report alongside the
    baseline so recovery-rate uplift has a meaningful frame of reference."""
    reason = row["decline_reason"]
    return max(SUCCESS_PROB[reason], key=SUCCESS_PROB[reason].get)


if __name__ == "__main__":
    blind_results = evaluate_policy(blind_policy)
    print("=== Blind Retry Baseline (RETRY_SAME for everything) ===")
    for k, v in blind_results.items():
        print(f"{k}: {v}")

    oracle_results = evaluate_policy(oracle_policy)
    print("\n=== Oracle Ceiling (always picks the true-best action) ===")
    for k, v in oracle_results.items():
        print(f"{k}: {v}")

    uplift = oracle_results["recovery_rate"] - blind_results["recovery_rate"]
    print(f"\nMax possible recovery-rate uplift for your agent to chase: +{uplift:.1%}")
