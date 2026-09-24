"""
Evaluates the Cost-Aware Optimizer's policy on the held-out test set,
using the same methodology as baseline/blind_retry_baseline.py so the
three numbers are directly comparable:
  - Blind retry baseline
  - Cost-Aware Optimizer (this script)
  - Oracle ceiling
"""

import os
import sys
import random
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "baseline"))
from ground_truth import SUCCESS_PROB, ACTION_COST  # noqa: E402
from blind_retry_baseline import evaluate_policy, blind_policy, oracle_policy  # noqa: E402
from train_optimizer import CostAwareOptimizer  # noqa: E402

random.seed(7)

TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions_test.csv")


def make_optimizer_policy(optimizer: CostAwareOptimizer):
    def policy(row):
        context = {
            "decline_reason": row["decline_reason"],
            "payment_method": row["payment_method"],
            "gateway": row["gateway"],
            "bank": row["bank"],
            "hour_of_day": int(row["hour_of_day"]),
            "day_of_week": int(row["day_of_week"]),
            "retry_attempt_number": int(row["retry_attempt_number"]),
            "amount_inr": float(row["amount_inr"]),
        }
        return optimizer.choose_action(context)
    return policy


if __name__ == "__main__":
    optimizer = CostAwareOptimizer()
    optimizer_policy = make_optimizer_policy(optimizer)

    blind_results = evaluate_policy(blind_policy)
    optimizer_results = evaluate_policy(optimizer_policy)
    oracle_results = evaluate_policy(oracle_policy)

    print(f"{'Policy':<22} {'Recovery Rate':<15} {'Revenue Recovered (INR)':<26} {'Avg Cost/Recovery'}")
    for name, r in [("Blind Retry", blind_results),
                     ("Cost-Aware Optimizer", optimizer_results),
                     ("Oracle Ceiling", oracle_results)]:
        print(f"{name:<22} {r['recovery_rate']*100:>6.2f}%       "
              f"Rs.{r['revenue_recovered_inr']:>15,.0f}          "
              f"{r['avg_cost_per_recovered']}")

    uplift = optimizer_results["recovery_rate"] - blind_results["recovery_rate"]
    ceiling_captured = uplift / (oracle_results["recovery_rate"] - blind_results["recovery_rate"])
    extra_revenue = optimizer_results["revenue_recovered_inr"] - blind_results["revenue_recovered_inr"]

    print(f"\nRecovery-rate uplift over blind retry: +{uplift*100:.2f} points")
    print(f"Extra revenue recovered vs blind retry: Rs.{extra_revenue:,.0f} (on this 5000-transaction test set)")
    print(f"% of the theoretical max uplift captured: {ceiling_captured*100:.1f}%")
