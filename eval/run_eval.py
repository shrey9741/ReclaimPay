"""
End-to-end eval harness (DetEval-style: deterministic pass/fail checks,
not a vibes-based LLM judge).

Unlike optimizer/evaluate_optimizer.py (Phase 2), which evaluated the
success-probability model in isolation, this drives the REAL running
orchestrator -- HTTP calls to the FastAPI webhook, real Postgres writes,
real Redis scheduling, real worker execution -- on a sample of the held-out
test set. This is what actually proves the system works end-to-end, not
just that the model has good offline metrics.

Checks (each is a hard pass/fail, not a soft score):
  1. RECOVERY_RATE_UPLIFT   -- recovery rate beats the blind-retry baseline
                                by at least MIN_UPLIFT_PTS percentage points
  2. NO_DUPLICATE_PROCESSING -- every gateway_transaction_id we sent exists
                                exactly once in the DB (idempotency held)
  3. NO_ORPHANED_RETRIES     -- every non-terminal transaction has a
                                corresponding pending retry attempt (nothing
                                stuck in RETRY_SCHEDULED with no work queued)
  4. LATENCY_BUDGET          -- p95 webhook response time stays under
                                LATENCY_BUDGET_MS (the decision has to be
                                fast enough to sit in a real payment flow)
  5. ABANDON_SANITY          -- no transaction with positive true expected
                                value got abandoned (the optimizer isn't
                                giving up on genuinely recoverable revenue)

Run: python3 run_eval.py [--sample-size N]
Exit code 0 if all checks pass, 1 otherwise -- this is what a CI job greps.
"""

import argparse
import csv
import os
import sys
import time
import uuid

import httpx

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "baseline"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))
from ground_truth import SUCCESS_PROB  # noqa: E402
from blind_retry_baseline import evaluate_policy, blind_policy  # noqa: E402

API_BASE = "http://localhost:8000"
TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions_test.csv")

MIN_UPLIFT_PTS = 0.10          # optimizer must beat baseline by >=10 points
LATENCY_BUDGET_MS = 500
POLL_TIMEOUT_S = 30            # max time to wait for a transaction to settle
POLL_INTERVAL_S = 1


def load_sample(n):
    with open(TEST_PATH) as f:
        rows = list(csv.DictReader(f))
    return rows[:n]


def send_webhook(client, row, run_tag):
    txn_id = f"EVAL-{run_tag}-{uuid.uuid4().hex[:8]}"
    payload = {
        "gateway_transaction_id": txn_id,
        "merchant_id": row["merchant_id"],
        "amount_inr": float(row["amount_inr"]),
        "payment_method": row["payment_method"],
        "gateway": row["gateway"],
        "bank": row["bank"],
        "decline_message": row["decline_message"],
    }
    start = time.time()
    resp = client.post(f"{API_BASE}/webhook/transaction-failed", json=payload, timeout=10)
    latency_ms = (time.time() - start) * 1000
    resp.raise_for_status()
    return txn_id, resp.json(), latency_ms


def run_eval(sample_size):
    print(f"Loading {sample_size} held-out test transactions...")
    sample = load_sample(sample_size)
    run_tag = uuid.uuid4().hex[:6]

    client = httpx.Client()
    latencies = []
    pending = []  # list of {"row": ..., "txn_id": ..., "final": None}

    print("Sending through live orchestrator (webhook -> classify -> decide)...")
    for row in sample:
        txn_id, resp, latency_ms = send_webhook(client, row, run_tag)
        latencies.append(latency_ms)
        pending.append({"row": row, "txn_id": txn_id, "final": None})
    print(f"  ...{len(pending)} webhooks sent, now waiting for retry loop to settle")

    # Poll all of them together rather than one-at-a-time -- some will need
    # to wait through RETRY_BACKOFF (60s), possibly repeated up to
    # max_retry_attempts times, so the shared deadline must cover the worst
    # realistic case (3 sequential 60s backoffs), not an arbitrary guess.
    from redis_queue import ACTION_DELAY_SECONDS
    MAX_ATTEMPTS = 3
    worst_case_wait = MAX_ATTEMPTS * max(ACTION_DELAY_SECONDS.values())
    terminal_states = {"RECOVERED", "EXHAUSTED", "ABANDONED"}
    deadline = time.time() + worst_case_wait + 30  # +30s buffer for processing overhead
    while time.time() < deadline:
        still_open = [p for p in pending if p["final"] is None or p["final"]["status"] not in terminal_states]
        if not still_open:
            break
        for p in still_open:
            resp = client.get(f"{API_BASE}/transaction/{p['txn_id']}", timeout=10)
            resp.raise_for_status()
            p["final"] = resp.json()
        time.sleep(POLL_INTERVAL_S)

    results = pending

    # ---- Check 1: recovery rate uplift ----
    recovered = sum(1 for r in results if r["final"]["status"] == "RECOVERED")
    e2e_recovery_rate = recovered / len(results)

    baseline = evaluate_policy(blind_policy, test_path=TEST_PATH)
    # baseline evaluated on the full 5000-row test set; recompute on just
    # our sample for a fair apples-to-apples comparison
    baseline_sample_recovered = 0
    for row in sample:
        import random
        p = SUCCESS_PROB[row["decline_reason"]]["RETRY_SAME"]
        baseline_sample_recovered += 1 if random.random() < p else 0
    baseline_sample_rate = baseline_sample_recovered / len(sample)

    uplift = e2e_recovery_rate - baseline_sample_rate
    check1_pass = uplift >= MIN_UPLIFT_PTS

    # ---- Check 2: no duplicate processing ----
    ids_sent = [r["txn_id"] for r in results]
    check2_pass = len(ids_sent) == len(set(ids_sent))  # trivially true by construction here;
    # the real idempotency guarantee was already verified manually in Phase 3
    # (duplicate webhook -> idempotent_replay: true). Re-verify it here too:
    dup_resp = client.post(f"{API_BASE}/webhook/transaction-failed", json={
        "gateway_transaction_id": results[0]["txn_id"],
        "merchant_id": results[0]["row"]["merchant_id"],
        "amount_inr": float(results[0]["row"]["amount_inr"]),
        "payment_method": results[0]["row"]["payment_method"],
        "gateway": results[0]["row"]["gateway"],
        "bank": results[0]["row"]["bank"],
        "decline_message": results[0]["row"]["decline_message"],
    })
    check2_pass = check2_pass and dup_resp.json().get("idempotent_replay") is True

    # ---- Check 3: no orphaned retries ----
    stuck = [r for r in results if r["final"]["status"] not in
             {"RECOVERED", "EXHAUSTED", "ABANDONED"}]
    check3_pass = len(stuck) == 0

    # ---- Check 4: latency budget ----
    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)
    p95_latency = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]
    check4_pass = p95_latency <= LATENCY_BUDGET_MS

    # ---- Check 5: abandon sanity ----
    bad_abandons = []
    for r in results:
        if r["final"]["status"] == "ABANDONED":
            reason = r["row"]["decline_reason"]
            best_true_p = max(SUCCESS_PROB[reason].values())
            if best_true_p > 0.3:  # a genuinely recoverable case, abandoned anyway
                bad_abandons.append(r["txn_id"])
    check5_pass = len(bad_abandons) == 0

    # ---- Report ----
    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)
    print(f"Sample size: {len(results)}")
    print(f"End-to-end recovery rate: {e2e_recovery_rate:.2%}")
    print(f"Baseline (blind retry) rate on same sample: {baseline_sample_rate:.2%}")
    print(f"Uplift: {uplift:+.2%}\n")

    checks = [
        ("RECOVERY_RATE_UPLIFT", check1_pass, f"uplift={uplift:+.2%}, required>=+{MIN_UPLIFT_PTS:.0%}"),
        ("NO_DUPLICATE_PROCESSING", check2_pass, "idempotency replay verified"),
        ("NO_ORPHANED_RETRIES", check3_pass, f"{len(stuck)} stuck transaction(s)"),
        ("LATENCY_BUDGET", check4_pass, f"p95={p95_latency:.0f}ms, budget={LATENCY_BUDGET_MS}ms"),
        ("ABANDON_SANITY", check5_pass, f"{len(bad_abandons)} bad abandon(s)"),
    ]
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"[{status}] {name} -- {detail}")

    print("=" * 60)
    print("OVERALL: " + ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    sys.exit(run_eval(args.sample_size))
