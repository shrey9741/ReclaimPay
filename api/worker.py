"""
Worker: consumes retry jobs from the Redis Stream (via consumer group,
so you can run several of these in parallel for horizontal scaling)
and executes them.

NOTE on "executing" a retry: this is a portfolio project without a real
payment gateway to call, so execution is simulated by sampling from the
same ground-truth probability model the synthetic training data came
from (data/ground_truth.py). In a real deployment, this is the exact
spot where you'd call the actual gateway/bank retry API and use its
real response -- everything else in this file (state transitions,
re-deciding the next action, idempotent ack) stays the same.
"""

import sys
import os
import json
import time
import random
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "optimizer"))

from db import SessionLocal, Transaction, RetryAttempt, TxnStatus  # noqa: E402
from redis_queue import read_pending, ack, schedule_retry  # noqa: E402
from ground_truth import SUCCESS_PROB  # noqa: E402
from train_optimizer import CostAwareOptimizer  # noqa: E402

optimizer = CostAwareOptimizer()


def simulate_gateway_retry(decline_reason: str, action: str) -> bool:
    """Stand-in for a real gateway API call -- see module docstring."""
    p = SUCCESS_PROB[decline_reason][action]
    return random.random() < p


def process_message(db, transaction_id: int, action: str, attempt_number: int):
    txn = db.query(Transaction).filter_by(id=transaction_id).first()
    if txn is None:
        return  # shouldn't happen, but don't crash the worker on bad data

    attempt = (
        db.query(RetryAttempt)
        .filter_by(transaction_id=transaction_id, attempt_number=attempt_number)
        .first()
    )

    success = simulate_gateway_retry(txn.decline_reason, action)
    now = datetime.utcnow()
    if attempt:
        attempt.executed_at = now
        attempt.outcome_success = int(success)

    if success:
        txn.status = TxnStatus.RECOVERED
        db.commit()
        print(f"[worker] txn={transaction_id} attempt={attempt_number} action={action} -> RECOVERED")
        return

    if attempt_number >= txn.max_retry_attempts:
        txn.status = TxnStatus.EXHAUSTED
        db.commit()
        print(f"[worker] txn={transaction_id} attempt={attempt_number} action={action} -> EXHAUSTED")
        return

    # Failed but attempts remain -- re-decide the next action. This is NOT
    # just "retry the same action again": the optimizer gets an updated
    # retry_attempt_number as context, so it can (and does, if the training
    # data reflects it) behave differently on a 2nd/3rd attempt.
    next_attempt_number = attempt_number + 1
    context = {
        "decline_reason": txn.decline_reason,
        "payment_method": txn.payment_method,
        "gateway": txn.gateway,
        "bank": txn.bank,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "retry_attempt_number": next_attempt_number,
        "amount_inr": txn.amount_inr,
    }
    decision = optimizer.score_actions(context)[0]

    if decision["action"] == "ABANDON":
        txn.status = TxnStatus.ABANDONED
        db.commit()
        print(f"[worker] txn={transaction_id} attempt={attempt_number} action={action} -> failed, then ABANDONED")
        return

    txn.retry_attempt_number = next_attempt_number
    txn.status = TxnStatus.RETRY_SCHEDULED
    new_attempt = RetryAttempt(
        transaction_id=txn.id, attempt_number=next_attempt_number, action=decision["action"],
        p_success_predicted=decision["p_success"], expected_value_inr=decision["expected_value_inr"],
        scheduled_at=now,
    )
    db.add(new_attempt)
    db.commit()
    schedule_retry(txn.id, decision["action"], attempt_number=next_attempt_number)
    print(f"[worker] txn={transaction_id} attempt={attempt_number} action={action} -> failed, "
          f"scheduled attempt {next_attempt_number} action={decision['action']}")


def run_worker(consumer_name="worker-1", max_iterations=None):
    db = SessionLocal()
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            messages = read_pending(consumer_name, count=10, block_ms=2000)
            for message_id, fields in messages:
                data = json.loads(fields["data"])
                process_message(db, data["transaction_id"], data["action"], data["attempt_number"])
                ack(message_id)
            iterations += 1
            if not messages:
                time.sleep(1)
    finally:
        db.close()


if __name__ == "__main__":
    run_worker()
