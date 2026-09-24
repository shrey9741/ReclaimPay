"""
Generates a synthetic dataset of failed payment transactions with logged
recovery-action outcomes.

Why "logged bandit feedback" and not just random labels:
To later train a success-probability model (the Cost-Aware Optimizer agent),
we need examples of *which action was taken* and *what happened* -- not just
the best action. So this script simulates an exploration policy (mostly
epsilon-greedy over a noisy prior) choosing an action for each failed
transaction, then samples the real outcome from the ground-truth model.
This mirrors how you'd actually bootstrap an offline dataset in production
before you have a live policy to learn from.

Output: data/transactions.csv, data/transactions_test.csv (held-out)
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

from ground_truth import (
    DECLINE_REASONS,
    RECOVERY_ACTIONS,
    SUCCESS_PROB,
    REASON_BASE_RATE,
    GATEWAYS,
    METHODS,
    BANKS,
)

random.seed(42)

N_TRAIN = 20000
N_TEST = 5000

FIELDNAMES = [
    "transaction_id",
    "merchant_id",
    "amount_inr",
    "payment_method",
    "gateway",
    "bank",
    "decline_reason",
    "decline_message",
    "hour_of_day",
    "day_of_week",
    "retry_attempt_number",
    "action_taken",
    "outcome_success",
]

# A handful of raw bank-style decline messages per reason, so the
# Classifier Agent has to do real text -> category mapping, not a lookup.
DECLINE_MESSAGES = {
    "OTP_TIMEOUT": [
        "OTP validation timed out",
        "Txn declined: OTP not entered within time limit",
        "3DS authentication timeout",
        "One time password expired before submission",
        "Customer did not complete 2FA in time",
        "Authentication step incomplete - session expired",
    ],
    "INSUFFICIENT_FUNDS": [
        "Insufficient balance in account",
        "Txn declined by issuer: low funds",
        "Available limit exceeded",
        "Not enough balance to complete transaction",
        "Account balance below transaction amount",
        "Declined - insufficient credit limit available",
    ],
    "BANK_SERVER_ERROR": [
        "Issuing bank server not responding",
        "Bank system temporarily unavailable",
        "Issuer timeout - please retry",
        "Unable to reach issuing bank",
        "Bank system error, transaction not processed",
        "Issuer connection failure",
    ],
    "GATEWAY_TIMEOUT": [
        "Gateway request timed out",
        "No response from payment processor",
        "Upstream timeout at gateway",
        "Processor did not respond in time",
        "Gateway connection timed out",
        "Payment service unavailable, try again",
    ],
    "RISK_BLOCK": [
        "Transaction blocked by risk engine",
        "Declined: suspected fraud",
        "Blocked due to unusual transaction pattern",
        "Transaction flagged for manual review",
        "Declined by fraud prevention system",
        "Unusual activity detected, transaction held",
    ],
    "CARD_EXPIRED": [
        "Card has expired",
        "Invalid expiry date",
        "Expired card - transaction declined",
        "Card validity has lapsed",
        "Expiry date does not match records",
    ],
    "NETWORK_ERROR": [
        "Network error during transaction",
        "Connection dropped mid-transaction",
        "Request failed: network issue",
        "Transaction interrupted due to connectivity issue",
        "Timed out due to poor network",
    ],
}

MERCHANT_IDS = [f"M{100+i}" for i in range(15)]

# Generic, ambiguous messages that genuinely could belong to more than one
# category -- real bank/gateway systems produce these often. These are
# intentionally hard: the classifier SHOULD have lower confidence on them,
# which is exactly what should trigger LLM fallback in production.
AMBIGUOUS_MESSAGES = [
    "Transaction declined",
    "Transaction could not be completed",
    "Payment failed, please try again",
    "Declined by bank",
    "Unable to process transaction at this time",
]

def add_typo_noise(text, p=0.12):
    """Randomly drop/duplicate a character to mimic real inconsistent
    logging (truncated logs, encoding glitches, etc.)."""
    if random.random() > p or len(text) < 5:
        return text
    idx = random.randint(1, len(text) - 2)
    action = random.choice(["drop", "dup", "lower"])
    if action == "drop":
        return text[:idx] + text[idx + 1:]
    elif action == "dup":
        return text[:idx] + text[idx] + text[idx:]
    else:
        return text.lower()


def sample_decline_message(reason):
    # 15% of the time, use a generic ambiguous message shared across
    # categories -- forces the classifier to genuinely generalize and
    # gives it real low-confidence cases to route to LLM fallback.
    if random.random() < 0.15:
        msg = random.choice(AMBIGUOUS_MESSAGES)
    else:
        msg = random.choice(DECLINE_MESSAGES[reason])
    return add_typo_noise(msg)


def sample_decline_reason():
    reasons, weights = zip(*REASON_BASE_RATE.items())
    return random.choices(reasons, weights=weights, k=1)[0]


def exploration_policy(reason):
    """
    Epsilon-greedy-ish logging policy: mostly picks a 'reasonable-looking'
    action for the reason (imitating an imperfect existing system) but
    explores other actions often enough to give the optimizer full coverage.
    This is intentionally NOT the optimal policy -- it's what you'd expect
    a naive existing retry system to do, which is exactly the baseline
    the trained agent needs to beat.
    """
    naive_default = {
        "OTP_TIMEOUT": "RETRY_SAME",
        "INSUFFICIENT_FUNDS": "RETRY_SAME",
        "BANK_SERVER_ERROR": "RETRY_SAME",
        "GATEWAY_TIMEOUT": "RETRY_SAME",
        "RISK_BLOCK": "RETRY_SAME",
        "CARD_EXPIRED": "RETRY_SAME",
        "NETWORK_ERROR": "RETRY_SAME",
    }
    epsilon = 0.5  # high exploration so every action gets enough data
    if random.random() < epsilon:
        return random.choice(RECOVERY_ACTIONS)
    return naive_default[reason]


def sample_outcome(reason, action):
    p = SUCCESS_PROB[reason][action]
    return 1 if random.random() < p else 0


def generate_row():
    reason = sample_decline_reason()
    action = exploration_policy(reason)
    outcome = sample_outcome(reason, action)

    return {
        "transaction_id": str(uuid.uuid4()),
        "merchant_id": random.choice(MERCHANT_IDS),
        "amount_inr": round(random.uniform(50, 25000), 2),
        "payment_method": random.choice(METHODS),
        "gateway": random.choice(GATEWAYS),
        "bank": random.choice(BANKS),
        "decline_reason": reason,
        "decline_message": sample_decline_message(reason),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "retry_attempt_number": random.choices([1, 2, 3], weights=[0.7, 0.22, 0.08])[0],
        "action_taken": action,
        "outcome_success": outcome,
    }


def write_csv(path, n_rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for _ in range(n_rows):
            writer.writerow(generate_row())


if __name__ == "__main__":
    write_csv("transactions_train.csv", N_TRAIN)
    write_csv("transactions_test.csv", N_TEST)
    print(f"Wrote {N_TRAIN} training rows -> transactions_train.csv")
    print(f"Wrote {N_TEST} test rows -> transactions_test.csv")
