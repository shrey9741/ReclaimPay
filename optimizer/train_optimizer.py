"""
Cost-Aware Optimizer Agent

This is the core decision-maker. It does NOT just predict "will this
succeed" for the action that was actually taken -- it needs to answer
"what's P(success) for EACH possible action on THIS transaction", so it
can pick the one that maximizes expected value:

    expected_value(action) = P(success | context, action) * amount_inr
                              - cost_units(action) * COST_UNIT_INR

This is a genuinely different problem from a normal classifier: at
inference time we don't know which action will be taken (that's what
we're deciding), so for every failed transaction we score ALL 5
candidate actions by substituting each one into the feature vector,
then pick the argmax. This is a simple offline-bandit / counterfactual
policy -- the honest, buildable version of "contextual bandit" without
needing a live RL loop for a portfolio project.

Model: HistGradientBoostingClassifier (handles categorical-ish features
+ interactions between reason and action better than plain logistic
regression, and is well-calibrated out of the box).
"""

import os
import sys
import random
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from ground_truth import RECOVERY_ACTIONS, SUCCESS_PROB, ACTION_COST  # noqa: E402

random.seed(11)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "optimizer_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "optimizer_encoder.joblib")

CATEGORICAL_COLS = ["decline_reason", "payment_method", "gateway", "bank", "action_taken"]
NUMERIC_COLS = ["hour_of_day", "day_of_week", "retry_attempt_number"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# What one abstract "cost unit" (from ground_truth.ACTION_COST) is worth in
# INR -- an explicit, documented assumption so the expected-value objective
# is honest about what it's trading off, not a magic number.
COST_UNIT_INR = 40.0


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "transactions_test.csv"))
    return train, test


def build_encoder(df):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(df[CATEGORICAL_COLS])
    return enc


def encode(df, enc):
    cat = enc.transform(df[CATEGORICAL_COLS])
    num = df[NUMERIC_COLS].values
    import numpy as np
    return np.hstack([cat, num])


def train_optimizer():
    train, test = load_data()
    enc = build_encoder(pd.concat([train, test])[CATEGORICAL_COLS])

    X_train = encode(train, enc)
    y_train = train["outcome_success"].values
    X_test = encode(test, enc)
    y_test = test["outcome_success"].values

    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"Optimizer success-probability model -- test AUC: {auc:.4f}, Brier score: {brier:.4f}")
    print("(Brier score closer to 0 = well-calibrated probabilities, which matters")
    print(" here since we act on the probability value, not just a class label.)\n")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(enc, ENCODER_PATH)
    return model, enc


class CostAwareOptimizer:
    """Wraps the trained model: given a failed transaction's context,
    scores every candidate action and returns the one with highest
    expected value."""

    def __init__(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
        self.model = joblib.load(model_path)
        self.enc = joblib.load(encoder_path)

    def score_actions(self, row: dict):
        import numpy as np
        rows = []
        for action in RECOVERY_ACTIONS:
            r = dict(row)
            r["action_taken"] = action
            rows.append(r)
        df = pd.DataFrame(rows)
        X = encode(df, self.enc)
        proba = self.model.predict_proba(X)[:, 1]

        results = []
        for action, p in zip(RECOVERY_ACTIONS, proba):
            cost_inr = ACTION_COST[action] * COST_UNIT_INR
            ev = float(p) * float(row["amount_inr"]) - cost_inr
            results.append({"action": action, "p_success": round(float(p), 4),
                             "cost_inr": float(cost_inr), "expected_value_inr": round(float(ev), 2)})
        return sorted(results, key=lambda x: -x["expected_value_inr"])

    def choose_action(self, row: dict):
        return self.score_actions(row)[0]["action"]


if __name__ == "__main__":
    train_optimizer()

    optimizer = CostAwareOptimizer()
    sample_row = {
        "decline_reason": "BANK_SERVER_ERROR",
        "payment_method": "CARD",
        "gateway": "GatewayA",
        "bank": "BankX",
        "hour_of_day": 14,
        "day_of_week": 2,
        "retry_attempt_number": 1,
        "amount_inr": 5000.0,
    }
    print("--- Sample decision for a BANK_SERVER_ERROR failure, Rs.5000 ---")
    for r in optimizer.score_actions(sample_row):
        print(r)
    print(f"\nChosen action: {optimizer.choose_action(sample_row)}")
