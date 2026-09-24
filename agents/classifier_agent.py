"""
Classifier Agent

Takes the raw decline_message text (what a bank/gateway actually sends,
often inconsistent phrasing) and predicts the underlying decline_reason
category. In production this is the first agent in the pipeline -- the
orchestrator hands it a failed transaction's log/message, and it outputs
a clean category the rest of the system can reason over.

Model: TF-IDF (char + word n-grams, since bank messages are short and
inconsistent) + Logistic Regression. Deliberately simple and fast --
this needs to run inline on every failed transaction, so we're not
reaching for an LLM call here (that's reserved for genuinely ambiguous/
novel messages the model has low confidence on -- see `agents/classifier_agent.py`
in Phase 3 for the LLM-fallback wrapper).
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_model.joblib")


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "transactions_train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "transactions_test.csv"))
    return train, test


def train_classifier():
    train, test = load_data()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])

    pipeline.fit(train["decline_message"], train["decline_reason"])

    preds = pipeline.predict(test["decline_message"])
    acc = accuracy_score(test["decline_reason"], preds)

    print(f"Classifier Agent test accuracy: {acc:.4f}\n")
    print(classification_report(test["decline_reason"], preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return pipeline, acc


class ClassifierAgent:
    """Wraps the trained model for use by the orchestrator."""

    def __init__(self, model_path=MODEL_PATH, confidence_threshold=0.55):
        self.pipeline = joblib.load(model_path)
        self.confidence_threshold = confidence_threshold

    def classify(self, decline_message: str):
        proba = self.pipeline.predict_proba([decline_message])[0]
        classes = self.pipeline.classes_
        best_idx = proba.argmax()
        category = classes[best_idx]
        confidence = proba[best_idx]

        return {
            "category": category,
            "confidence": round(float(confidence), 4),
            "needs_llm_fallback": confidence < self.confidence_threshold,
        }


if __name__ == "__main__":
    train_classifier()

    agent = ClassifierAgent()
    samples = [
        "OTP validation timed out",
        "Issuer timeout - please retry",
        "some completely novel bank error we never trained on",
    ]
    print("\n--- Sample classifications ---")
    for msg in samples:
        result = agent.classify(msg)
        print(f"{msg!r} -> {result}")
