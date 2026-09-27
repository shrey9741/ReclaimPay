"""
FastAPI orchestrator.

Endpoints:
  POST /webhook/transaction-failed  -- gateway calls this when a payment fails
  GET  /transaction/{gateway_transaction_id} -- status + full retry history
  GET  /merchant/{merchant_id}/report -- aggregated recovery stats

Flow on webhook receipt:
  1. Idempotency check (DB unique constraint on gateway_transaction_id)
  2. Classifier Agent: raw decline message -> decline_reason category
  3. Cost-Aware Optimizer: decide the best recovery action
  4. Persist + schedule the retry (or mark ABANDONED if that's the
     optimizer's choice -- no point queuing a retry that shouldn't happen)
"""

import sys
import os
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "optimizer"))

from db import SessionLocal, Transaction, RetryAttempt, TxnStatus, init_db  # noqa: E402
from redis_queue import schedule_retry, ensure_consumer_group  # noqa: E402
from classifier_agent import ClassifierAgent  # noqa: E402
from train_optimizer import CostAwareOptimizer  # noqa: E402

app = FastAPI(title="ReclaimPay Orchestrator")

# React dev server (Vite default) needs CORS to call this API directly.
# Tighten allow_origins to your actual deployed frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = ClassifierAgent()
optimizer = CostAwareOptimizer()


class FailedTransactionIn(BaseModel):
    gateway_transaction_id: str
    merchant_id: str
    amount_inr: float
    payment_method: str
    gateway: str
    bank: str
    decline_message: str


@app.on_event("startup")
def startup():
    init_db()
    ensure_consumer_group()


def transaction_to_dict(txn: Transaction, attempts=None):
    return {
        "id": txn.id,
        "gateway_transaction_id": txn.gateway_transaction_id,
        "merchant_id": txn.merchant_id,
        "amount_inr": txn.amount_inr,
        "decline_reason": txn.decline_reason,
        "classifier_confidence": txn.classifier_confidence,
        "status": txn.status,
        "retry_attempt_number": txn.retry_attempt_number,
        "created_at": txn.created_at.isoformat() if txn.created_at else None,
        "updated_at": txn.updated_at.isoformat() if txn.updated_at else None,
        "attempts": [
            {
                "attempt_number": a.attempt_number,
                "action": a.action,
                "p_success_predicted": a.p_success_predicted,
                "expected_value_inr": a.expected_value_inr,
                "scheduled_at": a.scheduled_at.isoformat() if a.scheduled_at else None,
                "executed_at": a.executed_at.isoformat() if a.executed_at else None,
                "outcome_success": a.outcome_success,
            } for a in (attempts or [])
        ],
    }


@app.post("/webhook/transaction-failed")
def transaction_failed(payload: FailedTransactionIn):
    db = SessionLocal()
    try:
        existing = db.query(Transaction).filter_by(
            gateway_transaction_id=payload.gateway_transaction_id
        ).first()
        if existing:
            # Idempotency: same failure webhook delivered twice (real
            # gateways do this) -- return the existing record instead of
            # starting a second, duplicate retry flow.
            return {"idempotent_replay": True, **transaction_to_dict(existing, existing.attempts)}

        classification = classifier.classify(payload.decline_message)
        now = datetime.utcnow()

        context = {
            "decline_reason": classification["category"],
            "payment_method": payload.payment_method,
            "gateway": payload.gateway,
            "bank": payload.bank,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "retry_attempt_number": 1,
            "amount_inr": payload.amount_inr,
        }
        decision = optimizer.score_actions(context)[0]

        txn = Transaction(
            gateway_transaction_id=payload.gateway_transaction_id,
            merchant_id=payload.merchant_id,
            amount_inr=payload.amount_inr,
            payment_method=payload.payment_method,
            gateway=payload.gateway,
            bank=payload.bank,
            decline_message_raw=payload.decline_message,
            decline_reason=classification["category"],
            classifier_confidence=classification["confidence"],
        )

        if decision["action"] == "ABANDON":
            txn.status = TxnStatus.ABANDONED
            db.add(txn)
            db.flush()
            attempt = RetryAttempt(
                transaction_id=txn.id, attempt_number=0, action="ABANDON",
                p_success_predicted=decision["p_success"],
                expected_value_inr=decision["expected_value_inr"],
                scheduled_at=now, executed_at=now, outcome_success=0,
            )
            db.add(attempt)
        else:
            txn.status = TxnStatus.RETRY_SCHEDULED
            txn.retry_attempt_number = 1
            db.add(txn)
            db.flush()  # get txn.id before commit
            attempt = RetryAttempt(
                transaction_id=txn.id, attempt_number=1, action=decision["action"],
                p_success_predicted=decision["p_success"],
                expected_value_inr=decision["expected_value_inr"],
                scheduled_at=now,
            )
            db.add(attempt)
            db.commit()
            schedule_retry(txn.id, decision["action"], attempt_number=1)
            db.refresh(txn)
            return transaction_to_dict(txn, txn.attempts)

        db.commit()
        db.refresh(txn)
        return transaction_to_dict(txn, txn.attempts)

    except IntegrityError:
        db.rollback()
        existing = db.query(Transaction).filter_by(
            gateway_transaction_id=payload.gateway_transaction_id
        ).first()
        return {"idempotent_replay": True, **transaction_to_dict(existing, existing.attempts)}
    finally:
        db.close()


@app.get("/transaction/{gateway_transaction_id}")
def get_transaction(gateway_transaction_id: str):
    db = SessionLocal()
    try:
        txn = db.query(Transaction).filter_by(
            gateway_transaction_id=gateway_transaction_id
        ).first()
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction_to_dict(txn, txn.attempts)
    finally:
        db.close()


@app.get("/merchant/{merchant_id}/report")
def merchant_report(merchant_id: str):
    db = SessionLocal()
    try:
        txns = db.query(Transaction).filter_by(merchant_id=merchant_id).all()
        total = len(txns)
        recovered = sum(1 for t in txns if t.status == TxnStatus.RECOVERED)
        exhausted = sum(1 for t in txns if t.status == TxnStatus.EXHAUSTED)
        abandoned = sum(1 for t in txns if t.status == TxnStatus.ABANDONED)
        pending = sum(1 for t in txns if t.status == TxnStatus.RETRY_SCHEDULED)
        revenue_recovered = sum(t.amount_inr for t in txns if t.status == TxnStatus.RECOVERED)

        reason_breakdown = {}
        for t in txns:
            if t.status == TxnStatus.EXHAUSTED and t.decline_reason:
                reason_breakdown[t.decline_reason] = reason_breakdown.get(t.decline_reason, 0) + 1

        return {
            "merchant_id": merchant_id,
            "total_failed_transactions": total,
            "recovered": recovered,
            "exhausted": exhausted,
            "abandoned": abandoned,
            "pending_retry": pending,
            "recovery_rate": round(recovered / total, 4) if total else None,
            "revenue_recovered_inr": round(revenue_recovered, 2),
            "top_unrecovered_failure_reasons": reason_breakdown,
        }
    finally:
        db.close()


# Recovery rate this system replaced -- what a naive "just retry the same
# gateway" system achieves (measured in Phase 1 on the held-out test set).
# Used as the dashed reference line on the Recovery Trends chart.
BLIND_RETRY_BASELINE_RATE = 0.3128


@app.get("/transactions")
def list_transactions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str = Query(None, description="Filter by status, e.g. RECOVERED"),
    merchant_id: str = Query(None),
):
    """Powers the Live Feed screen -- most recent failed transactions first,
    with everything the table needs to render a row without a second call
    per row."""
    db = SessionLocal()
    try:
        q = db.query(Transaction)
        if status:
            q = q.filter(Transaction.status == status)
        if merchant_id:
            q = q.filter(Transaction.merchant_id == merchant_id)
        total = q.count()
        rows = q.order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "transactions": [
                {
                    "gateway_transaction_id": t.gateway_transaction_id,
                    "merchant_id": t.merchant_id,
                    "amount_inr": t.amount_inr,
                    "decline_reason": t.decline_reason,
                    "status": t.status,
                    "retry_attempt_number": t.retry_attempt_number,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                } for t in rows
            ],
        }
    finally:
        db.close()


@app.get("/transaction/{gateway_transaction_id}/trace")
def get_decision_trace(gateway_transaction_id: str):
    """Powers the Decision Trace panel: for the LATEST attempt on this
    transaction, shows every action the optimizer considered (not just the
    one it picked) so the UI can render the full comparison, plus the
    original classification and the full retry history."""
    db = SessionLocal()
    try:
        txn = db.query(Transaction).filter_by(
            gateway_transaction_id=gateway_transaction_id
        ).first()
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")

        # Re-score all actions for the current context so the UI can show
        # "here's what the optimizer considered" even for older transactions
        # (we don't persist every candidate action's score, only the chosen
        # one, to keep the retry_attempts table lean -- this recomputes it
        # on demand instead).
        now = datetime.utcnow()
        context = {
            "decline_reason": txn.decline_reason,
            "payment_method": txn.payment_method,
            "gateway": txn.gateway,
            "bank": txn.bank,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "retry_attempt_number": txn.retry_attempt_number or 1,
            "amount_inr": txn.amount_inr,
        }
        all_actions_considered = optimizer.score_actions(context)

        return {
            "gateway_transaction_id": txn.gateway_transaction_id,
            "decline_message_raw": txn.decline_message_raw,
            "classifier_category": txn.decline_reason,
            "classifier_confidence": txn.classifier_confidence,
            "actions_considered": all_actions_considered,
            "chosen_action": all_actions_considered[0]["action"] if all_actions_considered else None,
            "retry_history": [
                {
                    "attempt_number": a.attempt_number,
                    "action": a.action,
                    "p_success_predicted": a.p_success_predicted,
                    "expected_value_inr": a.expected_value_inr,
                    "scheduled_at": a.scheduled_at.isoformat() if a.scheduled_at else None,
                    "executed_at": a.executed_at.isoformat() if a.executed_at else None,
                    "outcome_success": a.outcome_success,
                } for a in txn.attempts
            ],
            "final_status": txn.status,
        }
    finally:
        db.close()


@app.get("/trends")
def recovery_trends(merchant_id: str = Query(None), days: int = Query(30, ge=1, le=90)):
    """Powers the Recovery Trends screen: daily recovery rate vs the blind-
    retry baseline, outcome breakdown by failure reason, and the headline
    revenue-recovered number."""
    db = SessionLocal()
    try:
        q = db.query(Transaction).filter(
            Transaction.created_at >= datetime.utcnow() - timedelta(days=days)
        )
        if merchant_id:
            q = q.filter(Transaction.merchant_id == merchant_id)
        txns = q.all()

        total = len(txns)
        recovered = sum(1 for t in txns if t.status == TxnStatus.RECOVERED)
        revenue_recovered = sum(t.amount_inr for t in txns if t.status == TxnStatus.RECOVERED)
        recovery_rate = round(recovered / total, 4) if total else None

        # Daily time series
        by_day = {}
        for t in txns:
            if t.status not in (TxnStatus.RECOVERED, TxnStatus.EXHAUSTED, TxnStatus.ABANDONED):
                continue  # only count settled transactions in the trend line
            day = t.created_at.date().isoformat()
            by_day.setdefault(day, {"total": 0, "recovered": 0})
            by_day[day]["total"] += 1
            if t.status == TxnStatus.RECOVERED:
                by_day[day]["recovered"] += 1

        timeseries = [
            {
                "date": day,
                "recovery_rate": round(vals["recovered"] / vals["total"], 4) if vals["total"] else 0,
                "recovered": vals["recovered"],
                "total": vals["total"],
                "baseline_recovery_rate": BLIND_RETRY_BASELINE_RATE,
            }
            for day, vals in sorted(by_day.items())
        ]

        # Outcome breakdown by failure reason
        by_reason = {}
        for t in txns:
            if not t.decline_reason or t.status not in (TxnStatus.RECOVERED, TxnStatus.EXHAUSTED):
                continue
            by_reason.setdefault(t.decline_reason, {"recovered": 0, "exhausted": 0})
            key = "recovered" if t.status == TxnStatus.RECOVERED else "exhausted"
            by_reason[t.decline_reason][key] += 1

        outcome_by_reason = [
            {"reason": reason, **counts} for reason, counts in by_reason.items()
        ]

        return {
            "period_days": days,
            "total_revenue_recovered_inr": round(revenue_recovered, 2),
            "recovery_rate": recovery_rate,
            "baseline_recovery_rate": BLIND_RETRY_BASELINE_RATE,
            "uplift_pts": round(recovery_rate - BLIND_RETRY_BASELINE_RATE, 4) if recovery_rate is not None else None,
            "timeseries": timeseries,
            "outcome_by_reason": outcome_by_reason,
        }
    finally:
        db.close()