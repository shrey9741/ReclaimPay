"""
Database models for the orchestrator.

Transaction state machine:

    FAILED --(classified + action decided)--> RETRY_SCHEDULED
    RETRY_SCHEDULED --(retry executed, succeeded)--> RECOVERED
    RETRY_SCHEDULED --(retry executed, failed, attempts remain)--> RETRY_SCHEDULED
    RETRY_SCHEDULED --(retry executed, failed, attempts exhausted)--> EXHAUSTED
    FAILED --(optimizer chose ABANDON)--> ABANDONED

Idempotency: `transaction_id` (the payment gateway's own transaction id, NOT
our internal primary key) has a UNIQUE constraint. If the same failed-payment
webhook is delivered twice (which real payment gateways do -- at-least-once
delivery is standard), the second insert is rejected at the DB level and the
API returns the existing record instead of creating a duplicate retry flow.
This is what actually prevents double-charging a customer.
"""

import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Enum, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

DATABASE_URL = "postgresql://reclaimpay:reclaimpay@localhost:5432/reclaimpay"


class TxnStatus(str, enum.Enum):
    FAILED = "FAILED"                     # just ingested, not yet processed
    RETRY_SCHEDULED = "RETRY_SCHEDULED"   # an action was decided, waiting to execute
    RECOVERED = "RECOVERED"               # a retry succeeded -- terminal
    EXHAUSTED = "EXHAUSTED"               # ran out of retry attempts -- terminal
    ABANDONED = "ABANDONED"               # optimizer chose not to retry at all -- terminal


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (UniqueConstraint("gateway_transaction_id", name="uq_gateway_txn_id"),)

    id = Column(Integer, primary_key=True)
    gateway_transaction_id = Column(String, nullable=False, index=True)  # idempotency key
    merchant_id = Column(String, nullable=False, index=True)
    amount_inr = Column(Float, nullable=False)
    payment_method = Column(String, nullable=False)
    gateway = Column(String, nullable=False)
    bank = Column(String, nullable=False)
    decline_message_raw = Column(String, nullable=False)
    decline_reason = Column(String, nullable=True)       # filled by Classifier Agent
    classifier_confidence = Column(Float, nullable=True)
    status = Column(Enum(TxnStatus), nullable=False, default=TxnStatus.FAILED, index=True)
    retry_attempt_number = Column(Integer, nullable=False, default=0)
    max_retry_attempts = Column(Integer, nullable=False, default=3)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    attempts = relationship("RetryAttempt", back_populates="transaction")


class RetryAttempt(Base):
    __tablename__ = "retry_attempts"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    action = Column(String, nullable=False)          # RETRY_SAME / SWITCH_GATEWAY / etc.
    p_success_predicted = Column(Float, nullable=False)
    expected_value_inr = Column(Float, nullable=False)
    scheduled_at = Column(DateTime, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    outcome_success = Column(Integer, nullable=True)  # null until executed
    created_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="attempts")


engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_db()
    print("Tables created.")
