"""
Delayed retry queue on Redis.

Why not just push straight to a Redis Stream: retries often need a
delay (RETRY_BACKOFF should wait minutes, not fire instantly), and
Streams don't support delayed delivery natively. Standard pattern:

  1. Schedule: push (due_timestamp, transaction_id) into a Sorted Set
     (ZSET), score = due unix timestamp.
  2. Scheduler loop: repeatedly pop everything from the ZSET with
     score <= now, and XADD each into a Stream for processing.
  3. Workers: consume from the Stream using a Consumer Group, so
     multiple worker processes can run in parallel with no duplicate
     processing (each message is delivered to exactly one consumer
     in the group), and XACK on success. Unacked messages can be
     re-claimed (XCLAIM) if a worker crashes mid-processing --
     that's the at-least-once delivery guarantee that makes the
     idempotency check in db.py necessary in the first place.
"""

import json
import time
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

SCHEDULE_ZSET = "retry:schedule"
STREAM_KEY = "retry:stream"
CONSUMER_GROUP = "retry-workers"

# Backoff delay in seconds per action -- how long to wait before executing.
ACTION_DELAY_SECONDS = {
    "RETRY_SAME": 5,
    "RETRY_BACKOFF": 60,       # give the bank/balance situation time to change
    "SWITCH_GATEWAY": 5,
    "SWITCH_METHOD": 5,
    "ABANDON": 0,              # never actually scheduled, handled synchronously
}


def ensure_consumer_group():
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise  # group already exists -- fine


def schedule_retry(transaction_id: int, action: str, attempt_number: int):
    delay = ACTION_DELAY_SECONDS.get(action, 30)
    due_at = time.time() + delay
    payload = json.dumps({
        "transaction_id": transaction_id,
        "action": action,
        "attempt_number": attempt_number,
    })
    # score must be unique-ish per member for ZSET semantics; use the
    # payload itself as the member since it's already unique per attempt.
    r.zadd(SCHEDULE_ZSET, {payload: due_at})


def promote_due_retries() -> int:
    """Moves everything in the ZSET whose due time has passed into the
    Stream for workers to pick up. Returns how many were promoted.
    Call this from a scheduler loop running every few seconds."""
    now = time.time()
    due = r.zrangebyscore(SCHEDULE_ZSET, min=0, max=now)
    for payload in due:
        r.xadd(STREAM_KEY, {"data": payload})
        r.zrem(SCHEDULE_ZSET, payload)
    return len(due)


def read_pending(consumer_name: str, count: int = 10, block_ms: int = 2000):
    """Worker-side: read up to `count` new messages for this consumer."""
    ensure_consumer_group()
    resp = r.xreadgroup(CONSUMER_GROUP, consumer_name, {STREAM_KEY: ">"},
                         count=count, block=block_ms)
    if not resp:
        return []
    _, messages = resp[0]
    return messages  # list of (message_id, {"data": payload_json})


def ack(message_id: str):
    r.xack(STREAM_KEY, CONSUMER_GROUP, message_id)
