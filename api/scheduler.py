"""
Scheduler: polls the Redis delay ZSET every couple seconds and promotes
anything whose due time has passed into the Stream for workers to consume.

Run this as its own process alongside the worker(s) and the API.
"""

import time
import sys
import os

sys.path.append(os.path.dirname(__file__))
from redis_queue import promote_due_retries  # noqa: E402


def run_scheduler(poll_interval_seconds=2, max_iterations=None):
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        promoted = promote_due_retries()
        if promoted:
            print(f"[scheduler] promoted {promoted} due retr{'y' if promoted == 1 else 'ies'}")
        time.sleep(poll_interval_seconds)
        iterations += 1


if __name__ == "__main__":
    run_scheduler()
