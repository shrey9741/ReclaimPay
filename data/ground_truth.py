"""
Ground-truth simulation model for the Payment Failure Recovery Agent.

This is the "world model" the synthetic data is sampled from. It encodes
realistic (but invented) success probabilities for each
(failure_reason, recovery_action) pair. Nothing here is real gateway data --
it's a plausible causal model we're building on purpose so that:
  1. we can generate labeled training data for the cost-aware optimizer
  2. we have a known ground truth to evaluate agent decisions against later

DECLINE REASONS (what actually went wrong):
  OTP_TIMEOUT        - user didn't enter OTP in time
  INSUFFICIENT_FUNDS - account/card didn't have enough balance
  BANK_SERVER_ERROR  - issuing bank's server was down/slow
  GATEWAY_TIMEOUT     - payment gateway itself timed out
  RISK_BLOCK          - transaction blocked by fraud/risk rules
  CARD_EXPIRED        - card expiry issue
  NETWORK_ERROR        - generic network blip on user's side

RECOVERY ACTIONS (what the agent can do next):
  RETRY_SAME          - retry same gateway + same method immediately
  RETRY_BACKOFF        - retry same gateway + same method after a delay
  SWITCH_GATEWAY        - retry via a different payment gateway, same method
  SWITCH_METHOD        - retry via a different method (e.g. card -> UPI)
  ABANDON               - give up, don't retry (avoid wasting gateway fees)

Each cell below is P(success | reason, action). These are hand-designed to be
causally sensible (e.g. retrying INSUFFICIENT_FUNDS immediately almost never
helps -- the money isn't magically there -- but waiting a bit does since
salary/UPI credits land later in the day).
"""

DECLINE_REASONS = [
    "OTP_TIMEOUT",
    "INSUFFICIENT_FUNDS",
    "BANK_SERVER_ERROR",
    "GATEWAY_TIMEOUT",
    "RISK_BLOCK",
    "CARD_EXPIRED",
    "NETWORK_ERROR",
]

RECOVERY_ACTIONS = [
    "RETRY_SAME",
    "RETRY_BACKOFF",
    "SWITCH_GATEWAY",
    "SWITCH_METHOD",
    "ABANDON",
]

# P(success | reason, action) -- the causal ground truth.
SUCCESS_PROB = {
    "OTP_TIMEOUT": {
        "RETRY_SAME": 0.65,       # user just re-enters OTP correctly
        "RETRY_BACKOFF": 0.55,
        "SWITCH_GATEWAY": 0.30,   # doesn't fix a user-side OTP problem
        "SWITCH_METHOD": 0.40,    # UPI has no OTP step, so this can help
        "ABANDON": 0.0,
    },
    "INSUFFICIENT_FUNDS": {
        "RETRY_SAME": 0.04,       # money isn't there yet
        "RETRY_BACKOFF": 0.28,    # salary/credit may land within a few hours
        "SWITCH_GATEWAY": 0.03,   # doesn't fix the underlying balance issue
        "SWITCH_METHOD": 0.12,    # maybe a different linked account/wallet
        "ABANDON": 0.0,
    },
    "BANK_SERVER_ERROR": {
        "RETRY_SAME": 0.35,
        "RETRY_BACKOFF": 0.50,    # bank's server often recovers within minutes
        "SWITCH_GATEWAY": 0.58,   # different gateway may route to bank differently
        "SWITCH_METHOD": 0.45,
        "ABANDON": 0.0,
    },
    "GATEWAY_TIMEOUT": {
        "RETRY_SAME": 0.30,
        "RETRY_BACKOFF": 0.45,
        "SWITCH_GATEWAY": 0.60,   # the gateway itself was the problem
        "SWITCH_METHOD": 0.35,
        "ABANDON": 0.0,
    },
    "RISK_BLOCK": {
        "RETRY_SAME": 0.05,       # will likely get blocked again
        "RETRY_BACKOFF": 0.10,
        "SWITCH_GATEWAY": 0.15,   # different risk engine, marginal help
        "SWITCH_METHOD": 0.20,
        "ABANDON": 0.0,           # often the *correct* action -- avoid retry fraud loops
    },
    "CARD_EXPIRED": {
        "RETRY_SAME": 0.01,       # will never succeed without user action
        "RETRY_BACKOFF": 0.01,
        "SWITCH_GATEWAY": 0.01,
        "SWITCH_METHOD": 0.55,    # switching off the expired card actually fixes it
        "ABANDON": 0.0,
    },
    "NETWORK_ERROR": {
        "RETRY_SAME": 0.60,       # transient, usually just works
        "RETRY_BACKOFF": 0.65,
        "SWITCH_GATEWAY": 0.40,
        "SWITCH_METHOD": 0.40,
        "ABANDON": 0.0,
    },
}

# Relative cost of each action (gateway fees + time + opportunity cost),
# in arbitrary cost units. Used by the optimizer to trade off
# P(success) against cost, not just chase the highest success rate.
ACTION_COST = {
    "RETRY_SAME": 1.0,
    "RETRY_BACKOFF": 1.2,     # small extra cost for delayed settlement
    "SWITCH_GATEWAY": 2.5,    # new gateway fee structure, integration overhead
    "SWITCH_METHOD": 2.0,
    "ABANDON": 0.0,
}

# Roughly how often each decline reason occurs in the wild (invented but
# ballpark-realistic weights for the Indian UPI/card payments context).
REASON_BASE_RATE = {
    "OTP_TIMEOUT": 0.22,
    "INSUFFICIENT_FUNDS": 0.20,
    "BANK_SERVER_ERROR": 0.18,
    "GATEWAY_TIMEOUT": 0.15,
    "RISK_BLOCK": 0.10,
    "CARD_EXPIRED": 0.07,
    "NETWORK_ERROR": 0.08,
}

GATEWAYS = ["GatewayA", "GatewayB", "GatewayC"]
METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]
BANKS = ["BankX", "BankY", "BankZ", "BankW"]
