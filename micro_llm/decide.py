"""Step 2: decides which actions a ticket needs, and which of them can run yet.

Each decider answers one question from the step-1 answers. They are rules for now, because no
record of the actions past tickets needed exists to train them on. The account gate stays a
rule for good: it must never guess whose account a stranger may act on.
"""

import re
from dataclasses import dataclass

# A decider says yes when its strongest step-1 answer reaches this chance
DECIDE_MIN = 0.5


@dataclass(frozen=True)
class Decider:
    action: str
    answers: tuple  # the step-1 answers it reads; the strongest one decides
    needs_account: bool


@dataclass(frozen=True)
class Step:
    action: str
    reason: str
    inputs: dict
    needs_account: bool


DECIDERS = [
    # Was the customer charged wrongly?
    Decider(
        "refund_review",
        ("duplicate_charge", "refund_requested", "refund_implied"),
        True,
    ),
    # Is the customer locked out?
    Decider("login_help", ("cannot_log_in",), True),
    # Did the customer ask for a feature? A vote counts once per customer, so it waits too
    Decider("feature_vote", ("mentions_feature_request",), True),
]


# Entity kinds whose text carries a label in front of the identifier
LABELLED = {"order_number", "username"}


def value(text):
    """The identifier inside an entity: "#98423" and "order 98423" give 98423, "username: samr" gives samr."""
    return re.search(r"[\w.-]+$", text).group()


def inputs(action, result):
    """What an action needs from the ticket, saved so it can still run after later messages."""
    found = {}
    for e in result["entities"]:
        text = value(e["text"]) if e["kind"] in LABELLED else e["text"]
        found.setdefault(e["kind"], text)
    if action == "verify_identity":
        return {k: found.get(k) for k in ("order_number", "email", "username")}
    if action == "refund_review":
        return {"order_number": found.get("order_number")}
    if action == "feature_vote":
        asks = [p["text"] for p in result["pieces"] if p["kind"] == "request"]
        named = [
            e["text"]
            for e in result["entities"]
            if e["kind"] == "payment_method" and any(e["text"] in a for a in asks)
        ]
        return {
            "feature": named[0] if named else None,
            "request": asks[0] if asks else None,
        }
    return {}


def plan(result, account=None):
    """The steps a ticket needs, in order. Without a verified account, verifying it comes first."""
    steps = []
    for d in DECIDERS:
        p, q = max((result["answers"][q], q) for q in d.answers)
        if p >= DECIDE_MIN:
            steps.append(
                Step(
                    d.action, f"{q} {p:.2f}", inputs(d.action, result), d.needs_account
                )
            )
    held = [s.action for s in steps if s.needs_account]
    if account is None and held:
        reason = f"no verified account, needed by {', '.join(held)}"
        steps.insert(
            0, Step("verify_identity", reason, inputs("verify_identity", result), False)
        )
    return steps


def ready(step, account):
    """Whether a step can run now: it needs no account, or the account is verified."""
    return account is not None or not step.needs_account
