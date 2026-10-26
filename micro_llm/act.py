"""Step 3: actions that each do one job through the systems' tools and report what they did.

An action gets the inputs step 2 saved, the verified account (None before verification) and
the systems. An action that touches an account first checks the account is the verified one.
"""

from dataclasses import dataclass

CHECKING = "We're checking the charge and will update you shortly."


@dataclass(frozen=True)
class Outcome:
    status: str  # who acts next: "done" for no one, "with_staff" or "with_customer"
    did: str  # one line for the audit log
    reply: str  # one line for the customer


def verify_identity(inputs, account, systems):
    """Sends a one-time link to the email on file, never to an address typed into the ticket."""
    number, email, username = (
        inputs.get(k) for k in ("order_number", "email", "username")
    )
    order = systems.find_order(number) if number else None
    found = order["account"] if order else None
    if found is None and (email or username):
        found = systems.find_account(email, username)
    if found is None:
        return Outcome(
            "with_customer",
            "found no account to verify",
            "Please reply with the email address or username on your account, so we can confirm it's you.",
        )
    sent = systems.send_verification_link(found)
    where = f"order #{number}" if order else "your account"
    return Outcome(
        "with_customer",
        f"sent a verification link to {sent}, the email on {where}",
        f"We sent a link to {sent}, the email address on {where}. Open it to confirm it's you, "
        "and we'll pick up the rest.",
    )


def refund_review(inputs, account, systems):
    """Asks staff to refund each settled charge that repeats an earlier one on the order."""
    number = inputs.get("order_number")
    order = systems.find_order(number) if number else None
    # Another account's order gets the same reply as a missing one, so the reply reveals nothing
    if order is None or order["account"] != account:
        what = f"order #{number}" if number else "an order number"
        return Outcome(
            "with_staff",
            f"found no {what} on account {account}; staff to check",
            CHECKING,
        )
    settled = [c for c in systems.order_charges(number) if c["status"] == "settled"]
    repeats = [
        c
        for i, c in enumerate(settled)
        if any(c["amount"] == d["amount"] for d in settled[:i])
    ]
    if not repeats:
        return Outcome(
            "with_staff",
            f"no repeated charge on order #{number}; staff to check",
            CHECKING,
        )
    ids = [
        systems.open_refund_request(number, c["id"], "duplicate charge")
        for c in repeats
    ]
    amount = repeats[0]["amount"]
    return Outcome(
        "with_staff",
        f"opened refund request {', '.join(ids)} for charge {', '.join(c['id'] for c in repeats)} "
        f"(${amount}) on order #{number}",
        f"We found the repeated ${amount} charge on order #{number} and asked our billing team to refund it.",
    )


def login_help(inputs, account, systems):
    """Joins the customer to an open login incident, or else sends a reset link to the email on file."""
    incidents = systems.open_incidents("login")
    if incidents:
        incident = incidents[0]
        systems.add_to_incident(incident["id"], account)
        return Outcome(
            "done",
            f"added the account to {incident['id']}: {incident['title']}",
            "The login problem is a known fault we're fixing now, and we'll email you when it's fixed.",
        )
    sent = systems.send_password_reset(account)
    return Outcome(
        "done",
        f"sent a password reset link to {sent}",
        f"We sent a password reset link to {sent}.",
    )


def feature_vote(inputs, account, systems):
    """Logs one vote for the feature, tied to the account so each customer counts once."""
    feature = inputs["feature"] or inputs["request"]
    systems.log_feature_vote(feature, account)
    thanks = (
        f"Thanks for the {inputs['feature']} idea."
        if inputs["feature"]
        else "Thanks for the idea."
    )
    return Outcome(
        "done",
        f"logged a vote for {feature}",
        f"{thanks} We've passed it to the product team.",
    )


ACTIONS = {
    a.__name__: a for a in (verify_identity, refund_review, login_help, feature_vote)
}
