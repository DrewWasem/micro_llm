"""Stand-ins for the systems the step-3 actions call: accounts, orders, payments, incidents and
the product board.

They read `systems.json` and record every call, so tests and the audit log show what each
action did. None of them moves money: a refund can only be requested for staff to approve.
"""

import json

from micro_llm.config import SAMPLE_SYSTEMS


def masked(email):
    """sam.rivera@example.com becomes s***@example.com, so a reply never shows the whole address."""
    name, _, domain = email.partition("@")
    return f"{name[:1]}***@{domain}"


class Systems:
    def __init__(self, path=SAMPLE_SYSTEMS):
        data = json.loads(path.read_text())
        self.accounts = {a["id"]: a for a in data["accounts"]}
        self.orders = {o["number"]: o for o in data["orders"]}
        self.charges = data["charges"]
        self.incidents = data["incidents"]
        self.calls = []

    def record(self, tool, **args):
        self.calls.append({"tool": tool, **args})

    # Reads

    def find_order(self, number):
        self.record("find_order", number=number)
        return self.orders.get(number)

    def find_account(self, email=None, username=None):
        """The id of the account with this email or username, or None."""
        self.record("find_account", email=email, username=username)
        for a in self.accounts.values():
            if (email and a["email"].lower() == email.lower()) or (
                username and a["username"].lower() == username.lower()
            ):
                return a["id"]
        return None

    def order_charges(self, number):
        self.record("order_charges", number=number)
        return [c for c in self.charges if c["order"] == number]

    def open_incidents(self, topic):
        self.record("open_incidents", topic=topic)
        return [
            i for i in self.incidents if i["topic"] == topic and i["status"] == "open"
        ]

    # Writes

    def send_verification_link(self, account):
        """Emails a one-time link to the address on file and returns that address masked."""
        self.record("send_verification_link", account=account)
        return masked(self.accounts[account]["email"])

    def send_password_reset(self, account):
        self.record("send_password_reset", account=account)
        return masked(self.accounts[account]["email"])

    def open_refund_request(self, order, charge, reason):
        """Queues a refund for staff to approve and returns the request's id."""
        self.record("open_refund_request", order=order, charge=charge, reason=reason)
        return f"RR-{sum(c['tool'] == 'open_refund_request' for c in self.calls)}"

    def add_to_incident(self, incident, account):
        self.record("add_to_incident", incident=incident, account=account)

    def log_feature_vote(self, feature, account):
        self.record("log_feature_vote", feature=feature, account=account)
