"""Which model answers which questions, and which specialists read a piece."""

MODEL_QUESTIONS = {
    "router": ["piece_kind", "primary_team"],
    "tone": [
        "frustration",
        "sentiment",
        "politeness",
        "time_pressure",
        "threatens_to_leave",
        "asks_for_human",
        "prior_contact",
        "frustration_rising",
    ],
    "billing": [
        "duplicate_charge",
        "refund_requested",
        "refund_implied",
        "payment_method_used",
        "chargeback_mentioned",
    ],
    "account": ["cannot_log_in", "security_related", "tried_fix_themselves"],
    "bug": [
        "bug_severity",
        "has_reproducible_steps",
        "problem_after_update",
        "blames_update",
        "possible_widespread_issue",
    ],
    "feature": ["wants_new_payment_method", "feature_request_strength"],
    "linker": ["issues_related"],
}
TEAM_SPECIALIST = {
    "billing_and_payments": "billing",
    "account_support": "account",
    "engineering": "bug",
    "product_management": "feature",
}


def is_issue(kind):
    return kind in ("problem", "request")


def specialists(kind, team):
    """Specialist models that read a piece, given what the router says about it."""
    if not is_issue(kind) or team not in TEAM_SPECIALIST:
        return []
    names = [TEAM_SPECIALIST[team]]
    if team == "account_support" and kind == "problem":
        names.append("bug")
    return names
