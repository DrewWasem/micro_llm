"""Turns per-piece model answers into the 47 ticket answers. 21 come from rules here.

Ticket answers keep the model answer shapes: a probability, an option with probabilities,
or a level with probabilities. "Any piece" questions take the strongest piece.
"""

from micro_llm.models import to_answer
from micro_llm.questions import bank, classes, qtype
from micro_llm.routing import MODEL_QUESTIONS

CATEGORY = {
    "billing_and_payments": "billing",
    "account_support": "account",
    "engineering": "bug_report",
    "product_management": "feature_request",
}
# A problem for this team can't be fixed without one of these details
NEEDS = {
    "billing_and_payments": {"order_number", "money", "date"},
    "account_support": {"error_text", "device", "email", "username"},
    "engineering": {"error_text", "device"},
}
FLAGS = [
    "time_pressure",
    "threatens_to_leave",
    "asks_for_human",
    "prior_contact",
    "frustration_rising",
]
SPECIALISTS = ["billing", "account", "bug", "feature"]
TICKET_QUESTIONS = [q for q in bank() if q != "piece_kind"]


def kind_p(piece, kind):
    return piece["router"]["piece_kind"]["p"].get(kind, 0.0)


def team_p(piece, team):
    return piece["router"]["primary_team"]["p"].get(team, 0.0)


def issue_p(piece):
    return kind_p(piece, "problem") + kind_p(piece, "request")


def top(values):
    return round(max(values, default=0.0), 3)


def strongest(qid, answers):
    """The answer with the highest level, or level 0 when no piece was asked."""
    return max(answers, key=lambda a: a["level"], default=to_answer(qid, {0: 1.0}))


def weakest(qid, answers):
    return min(answers, key=lambda a: a["level"], default=to_answer(qid, {0: 1.0}))


def average(qid, answers):
    """The mean of the pieces' level probabilities."""
    if not answers:
        return to_answer(qid, {0: 1.0})
    return to_answer(
        qid,
        {lvl: sum(a["p"][lvl] for a in answers) / len(answers) for lvl in classes(qid)},
    )


def stated(answer):
    return 1 - answer["p"].get("not_stated", 0.0)


def answer(pieces, pairs, kinds, apple):
    """Every ticket answer.

    pieces: each piece's model outputs, keyed by model name, with only the models that read it.
    pairs: the linker's issues_related probability for each pair of issue pieces.
    kinds: the entity kinds found in each piece. apple: an Apple product or Apple Pay was named.
    """
    out = {}
    issues = [p for p in pieces if issue_p(p) >= 0.5]
    talk = [p for p in pieces if kind_p(p, "greeting") < 0.5] or pieces
    found = set().union(*kinds)

    # Tone: the angriest piece, the average mood, the rudest piece, and any flag
    out["frustration"] = strongest(
        "frustration", [p["tone"]["frustration"] for p in talk]
    )
    out["sentiment"] = average("sentiment", [p["tone"]["sentiment"] for p in talk])
    out["politeness"] = weakest("politeness", [p["tone"]["politeness"] for p in talk])
    for q in FLAGS:
        out[q] = top(p["tone"][q] for p in pieces)

    # Specialists: the strongest answer from any piece the model read
    for model in SPECIALISTS:
        for q in MODEL_QUESTIONS[model]:
            said = [p[model][q] for p in pieces if model in p]
            if qtype(q) == "noul":
                out[q] = top(said)
            elif qtype(q) == "score":
                out[q] = strongest(q, said)
            else:
                out[q] = max(
                    said, key=stated, default=to_answer(q, {"not_stated": 1.0})
                )
    out["issues_related"] = top(pairs)

    # Router: the first issue owns the ticket and counts double toward its category
    first = issues[0] if issues else pieces[0]
    out["primary_team"] = first["router"]["primary_team"]
    votes = dict.fromkeys(classes("category"), 0.0)
    for p in issues:
        for team, category in CATEGORY.items():
            votes[category] += (2.0 if p is first else 1.0) * team_p(p, team)
    total = sum(votes.values()) or 1.0
    out["category"] = to_answer("category", {c: v / total for c, v in votes.items()})
    # Code has no position bias, so the reordered probe gets the same answer
    out["category_reordered"] = to_answer("category_reordered", out["category"]["p"])
    out["mentions_billing_issue"] = top(
        issue_p(p) * team_p(p, "billing_and_payments") for p in pieces
    )
    out["mentions_account_issue"] = top(
        issue_p(p) * team_p(p, "account_support") for p in pieces
    )
    out["mentions_bug"] = top(
        kind_p(p, "problem") * (team_p(p, "engineering") + team_p(p, "account_support"))
        for p in pieces
    )
    out["mentions_feature_request"] = top(
        kind_p(p, "request") * team_p(p, "product_management") for p in pieces
    )

    # issue_count: the chance of each count, treating pieces as independent
    counts = [1.0]
    for p in pieces:
        q = issue_p(p)
        counts = [a * (1 - q) + b * q for a, b in zip(counts + [0.0], [0.0] + counts)]
    counts += [0.0] * 4
    out["issue_count"] = to_answer(
        "issue_count",
        {
            "one": counts[0] + counts[1],
            "two": counts[2],
            "three": counts[3],
            "four_or_more": sum(counts[4:]),
        },
    )
    problem = top(kind_p(p, "problem") for p in pieces)
    request = top(kind_p(p, "request") for p in pieces)
    out["message_type"] = to_answer(
        "message_type",
        {
            "problem_report": problem,
            "request": (1 - problem) * request,
            "other": (1 - problem) * (1 - request),
        },
    )

    # Customer: an order number or a billing issue means they already bought something
    out["existing_customer"] = (
        1.0 if "order_number" in found else out["mentions_billing_issue"]
    )
    out["journey_stage"] = to_answer(
        "journey_stage",
        {
            "post_purchase": out["existing_customer"],
            "unclear": 1 - out["existing_customer"],
        },
    )
    out["has_account"] = top(
        [out["mentions_account_issue"], out["cannot_log_in"], out["existing_customer"]]
    )
    leave, anger = out["threatens_to_leave"], out["frustration"]["level"] / 2
    out["churn_risk"] = to_answer(
        "churn_risk", {2: leave, 1: (1 - leave) * anger, 0: (1 - leave) * (1 - anger)}
    )

    # Priority: urgent when a blocking bug may hit many users, high when money or access is at stake
    urgent = out["possible_widespread_issue"] * out["bug_severity"]["p"][2]
    money = max(
        out["duplicate_charge"], out["refund_implied"], out["chargeback_mentioned"]
    )
    access = max(out["cannot_log_in"], out["security_related"])
    high = (1 - urgent) * max(money, access, out["time_pressure"])
    normal = (1 - urgent - high) * problem
    out["priority"] = to_answer(
        "priority", {3: urgent, 2: high, 1: normal, 0: 1 - urgent - high - normal}
    )
    out["needs_staff_action"] = top(
        [
            out["refund_requested"],
            out["refund_implied"],
            out["duplicate_charge"],
            out["cannot_log_in"],
            out["security_related"],
        ]
    )
    out["needs_more_info"] = top(
        kind_p(p, "problem") * team_p(p, team)
        for p, k in zip(pieces, kinds)
        for team, wanted in NEEDS.items()
        if not wanted & k
    )

    # Entities: exact spans, so these are yes or no
    out["likely_apple_user"] = float(apple)
    out["charge_amount_given"] = float("money" in found)
    out["order_date_given"] = float(
        any(
            "date" in k
            and (team_p(p, "billing_and_payments") >= 0.5 or "order_number" in k)
            for p, k in zip(pieces, kinds)
        )
    )
    out["error_message_given"] = float("error_text" in found)
    out["device_or_browser_given"] = float("device" in found)
    out["identity_details_given"] = float(bool(found & {"email", "name", "username"}))
    out["sensitive_data_present"] = float(
        bool(found & {"card_number", "password", "address"})
    )
    return {q: out[q] for q in TICKET_QUESTIONS}
