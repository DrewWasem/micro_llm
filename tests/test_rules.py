from micro_llm.models import to_answer
from micro_llm.questions import bank, qtype
from micro_llm.routing import MODEL_QUESTIONS
from micro_llm.rules import TICKET_QUESTIONS, answer


def piece(kind, team, **labels):
    def one(q):
        default = {"noul": False, "choice": "not_stated"}.get(qtype(q), 0)
        return to_answer(q, {labels.get(q, default): 1.0})

    out = {
        "router": {
            "piece_kind": to_answer("piece_kind", {kind: 1.0}),
            "primary_team": to_answer("primary_team", {team: 1.0}),
        },
        "tone": {q: one(q) for q in MODEL_QUESTIONS["tone"]},
    }
    for model in labels.pop("models", []):
        out[model] = {q: one(q) for q in MODEL_QUESTIONS[model]}
    return out


def test_every_ticket_question_gets_an_answer():
    assert len(TICKET_QUESTIONS) == 47 and set(TICKET_QUESTIONS) == set(bank()) - {"piece_kind"}
    pieces = [piece("problem", "billing_and_payments", duplicate_charge=True, models=["billing"])]
    out = answer(pieces, [], [{"order_number"}], apple=False)
    assert set(out) == set(TICKET_QUESTIONS)


def test_a_double_charge_is_high_priority_billing():
    pieces = [
        piece("problem", "billing_and_payments", duplicate_charge=True, refund_implied=True, models=["billing"]),
        piece("feeling", "other"),
    ]
    out = answer(pieces, [], [{"order_number"}, set()], apple=False)
    assert out["category"]["answer"] == "billing"
    assert out["priority"]["level"] == 2
    assert out["issue_count"]["answer"] == "one"
    assert out["needs_staff_action"] == 1.0
    assert out["existing_customer"] == 1.0
    assert out["needs_more_info"] == 0.0


def test_a_bug_without_details_needs_more_info():
    pieces = [piece("problem", "engineering", bug_severity=2, models=["bug"])]
    out = answer(pieces, [], [set()], apple=False)
    assert out["mentions_bug"] == 1.0
    assert out["needs_more_info"] == 1.0
    assert out["bug_severity"]["level"] == 2
