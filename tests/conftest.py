import pytest

from micro_llm.rules import TICKET_QUESTIONS


@pytest.fixture
def sample_result():
    """Step 1's output for the sample ticket, cut down to what steps 2 and 3 read."""
    answers = dict.fromkeys(TICKET_QUESTIONS, 0.0)
    answers.update(
        duplicate_charge=0.905,
        refund_implied=0.717,
        cannot_log_in=0.969,
        mentions_feature_request=0.775,
    )
    return {
        "answers": answers,
        "pieces": [
            {
                "id": "P1",
                "kind": "problem",
                "team": "billing_and_payments",
                "text": "I placed an order (#98423) last Thursday and was charged twice.",
            },
            {
                "id": "P2",
                "kind": "problem",
                "team": "account_support",
                "text": "I also can't log in after the site update",
            },
            {
                "id": "P3",
                "kind": "request",
                "team": "product_management",
                "text": "adding Apple Pay would be really helpful.",
            },
            {
                "id": "P4",
                "kind": "feeling",
                "team": "other",
                "text": "This is getting frustrating.",
            },
        ],
        "entities": [
            {"kind": "order_number", "text": "#98423", "start": 23, "end": 29},
            {"kind": "date", "text": "last Thursday", "start": 31, "end": 44},
            {"kind": "payment_method", "text": "Apple Pay", "start": 122, "end": 131},
        ],
    }
