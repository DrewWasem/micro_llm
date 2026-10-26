from micro_llm.decide import plan, ready, value


def test_an_unknown_account_is_verified_first_and_the_rest_waits(sample_result):
    steps = plan(sample_result)
    assert [s.action for s in steps] == [
        "verify_identity",
        "refund_review",
        "login_help",
        "feature_vote",
    ]
    assert steps[0].inputs == {"order_number": "98423", "email": None, "username": None}
    assert [ready(s, None) for s in steps] == [True, False, False, False]


def test_a_verified_account_skips_verification(sample_result):
    steps = plan(sample_result, account="A-1001")
    assert [s.action for s in steps] == ["refund_review", "login_help", "feature_vote"]
    assert all(ready(s, "A-1001") for s in steps)
    assert steps[0].reason == "duplicate_charge 0.91"
    assert steps[2].inputs == {
        "feature": "Apple Pay",
        "request": "adding Apple Pay would be really helpful.",
    }


def test_weak_answers_plan_nothing_not_even_verification(sample_result):
    sample_result["answers"].update(
        duplicate_charge=0.4,
        refund_implied=0.3,
        cannot_log_in=0.2,
        mentions_feature_request=0.1,
    )
    assert plan(sample_result) == []


def test_a_typed_email_and_username_reach_verification_whole(sample_result):
    sample_result["entities"] = [
        {"kind": "email", "text": "lee.chen@example.com", "start": 0, "end": 20},
        {"kind": "username", "text": "username: leec", "start": 21, "end": 35},
    ]
    assert plan(sample_result)[0].inputs == {
        "order_number": None,
        "email": "lee.chen@example.com",
        "username": "leec",
    }


def test_identifiers_lose_their_labels():
    assert (
        value("#98423") == value("order 98423") == value("Order no. 98423") == "98423"
    )
    assert value("username: samr") == value("user id is @samr") == "samr"
