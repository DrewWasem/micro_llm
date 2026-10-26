from micro_llm.act import feature_vote, login_help, refund_review, verify_identity
from micro_llm.systems import Systems

WRITES = {
    "send_verification_link",
    "send_password_reset",
    "open_refund_request",
    "add_to_incident",
    "log_feature_vote",
}


def writes(systems):
    return [c for c in systems.calls if c["tool"] in WRITES]


def test_verification_goes_to_the_email_on_file_for_the_order():
    s = Systems()
    out = verify_identity(
        {"order_number": "98423", "email": None, "username": None}, None, s
    )
    assert out.status == "with_customer"
    assert writes(s) == [{"tool": "send_verification_link", "account": "A-1001"}]
    assert "s***@example.com" in out.reply and "sam.rivera@example.com" not in out.reply


def test_a_typed_email_only_picks_which_account_to_verify():
    s = Systems()
    verify_identity(
        {"order_number": None, "email": "LEE.CHEN@example.com", "username": None},
        None,
        s,
    )
    assert writes(s) == [{"tool": "send_verification_link", "account": "A-1002"}]


def test_with_nothing_to_match_verification_asks_the_customer():
    s = Systems()
    out = verify_identity(
        {"order_number": "11111", "email": None, "username": None}, None, s
    )
    assert out.status == "with_customer" and "email address or username" in out.reply
    assert writes(s) == []


def test_a_repeated_charge_becomes_a_refund_request_for_staff():
    s = Systems()
    out = refund_review({"order_number": "98423"}, "A-1001", s)
    assert out.status == "with_staff"
    assert writes(s) == [
        {
            "tool": "open_refund_request",
            "order": "98423",
            "charge": "ch_302",
            "reason": "duplicate charge",
        }
    ]
    assert "$49.99" in out.reply


def test_another_accounts_order_gets_no_refund_request_and_a_neutral_reply():
    s = Systems()
    out = refund_review({"order_number": "98423"}, "A-1002", s)
    missing = refund_review({"order_number": "11111"}, "A-1002", Systems())
    assert out.status == "with_staff" and writes(s) == []
    assert out.reply == missing.reply


def test_login_help_joins_an_open_incident_before_sending_a_reset():
    s = Systems()
    assert login_help({}, "A-1001", s).status == "done"
    assert writes(s) == [
        {"tool": "add_to_incident", "incident": "INC-77", "account": "A-1001"}
    ]

    s = Systems()
    s.incidents = []
    out = login_help({}, "A-1001", s)
    assert writes(s) == [{"tool": "send_password_reset", "account": "A-1001"}]
    assert "s***@example.com" in out.reply


def test_a_feature_vote_is_tied_to_the_account():
    s = Systems()
    out = feature_vote(
        {"feature": "Apple Pay", "request": "adding Apple Pay would help."}, "A-1001", s
    )
    assert s.calls == [
        {"tool": "log_feature_vote", "feature": "Apple Pay", "account": "A-1001"}
    ]
    assert out.reply.startswith("Thanks for the Apple Pay idea.")
