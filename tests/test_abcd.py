from micro_llm.abcd import (
    ALL,
    GROUPS,
    customer_text,
    details,
    entered,
    follows,
    intent,
    our_truth,
    typed_in,
)


def turn(speaker, text, action=None, values=()):
    return {
        "speaker": speaker,
        "text": text,
        "targets": ["recover_password", None, action, list(values), -1],
    }


CHAT = {
    "scenario": {
        "flow": "account_access",
        "subflow": "recover_password",
        "personal": {"customer_name": "Chloe Zhang", "member_level": "gold"},
        "order": {},
    },
    "original": [
        ["agent", "Hi! How can I help?"],
        ["customer", "I can't get into my account"],
        ["customer", "I forgot my password"],
        ["agent", "Can I have your name and email?"],
        ["customer", "Chloe Zhang"],
        ["customer", "chloe@email.com"],
        ["action", "Account has been pulled up for Chloe Zhang."],
        ["action", "A new password has been made."],
    ],
    "delexed": [
        turn("agent", "hi! how can i help?"),
        turn("customer", "i can't get into my account"),
        turn("customer", "i forgot my password"),
        turn("agent", "can i have your name and email?"),
        turn("customer", "chloe zhang"),
        turn("customer", "<email>"),
        turn("action", "", "pull-up-account", ["chloe zhang"]),
        turn("action", "", "make-password"),
    ],
}


def test_the_opening_is_every_message_before_the_agent_replies():
    opening = customer_text(CHAT, "opening")
    assert opening == "I can't get into my account\nI forgot my password"
    assert customer_text(CHAT, "all").count("\n") == 3


def test_details_come_from_placeholders_and_the_customers_name():
    assert details(CHAT) == {"name": 2, "email": 3}
    assert typed_in(CHAT, "opening") == set()
    assert typed_in(CHAT, "all") == {"name", "email"}


def test_our_questions_read_the_intent_and_the_actions():
    t = our_truth(CHAT, "opening")
    assert t["primary_team"] == "account_support" and t["category"] == "account"
    assert t["cannot_log_in"] and t["refund_requested"] is False
    # make-password changes something, so staff had to act
    assert t["needs_staff_action"]
    assert not t["identity_details_given"]
    assert our_truth(CHAT, "all")["identity_details_given"]
    # A lost password says nothing about whether they bought anything
    assert t["existing_customer"] is None


def test_the_policy_order_allows_other_actions_between():
    need = ["pull-up-account", "make-password"]
    assert follows(["pull-up-account", "search-faq", "make-password"], need)
    assert not follows(["make-password", "pull-up-account"], need)


def test_faq_chats_map_back_to_their_topic():
    assert (
        intent({"scenario": {"flow": "single_item_query", "subflow": "boots_how_1"}})
        == "boots"
    )


def test_every_intent_sits_in_exactly_one_mapping_group():
    assert sum(map(len, GROUPS)) == len(ALL) == 56


def test_entered_values_survive_typos_and_need_the_right_action():
    chat = {"delexed": [turn("action", "", "notify-team", ["purchasing team"])]}
    teams = ["manager", "website team", "purchasing department"]
    assert entered(chat, teams, {"notify-team"}) == "purchasing department"
    assert entered(chat, teams, {"update-order"}) == "none"
