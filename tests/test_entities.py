import json

from micro_llm.config import SAMPLE_TICKET
from micro_llm.entities import find, is_apple


def kinds(text):
    return {(e.kind, e.text) for e in find(text)}


def test_sample_ticket_entities():
    text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    assert kinds(text) == {("order_number", "#98423"), ("date", "last Thursday"), ("payment_method", "Apple Pay")}
    assert any(is_apple(e) for e in find(text))


def test_card_numbers_need_a_valid_check_digit():
    assert ("card_number", "4111 1111 1111 1111") in kinds("My card 4111 1111 1111 1111 was charged.")
    assert not any(k == "card_number" for k, _ in kinds("Reference 1234 5678 9012 3456 failed."))


def test_personal_details():
    found = kinds('Error: "Session expired (code 403)" on my iPhone 15. Email me at sam@example.com. Thanks, Sam')
    assert ("error_text", '"Session expired (code 403)"') in found
    assert ("device", "iPhone 15") in found
    assert ("email", "sam@example.com") in found
    assert ("name", "Sam") in found
