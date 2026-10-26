import json

from micro_llm.config import SAMPLE_TICKET
from micro_llm.splitter import split


def test_sample_ticket_matches_the_context_pieces():
    text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    expected = json.loads(SAMPLE_TICKET.with_name("context_pieces.json").read_text())["pieces"]
    got = [{"id": p.id, "text": p.text, "start": p.start, "end": p.end} for p in split(text)]
    assert got == expected


def test_spans_point_at_the_text():
    text = "Hello team, my card was charged $49.99 twice. Please fix it, but don't cancel my plan!"
    for p in split(text):
        assert text[p.start : p.end] == p.text
    assert [p.text for p in split(text)] == [
        "my card was charged $49.99 twice.",
        "Please fix it",
        "don't cancel my plan!",
    ]


def test_empty_text_has_no_pieces():
    assert split("") == []
    assert split("   ") == []
