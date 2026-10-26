import random

from micro_llm import generate as g


def test_parse_reads_numbered_pieces_under_each_id():
    reply = "### t1\n1. Hi there,\n2. I was charged twice.\n\n### t2: extra words\n1) My login fails."
    assert g.parse(reply) == {"t1": ["Hi there,", "I was charged twice."], "t2": ["My login fails."]}


def test_assembled_spans_match_the_pieces():
    rng = random.Random(1)
    for k in range(200):
        spec = g.sample_spec(rng, f"s{k}")
        texts = [f"Piece number {i} is here." for i in range(len(spec["pieces"]))]
        message, spans = g.assemble(spec, texts)
        for (start, end), text in zip(spans, texts):
            assert message[start:end].rstrip(".").lower() == text.rstrip(".").lower()


def test_valid_rejects_a_missing_order_number():
    spec = {"pieces": [{"kind": "problem", "entities": {"order_number": "an order number"}}]}
    assert not g.valid(spec, ["I was charged twice for my order."])
    assert g.valid(spec, ["I was charged twice for order 55123."])
    assert not g.valid(spec, ["Too", "many"])
