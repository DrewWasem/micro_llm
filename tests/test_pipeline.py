import json

import pytest

from micro_llm.config import MODELS_DIR, SAMPLE_TICKET
from micro_llm.rules import TICKET_QUESTIONS

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "router.pkl").exists(), reason="models not trained"
)


def test_sample_ticket_runs_end_to_end():
    from micro_llm.pipeline import Pipeline

    text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    result = Pipeline().run(text)
    assert set(result["answers"]) == set(TICKET_QUESTIONS)
    assert len(result["pieces"]) == 4
    assert result["ms"]["total"] < 1000


def test_sample_ticket_verifies_the_account_before_anything_else(tmp_path):
    from micro_llm.pipeline import Pipeline
    from micro_llm.resolve import handle
    from micro_llm.store import Store
    from micro_llm.systems import Systems

    text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    store = Store(tmp_path / "threads.sqlite")
    out = handle(store, Systems(), Pipeline().run(text))
    store.close()
    assert [a for a, _ in out["acted"]] == ["verify_identity"]
    assert out["waiting"] == 3
