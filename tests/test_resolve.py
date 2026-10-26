import pytest

from micro_llm.resolve import handle
from micro_llm.store import Store
from micro_llm.systems import Systems


def acted(out):
    return [(action, o.status) for action, o in out["acted"]]


def test_steps_wait_for_a_verified_account_then_run_once(tmp_path, sample_result):
    path = tmp_path / "threads.sqlite"
    store = Store(path)
    first = handle(store, Systems(), sample_result)
    store.close()
    assert acted(first) == [("verify_identity", "with_customer")]
    assert first["waiting"] == 3

    # A new process: the waiting steps come back from the file
    store = Store(path)
    second = handle(store, Systems(), None, thread=first["thread"], account="A-1001")
    assert second["released"] == ["refund_review", "login_help", "feature_vote"]
    assert acted(second) == [
        ("refund_review", "with_staff"),
        ("login_help", "done"),
        ("feature_vote", "done"),
    ]
    assert second["waiting"] == 0

    again = handle(
        store, Systems(), sample_result, thread=first["thread"], account="A-1001"
    )
    assert again["planned"] == [] and acted(again) == []

    events = [e for (e,) in store.db.execute("SELECT event FROM audit ORDER BY id")]
    assert events.count("planned") == 4 and events.count("released") == 3
    assert "account connected" in events
    status = dict(store.db.execute("SELECT action, status FROM steps"))
    assert status["verify_identity"] == "done"
    store.close()


def test_a_thread_keeps_its_first_account(tmp_path, sample_result):
    store = Store(tmp_path / "threads.sqlite")
    thread = handle(store, Systems(), sample_result)["thread"]
    handle(store, Systems(), None, thread=thread, account="A-1001")
    with pytest.raises(ValueError, match="another account"):
        handle(store, Systems(), None, thread=thread, account="A-1002")
    with pytest.raises(ValueError, match="no thread"):
        handle(store, Systems(), None, thread=thread + 1)
    store.close()


def test_a_verified_customer_skips_straight_to_the_actions(tmp_path, sample_result):
    store = Store(tmp_path / "threads.sqlite")
    out = handle(store, Systems(), sample_result, account="A-1001")
    assert [a for a, _ in out["acted"]] == [
        "refund_review",
        "login_help",
        "feature_vote",
    ]
    store.close()
