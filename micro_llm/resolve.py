"""Steps 1 to 3 for one message: answer the 47 questions, decide the next steps, then act.

python -m micro_llm.resolve                     # the sample ticket, from an unknown account
python -m micro_llm.resolve --thread 1 --account A-1001 "I confirmed it's me."

--account stands for the account the customer proved is theirs by opening the link. A live
system takes it from their signed-in session, never from the message text.
"""

import argparse
import json
import sys

from micro_llm import act, decide
from micro_llm.config import SAMPLE_TICKET, STORE_FILE
from micro_llm.pipeline import Pipeline
from micro_llm.store import Store
from micro_llm.systems import Systems


def handle(store, systems, result, thread=None, account=None):
    """Steps 2 and 3 for one message on a thread. result is step 1's output, or None for no text."""
    thread = store.open(thread)
    if account:
        store.connect(thread, account)
    account = store.account(thread)
    planned = []
    for step in decide.plan(result, account) if result else []:
        status = "ready" if decide.ready(step, account) else "waiting"
        if store.add(thread, step, status):
            planned.append((step, status))
    released = store.release(thread) if account else []
    acted = []
    for action, inputs in store.ready(thread):
        start = len(systems.calls)
        outcome = act.ACTIONS[action](inputs, account, systems)
        store.finish(thread, action, outcome, systems.calls[start:])
        acted.append((action, outcome))
    return {
        "thread": thread,
        "account": account,
        "planned": planned,
        "released": released,
        "acted": acted,
        "waiting": store.waiting(thread),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "text", nargs="*", help="the message; the sample ticket when no thread is given"
    )
    parser.add_argument("--thread", type=int, help="the thread this message continues")
    parser.add_argument("--account", help="the account the customer verified")
    args = parser.parse_args()
    text = " ".join(args.text)
    if not text and args.thread is None:
        text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]

    result = Pipeline().run(text) if text else None
    systems = Systems()
    store = Store(STORE_FILE)
    try:
        out = handle(store, systems, result, args.thread, args.account)
    except ValueError as e:
        sys.exit(f"error: {e}")
    finally:
        store.close()

    who = (
        f"account {out['account']} verified"
        if out["account"]
        else "no verified account"
    )
    print(f"thread {out['thread']}, {who}")
    if result:
        print(
            f"step 1  pieces {len(result['pieces'])}, entities {len(result['entities'])}, "
            f"model runs {result['model_runs']}"
        )
    for step, status in out["planned"]:
        print(f"step 2  {step.action:15} {status:13} {step.reason}")
    for action in out["released"]:
        print(f"step 2  {action:15} {'ready':13} released now the account is verified")
    for action, outcome in out["acted"]:
        print(f"step 3  {action:15} {outcome.status:13} {outcome.did}")
    if not (out["planned"] or out["released"] or out["acted"]):
        print("step 2  nothing new to do on this thread")
    reply = " ".join(o.reply for _, o in out["acted"])
    if reply:
        print(f"reply   {reply}")
    if out["waiting"]:
        print(f"{out['waiting']} steps wait for a verified account.")
        sent = [
            c["account"] for c in systems.calls if c["tool"] == "send_verification_link"
        ]
        if sent:
            print(
                "Once the link is opened, continue with:\n"
                f'python3 -m micro_llm.resolve --thread {out["thread"]} --account {sent[0]} "I confirmed it\'s me."'
            )


if __name__ == "__main__":
    main()
