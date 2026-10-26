"""Trains the 7 learned models on the synthetic tickets and saves them to models/.

python -m micro_llm.train
"""

import json
import random
import time

from micro_llm.config import SEED, TEST_SHARE, TICKETS_FILE
from micro_llm.models import PairModel, TextModel, save
from micro_llm.questions import qtype
from micro_llm.routing import MODEL_QUESTIONS, is_issue, specialists

SPECIALISTS = ["billing", "account", "bug", "feature"]


def load_records(path=TICKETS_FILE):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_records(records):
    """Train and test tickets, split by ticket so no ticket has pieces on both sides."""
    ids = sorted(r["id"] for r in records)
    random.Random(SEED).shuffle(ids)
    test = set(ids[: round(len(ids) * TEST_SHARE)])
    return [r for r in records if r["id"] not in test], [
        r for r in records if r["id"] in test
    ]


def default(q):
    """A specialist's answer for a piece outside its team: no, not stated, or level 0."""
    return {"noul": False, "choice": "not_stated"}.get(qtype(q), 0)


def owns(name, labels):
    return name in specialists(labels["piece_kind"], labels["primary_team"])


def model_data(name, records, owned_only=False):
    """One model's inputs and per-question labels.

    A specialist trains on every issue piece. Pieces outside its team carry default labels,
    so a misrouted piece gets "no" answers instead of guesses.
    """
    if name == "linker":
        rows = [
            (
                (r["pieces"][x["a"]]["text"], r["pieces"][x["b"]]["text"]),
                {"issues_related": x["issues_related"]},
            )
            for r in records
            for x in r["pairs"]
        ]
    else:
        pieces = [p for r in records for p in r["pieces"]]
        if name in SPECIALISTS:
            pieces = [p for p in pieces if is_issue(p["labels"]["piece_kind"])]
            if owned_only:
                pieces = [p for p in pieces if owns(name, p["labels"])]
        rows = [
            (
                p["text"],
                {
                    q: p["labels"].get(q, default(q))
                    if name not in SPECIALISTS or owns(name, p["labels"])
                    else default(q)
                    for q in MODEL_QUESTIONS[name]
                },
            )
            for p in pieces
        ]
    return [x for x, _ in rows], {
        q: [y[q] for _, y in rows] for q in MODEL_QUESTIONS[name]
    }


def train(records):
    models = {}
    for name in MODEL_QUESTIONS:
        inputs, labels = model_data(name, records)
        models[name] = (PairModel() if name == "linker" else TextModel(name)).fit(
            inputs, labels
        )
    return models


def main():
    train_set, test_set = split_records(load_records())
    started = time.perf_counter()
    models = train(train_set)
    save(models)
    print(
        f"Trained {len(models)} models on {len(train_set)} tickets in {time.perf_counter() - started:.1f} s; "
        f"{len(test_set)} tickets held out"
    )


if __name__ == "__main__":
    main()
