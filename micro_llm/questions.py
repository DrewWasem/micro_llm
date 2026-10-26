"""The question bank: every answer the models and rules produce, worded as in the Jev Playground."""

import json
from functools import cache
from pathlib import Path

QUESTIONS_FILE = Path(__file__).with_name("questions.json")


@cache
def bank():
    return json.loads(QUESTIONS_FILE.read_text())


def qtype(qid):
    return bank()[qid]["type"]


def classes(qid):
    """Every value an answer can take: option names, level indexes, or False/True."""
    q = bank()[qid]
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return list(range(len(q["criteria"])))
    return [False, True]


def describe(qid, value):
    """One line telling a writer what a label means for the text."""
    q = bank()[qid]
    if q["type"] == "noul":
        return f"{'YES' if value else 'NO'}: {q['instructions']}"
    return f"{q['instructions']}: {q['criteria'][value]}"
