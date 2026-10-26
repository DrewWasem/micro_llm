"""Scores the trained models on held-out tickets, then runs the sample ticket.

Every score sits next to its majority-class baseline: the accuracy of always giving the
most common answer. Results go to models/eval.json.

    python -m micro_llm.evaluate
"""

import json
import statistics
from collections import Counter

from micro_llm import splitter
from micro_llm.config import MODELS_DIR, SAMPLE_TICKET
from micro_llm.models import load, to_answer
from micro_llm.pipeline import Pipeline
from micro_llm.questions import qtype
from micro_llm.routing import MODEL_QUESTIONS, specialists
from micro_llm.rules import TICKET_QUESTIONS, answer
from micro_llm.train import default, load_records, model_data, split_records

# The generator's entity names, as the entity finder names them
TRUTH_KIND = {"amount": "money", "order_date": "date", "card": "card_number"}
ENTITY_QUESTIONS = [
    "likely_apple_user",
    "charge_amount_given",
    "order_date_given",
    "error_message_given",
    "device_or_browser_given",
    "identity_details_given",
    "sensitive_data_present",
]
MODEL_OWNED = {q for qs in MODEL_QUESTIONS.values() for q in qs}


def hard(q, a):
    """An answer as one value: yes or no, an option, or a whole level."""
    if qtype(q) == "noul":
        return a >= 0.5
    return a["answer"] if qtype(q) == "choice" else round(a["level"])


def score(q, truth, pred):
    """Accuracy against the majority-class baseline, plus balanced accuracy for yes/no questions."""
    truth_h, pred_h = [hard(q, t) for t in truth], [hard(q, p) for p in pred]
    n = len(truth_h)
    row = {
        "n": n,
        "accuracy": round(sum(t == p for t, p in zip(truth_h, pred_h)) / n, 3),
        "baseline": round(Counter(truth_h).most_common(1)[0][1] / n, 3),
    }
    if qtype(q) == "noul":
        pos = [p for t, p in zip(truth_h, pred_h) if t]
        neg = [not p for t, p in zip(truth_h, pred_h) if not t]
        if pos and neg:
            row["balanced"] = round((sum(pos) / len(pos) + sum(neg) / len(neg)) / 2, 3)
    if qtype(q) == "score":
        row["mae"] = round(
            statistics.mean(abs(t["level"] - p["level"]) for t, p in zip(truth, pred)),
            3,
        )
    return row


def model_scores(models, test):
    """Each model on the held-out pieces it is meant for: specialists only on their own team's pieces."""
    out = {}
    for name, model in models.items():
        inputs, labels = model_data(name, test, owned_only=True)
        if not inputs:
            continue
        preds = model.predict(inputs)
        out[name] = {
            q: score(
                q, [to_answer(q, {y: 1.0}) for y in labels[q]], [p[q] for p in preds]
            )
            for q in MODEL_QUESTIONS[name]
        }
    return out


def truth_answers(record):
    """The ticket answers that perfect models would produce under the same rules."""
    pieces, kinds = [], []
    for p in record["pieces"]:
        labels = p["labels"]

        def one(q):
            return to_answer(q, {labels.get(q, default(q)): 1.0})

        piece = {
            name: {q: one(q) for q in MODEL_QUESTIONS[name]}
            for name in ("router", "tone")
        }
        for name in specialists(labels["piece_kind"], labels["primary_team"]):
            piece[name] = {q: one(q) for q in MODEL_QUESTIONS[name]}
        pieces.append(piece)
        kinds.append({TRUTH_KIND.get(k, k) for k in p["entities"]})
    pairs = [float(x["issues_related"]) for x in record["pairs"]]
    return answer(pieces, pairs, kinds, record["truth"]["likely_apple_user"])


def splitter_scores(test):
    """How often the splitter cuts each non-greeting piece exactly where the generator joined it."""
    found = total = whole = 0
    for r in test:
        want = {
            (p["start"], p["end"])
            for p in r["pieces"]
            if p["labels"]["piece_kind"] != "greeting"
        }
        got = {(p.start, p.end) for p in splitter.split(r["text"])}
        found += len(want & got)
        total += len(want)
        whole += want <= got
    return {
        "pieces_exact": round(found / total, 3),
        "tickets_all_exact": round(whole / len(test), 3),
    }


def sample_comparison(pipeline):
    """The sample ticket's answers next to the predictions written before any model existed."""
    text = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    result = pipeline.run(text)
    expected = json.loads(SAMPLE_TICKET.with_name("my_predictions.json").read_text())[
        "predictions"
    ]
    rows = {}
    for q, e in expected.items():
        a = result["answers"][q]
        got = (
            a
            if qtype(q) == "noul"
            else a["answer"]
            if qtype(q) == "choice"
            else a["level"]
        )
        gap = (
            (got != e["expect"])
            if qtype(q) == "choice"
            else round(abs(got - e["expect"]), 2)
        )
        rows[q] = {"expect": e["expect"], "got": got, "gap": gap}
    return result, rows


def main():
    models = load()
    _, test = split_records(load_records())
    pipeline = Pipeline(models)
    pipeline.run("warm up")

    truth, pred, latency = (
        {q: [] for q in TICKET_QUESTIONS},
        {q: [] for q in TICKET_QUESTIONS},
        [],
    )
    for r in test:
        result = pipeline.run(r["text"])
        latency.append(result["ms"]["total"])
        for q, a in truth_answers(r).items():
            truth[q].append(a)
            pred[q].append(result["answers"][q])
    ticket = {q: score(q, truth[q], pred[q]) for q in TICKET_QUESTIONS}
    result, sample = sample_comparison(pipeline)

    report = {
        "test_tickets": len(test),
        "models": model_scores(models, test),
        "splitter": splitter_scores(test),
        "ticket_answers": ticket,
        "latency_ms": {
            "median": round(statistics.median(latency), 2),
            "p95": round(sorted(latency)[int(0.95 * (len(latency) - 1))], 2),
        },
        "sample": {
            "pieces": result["pieces"],
            "model_runs": result["model_runs"],
            "ms": result["ms"],
            "vs_predictions": sample,
        },
    }
    (MODELS_DIR / "eval.json").write_text(json.dumps(report, indent=1))

    for name, qs in report["models"].items():
        cells = ", ".join(
            f"{q} {s['accuracy']:.2f}/{s['baseline']:.2f}" for q, s in qs.items()
        )
        print(f"{name:8} accuracy/baseline: {cells}")
    print(f"splitter: {report['splitter']}")
    for group, qs in [
        ("from models", [q for q in TICKET_QUESTIONS if q in MODEL_OWNED]),
        (
            "from code rules",
            [
                q
                for q in TICKET_QUESTIONS
                if q not in MODEL_OWNED and q not in ENTITY_QUESTIONS
            ],
        ),
        ("from entity rules", ENTITY_QUESTIONS),
    ]:
        acc = statistics.mean(ticket[q]["accuracy"] for q in qs)
        base = statistics.mean(ticket[q]["baseline"] for q in qs)
        worst = sorted(qs, key=lambda q: ticket[q]["accuracy"] - ticket[q]["baseline"])[
            :3
        ]
        print(
            f"ticket answers {group} ({len(qs)}): accuracy {acc:.3f} vs baseline {base:.3f}; weakest {worst}"
        )
    print(
        f"latency per ticket: {report['latency_ms']} over {len(test)} held-out tickets"
    )
    choices = [r for q, r in sample.items() if qtype(q) == "choice"]
    numbers = [r["gap"] for q, r in sample.items() if qtype(q) != "choice"]
    print(
        f"sample ticket: {result['model_runs']} model runs in {result['ms']['total']} ms; "
        f"{sum(not r['gap'] for r in choices)}/{len(choices)} choices match my predictions; "
        f"mean gap on the other {len(numbers)} answers {statistics.mean(numbers):.2f}"
    )


if __name__ == "__main__":
    main()
