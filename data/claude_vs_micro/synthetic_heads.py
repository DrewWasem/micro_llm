"""Did the synthetic system lose on ABCD because of its data or because of its design?

Trains compare.py's heads (same features, same LogisticRegression) on the 383 synthetic
training tickets, one head per shared question, with each ticket's truth built the way
micro_llm.evaluate builds it. Scores them on compare.py's 300 ABCD test openings and on
the 96 held-out synthetic tickets, with the metric the paper reports: accuracy for the 3
choice questions, balanced accuracy for the 10 yes/no questions. The same design trained
on 383 ABCD chats comes from learning_curve_<model>.json, so only the data differs.

python3 -I synthetic_heads.py <repo> <out_dir> [model=haiku]

Writes <out_dir>/synthetic_heads_<model>.json and prints a table. No model calls; it costs $0.
"""

import gzip
import json
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from micro_llm import abcd  # noqa: E402
from micro_llm.evaluate import hard, truth_answers  # noqa: E402
from micro_llm.models import text_features  # noqa: E402
from micro_llm.train import load_records, split_records  # noqa: E402

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
OURS = CHOICE + YES_NO

rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
curve = json.loads((OUT / f"learning_curve_{MODEL}.json").read_text())
data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
by_id = {c["convo_id"]: c for c in data["test"]}
openings = [abcd.customer_text(by_id[r["id"]], "opening") for r in rows]


def norm(q, v):
    """Same as compare.py: one comparable string per answer; anything malformed is wrong."""
    if q in YES_NO:
        if isinstance(v, str):
            v = {"true": True, "false": False}.get(v.strip().lower(), v)
        return (
            {True: "yes", False: "no"}.get(v, "malformed")
            if isinstance(v, bool)
            else "malformed"
        )
    return v.strip() if isinstance(v, str) else "malformed"


def score(q, truths, preds):
    """Accuracy for a choice question; balanced accuracy for a yes/no question."""
    pairs = [(norm(q, t), norm(q, p)) for t, p in zip(truths, preds) if t is not None]
    if q in CHOICE:
        return sum(t == p for t, p in pairs) / len(pairs)
    recall = {}
    for t, p in pairs:
        recall.setdefault(t, []).append(t == p)
    return statistics.mean(statistics.mean(v) for v in recall.values())


def labels(records):
    """Each ticket's truth on the 13 shared questions, as micro_llm.evaluate scores it."""
    out = []
    for r in records:
        t = truth_answers(r)
        out.append({q: hard(q, t[q]) for q in OURS})
    return out


train, held = split_records(load_records())
train_truth, held_truth = labels(train), labels(held)
features = text_features()
X = features.fit_transform([r["text"] for r in train])
X_abcd = features.transform(openings)
X_held = features.transform([r["text"] for r in held])

abcd_pred, held_pred, classes = {}, {}, {}
for q in OURS:
    y = [t[q] for t in train_truth]
    classes[q] = dict(Counter(norm(q, v) for v in y))
    model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    abcd_pred[q] = [
        v.item() if hasattr(v, "item") else v for v in model.predict(X_abcd)
    ]
    held_pred[q] = [
        v.item() if hasattr(v, "item") else v for v in model.predict(X_held)
    ]

gold = {q: [r["truth"].get(q) for r in rows] for q in OURS}
abcd_classes = {
    q: dict(Counter(norm(q, v) for v in gold[q] if v is not None)) for q in OURS
}
runs_383 = [x for x in curve["runs"] if x["size"] == 383]
per_q = {}
for q in OURS:
    per_q[q] = {
        "synthetic_heads_on_abcd": round(score(q, gold[q], abcd_pred[q]), 6),
        "synthetic_heads_on_held_out_synthetic": round(
            score(q, [t[q] for t in held_truth], held_pred[q]), 6
        ),
        "synthetic_system_on_abcd": round(
            score(q, gold[q], [(r["micro_syn"] or {}).get(q) for r in rows]), 6
        ),
        "abcd_heads_383_on_abcd": round(
            statistics.mean(x["questions_as_reported"][q] for x in runs_383), 6
        ),
        "abcd_heads_8034_on_abcd": round(
            score(q, gold[q], [(r["micro_abcd"] or {}).get(q) for r in rows]), 6
        ),
        "claude_on_abcd": round(
            score(q, gold[q], [(r["claude"] or {}).get(q) for r in rows]), 6
        ),
        "answers_in_abcd_truth_never_in_synthetic_training": sorted(
            set(abcd_classes[q]) - set(classes[q])
        ),
    }
means = {
    k: round(statistics.mean(v[k] for v in per_q.values()), 6)
    for k in per_q[OURS[0]]
    if k != "answers_in_abcd_truth_never_in_synthetic_training"
}
# The questions whose ABCD answers all appear in the synthetic training labels
covered = [q for q, v in per_q.items() if not v["answers_in_abcd_truth_never_in_synthetic_training"]]
means_covered = {
    k: round(statistics.mean(per_q[q][k] for q in covered), 6) for k in means
}
out = {
    "model": MODEL,
    "synthetic_training_tickets": len(train),
    "synthetic_held_out_tickets": len(held),
    "abcd_openings": len(rows),
    "mean_13_shared": means,
    "covered_questions": covered,
    "mean_covered": means_covered,
    "questions": per_q,
    "synthetic_training_classes": classes,
    "abcd_truth_classes": abcd_classes,
}
(OUT / f"synthetic_heads_{MODEL}.json").write_text(json.dumps(out, indent=1))

cols = list(means)
print(f"{'question':24}" + "".join(f"{c[:14]:>16}" for c in cols))
for q, v in per_q.items():
    print(f"{q:24}" + "".join(f"{100 * v[c]:16.1f}" for c in cols))
print(f"{'mean of 13':24}" + "".join(f"{100 * means[c]:16.1f}" for c in cols))
print(f"{f'mean of {len(covered)} covered':24}" + "".join(f"{100 * means_covered[c]:16.1f}" for c in cols))
