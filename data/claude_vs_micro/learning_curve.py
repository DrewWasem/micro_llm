"""How many labeled chats do the heads need to catch Claude? A learning curve on ABCD.

Retrains compare.py's 15 heads (same features, same LogisticRegression) on nested random
subsets of ABCD's 8,034 training chats and scores each on the same 300 test openings, with
the three scoring rules in metrics.py. Claude's zero-shot score is a flat line read from
rows_<model>.json. The full-size run must reproduce compare.py's heads answer for answer,
which checks this code against the reported numbers.

python3 -I learning_curve.py <repo> <out_dir> [model=haiku] [seeds=5]

Writes <out_dir>/learning_curve_<model>.json and prints a table. No model calls; it costs $0.
"""

import gzip
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from micro_llm import abcd  # noqa: E402
from micro_llm.models import text_features  # noqa: E402

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
OURS = CHOICE + YES_NO
QUESTIONS = ["flow", "intent"] + OURS
RULES = {
    "as_reported": lambda q: "accuracy" if q in CHOICE else "balanced",
    "plain_throughout": lambda q: "accuracy",
    "balanced_throughout": lambda q: "balanced",
}
SIZES = [50, 100, 200, 383, 1000, 2000, 4000]  # plus the full training set, run once
SYNTHETIC_SIZE = (
    383  # tickets the synthetic system trained on (micro_llm/train.py split)
)

data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
train, test = data["train"], data["test"]
rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
by_id = {c["convo_id"]: c for c in test}
sample = [by_id[r["id"]] for r in rows]
assert [c["convo_id"] for c in random.Random(0).sample(test, len(rows))] == [
    r["id"] for r in rows
], "rows_<model>.json is not compare.py's seed-0 sample"


def truth(c):
    """Same as compare.py."""
    t = abcd.our_truth(c, "opening")
    return {
        "flow": c["scenario"]["flow"],
        "intent": abcd.intent(c),
        **{q: t[q] for q in OURS},
    }


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


def score(q, truths, preds, kind):
    """Accuracy, or balanced accuracy: the mean recall over the classes the truth holds."""
    pairs = [(norm(q, t), norm(q, p)) for t, p in zip(truths, preds) if t is not None]
    if kind == "accuracy":
        return sum(t == p for t, p in pairs) / len(pairs)
    recall = {}
    for t, p in pairs:
        recall.setdefault(t, []).append(t == p)
    return statistics.mean(statistics.mean(v) for v in recall.values())


def means(preds, qs):
    return {
        rule: statistics.mean(score(q, gold[q], preds[q], pick(q)) for q in qs)
        for rule, pick in RULES.items()
    }


def heads(X, Xs, truths):
    """compare.py's head() for every question; a subset holding one answer always gives it."""
    answers = {}
    for q in QUESTIONS:
        keep = [i for i, t in enumerate(truths) if t[q] is not None]
        labels = [truths[i][q] for i in keep]
        if len(set(labels)) < 2:
            answers[q] = [labels[0] if labels else None] * Xs.shape[0]
            continue
        model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[keep], labels)
        answers[q] = model.predict(Xs).tolist()
    return answers


started = time.perf_counter()
train_text = [abcd.customer_text(c, "opening") for c in train]
train_truth = [truth(c) for c in train]
sample_text = [abcd.customer_text(c, "opening") for c in sample]
gold = {q: [r["truth"].get(q) for r in rows] for q in QUESTIONS}

# Nested subsets: each seed draws one order of the training chats and takes its first k.
# Chats keep their file order inside a subset, so the full set matches compare.py exactly.
runs = [{"size": len(train), "seed": None, "idx": list(range(len(train)))}]
for seed in range(SEEDS):
    order = random.Random(seed).sample(range(len(train)), len(train))
    runs += [{"size": k, "seed": seed, "idx": sorted(order[:k])} for k in SIZES]

jobs = []
for r in runs:
    f = text_features()
    X = f.fit_transform([train_text[i] for i in r["idx"]])
    jobs.append((X, f.transform(sample_text), [train_truth[i] for i in r["idx"]]))
answers = joblib.Parallel(n_jobs=-1)(joblib.delayed(heads)(*j) for j in jobs)

full = answers[0]
mismatches = {
    q: sum(a != (r["micro_abcd"] or {}).get(q) for a, r in zip(full[q], rows))
    for q in QUESTIONS
}
assert not any(mismatches.values()), (
    f"full-size heads differ from compare.py: {mismatches}"
)

claude = {q: [(r["claude"] or {}).get(q) for r in rows] for q in QUESTIONS}
syn = {q: [(r["micro_syn"] or {}).get(q) for r in rows] for q in OURS}
reference = {
    "claude_15": means(claude, QUESTIONS),
    "claude_13_shared": means(claude, OURS),
    "micro_syn_13_shared": means(syn, OURS),
}

results = []
for r, a in zip(runs, answers):
    results.append(
        {
            "size": r["size"],
            "seed": r["seed"],
            "mean_15": means(a, QUESTIONS),
            "mean_13_shared": means(a, OURS),
            "questions_as_reported": {
                q: round(score(q, gold[q], a[q], RULES["as_reported"](q)), 4)
                for q in QUESTIONS
            },
        }
    )


def spread(xs):
    return {
        "mean": round(statistics.mean(xs), 4),
        "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


curve = []
for k in SIZES + [len(train)]:
    at = [x for x in results if x["size"] == k]
    claude_q = {
        q: score(q, gold[q], claude[q], RULES["as_reported"](q)) for q in QUESTIONS
    }
    curve.append(
        {
            "size": k,
            "runs": len(at),
            "mean_15": {
                rule: spread([x["mean_15"][rule] for x in at]) for rule in RULES
            },
            "mean_13_shared": {
                rule: spread([x["mean_13_shared"][rule] for x in at]) for rule in RULES
            },
            "questions_ahead_of_claude_as_reported": spread(
                [
                    sum(x["questions_as_reported"][q] > claude_q[q] for q in QUESTIONS)
                    for x in at
                ]
            ),
        }
    )

catch_up = {
    rule: next(
        (
            c["size"]
            for c in curve
            if c["mean_15"][rule]["mean"] > reference["claude_15"][rule]
        ),
        None,
    )
    for rule in RULES
}
out = {
    "model": MODEL,
    "seeds": SEEDS,
    "train_chats": len(train),
    "test_openings": len(rows),
    "synthetic_training_tickets": SYNTHETIC_SIZE,
    "full_size_mismatches_with_compare_py": sum(mismatches.values()),
    "reference": {
        k: {rule: round(v, 4) for rule, v in m.items()} for k, m in reference.items()
    },
    "smallest_size_whose_seed_mean_beats_claude": catch_up,
    "curve": curve,
    "runs": results,
    "seconds": round(time.perf_counter() - started, 1),
}
(OUT / f"learning_curve_{MODEL}.json").write_text(json.dumps(out, indent=1))

pct = lambda s: f"{100 * s['mean']:5.1f} ±{100 * s['sd']:3.1f}"  # noqa: E731
print(
    f"{'chats':>6} {'runs':>4}  {'reported':>11} {'plain':>11} {'balanced':>11}  {'13 shared':>11}"
)
for c in curve:
    m = c["mean_15"]
    print(f"{c['size']:6} {c['runs']:4}  {pct(m['as_reported'])} {pct(m['plain_throughout'])} "
          f"{pct(m['balanced_throughout'])}  {pct(c['mean_13_shared']['as_reported'])}")  # fmt: skip
ref = out["reference"]
print(f"claude      {100 * ref['claude_15']['as_reported']:11.1f} {100 * ref['claude_15']['plain_throughout']:11.1f} "
      f"{100 * ref['claude_15']['balanced_throughout']:11.1f}  {100 * ref['claude_13_shared']['as_reported']:11.1f}")  # fmt: skip
print(
    f"micro_syn (383 synthetic tickets), 13 shared: {100 * ref['micro_syn_13_shared']['as_reported']:.1f}"
)
print(f"smallest size beating Claude: {catch_up}; {out['seconds']} s")
