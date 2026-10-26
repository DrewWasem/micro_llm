"""Does the micro-vs-Claude gap depend on the scoring rule?

stats.py scores flow, intent and the yes/no questions by balanced accuracy and our 3 choice
questions by plain accuracy. This rescores the same rows three ways (as reported, plain
accuracy throughout, balanced accuracy throughout), each with a joint bootstrap over chats,
and splits each yes/no score into recall on "yes" and recall on "no". It also stores the
committed-chat counts behind robustness.py, so no figure has to be rounded twice.

python3 -I metrics.py <out_dir> [model=haiku] [reps=10000]

Writes <out_dir>/metrics_<model>.json. No model calls; it costs $0.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

OUT = Path(sys.argv[1])
MODEL = sys.argv[2] if len(sys.argv) > 2 else "haiku"
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
QUESTIONS = ["flow", "intent"] + CHOICE + YES_NO
SYSTEMS = ["micro_abcd", "claude"]
RULES = {
    "as_reported": lambda q: "accuracy" if q in CHOICE else "balanced",
    "plain_throughout": lambda q: "accuracy",
    "balanced_throughout": lambda q: "balanced",
}

rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
rng = np.random.default_rng(0)
joint_idx = rng.integers(0, len(rows), size=(REPS, len(rows)))


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


def score(t, c, k, kind):
    """Accuracy or balanced accuracy per row; t = -1 marks a chat with no truth."""
    keep = t >= 0
    if kind == "accuracy":
        return (c & keep).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
    reps = t.shape[0]
    t = np.where(keep, t, k)
    flat = (t + (k + 1) * np.arange(reps)[:, None]).ravel()
    total = np.bincount(flat, minlength=reps * (k + 1)).reshape(reps, k + 1)[:, :-1]
    hits = np.bincount(
        flat, weights=(c & keep).ravel(), minlength=reps * (k + 1)
    ).reshape(reps, k + 1)[:, :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(total > 0, hits / np.maximum(total, 1), np.nan)
    return np.nanmean(recall, axis=1)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


per_q, point, boot = {}, {}, {}
for q in QUESTIONS:
    graded = [r for r in rows if r["truth"].get(q) is not None]
    truths = [norm(q, r["truth"][q]) for r in graded]
    classes = sorted(set(truths))
    pos = {r["id"]: i for i, r in enumerate(rows)}
    t_all = np.full(len(rows), -1)
    for r, x in zip(graded, truths):
        t_all[pos[r["id"]]] = classes.index(x)
    correct = {}
    for s in SYSTEMS:
        c_all = np.zeros(len(rows), dtype=bool)
        for r, x in zip(graded, truths):
            c_all[pos[r["id"]]] = norm(q, (r.get(s) or {}).get(q)) == x
        correct[s] = c_all
    a, b = correct["micro_abcd"], correct["claude"]
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    row = {
        "n": len(graded),
        "correct": {s: int(correct[s].sum()) for s in SYSTEMS},
        "accuracy": {
            s: round(float(correct[s].sum()) / len(graded), 6) for s in SYSTEMS
        },
        "only_micro_right": only_a,
        "only_claude_right": only_b,
        "mcnemar_p": float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)
        if only_a + only_b
        else 1.0,
    }
    if q in YES_NO:
        for label in ("yes", "no"):
            mask = t_all == classes.index(label)
            row[f"recall_{label}"] = {
                s: [int((correct[s] & mask).sum()), int(mask.sum())] for s in SYSTEMS
            }
    per_q[q] = row
    for name, rule in RULES.items():
        kind = rule(q)
        for s in SYSTEMS:
            point.setdefault(name, {}).setdefault(s, []).append(
                float(score(t_all[None, :], correct[s][None, :], len(classes), kind)[0])
            )
            boot.setdefault(name, {}).setdefault(s, []).append(
                score(t_all[joint_idx], correct[s][joint_idx], len(classes), kind)
            )

for q, p in zip(QUESTIONS, holm([per_q[q]["mcnemar_p"] for q in QUESTIONS])):
    per_q[q]["mcnemar_p_holm"] = round(p, 4)
    per_q[q]["mcnemar_p"] = round(per_q[q]["mcnemar_p"], 6)

means = {}
for name in RULES:
    m = {s: statistics.mean(point[name][s]) for s in SYSTEMS}
    bm = {s: np.mean(boot[name][s], axis=0) for s in SYSTEMS}
    means[name] = {
        "micro_abcd": round(m["micro_abcd"], 4),
        "claude": round(m["claude"], 4),
        "ci95": {s: ci(bm[s]) for s in SYSTEMS},
        "diff_points": round(100 * (m["micro_abcd"] - m["claude"]), 1),
        "diff_ci95_points": [
            round(100 * x, 1) for x in ci(bm["micro_abcd"] - bm["claude"])
        ],
    }

pooled = {
    label: {
        s: [sum(per_q[q][f"recall_{label}"][s][i] for q in YES_NO) for i in (0, 1)]
        for s in SYSTEMS
    }
    for label in ("yes", "no")
}
claude_ahead_plain = [
    q
    for q in YES_NO
    if per_q[q]["accuracy"]["claude"] > per_q[q]["accuracy"]["micro_abcd"]
]

# The committed-chat check from robustness.py, kept as counts
committed = {}
for q in ["flow", "intent"] + CHOICE:
    takes = {r["truth"][q] for r in rows if r["truth"].get(q) is not None}
    graded = [r for r in rows if r["truth"].get(q) is not None]
    kept = [
        r for r in graded
        if isinstance((r["claude"] or {}).get(q), str) and r["claude"][q].strip() in takes
    ]  # fmt: skip
    hm = sum(r["micro_abcd"][q] == r["truth"][q] for r in kept)
    hc = sum(r["claude"][q].strip() == r["truth"][q] for r in kept)
    committed[q] = {
        "graded": len(graded),
        "committed": len(kept),
        "micro_correct": hm,
        "claude_correct": hc,
        "diff_points": round(100 * (hm - hc) / len(kept), 1),
    }

out = {
    "reps": REPS,
    "means": means,
    "questions": per_q,
    "yes_no_pooled_recall": pooled,
    "yes_no_claude_ahead_on_plain_accuracy": claude_ahead_plain,
    "mcnemar_holm_005": {
        "micro_ahead": [
            q
            for q in QUESTIONS
            if per_q[q]["only_micro_right"] > per_q[q]["only_claude_right"]
            and per_q[q]["mcnemar_p_holm"] < 0.05
        ],
        "claude_ahead": [
            q
            for q in QUESTIONS
            if per_q[q]["only_claude_right"] > per_q[q]["only_micro_right"]
            and per_q[q]["mcnemar_p_holm"] < 0.05
        ],
    },  # fmt: skip
    "committed": committed,
}
(OUT / f"metrics_{MODEL}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("means", "yes_no_pooled_recall",
      "yes_no_claude_ahead_on_plain_accuracy", "mcnemar_holm_005", "committed")}, indent=1))  # fmt: skip
