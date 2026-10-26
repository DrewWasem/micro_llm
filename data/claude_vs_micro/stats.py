"""Confidence intervals and paired significance tests for the Claude vs micro comparison.

Reads the per-chat rows that compare.py wrote and scores them with the same rules:
balanced accuracy for flow, intent and the yes/no questions, plain accuracy for our 3
choice questions, and any malformed or missing answer counts as wrong.

python3 -I stats.py <out_dir> [model=haiku] [reps=10000]

Writes <out_dir>/stats_<model>.json and prints a table. No model calls; it costs $0.
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
ABCD_QS = ["flow", "intent"]
QUESTIONS = ABCD_QS + CHOICE + YES_NO
SYSTEMS = ["micro_abcd", "claude", "micro_syn"]

rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
replies = {}
for line in (OUT / f"claude_{MODEL}.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("answers") is not None:
        replies[r["id"]] = r  # the last good reply per chat wins, as in compare.py
replies = [replies[r["id"]] for r in rows]
rng = np.random.default_rng(0)


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


def metric(q):
    return "accuracy" if q in CHOICE else "balanced"


def score(truth_ids, correct, k, kind):
    """Accuracy or balanced accuracy for each row of a 2-D correctness matrix.

    truth_ids: (reps, n) class index of each sampled chat; correct: (reps, n) bool.
    Balanced accuracy is the mean recall over the classes present in that row, which is
    what sklearn's balanced_accuracy_score does when a prediction names an unseen class.
    """
    if kind == "accuracy":
        return correct.mean(axis=1)
    reps = truth_ids.shape[0]
    flat = (truth_ids + k * np.arange(reps)[:, None]).ravel()
    total = np.bincount(flat, minlength=reps * k).reshape(reps, k)
    hits = np.bincount(flat, weights=correct.ravel(), minlength=reps * k).reshape(
        reps, k
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(total > 0, hits / np.maximum(total, 1), np.nan)
    return np.nanmean(recall, axis=1)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


results, boot_diffs, perm_p, points = {}, {}, {}, {}
for q in QUESTIONS:
    graded = [r for r in rows if r["truth"].get(q) is not None]
    truths = [norm(q, r["truth"][q]) for r in graded]
    classes = sorted(set(truths))
    k = len(classes)
    t = np.array([classes.index(x) for x in truths])
    n = len(t)
    correct, invalid = {}, {}
    for s in SYSTEMS:
        if graded[0].get(s) is None or q not in (graded[0].get(s) or {}):
            continue
        preds = [norm(q, (r.get(s) or {}).get(q)) for r in graded]
        correct[s] = np.array([p == x for p, x in zip(preds, truths)])
        invalid[s] = sum(
            p == "malformed" or (q not in YES_NO and p not in classes) for p in preds
        )
    kind = metric(q)
    idx = rng.integers(0, n, size=(REPS, n))
    row = {"n": n, "metric": kind, "classes": k, "systems": {}}
    boots, points[q] = {}, {}
    for s, c in correct.items():
        points[q][s] = float(score(t[None, :], c[None, :], k, kind)[0])
        boots[s] = score(t[idx], c[idx], k, kind)
        row["systems"][s] = {
            "score": round(points[q][s], 4),
            "ci95": ci(boots[s]),
            "answers_outside_labels": invalid[s],
        }
    a, b = correct["micro_abcd"], correct["claude"]
    # Subtract unrounded scores, so the reported gap is the rounded true gap
    diff = points[q]["micro_abcd"] - points[q]["claude"]
    boot_diffs[q] = boots["micro_abcd"] - boots["claude"]
    # Paired approximate randomization: swap the two systems' answers on a random half of chats
    swap = rng.random((REPS, n)) < 0.5
    pa = score(np.broadcast_to(t, (REPS, n)), np.where(swap, b, a), k, kind)
    pb = score(np.broadcast_to(t, (REPS, n)), np.where(swap, a, b), k, kind)
    perm_p[q] = float((np.sum(np.abs(pa - pb) >= abs(diff) - 1e-12) + 1) / (REPS + 1))
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    row["micro_abcd_minus_claude"] = {
        "points": round(100 * diff, 1),
        "ci95_points": [round(100 * x, 1) for x in ci(boot_diffs[q])],
        "permutation_p": round(perm_p[q], 4),
        "mcnemar": {
            "only_micro_right": only_a,
            "only_claude_right": only_b,
            "p": round(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue, 4)
            if only_a + only_b
            else 1.0,
        },
    }
    results[q] = row

adjusted = holm([perm_p[q] for q in QUESTIONS])
for q, p in zip(QUESTIONS, adjusted):
    results[q]["micro_abcd_minus_claude"]["permutation_p_holm"] = round(p, 4)

# One headline: the mean over the 15 questions, resampling chats jointly
# (each question's bootstrap reused the same rng stream, so recompute jointly here).
# A fresh seed-0 generator makes the same draw as metrics.py, so both files agree.
joint_idx = np.random.default_rng(0).integers(0, len(rows), size=(REPS, len(rows)))
pos = {r["id"]: i for i, r in enumerate(rows)}
macro = {"micro_abcd": [], "claude": []}
for q in QUESTIONS:
    graded = [r for r in rows if r["truth"].get(q) is not None]
    truths = [norm(q, r["truth"][q]) for r in graded]
    classes = sorted(set(truths))
    t_all = np.full(len(rows), -1)
    for r, x in zip(graded, truths):
        t_all[pos[r["id"]]] = classes.index(x)
    for s in macro:
        c_all = np.zeros(len(rows), dtype=bool)
        for r, x in zip(graded, truths):
            c_all[pos[r["id"]]] = norm(q, (r.get(s) or {}).get(q)) == x
        ti, ci_ = t_all[joint_idx], c_all[joint_idx]
        keep = ti >= 0
        # Drop ungraded chats by giving them their own class with no weight
        k = len(classes) + 1
        ti = np.where(keep, ti, len(classes))
        if metric(q) == "accuracy":
            vals = (ci_ & keep).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
        else:
            reps = ti.shape[0]
            flat = (ti + k * np.arange(reps)[:, None]).ravel()
            total = np.bincount(flat, minlength=reps * k).reshape(reps, k)[:, :-1]
            hits = np.bincount(
                flat, weights=(ci_ & keep).ravel(), minlength=reps * k
            ).reshape(reps, k)[:, :-1]
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = np.nanmean(
                    np.where(total > 0, hits / np.maximum(total, 1), np.nan), axis=1
                )
        macro[s].append(vals)
mean_point = {s: statistics.mean(points[q][s] for q in QUESTIONS) for s in macro}
macro_points = {s: round(v, 4) for s, v in mean_point.items()}
m_a, m_c = np.mean(macro["micro_abcd"], axis=0), np.mean(macro["claude"], axis=0)
summary = {
    "questions": len(QUESTIONS),
    "reps": REPS,
    "mean_score": macro_points,
    "mean_ci95": {"micro_abcd": ci(m_a), "claude": ci(m_c)},
    "mean_diff_points": round(100 * (mean_point["micro_abcd"] - mean_point["claude"]), 1),
    "mean_diff_ci95_points": [round(100 * x, 1) for x in ci(m_a - m_c)],
    "micro_ahead_holm_005": [
        q for q in QUESTIONS
        if results[q]["micro_abcd_minus_claude"]["points"] > 0
        and results[q]["micro_abcd_minus_claude"]["permutation_p_holm"] < 0.05
    ],
    "claude_ahead_holm_005": [
        q for q in QUESTIONS
        if results[q]["micro_abcd_minus_claude"]["points"] < 0
        and results[q]["micro_abcd_minus_claude"]["permutation_p_holm"] < 0.05
    ],
}  # fmt: skip

# Claude's time, tokens and cost per chat
wall = [r["wall_s"] for r in replies]
api = [(r.get("api_ms") or 0) / 1000 for r in replies]
out_tok = [r["usage"]["output_tokens"] for r in replies]
think = [
    (r["usage"].get("output_tokens_details") or {}).get("thinking_tokens", 0)
    for r in replies
]
in_tok = [
    r["usage"]["input_tokens"]
    + r["usage"].get("cache_read_input_tokens", 0)
    + r["usage"].get("cache_creation_input_tokens", 0)
    for r in replies
]
cost = [r.get("cost") or 0 for r in replies]


def pcts(x):
    p50, p90, p99 = np.percentile(x, [50, 90, 99])
    return {"p50": round(float(p50), 2), "p90": round(float(p90), 2), "p99": round(float(p99), 2),
            "min": round(float(min(x)), 2), "max": round(float(max(x)), 2)}  # fmt: skip


claude_cost = {
    "chats": len(replies),
    "wall_s": pcts(wall),
    "api_s": pcts(api),
    "output_tokens": pcts(out_tok),
    "thinking_tokens": pcts(think),
    "thinking_share_of_output": round(sum(think) / sum(out_tok), 4),
    "input_tokens": pcts(in_tok),
    "cache_read_calls": sum(
        1 for r in replies if r["usage"].get("cache_read_input_tokens", 0) > 0
    ),
    "cost_usd": {
        "total": round(sum(cost), 4),
        "mean": round(statistics.mean(cost), 6),
        "p50": round(float(np.percentile(cost, 50)), 6),
    },  # fmt: skip
}

(OUT / f"stats_{MODEL}.json").write_text(
    json.dumps(
        {"summary": summary, "claude_cost": claude_cost, "questions": results}, indent=1
    )
)
print(json.dumps(summary, indent=1))
print(json.dumps(claude_cost, indent=1))
print(
    f"\n{'question':24} {'n':>4} {'metric':>8} {'micro (95% CI)':>20} {'claude (95% CI)':>20} {'diff':>6} {'diff CI':>14} {'p_holm':>7}"
)
for q, r in results.items():
    m, c, d = (
        r["systems"]["micro_abcd"],
        r["systems"]["claude"],
        r["micro_abcd_minus_claude"],
    )
    f = (
        lambda s: f"{100 * s['score']:.1f} ({100 * s['ci95'][0]:.0f}-{100 * s['ci95'][1]:.0f})"
    )  # noqa: E731
    print(f"{q:24} {r['n']:4} {r['metric'][:8]:>8} {f(m):>20} {f(c):>20} {d['points']:6.1f} "
          f"{str(d['ci95_points']):>14} {d['permutation_p_holm']:7.4f}  claude_bad={c['answers_outside_labels']}")  # fmt: skip
