"""Checks added after the second cold read of the paper.

1. Routing under two fallback rules. checks.py keeps a system's direct answer whenever its
   intent answer routes nowhere: an intent the mapping does not know, or a known intent that
   the mapping gives no answer for that question. The strict rule keeps the direct answer
   only for the first kind and counts the second kind wrong.
2. Paired intervals for the learning curve: each run's lead over Claude, from the joint
   bootstrap of stats.py (NumPy seed 0, chats resampled jointly over the 15 questions).
3. Heads trained without balanced class weights, which tilt them toward balanced accuracy.
4. The synthetic system's lead over the synthetic-trained heads, split into its two
   regular-expression answers and its 11 learned ones.
5. Misses and false flags on the 9 rare-yes questions, counted per system.

python3 -I revision_checks.py <repo> <out_dir> [model=haiku] [seeds=5] [reps=10000]

Writes <out_dir>/revision_checks_<model>.json and prints a summary. No model calls; it costs $0.
"""

import gzip
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
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
BY_INTENT = CHOICE + YES_NO[:7]
REGEX = [
    "identity_details_given",
    "charge_amount_given",
]  # the synthetic entity finder's
RARE_YES = [
    q for q in YES_NO if q != "needs_staff_action"
]  # yes is the minority on these 9
RULES = {
    "as_reported": lambda q: "accuracy" if q in CHOICE else "balanced",
    "plain_throughout": lambda q: "accuracy",
    "balanced_throughout": lambda q: "balanced",
}
SIZES = [50, 100, 200, 383, 1000, 2000, 4000]  # learning_curve.py's sizes

data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
train, test = data["train"], data["test"]
rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
by_id = {c["convo_id"]: c for c in test}
sample = [by_id[r["id"]] for r in rows]
assert [c["convo_id"] for c in random.Random(0).sample(test, len(rows))] == [
    r["id"] for r in rows
], "rows_<model>.json is not compare.py's seed-0 sample"
r6 = lambda x: round(float(x), 6)  # noqa: E731
pts = lambda x: round(100 * float(x), 6)  # noqa: E731


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


gold = {q: [r["truth"].get(q) for r in rows] for q in QUESTIONS}


def score(q, preds, kind):
    """Accuracy, or balanced accuracy: the mean recall over the classes the truth holds."""
    pairs = [(norm(q, t), norm(q, p)) for t, p in zip(gold[q], preds) if t is not None]
    if kind == "accuracy":
        return sum(t == p for t, p in pairs) / len(pairs)
    recall = {}
    for t, p in pairs:
        recall.setdefault(t, []).append(t == p)
    return statistics.mean(statistics.mean(v) for v in recall.values())


def means(preds, qs=QUESTIONS):
    return {
        rule: statistics.mean(score(q, preds[q], pick(q)) for q in qs)
        for rule, pick in RULES.items()
    }


def answers_of(s):
    return {q: [(r[s] or {}).get(q) for r in rows] for q in QUESTIONS}


claude, heads_full = answers_of("claude"), answers_of("micro_abcd")
claude_point = means(claude)
out = {"model": MODEL, "chats": len(rows), "reps": REPS, "seeds": SEEDS}
checks = json.loads((OUT / f"checks_{MODEL}.json").read_text())
metrics = json.loads((OUT / f"metrics_{MODEL}.json").read_text())

# 1. Routing under two fallback rules (checks.py's rule is "fallback")


def mapped(i):
    """Same as checks.py: what the truth mapping says for an intent, on the 10 questions."""
    m = {
        "primary_team": abcd.TEAM.get(i),
        "journey_stage": abcd.JOURNEY.get(i),
        "message_type": abcd.MESSAGE.get(i),
    }
    for q in YES_NO[:7]:
        yes, no = abcd.YES_NO[q]
        m[q] = True if i in yes else False if i in no else None
    return m


for r in rows:
    m = mapped(r["truth"]["intent"])
    assert all(m[q] == r["truth"][q] for q in BY_INTENT), r["id"]

truth_intents = {norm("intent", t) for t in gold["intent"]}
routing = {"questions": BY_INTENT, "systems": {}}
for s in ("micro_abcd", "claude"):
    direct = answers_of(s)
    intents = [norm("intent", i) for i in direct["intent"]]
    known = [i in abcd.ALL for i in intents]
    fallback, strict, pairs = {}, {}, {}
    for q in BY_INTENT:
        f, st, n_unknown, n_nowhere = [], [], 0, 0
        for j in range(len(rows)):
            graded = gold[q][j] is not None
            v = mapped(intents[j])[q] if known[j] else None
            if not known[j]:
                f.append(direct[q][j])
                st.append(direct[q][j])
                n_unknown += graded
            elif v is None:
                f.append(direct[q][j])
                st.append(
                    None
                )  # strict: the known intent settles nothing, so no answer
                n_nowhere += graded
            else:
                f.append(v)
                st.append(v)
        fallback[q], strict[q] = f, st
        pairs[q] = {
            "unknown_intent": n_unknown,
            "known_intent_routes_nowhere": n_nowhere,
        }
    per_q = {
        q: {
            "direct": r6(score(q, direct[q], RULES["as_reported"](q))),
            "routed_fallback": r6(score(q, fallback[q], RULES["as_reported"](q))),
            "routed_strict": r6(score(q, strict[q], RULES["as_reported"](q))),
        }
        for q in BY_INTENT
    }
    mean10 = {
        k: r6(statistics.mean(v[k] for v in per_q.values()))
        for k in ("direct", "routed_fallback", "routed_strict")
    }
    routing["systems"][s] = {
        "intent_known_to_mapping": sum(known),
        "intent_a_sampled_truth_holds": sum(i in truth_intents for i in intents),
        "graded_pairs_falling_back": {
            "unknown_intent": sum(p["unknown_intent"] for p in pairs.values()),
            "known_intent_routes_nowhere": sum(
                p["known_intent_routes_nowhere"] for p in pairs.values()
            ),
        },
        "pairs_by_question": pairs,
        "per_question": per_q,
        "mean_10": mean10,
    }
    # The fallback rule must reproduce checks.py's routed means
    assert (
        abs(mean10["routed_fallback"] - checks["routed"]["mean"][f"{s}_routed"]) < 1e-6
    )
    assert abs(mean10["direct"] - checks["routed"]["mean"][f"{s}_direct"]) < 1e-6

h, c = (
    routing["systems"]["micro_abcd"]["mean_10"],
    routing["systems"]["claude"]["mean_10"],
)
routing["gaps_points"] = {
    "direct": pts(h["direct"] - c["direct"]),
    "claude_routed_vs_heads_direct": {
        rule: pts(h["direct"] - c[f"routed_{rule}"]) for rule in ("fallback", "strict")
    },
    "both_routed": {
        rule: pts(h[f"routed_{rule}"] - c[f"routed_{rule}"])
        for rule in ("fallback", "strict")
    },
    "share_of_direct_gap_closed_both_routed": {
        rule: r6(
            1
            - (h[f"routed_{rule}"] - c[f"routed_{rule}"]) / (h["direct"] - c["direct"])
        )
        for rule in ("fallback", "strict")
    },
}
out["routing"] = routing

# 2. Retrain: the full set with and without class weights, then learning_curve.py's runs


def heads(X, Xs, truths, weight):
    """compare.py's head() for every question; a subset holding one answer always gives it."""
    answers = {}
    for q in QUESTIONS:
        keep = [i for i, t in enumerate(truths) if t[q] is not None]
        labels = [truths[i][q] for i in keep]
        if len(set(labels)) < 2:
            answers[q] = [labels[0] if labels else None] * Xs.shape[0]
            continue
        model = LogisticRegression(C=4.0, class_weight=weight, max_iter=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[keep], labels)
        answers[q] = model.predict(Xs).tolist()
    return answers


started = time.perf_counter()
train_text = [abcd.customer_text(c, "opening") for c in train]
train_truth = [truth(c) for c in train]
sample_text = [abcd.customer_text(c, "opening") for c in sample]
everything = list(range(len(train)))
runs = [
    {"size": len(train), "seed": None, "idx": everything, "weight": "balanced"},
    {"size": len(train), "seed": None, "idx": everything, "weight": None},
]
for seed in range(SEEDS):
    order = random.Random(seed).sample(range(len(train)), len(train))
    runs += [
        {"size": k, "seed": seed, "idx": sorted(order[:k]), "weight": "balanced"}
        for k in SIZES
    ]
jobs = []
for r in runs:
    f = text_features()
    X = f.fit_transform([train_text[i] for i in r["idx"]])
    jobs.append(
        (X, f.transform(sample_text), [train_truth[i] for i in r["idx"]], r["weight"])
    )
answers = joblib.Parallel(n_jobs=-1)(joblib.delayed(heads)(*j) for j in jobs)
out["training_seconds"] = round(time.perf_counter() - started, 1)

mismatches = sum(
    a != b for q in QUESTIONS for a, b in zip(answers[0][q], heads_full[q])
)
assert mismatches == 0, (
    f"full-size heads differ from compare.py on {mismatches} answers"
)

# The joint paired bootstrap, drawn exactly as stats.py and metrics.py draw it
joint_idx = np.random.default_rng(0).integers(0, len(rows), size=(REPS, len(rows)))
classes = {q: sorted({norm(q, t) for t in gold[q] if t is not None}) for q in QUESTIONS}
truth_ids = {
    q: np.array(
        [classes[q].index(norm(q, t)) if t is not None else -1 for t in gold[q]]
    )
    for q in QUESTIONS
}


def boot(preds):
    """Each resample's mean over the 15 questions under each rule, as in stats.py."""
    per_q = {}
    for q in QUESTIONS:
        right = np.array(
            [
                t is not None and norm(q, p) == norm(q, t)
                for t, p in zip(gold[q], preds[q])
            ]
        )
        ti, ri = truth_ids[q][joint_idx], right[joint_idx]
        keep = ti >= 0
        k = len(classes[q]) + 1  # ungraded chats get their own class with no weight
        ti = np.where(keep, ti, len(classes[q]))
        acc = (ri & keep).sum(axis=1) / np.maximum(keep.sum(axis=1), 1)
        flat = (ti + k * np.arange(REPS)[:, None]).ravel()
        total = np.bincount(flat, minlength=REPS * k).reshape(REPS, k)[:, :-1]
        hits = np.bincount(
            flat, weights=(ri & keep).ravel(), minlength=REPS * k
        ).reshape(REPS, k)[:, :-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            bal = np.nanmean(
                np.where(total > 0, hits / np.maximum(total, 1), np.nan), axis=1
            )
        per_q[q] = {"accuracy": acc, "balanced": bal}
    return {
        rule: np.mean([per_q[q][pick(q)] for q in QUESTIONS], axis=0)
        for rule, pick in RULES.items()
    }


def ci_points(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [pts(lo), pts(hi)]


claude_boot = boot(claude)


def lead(a):
    """Heads minus Claude on the mean of 15, with its paired 95% interval, per rule."""
    point, b = means(a), boot(a)
    return {
        rule: {
            "heads": r6(point[rule]),
            "claude": r6(claude_point[rule]),
            "lead_points": pts(point[rule] - claude_point[rule]),
            "lead_ci95_points": ci_points(b[rule] - claude_boot[rule]),
        }
        for rule in RULES
    }


# The full-size weighted heads must reproduce the reported intervals
full = lead(answers[0])
for rule in RULES:
    reported = metrics["means"][rule]["diff_ci95_points"]
    mine = [round(x, 1) for x in full[rule]["lead_ci95_points"]]
    assert mine == reported, (rule, mine, reported)

# 3. Heads without class weights
unweighted = answers[1]
out["unweighted_heads"] = {
    "class_weight": None,
    "lead": lead(unweighted),
    "weighted_lead": full,
    "questions_as_reported": {
        q: {
            "unweighted": r6(score(q, unweighted[q], RULES["as_reported"](q))),
            "weighted": r6(score(q, heads_full[q], RULES["as_reported"](q))),
            "claude": r6(score(q, claude[q], RULES["as_reported"](q))),
        }
        for q in QUESTIONS
    },
    "yes_no_plain_accuracy": {
        q: {
            "unweighted": r6(score(q, unweighted[q], "accuracy")),
            "weighted": r6(score(q, heads_full[q], "accuracy")),
            "claude": r6(score(q, claude[q], "accuracy")),
        }
        for q in YES_NO
    },
}
plain = out["unweighted_heads"]["yes_no_plain_accuracy"]
out["unweighted_heads"]["yes_no_where_claude_plain_is_higher"] = {
    "unweighted": sum(v["claude"] > v["unweighted"] for v in plain.values()),
    "weighted": sum(v["claude"] > v["weighted"] for v in plain.values()),
}

# 2 (continued). Each seeded run's lead over Claude
seeded = []
for r, a in zip(runs[2:], answers[2:]):
    seeded.append({"size": r["size"], "seed": r["seed"], "lead": lead(a)})
lc = json.loads((OUT / f"learning_curve_{MODEL}.json").read_text())
recorded = {(x["size"], x["seed"]): x["mean_15"] for x in lc["runs"]}
for x in seeded:
    for rule in RULES:
        assert (
            abs(x["lead"][rule]["heads"] - recorded[(x["size"], x["seed"])][rule])
            < 1e-6
        )

curve = []
for k in SIZES:
    at = [x for x in seeded if x["size"] == k]
    curve.append(
        {
            "size": k,
            "runs": len(at),
            **{
                rule: {
                    "lead_points_by_seed": [x["lead"][rule]["lead_points"] for x in at],
                    "lead_ci95_points_by_seed": [
                        x["lead"][rule]["lead_ci95_points"] for x in at
                    ],
                    "runs_whose_interval_is_above_zero": sum(
                        x["lead"][rule]["lead_ci95_points"][0] > 0 for x in at
                    ),
                    "runs_whose_interval_is_below_zero": sum(
                        x["lead"][rule]["lead_ci95_points"][1] < 0 for x in at
                    ),
                }
                for rule in RULES
            },
        }
    )
out["learning_curve_leads"] = {"full_size": full, "curve": curve}

# 4. The synthetic system's two parts
syn = json.loads((OUT / f"synthetic_heads_{MODEL}.json").read_text())["questions"]
gain = {
    q: syn[q]["synthetic_system_on_abcd"] - syn[q]["synthetic_heads_on_abcd"]
    for q in OURS
}
out["synthetic_system_minus_synthetic_heads"] = {
    "questions": len(OURS),
    "regex_questions": REGEX,
    "total_points": pts(sum(gain.values()) / len(OURS)),
    "regex_part_points": pts(sum(gain[q] for q in REGEX) / len(OURS)),
    "learned_part_points": pts(
        sum(gain[q] for q in OURS if q not in REGEX) / len(OURS)
    ),
    "per_question_points": {q: pts(g) for q, g in gain.items()},
}

# 5. Misses and false flags on the 9 rare-yes questions
errors = {}
for s, a in (("micro_abcd", heads_full), ("claude", claude)):
    tally = {"yes": 0, "yes_found": 0, "no": 0, "no_kept": 0}
    for q in RARE_YES:
        for t, p in zip(gold[q], a[q]):
            if t is None:
                continue
            side = "yes" if t else "no"
            tally[side] += 1
            tally[f"{side}_{'found' if side == 'yes' else 'kept'}"] += norm(
                q, p
            ) == norm(q, t)
    tally["missed_yes"] = tally["yes"] - tally["yes_found"]
    tally["false_flags"] = tally["no"] - tally["no_kept"]
    tally["errors"] = tally["missed_yes"] + tally["false_flags"]
    errors[s] = tally
out["rare_yes_errors"] = {"questions": RARE_YES, "systems": errors}

(OUT / f"revision_checks_{MODEL}.json").write_text(json.dumps(out, indent=1))

g = routing["gaps_points"]
print(f"routing gaps: {json.dumps(g)}")
for s, v in routing["systems"].items():
    print(s, v["mean_10"], v["graded_pairs_falling_back"], v["intent_known_to_mapping"],
          v["intent_a_sampled_truth_holds"])  # fmt: skip
print("unweighted lead:", {r: (v["lead_points"], v["lead_ci95_points"]) for r, v in out["unweighted_heads"]["lead"].items()})  # fmt: skip
print("unweighted yes/no where Claude plain higher:", out["unweighted_heads"]["yes_no_where_claude_plain_is_higher"])  # fmt: skip
for x in curve:
    a = x["as_reported"]
    print(x["size"], [round(v, 1) for v in a["lead_points_by_seed"]],
          [[round(lo, 1), round(hi, 1)] for lo, hi in a["lead_ci95_points_by_seed"]],
          "above 0:", a["runs_whose_interval_is_above_zero"],
          "below 0:", a["runs_whose_interval_is_below_zero"])  # fmt: skip
print("synthetic parts:", {k: v for k, v in out["synthetic_system_minus_synthetic_heads"].items() if k.endswith("points") and k != "per_question_points"})  # fmt: skip
print("rare-yes errors:", errors)
print("training seconds:", out["training_seconds"])
