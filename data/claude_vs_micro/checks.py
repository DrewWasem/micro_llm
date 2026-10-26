"""Checks a careful reviewer asks for, computed from the saved rows. No model calls; it costs $0.

1. Routed intents: Claude's own intent answer, sent through the mapping that made the truth,
   for the 10 questions whose truth depends on the intent alone. The heads get the same.
2. Exclusions: the headline mean without each question a reviewer could call unfair.
3. Rare "yes" recall, pooled over the 9 yes/no questions whose "yes" is the minority.
4. Each question's share of the shrink from the reported gap to the plain-accuracy gap.
5. The false alarms one missed "yes" is worth under balanced accuracy (n_no / n_yes).
6. Constant-answer baselines under each question's reported metric, and who fails them.
7. Claude's empty and out-of-set answers on the 5 choice questions.
8. Interface overhead per call: wall time minus API time.
9. Test openings with a near-duplicate among the training openings, and the gap without them.
10. The heads on all 1,004 test openings, and the synthetic system on its own held-out tickets.

python3 -I checks.py <repo> <out_dir> [model=haiku]

Writes <out_dir>/checks_<model>.json and prints a summary.
"""

import gzip
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from micro_llm import abcd  # noqa: E402

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
OURS = CHOICE + YES_NO
QUESTIONS = ["flow", "intent"] + OURS
# Truth that depends on the intent alone; the other 3 come from agent actions or typed values
BY_INTENT = CHOICE + YES_NO[:7]
RARE_YES = [q for q in YES_NO if q != "needs_staff_action"]
SYSTEMS = ["micro_abcd", "claude"]
NEAR = 0.9  # cosine similarity that counts as a near-duplicate opening

rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
stats = json.loads((OUT / f"stats_{MODEL}.json").read_text())
metrics = json.loads((OUT / f"metrics_{MODEL}.json").read_text())
bank = json.loads((REPO / "micro_llm" / "questions.json").read_text())
analysis = json.loads((REPO / "data" / "abcd_analysis.json").read_text())
synthetic = json.loads((REPO / "models" / "eval.json").read_text())["ticket_answers"]
data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
train, test = data["train"], data["test"]
by_id = {c["convo_id"]: c for c in test}


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


def kind(q):
    return "accuracy" if q in CHOICE else "balanced"


def score(q, rs, answer, how=None):
    """The question's reported metric over rows rs; answer(r) gives the system's answer."""
    pairs = [
        (norm(q, r["truth"][q]), norm(q, answer(r)))
        for r in rs
        if r["truth"].get(q) is not None
    ]
    if (how or kind(q)) == "accuracy":
        return sum(t == p for t, p in pairs) / len(pairs)
    recall = {}
    for t, p in pairs:
        recall.setdefault(t, []).append(t == p)
    return statistics.mean(statistics.mean(v) for v in recall.values())


def direct(s, q):
    return lambda r: (r[s] or {}).get(q)


def mean(s, qs, rs=rows):
    return statistics.mean(score(q, rs, direct(s, q)) for q in qs)


def mapped(i):
    """What the truth mapping says for an intent, on the 10 intent-only questions."""
    out = {
        "primary_team": abcd.TEAM.get(i),
        "journey_stage": abcd.JOURNEY.get(i),
        "message_type": abcd.MESSAGE.get(i),
    }
    for q in YES_NO[:7]:
        yes, no = abcd.YES_NO[q]
        out[q] = True if i in yes else False if i in no else None
    return out


r4 = lambda x: round(x, 6)  # noqa: E731
out = {"model": MODEL, "chats": len(rows)}

# 1. Routed intents. The mapping must rebuild the truth from the true intent first.
for r in rows:
    m = mapped(r["truth"]["intent"])
    assert all(m[q] == r["truth"][q] for q in BY_INTENT), r["id"]


def routed(s, q):
    """The mapped answer from the system's own intent; its direct answer when that maps to nothing."""

    def answer(r):
        i = norm("intent", (r[s] or {}).get("intent"))
        v = mapped(i)[q] if i in abcd.ALL else None
        return (r[s] or {}).get(q) if v is None else v

    return answer


route = {}
for q in BY_INTENT:
    route[q] = {
        f"{s}_{how}": r4(score(q, rows, (direct if how == "direct" else routed)(s, q)))
        for s in SYSTEMS
        for how in ("direct", "routed")
    }
usable = sum(
    norm("intent", (r["claude"] or {}).get("intent")) in abcd.ALL for r in rows
)
out["routed"] = {
    "questions": BY_INTENT,
    "claude_intents_usable": usable,
    "per_question": route,
    "mean": {
        k: r4(statistics.mean(v[k] for v in route.values()))
        for k in route[BY_INTENT[0]]
    },
}

# 2. Exclusions from the 15-question mean
same = sum(r["truth"]["cannot_log_in"] == r["truth"]["security_related"] for r in rows)
drops = {
    "none": [],
    "charge_amount_given (2 positives)": ["charge_amount_given"],
    "needs_staff_action (follows the agent's later actions)": ["needs_staff_action"],
    "cannot_log_in (same truth as security_related)": ["cannot_log_in"],
    "all three": ["charge_amount_given", "needs_staff_action", "cannot_log_in"],
    "the 13 mapped questions (ABCD-native only)": OURS,
    "flow and intent (mapped only)": ["flow", "intent"],
}
excl = {}
for name, drop in drops.items():
    qs = [q for q in QUESTIONS if q not in drop]
    h, c = mean("micro_abcd", qs), mean("claude", qs)
    excl[name] = {
        "questions": len(qs),
        "heads": r4(h),
        "claude": r4(c),
        "gap_points": round(100 * (h - c), 1),
    }
out["exclusions"] = {
    "cannot_log_in_equals_security_related_on": f"{same} of {len(rows)} chats",
    "means": excl,
    "gap_range_points": [min(v["gap_points"] for v in excl.values()), max(v["gap_points"] for v in excl.values())],
}  # fmt: skip

# 3. Rare "yes" recall, pooled from metrics.py's counts
rec = metrics["questions"]
pool = {
    f"{label}_{s}": [sum(rec[q][f"recall_{label}"][s][i] for q in RARE_YES) for i in (0, 1)]
    for label in ("yes", "no")
    for s in SYSTEMS
}  # fmt: skip
out["rare_yes"] = {
    "questions": RARE_YES,
    "recall": {
        k: {"right": v[0], "of": v[1], "share": r4(v[0] / v[1])}
        for k, v in pool.items()
    },
    "needs_staff_action_yes": rec["needs_staff_action"]["recall_yes"],
}

# 4. Each question's share of the shrink from the reported gap to the plain-accuracy gap
shrink = {}
for q in QUESTIONS:
    gap = {
        how: score(q, rows, direct("micro_abcd", q), how) - score(q, rows, direct("claude", q), how)
        for how in (kind(q), "accuracy")
    }  # fmt: skip
    shrink[q] = round(100 * (gap[kind(q)] - gap["accuracy"]) / len(QUESTIONS), 2)
total = sum(shrink.values())
out["shrink"] = {
    "total_points": round(total, 1),
    "per_question_points": dict(sorted(shrink.items(), key=lambda x: -abs(x[1]))),
    "largest_share": r4(max(shrink.values()) / total),
}

# 5. What one missed "yes" costs in false alarms, under balanced accuracy
ratio = {}
for q in YES_NO:
    t = Counter(norm(q, r["truth"][q]) for r in rows if r["truth"].get(q) is not None)
    ratio[q] = {
        "yes": t["yes"],
        "no": t["no"],
        "no_per_yes": round(t["no"] / t["yes"], 1),
    }
out["false_alarms_per_missed_yes"] = ratio

# 6. Constant-answer baselines under each question's reported metric
base, fails = {}, []
for q in QUESTIONS:
    t = [norm(q, r["truth"][q]) for r in rows if r["truth"].get(q) is not None]
    c = Counter(t)
    b = c.most_common(1)[0][1] / len(t) if kind(q) == "accuracy" else 1 / len(c)
    plain = c.most_common(1)[0][1] / len(t)
    base[q] = {"reported_metric": r4(b), "plain_accuracy": r4(plain), "classes": len(c)}
    for s in SYSTEMS:
        sq = stats["questions"][q]["systems"][s]
        if sq["ci95"][0] <= b:
            fails.append(
                {
                    "question": q,
                    "system": s,
                    "score": sq["score"],
                    "ci95": sq["ci95"],
                    "baseline": r4(b),
                }
            )
        p = score(q, rows, direct(s, q), "accuracy")
        if p <= plain:
            fails.append(
                {
                    "question": q,
                    "system": s,
                    "plain_accuracy": r4(p),
                    "always_majority": r4(plain),
                }
            )
out["baselines"] = {"per_question": base, "interval_or_score_not_above_baseline": fails}  # fmt: skip

# 7. Claude's empty and out-of-set answers on the 5 choice questions
offered = {
    "flow": {c["scenario"]["flow"] for c in train},
    "intent": {abcd.intent(c) for c in train},
    **{q: set(bank[q]["criteria"]) for q in CHOICE},
}
kinds = {}
for q in ["flow", "intent"] + CHOICE:
    true_set = {r["truth"][q] for r in rows if r["truth"].get(q) is not None}
    count = Counter()
    for s in SYSTEMS:
        for r in rows:
            if r["truth"].get(q) is None:
                continue
            v = (r[s] or {}).get(q)
            v = v.strip() if isinstance(v, str) else v
            if v in (None, ""):
                k = "empty"
            elif v not in offered[q]:
                k = "not_offered"
            elif v not in true_set:
                k = "offered_but_no_chat_has_it"
            else:
                k = "possible"
            count[(s, k)] += 1
    kinds[q] = {s: {k: count[(s, k)] for (s2, k) in count if s2 == s} for s in SYSTEMS}
    wrong_set = Counter(
        (r["micro_abcd"] or {}).get(q) for r in rows
        if r["truth"].get(q) is not None and (r["micro_abcd"] or {}).get(q) not in true_set
    )  # fmt: skip
    kinds[q]["micro_abcd_answers_no_chat_has"] = dict(wrong_set)
    kinds[q]["claude_answers_no_chat_has"] = dict(
        Counter(
            (r["claude"] or {}).get(q)
            for r in rows
            if r["truth"].get(q) is not None
            and (r["claude"] or {}).get(q) not in true_set
        )  # fmt: skip
    )
out["answer_kinds"] = kinds

# 8. Interface overhead per call (the last good reply per chat, as stats.py reads it)
replies = {}
for line in (OUT / f"claude_{MODEL}.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("answers") is not None:
        replies[r["id"]] = r
gap = np.array(
    [
        replies[r["id"]]["wall_s"] - (replies[r["id"]].get("api_ms") or 0) / 1000
        for r in rows
    ]
)
p10, p50, p90 = np.percentile(gap, [10, 50, 90])
out["overhead_s"] = {"p10": round(float(p10), 2), "p50": round(float(p50), 2), "p90": round(float(p90), 2)}  # fmt: skip


# 9. Near-duplicate openings between the training chats and the 300 test openings
def plain_text(x):
    return re.sub(r"[^a-z0-9 ]", "", " ".join(x.lower().split()))


train_open = [abcd.customer_text(c, "opening") for c in train]
test_open = [abcd.customer_text(by_id[r["id"]], "opening") for r in rows]
seen = {plain_text(x) for x in train_open}
exact = [plain_text(x) in seen for x in test_open]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit(train_open)
sims = (
    (vec.transform(test_open) @ vec.transform(train_open).T)
    .max(axis=1)
    .toarray()
    .ravel()
)
near = sims >= NEAR
fresh = [r for r, n in zip(rows, near) if not n]
h, c = mean("micro_abcd", QUESTIONS, fresh), mean("claude", QUESTIONS, fresh)
out["overlap"] = {
    "exact_match_after_lowercasing": int(sum(exact)),
    f"cosine_at_least_{NEAR}": int(near.sum()),
    "median_best_cosine": round(float(np.median(sims)), 3),
    "words_in_exact_matches_median": statistics.median(len(x.split()) for x, e in zip(test_open, exact) if e) if any(exact) else None,
    "without_near_duplicates": {"chats": len(fresh), "heads": r4(h), "claude": r4(c), "gap_points": round(100 * (h - c), 1)},
}  # fmt: skip

# 10. The heads on all 1,004 test openings; the synthetic system on its own held-out tickets
full = {
    q: (analysis["learn"][q] if q in ("flow", "intent") else analysis["retrained"][q])[
        "opening"
    ]["accuracy" if kind(q) == "accuracy" else "balanced"]
    for q in QUESTIONS
}
moves = {q: round(100 * (full[q] - stats["questions"][q]["systems"]["micro_abcd"]["score"]), 1) for q in QUESTIONS}  # fmt: skip
own = {
    q: synthetic[q]["accuracy" if kind(q) == "accuracy" else "balanced"] for q in OURS
}
out["full_test"] = {
    "openings": analysis["learn"]["flow"]["opening"]["n"],
    "heads_mean": r4(statistics.mean(full.values())),
    "heads_sample_mean": stats["summary"]["mean_score"]["micro_abcd"],
    "largest_moves_points": dict(sorted(moves.items(), key=lambda x: x[1])[:3]),
}
out["synthetic_own_tickets"] = {
    "tickets": synthetic["primary_team"]["n"],
    "mean_13_shared": r4(statistics.mean(own.values())),
    "primary_team_per_ticket": synthetic["primary_team"]["accuracy"],
}
team = {s: score("primary_team", rows, direct(s, "primary_team")) for s in SYSTEMS}
out["misroutes_per_1000"] = {s: round(1000 * (1 - a), 1) for s, a in team.items()}

# 11. Where each system's errors fall: truth -> answer, and the true intents behind each pair
conf = {}
for q in CHOICE + ["refund_implied", "needs_staff_action"]:
    conf[q] = {}
    for s in SYSTEMS:
        wrong = [
            (norm(q, r["truth"][q]), norm(q, direct(s, q)(r)), r["truth"]["intent"])
            for r in rows
            if r["truth"].get(q) is not None
            and norm(q, direct(s, q)(r)) != norm(q, r["truth"][q])
        ]
        pairs = Counter(f"{t} -> {a}" for t, a, _ in wrong)
        conf[q][s] = {
            "errors": len(wrong),
            "pairs": {
                p: {
                    "count": k,
                    "intents": dict(Counter(i for t, a, i in wrong if f"{t} -> {a}" == p).most_common(5)),
                }
                for p, k in pairs.most_common(4)
            },
        }  # fmt: skip
out["confusions"] = conf

(OUT / f"checks_{MODEL}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ("answer_kinds", "false_alarms_per_missed_yes")}, indent=1, default=str))  # fmt: skip
