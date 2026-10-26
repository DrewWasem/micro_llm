"""Checks micro_llm against ABCD: 10,042 store support chats that people wrote by role-play.

ABCD (ASAPP, MIT license) tags every chat with one of 55 intents and logs every action the
agent took, so it can grade questions our synthetic tickets can't. Results go to
data/abcd_analysis.json, in five parts:

- profile: how ABCD's chats differ from our synthetic tickets
- agents: which actions agents take, and how often they stay inside the intent's playbook
- details: which details customers type, and how many our entity finder catches
- transfer: our 9 models, trained only on synthetic tickets, on ABCD's 1,004 test chats
- learn: our model type trained on ABCD's 8,034 training chats, one head per question

Every model reads one of two inputs: the customer's opening message, which is all a triage
step sees, or every customer message in the chat.

Download ABCD into data/abcd/ first, at the commit this analysis used:

    mkdir -p data/abcd
    base=https://raw.githubusercontent.com/asappresearch/abcd/6b8700ce67c6b37b062dd7a60abc76d7ef832a97/data
    for f in abcd_v1.1.json.gz ontology.json kb.json; do curl -fsSL "$base/$f" -o data/abcd/$f; done
    echo 6b8700ce67c6b37b062dd7a60abc76d7ef832a97 > data/abcd/COMMIT

    python -m micro_llm.abcd
"""

import gzip
import json
import re
import statistics
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from difflib import get_close_matches
from functools import cache
from pathlib import Path
from typing import Callable

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from micro_llm import entities
from micro_llm.config import DATA_DIR, TICKETS_FILE
from micro_llm.evaluate import hard, score
from micro_llm.models import WORD, text_features, to_answer
from micro_llm.pipeline import Pipeline
from micro_llm.questions import qtype
from micro_llm.rules import TICKET_QUESTIONS

ABCD_DIR = DATA_DIR / "abcd"
OUT_FILE = DATA_DIR / "abcd_analysis.json"
INPUTS = ("opening", "all")
PLACEHOLDER = re.compile(r"<([a-z_]+)>")
FAQ_FLOWS = {"single_item_query", "storewide_query"}

# Actions that change something for the customer, rather than look something up
CHANGES = {
    "offer-refund",
    "make-purchase",
    "update-order",
    "update-account",
    "make-password",
    "promo-code",
    "send-link",
    "notify-team",
}
# Identity checks, strongest first
CHECKS = ["verify-identity", "validate-purchase", "pull-up-account"]
IDENTITY = {"name", "account_id", "username", "email", "phone", "order_id", "zip_code"}
DETAILS = sorted(IDENTITY | {"street_address", "pin_number", "amount"})
# The entity finder's kind for each detail it looks for
FINDER = {
    "name": "name",
    "username": "username",
    "email": "email",
    "phone": "phone",
    "order_id": "order_number",
    "street_address": "address",
    "amount": "money",
}
# Questions about a value the agent enters, and the ABCD value list each draws from
VALUES = {
    "escalation_team": "company_team",
    "account_change": "account_slotval",
    "order_change": "change_option",
    "order_option": "order_slotval",
    "recorded_reason": "reason_slotval",
    "refund_method": "refund_target",
    "product": "product",
}
# ABCD's "credit card" is both a refund method and a payment method, so refund_method
# only reads refund chats
ONLY_IN = {"refund_method": "product_defect"}
# Cuts for the two count questions: actions 1-2, 3, 4, 5+ and turns at ABCD's quartiles
ACTION_CUTS = [3, 4, 5]
LENGTH_CUTS = [18, 22, 27]

# My reading of which ABCD intents answer our questions. An intent left out of a question's
# sets doesn't fit its options cleanly, so that question skips it.
ACCESS = {"recover_username", "recover_password", "reset_2fa"}
PROFILE = {"manage_change_address", "manage_change_name", "manage_change_phone"}
SITE = {"shopping_cart", "search_results", "slow_speed"}
MONEY_PROBLEM = {
    "status_credit_missing",
    "status_mystery_fee",
    "refund_initiate",
    "refund_update",
    "refund_status",
    "bad_price_competitor",
    "bad_price_yesterday",
    "mistimed_billing_already_returned",
    "mistimed_billing_never_bought",
    "manage_dispute_bill",
}
# Billing-team intents that raise no problem: payment methods and subscription questions
BILLING = {
    "manage_payment_method",
    "status_payment_method",
    "status_active",
    "status_due_amount",
    "status_due_date",
    "status_questions",
    "manage_pay_bill",
    "manage_extension",
}
RETURNS = {"return_stain", "return_color", "return_size"}
ORDERS = {
    "status_shipping_question",
    "status_delivery_time",
    "status_quantity",
    "manage_upgrade",
    "manage_downgrade",
    "manage_create",
    "manage_cancel",
    "status",
    "manage",
    "missing",
    "out_of_stock_general",
    "out_of_stock_one_item",
}
QUERIES = {
    "boots",
    "shirt",
    "jeans",
    "jacket",
    "pricing",
    "membership",
    "timing",
    "policy",
}
UNCLEAR = {
    "status_service_added",
    "status_service_removed",
    "promo_code_invalid",
    "promo_code_out_of_date",
    "credit_card",
    "cost",
}
GROUPS = [
    ACCESS,
    PROFILE,
    SITE,
    MONEY_PROBLEM,
    BILLING,
    RETURNS,
    ORDERS,
    QUERIES,
    UNCLEAR,
]
ALL = set().union(*GROUPS)
SUBSCRIPTION = BILLING - {"manage_payment_method", "status_payment_method"} | {
    "manage_dispute_bill"
}
OUT_OF_STOCK = {"out_of_stock_general", "out_of_stock_one_item"}
BOUGHT = (
    (ORDERS - OUT_OF_STOCK - {"manage_create"})
    | RETURNS
    | {"refund_initiate", "refund_update", "refund_status", "status_mystery_fee"}
    | {
        "bad_price_competitor",
        "bad_price_yesterday",
        "mistimed_billing_already_returned",
    }
    | {"status_payment_method"}
)
NO_MONEY = (
    ACCESS
    | PROFILE
    | SITE
    | QUERIES
    | BILLING
    | (ORDERS - OUT_OF_STOCK - {"manage_cancel", "missing"})
)
TEAM = {
    **dict.fromkeys(ACCESS | PROFILE, "account_support"),
    **dict.fromkeys(MONEY_PROBLEM | BILLING, "billing_and_payments"),
    **dict.fromkeys(SITE, "engineering"),
    **dict.fromkeys(RETURNS | ORDERS | QUERIES, "other"),
}
CATEGORY = {
    "account_support": "account",
    "billing_and_payments": "billing",
    "engineering": "bug_report",
}
JOURNEY = {
    **dict.fromkeys({"boots", "shirt", "jeans", "jacket"} | OUT_OF_STOCK, "browsing"),
    "shopping_cart": "checkout",
    **dict.fromkeys(BOUGHT, "post_purchase"),
}
MESSAGE = {
    **dict.fromkeys(
        QUERIES
        | {s for s in ALL if s.startswith("status_") and s != "status_credit_missing"}
        | {"refund_status", "status", "cost"},
        "question",
    ),
    **dict.fromkeys(
        ACCESS
        | {s for s in ALL if s.startswith("manage") and s != "manage_dispute_bill"}
        | RETURNS
        | {"refund_initiate", "refund_update", "bad_price_competitor"}
        | {"bad_price_yesterday"},
        "request",
    ),
    **dict.fromkeys(
        SITE
        | {"credit_card", "missing", "status_credit_missing", "manage_dispute_bill"}
        | {"mistimed_billing_already_returned", "mistimed_billing_never_bought"}
        | {"promo_code_invalid", "promo_code_out_of_date"},
        "problem_report",
    ),
}
YES_NO = {  # question: (intents that mean yes, intents that mean no)
    "cannot_log_in": (ACCESS, ALL - ACCESS - UNCLEAR),
    "security_related": (ACCESS, ALL - ACCESS - UNCLEAR),
    "mentions_account_issue": (
        ACCESS | PROFILE,
        ALL - ACCESS - PROFILE - UNCLEAR - {"manage_payment_method"},
    ),
    "mentions_billing_issue": (
        MONEY_PROBLEM,
        ACCESS | PROFILE | SITE | ORDERS | QUERIES,
    ),
    "mentions_bug": (SITE, ALL - SITE - UNCLEAR),
    "refund_requested": ({"refund_initiate"}, NO_MONEY),
    "refund_implied": (
        RETURNS
        | {"refund_initiate", "bad_price_competitor", "bad_price_yesterday"}
        | {"mistimed_billing_already_returned", "mistimed_billing_never_bought"},
        NO_MONEY,
    ),
    "existing_customer": (BOUGHT, set()),
    "has_account": (ACCESS | PROFILE | SUBSCRIPTION | {"manage_payment_method"}, set()),
    # ABCD has none of these, so every yes is wrong
    "mentions_feature_request": (set(), ALL),
    "wants_new_payment_method": (
        set(),
        ALL - {"manage_payment_method", "status_payment_method", "credit_card"},
    ),
    "duplicate_charge": (
        set(),
        ALL
        - {"status_mystery_fee", "manage_dispute_bill"}
        - {"mistimed_billing_already_returned", "mistimed_billing_never_bought"},
    ),
    "chargeback_mentioned": (set(), ALL - {"manage_dispute_bill"}),
}


@dataclass
class Question:
    id: str
    kind: str  # noul, choice or score, as in questions.json
    group: str
    asks: str
    label: Callable  # a chat -> its answer, or None when ABCD can't say


def intent(chat):
    """One of ABCD's 55 intents. FAQ chats name their FAQ entry, so they map back to its topic."""
    s = chat["scenario"]
    if s["flow"] in FAQ_FLOWS:
        return s["subflow"].split("_")[0]
    return {"status_delivery_date": "status_delivery_time"}.get(
        s["subflow"], s["subflow"]
    )


def actions(chat):
    """Every action the agent took, in order, with the values it entered."""
    return [
        (t["targets"][2], t["targets"][3])
        for t in chat["delexed"]
        if t["speaker"] == "action"
    ]


def done(chat):
    return {a for a, _ in actions(chat)}


def opening_count(chat):
    """How many messages the customer sends before the agent first replies to them."""
    n = 0
    for speaker, _ in chat["original"]:
        if speaker == "customer":
            n += 1
        elif n:
            break
    return n


def customer_text(chat, part):
    """The customer's opening message or all their messages, one message per line."""
    said = [text for speaker, text in chat["original"] if speaker == "customer"]
    return "\n".join(said[: opening_count(chat)] if part == "opening" else said)


def details(chat):
    """Each detail the customer typed, mapped to the first of their messages that holds it.

    ABCD marks typed details with placeholders such as <email>. Names stay unmarked, so a
    name counts when the customer's full or first name appears.
    """
    full = chat["scenario"]["personal"]["customer_name"]
    name = re.compile(rf"\b(?:{re.escape(full)}|{re.escape(full.split()[0])})\b", re.I)
    marked = [t["text"] for t in chat["delexed"] if t["speaker"] == "customer"]
    said = [text for speaker, text in chat["original"] if speaker == "customer"]
    found = {}
    for i, (m, s) in enumerate(zip(marked, said)):
        for slot in PLACEHOLDER.findall(m):
            found.setdefault(slot, i)
        if name.search(s):
            found.setdefault("name", i)
    return found


def typed_in(chat, part):
    k = opening_count(chat) if part == "opening" else float("inf")
    return {d for d, i in details(chat).items() if i < k}


def follows(seq, required):
    """True when the required actions all appear in order, with others allowed between."""
    rest = iter(seq)
    return all(a in rest for a in required)


def level(n, cuts):
    return sum(n >= c for c in cuts)


def entered(chat, options, carriers, cutoff=0.6):
    """The first value an agent entered through these actions that names an option, typos allowed."""
    for a, values in actions(chat):
        if a in carriers:
            for v in values:
                hit = get_close_matches(str(v).lower().strip(), options, 1, cutoff)
                if hit:
                    return hit[0]
    return "none"


def our_truth(chat, part):
    """Our questions that ABCD answers for this chat, as plain values. None: it can't say."""
    i = intent(chat)
    team = TEAM.get(i)
    typed = typed_in(chat, part)
    out = {
        "primary_team": team,
        "category": CATEGORY.get(team),
        "category_reordered": CATEGORY.get(team),
        "journey_stage": JOURNEY.get(i),
        "message_type": MESSAGE.get(i),
        "issue_count": "one",
        "needs_staff_action": bool(CHANGES & done(chat)),
        "identity_details_given": bool(typed & {"name", "email", "username"}),
        "sensitive_data_present": "street_address" in typed,
        "charge_amount_given": "amount" in typed,
    }
    for q, (yes, no) in YES_NO.items():
        out[q] = True if i in yes else False if i in no else None
    return out


def new_questions(onto):
    """Every question ABCD's labels answer that our 47 don't."""
    flows = onto["intents"]["flows"]
    intents = [s for subs in onto["intents"]["subflows"].values() for s in subs]
    groups = onto["actions"].values()
    names = [a for g in groups for a in g]
    values = onto["values"]["enumerable"]
    qs = [
        Question(
            "flow", "choice", "intent", "Which of ABCD's 10 topics the chat is about",
            lambda c: c["scenario"]["flow"],
        ),
        Question(
            "intent", "choice", "intent", "Which of ABCD's 55 intents the chat is about",
            intent,
        ),
        Question(
            "faq_answer", "choice", "intent", "Which of 48 FAQ entries answers the question",
            lambda c: c["scenario"]["subflow"] if c["scenario"]["flow"] in FAQ_FLOWS else None,
        ),
    ]  # fmt: skip
    qs += [
        Question(
            f"flow:{f}", "noul", "intent", f"The chat is about {f}",
            lambda c, f=f: c["scenario"]["flow"] == f,
        )
        for f in flows
    ]  # fmt: skip
    qs += [
        Question(
            f"intent:{s}", "noul", "intent", f"The chat's intent is {s}",
            lambda c, s=s: intent(c) == s,
        )
        for s in intents
    ]  # fmt: skip
    qs += [
        Question(
            f"action:{a}", "noul", "action", f"The agent runs {a}",
            lambda c, a=a: a in done(c),
        )
        for a in names
    ]  # fmt: skip
    qs += [
        Question(
            "first_action", "choice", "action", "Which action the agent runs first",
            lambda c: next(iter(actions(c)), ("none",))[0],
        ),
        Question(
            "identity_check", "choice", "action", "The strongest identity check the agent runs",
            lambda c: next((a for a in CHECKS if a in done(c)), "none"),
        ),
        Question(
            "agent_changes_something", "noul", "action",
            "The agent changes something, not only looks it up",
            lambda c: bool(CHANGES & done(c)),
        ),
        Question(
            "action_count", "score", "action", "How many actions the agent runs: 1-2, 3, 4 or 5+",
            lambda c: level(len(actions(c)), ACTION_CUTS),
        ),
    ]  # fmt: skip
    for qid, slot in VALUES.items():
        carriers = {a for g in groups for a, slots in g.items() if slot in slots}

        def label(c, opts=values[slot], carriers=carriers, flow=ONLY_IN.get(qid)):
            if flow and c["scenario"]["flow"] != flow:
                return "none"
            return entered(c, opts, carriers)

        qs.append(
            Question(
                qid, "choice", "value", f"Which {slot} the agent enters, or none", label
            )
        )
    qs += [
        Question(
            "member_level", "choice", "scenario", "The customer's membership level",
            lambda c: c["scenario"]["personal"]["member_level"],
        ),
        Question(
            "payment_method_on_file", "choice", "scenario",
            "How the customer paid, where ABCD records it",
            lambda c: c["scenario"]["order"].get("payment_method") or "not_recorded",
        ),
    ]  # fmt: skip
    qs += [
        Question(
            f"needs:{d}", "noul", "detail", f"The customer will have to give their {d}",
            lambda c, d=d: d in details(c),
        )
        for d in DETAILS
    ]  # fmt: skip
    qs.append(
        Question(
            "chat_length", "score", "shape",
            "How many turns the chat takes: under 18, 18-21, 22-26 or 27+",
            lambda c: level(len(c["original"]), LENGTH_CUTS),
        )
    )  # fmt: skip
    return qs


def profile(data, tickets):
    """How ABCD's chats compare with our synthetic tickets."""
    chats = [c for split in data.values() for c in split]

    def words(texts):
        return [len(t.split()) for t in texts]

    opening = [customer_text(c, "opening") for c in chats]
    every = [customer_text(c, "all") for c in chats]
    ours = {w for t in tickets for w in WORD.findall(t.lower())}

    def unseen(texts):
        tokens = [w for t in texts for w in WORD.findall(t.lower())]
        return round(sum(w not in ours for w in tokens) / len(tokens), 3)

    return {
        "chats": {k: len(v) for k, v in data.items()},
        "flows": Counter(c["scenario"]["flow"] for c in chats).most_common(),
        "intents": Counter(intent(c) for c in chats).most_common(),
        "turns": spread([len(c["original"]) for c in chats]),
        "customer_messages": spread(
            [sum(s == "customer" for s, _ in c["original"]) for c in chats]
        ),
        "opening_messages": Counter(opening_count(c) for c in chats).most_common(),
        "agent_speaks_first": share(c["original"][0][0] == "agent" for c in chats),
        "words": {
            "abcd_opening": spread(words(opening)),
            "abcd_all_customer": spread(words(every)),
            "our_tickets": spread(words(tickets)),
        },
        "words_unseen_in_our_tickets": {
            "opening": unseen(opening),
            "all": unseen(every),
        },
        "examples": [
            {"intent": intent(c), "opening": customer_text(c, "opening")}
            for c in data["test"][:8]
        ],
    }


def agent_stats(chats, kb, teams):
    """Which actions agents take, and how often they stay inside their intent's playbook.

    ABCD's kb.json lists, in order, every action an intent's playbook allows. Many chats end
    early because the problem is solved, so a playbook is a ceiling, not a script.
    """
    n = len(chats)
    measures = ("within", "in_order", "whole")
    playbook = {k: Counter() for k in ("chats", *measures)}
    gate = Counter()
    sequences = Counter()
    escalated = Counter()
    for c in chats:
        seq = [a for a, _ in actions(c)]
        dedup = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
        sequences[" > ".join(dedup)] += 1
        flow = c["scenario"]["flow"]
        need = kb.get(intent(c))
        if need is not None:
            playbook["chats"][flow] += 1
            # Only playbook actions; those in playbook order; every playbook action in order
            playbook["within"][flow] += set(seq) <= set(need)
            playbook["in_order"][flow] += follows(need, dedup)
            playbook["whole"][flow] += follows(seq, need)
        changes = [k for k, a in enumerate(seq) if a in CHANGES]
        if changes:
            gate["changes"] += 1
            gate["checked_first"] += any(a in CHECKS for a in seq[: changes[0]])
        for a, values in actions(c):
            if a == "notify-team":
                hit = get_close_matches(str(next(iter(values), "")), teams, 1, 0.6)
                escalated[hit[0] if hit else "unclear"] += 1
    total = sum(playbook["chats"].values())
    return {
        "actions_per_chat": spread([len(actions(c)) for c in chats]),
        "chats_with_action": {
            a: round(k / n, 3)
            for a, k in Counter(a for c in chats for a in done(c)).most_common()
        },
        "playbook": {k: round(sum(playbook[k].values()) / total, 3) for k in measures},
        "playbook_by_flow": {
            f: {k: round(playbook[k][f] / playbook["chats"][f], 3) for k in measures}
            for f in sorted(playbook["chats"])
        },
        "chats_that_change_something": round(gate["changes"] / n, 3),
        "identity_checked_before_change": round(
            gate["checked_first"] / gate["changes"], 3
        ),
        "escalations": dict(escalated.most_common()),
        "escalated_chats": share("notify-team" in done(c) for c in chats),
        "top_sequences": sequences.most_common(10),
    }


def detail_stats(chats):
    """Which details customers type, and when; and how many the entity finder catches."""
    n = len(chats)
    anywhere, opening = Counter(), Counter()
    counts = {d: Counter() for d in FINDER}
    missed = {d: [] for d in FINDER}
    any_identity = 0
    for c in chats:
        found = details(c)
        k = opening_count(c)
        said = [t for s, t in c["original"] if s == "customer"]
        any_identity += any(i < k for d, i in found.items() if d in IDENTITY)
        for d, i in found.items():
            anywhere[d] += 1
            opening[d] += i < k
        kinds = {e.kind for e in entities.find(customer_text(c, "all"))}
        for d, kind in FINDER.items():
            truth, hit = d in found, kind in kinds
            counts[d][(truth, hit)] += 1
            if truth and not hit and len(missed[d]) < 4:
                missed[d].append(said[found[d]][:120])
    finder = {}
    for d, k in counts.items():
        tp, fn, fp = k[(True, True)], k[(True, False)], k[(False, True)]
        finder[d] = {
            "kind": FINDER[d],
            "typed_in": tp + fn,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "missed_examples": missed[d],
        }
    return {
        "typed_anywhere": {d: round(anywhere[d] / n, 3) for d in DETAILS},
        "typed_in_opening": {d: round(opening[d] / n, 3) for d in DETAILS},
        "opening_has_identity_detail": round(any_identity / n, 3),
        "entity_finder": finder,
    }


def transfer(test):
    """Our synthetic-trained pipeline on ABCD's test chats, for each question ABCD answers."""
    pipe = Pipeline()
    out = {}
    for part in INPUTS:
        truths = [our_truth(c, part) for c in test]
        runs, ms = [], []
        for c in test:
            t = time.perf_counter()
            runs.append(pipe.run(customer_text(c, part)))
            ms.append((time.perf_counter() - t) * 1000)
        preds = [r["answers"] for r in runs]
        scores = {}
        for q in truths[0]:
            pairs = [(t[q], p[q]) for t, p in zip(truths, preds) if t[q] is not None]
            truth = [to_answer(q, {t: 1.0}) for t, _ in pairs]
            scores[q] = score(q, truth, [p for _, p in pairs])
        teams = Counter(
            f"{t['primary_team']} -> {hard('primary_team', p['primary_team'])}"
            for t, p in zip(truths, preds)
            if t["primary_team"]
        )
        out[part] = {
            "scores": scores,
            "team_confusion": teams.most_common(),
            "pieces_per_chat": spread([len(r["pieces"]) for r in runs]),
            "ms_per_chat": spread(ms),
        }
    out["not_answerable"] = [q for q in TICKET_QUESTIONS if q not in scores]
    return out


@cache
def matrix(path):
    return joblib.load(path, mmap_mode="r")


def fit_and_grade(paths, kind, rows, y, test_rows, truth):
    """One logistic-regression head, set up as our models set theirs, graded on the test chats."""
    if len(set(y)) < 2:
        return None
    X, X_test = matrix(paths[0]), matrix(paths[1])
    model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[rows], y)
        pred = model.predict(X_test[test_rows]).tolist()
        return grade(kind, truth, pred)


def grade(kind, truth, pred):
    """Accuracy against always giving the most common answer, as evaluate.score reports it."""
    n = len(truth)
    row = {
        "n": n,
        "accuracy": round(sum(t == p for t, p in zip(truth, pred)) / n, 3),
        "baseline": round(Counter(truth).most_common(1)[0][1] / n, 3),
    }
    if kind != "score" and len(set(truth)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["balanced"] = round(balanced_accuracy_score(truth, pred), 3)
    if kind == "noul":
        row["positives"] = sum(truth)
    if kind == "score":
        row["mae"] = round(statistics.mean(abs(t - p) for t, p in zip(truth, pred)), 3)
    return row


def learn(train, test, questions):
    """Trains one head per question and input on ABCD's training chats; grades on its test chats.

    questions: (id, kind, label) where label takes a chat and an input name.
    """
    jobs, keys = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for part in INPUTS:
            f = text_features()
            X = f.fit_transform([customer_text(c, part) for c in train])
            X_test = f.transform([customer_text(c, part) for c in test])
            paths = (Path(tmp) / f"{part}_train.pkl", Path(tmp) / f"{part}_test.pkl")
            joblib.dump(X, paths[0])
            joblib.dump(X_test, paths[1])
            for qid, kind, label in questions:
                y = [label(c, part) for c in train]
                t = [label(c, part) for c in test]
                rows = [i for i, v in enumerate(y) if v is not None]
                test_rows = [i for i, v in enumerate(t) if v is not None]
                jobs.append(
                    joblib.delayed(fit_and_grade)(
                        tuple(map(str, paths)),
                        kind,
                        rows,
                        [y[i] for i in rows],
                        test_rows,
                        [t[i] for i in test_rows],
                    )
                )
                keys.append((qid, part))
        results = joblib.Parallel(n_jobs=-1)(jobs)
    out = {}
    for (qid, part), row in zip(keys, results):
        out.setdefault(qid, {})[part] = row
    return out


def spread(values):
    v = sorted(values)

    def pick(q):
        return v[min(len(v) - 1, int(q * len(v)))]

    return {"median": statistics.median(v), "p10": pick(0.1), "p90": pick(0.9)}


def share(flags):
    flags = list(flags)
    return round(sum(flags) / len(flags), 3)


def main():
    start = time.perf_counter()
    data = json.loads(gzip.decompress((ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
    onto = json.loads((ABCD_DIR / "ontology.json").read_text())
    kb = json.loads((ABCD_DIR / "kb.json").read_text())
    chats = [c for split in data.values() for c in split]
    assert sum(map(len, GROUPS)) == len(ALL), "an intent sits in two mapping groups"
    assert {intent(c) for c in chats} <= ALL, (
        "an ABCD intent is missing from the mapping"
    )
    tickets = [json.loads(line)["text"] for line in TICKETS_FILE.open()]

    new = new_questions(onto)
    labels = {q.id: Counter(map(str, (q.label(c) for c in chats))) for q in new}
    retrain = [
        (q, qtype(q), lambda c, part, q=q: our_truth(c, part)[q])
        for q in our_truth(chats[0], "all")
    ]
    result = {
        "source": {
            "commit": (ABCD_DIR / "COMMIT").read_text().strip()
            if (ABCD_DIR / "COMMIT").exists()
            else None,
            "inputs": {
                "opening": "the customer's first messages, up to the agent's first reply",
                "all": "every customer message in the chat",
            },
        },
        "profile": profile(data, tickets),
        "agents": agent_stats(chats, kb, onto["values"]["enumerable"]["company_team"]),
        "details": detail_stats(chats),
        "transfer": transfer(data["test"]),
        "questions": {
            q.id: {
                "kind": q.kind,
                "group": q.group,
                "asks": q.asks,
                "labeled_chats": sum(n for v, n in labels[q.id].items() if v != "None"),
                "options": len([v for v in labels[q.id] if v != "None"]),
                "top": labels[q.id].most_common(3),
            }
            for q in new
        },
        "learn": learn(
            data["train"],
            data["test"],
            [(q.id, q.kind, lambda c, part, q=q: q.label(c)) for q in new],
        ),
        "retrained": learn(data["train"], data["test"], retrain),
    }
    result["seconds"] = round(time.perf_counter() - start)
    OUT_FILE.write_text(json.dumps(result, indent=1, default=str))
    report(result)


def report(result):
    t = result["transfer"]
    print(f"{len(result['questions'])} new questions; results in {OUT_FILE}")
    for part in INPUTS:
        print(f"\nOur models on ABCD's {part} text (accuracy / baseline):")
        for q, row in t[part]["scores"].items():
            retrained = (result["retrained"].get(q) or {}).get(part)
            again = f"  retrained {retrained['accuracy']}" if retrained else ""
            print(
                f"  {q:26} {row['accuracy']:.3f} / {row['baseline']:.3f}  n={row['n']}{again}"
            )
    print("\nNew questions (opening accuracy / all accuracy / baseline):")
    for qid, rows in result["learn"].items():
        o, a = rows.get("opening"), rows.get("all")
        if o and a:
            print(
                f"  {qid:36} {o['accuracy']:.3f} / {a['accuracy']:.3f} / {a['baseline']:.3f}"
            )
    print(f"\n{result['seconds']} s")


if __name__ == "__main__":
    main()
