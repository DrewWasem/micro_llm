"""Writes labelled synthetic tickets with the Claude CLI.

Code samples every label first, so classes stay balanced and the truth is known before
any text exists. Claude only writes text that shows those labels.

    python -m micro_llm.generate --tickets 480
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from micro_llm.config import BATCHES_DIR, GENERATOR_MODEL, SEED, TICKETS_FILE
from micro_llm.questions import bank, qtype
from micro_llm.routing import is_issue

PRODUCTS = [
    "online clothing store",
    "food delivery app",
    "project management SaaS",
    "video streaming service",
    "mobile banking app",
    "online electronics shop",
    "fitness tracking app",
    "airline booking site",
    "cloud storage service",
    "online course platform",
    "ride-hailing app",
    "web hosting provider",
    "grocery delivery service",
    "photo editing app",
    "handmade goods marketplace",
    "concert ticketing site",
]
PERSONAS = [
    "small business owner",
    "college student",
    "retiree",
    "busy parent",
    "software developer",
    "non-native English speaker",
    "teenager",
    "office manager",
    "freelancer",
    "frequent traveler",
]
STYLES = [
    "short and blunt",
    "polite and formal",
    "chatty and rambling",
    "all lowercase with little punctuation",
    "a few typos",
    "precise and detailed",
    "emotional",
    "matter-of-fact",
]
# Tone levels a piece takes from the ticket's mood: frustration, sentiment, politeness
MOODS = {
    "calm": ([0], [1, 2], [2, 3]),
    "annoyed": ([1], [1], [1, 2]),
    "angry": ([2], [0], [0, 1]),
    "happy": ([0], [3, 4], [3]),
    "anxious": ([0, 1], [1], [2, 3]),
}
MOOD_WEIGHTS = {
    "calm": 0.3,
    "annoyed": 0.32,
    "angry": 0.16,
    "happy": 0.1,
    "anxious": 0.12,
}
TEAM_WEIGHTS = {
    "billing_and_payments": 0.25,
    "account_support": 0.22,
    "engineering": 0.22,
    "product_management": 0.16,
    "other": 0.15,
}
PAYMENT_WEIGHTS = {
    "card": 0.3,
    "paypal": 0.15,
    "apple_pay": 0.12,
    "other": 0.08,
    "not_stated": 0.35,
}

# Topic -> labels the topic fixes. A wrong charge is one the user should not have paid.
BILLING_PROBLEMS = {
    "charged twice for one order": {"duplicate_charge": True, "wrong_charge": True},
    "charged the wrong amount": {"wrong_charge": True},
    "charged after cancelling": {"wrong_charge": True},
    "an unexpected fee on the bill": {"wrong_charge": True},
    "a subscription that renewed without warning": {"wrong_charge": True},
    "a refund that never arrived": {"wrong_charge": True},
    "a declined payment": {},
}
BILLING_REQUESTS = {
    "a refund": {"refund_requested": True, "wrong_charge": True},
    "changing the payment method": {},
    "an invoice or receipt": {},
    "cancelling the subscription": {},
}
# Topic -> (kind, cannot_log_in, security_related, usual bug_severity)
ACCOUNT_TOPICS = {
    "cannot log in": ("problem", True, True, 2),
    "the password reset email never arrives": ("problem", True, True, 2),
    "the account is locked": ("problem", True, True, 2),
    "two-factor codes are rejected": ("problem", True, True, 2),
    "a suspicious login on the account": ("problem", False, True, 1),
    "an email change that did not save": ("problem", False, False, 1),
    "profile data that disappeared": ("problem", False, False, 1),
    "deleting the account": ("request", False, False, None),
    "changing the account email": ("request", False, False, None),
    "merging two accounts": ("request", False, False, None),
}
# Topic -> usual bug_severity
ENGINEERING_TOPICS = {
    "the app crashes on one screen": 2,
    "a page shows an error": 1,
    "a button does nothing": 1,
    "file uploads fail": 2,
    "data does not sync between devices": 1,
    "pages load very slowly": 1,
    "notifications stopped arriving": 1,
    "search returns no results": 1,
    "the layout looks broken": 0,
    "a label on a page has a typo": 0,
}
FEATURES = [
    "dark mode",
    "export to CSV",
    "an integration with another tool",
    "an offline mode",
    "more languages",
    "family or team sharing",
    "a home-screen widget",
]
NEW_PAYMENT_METHODS = ["Apple Pay", "Google Pay", "PayPal", "Venmo", "Klarna", "crypto"]
OTHER_METHODS = "Google Pay, Venmo, Klarna, a gift card or a bank transfer"
OTHER_TOPICS = {
    "a package that arrived late": "problem",
    "the wrong item delivered": "problem",
    "a rude delivery driver": "problem",
    "which countries they ship to": "request",
    "opening hours": "request",
    "a partnership": "request",
}
HUMAN_TOPIC = "talking to a real person"
CAUSES = [
    "the latest app update",
    "last week's website update",
    "switching to a new phone",
    "changing their email address",
    "the outage on Monday",
    "moving to the new pricing plan",
]
UPDATE_CAUSES = CAUSES[:2]
DEVICES = [
    "an iPhone",
    "an iPad",
    "a Mac",
    "Safari",
    "an Android phone",
    "a Windows laptop",
    "Chrome",
    "Firefox",
    "a Samsung tablet",
    "a Chromebook",
]
APPLE_DEVICES = DEVICES[:4]
ENTITY_HINTS = {
    "order_number": "an order or reference number",
    "amount": "the amount they were charged",
    "order_date": "when they placed the order",
    "error_text": "the exact error message they see, in quotes",
    "email": "their email address (an example.com address)",
    "card": "a full card number (a test number such as 4111 1111 1111 1111)",
    "address": "their home street address",
    "name": "their first name, as a sign-off",
}
TONE_FLAGS = [
    "time_pressure",
    "threatens_to_leave",
    "asks_for_human",
    "prior_contact",
    "frustration_rising",
]

# Claude writes only the pieces and code joins them. Asking for the whole ticket with the
# pieces copied out verbatim cost 20,000+ thinking tokens per ticket on 2026-10-24.
SYSTEM = (
    "You write realistic customer support messages for a dataset that trains small text classifiers. "
    "Reply only in the format asked, with no other text."
)
RULES = """Write the pieces of one support message for each spec below.

- Each piece is one sentence of 4 to 25 words that does only its own job. The pieces are joined in order into one message, so each must read naturally after the one before.
- Never write ", and", ", but" or ", plus" inside a piece. Code adds those joins between pieces.
- Show everything a piece lists, and nothing on its "Avoid" line.
- Add no detail a piece doesn't ask for: no order numbers, amounts, dates, devices, error messages, names, emails, addresses, payment methods or updates.
- Never mention urgency, earlier contact, leaving, growing frustration or wanting a human unless a piece lists it.
- Write as the persona, about the product, in the style given. Vary your wording from message to message.

Reply in exactly this format, one block per spec:
### <spec id>
1. <piece 1>
2. <piece 2>"""
HEADER = re.compile(r"^#+\s*([\w-]+)")
NUMBERED = re.compile(r"^\s*(\d+)[.):]\s+(.+?)\s*$")
# A piece asked to include one of these must show it, or its label would be wrong
ENTITY_CHECKS = {
    "order_number": re.compile(r"\d{3,}"),
    "amount": re.compile(r"\d"),
    "email": re.compile(r"@"),
    "card": re.compile(r"(?:\d[ -]?){12,}"),
}
JOINERS = {", and ": 0.6, ", but ": 0.3, ", plus ": 0.1}


def pick(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def bug_labels(rng, severity):
    if rng.random() < 0.3:
        severity = min(2, max(0, severity + rng.choice([-1, 1])))
    after = rng.random() < 0.35
    return {
        "bug_severity": severity,
        "has_reproducible_steps": rng.random() < 0.35,
        "problem_after_update": after,
        "blames_update": after and rng.random() < 0.5,
        "possible_widespread_issue": rng.random() < (0.55 if after else 0.2),
    }


def sample_issue(rng):
    """One problem or request piece: its kind, team, topic and specialist labels."""
    team = pick(rng, TEAM_WEIGHTS)
    if team == "billing_and_payments":
        if rng.random() < 0.35:
            kind = "request"
            topic = pick(rng, dict(zip(BILLING_REQUESTS, [0.4, 0.2, 0.2, 0.2])))
            fixed = BILLING_REQUESTS[topic]
        else:
            kind = "problem"
            topic = pick(
                rng, {t: 0.35 if "twice" in t else 0.65 / 6 for t in BILLING_PROBLEMS}
            )
            fixed = BILLING_PROBLEMS[topic]
        wrong = fixed.get("wrong_charge", False)
        duplicate = fixed.get("duplicate_charge", False) or (
            topic == "a refund" and rng.random() < 0.3
        )
        if duplicate and kind == "request":
            topic = "a refund for a double charge"
        requested = fixed.get("refund_requested", wrong and rng.random() < 0.4)
        labels = {
            "duplicate_charge": duplicate,
            "refund_requested": requested,
            "refund_implied": wrong or requested,
            "payment_method_used": pick(rng, PAYMENT_WEIGHTS),
            "chargeback_mentioned": wrong and rng.random() < 0.25,
        }
    elif team == "account_support":
        topic = pick(
            rng, {t: 0.25 if t == "cannot log in" else 0.75 / 9 for t in ACCOUNT_TOPICS}
        )
        kind, cannot, security, severity = ACCOUNT_TOPICS[topic]
        labels = {
            "cannot_log_in": cannot,
            "security_related": security,
            "tried_fix_themselves": kind == "problem" and rng.random() < 0.4,
        }
        if kind == "problem":
            labels |= bug_labels(rng, severity)
    elif team == "engineering":
        kind, topic = "problem", rng.choice(list(ENGINEERING_TOPICS))
        labels = bug_labels(rng, ENGINEERING_TOPICS[topic])
    elif team == "product_management":
        kind = "request"
        new_method = rng.random() < 0.35
        topic = (
            f"adding {rng.choice(NEW_PAYMENT_METHODS)} as a way to pay"
            if new_method
            else rng.choice(FEATURES)
        )
        labels = {
            "wants_new_payment_method": new_method,
            "feature_request_strength": pick(rng, {1: 0.35, 2: 0.45, 3: 0.2}),
        }
    else:
        topic = rng.choice(list(OTHER_TOPICS))
        kind, labels = OTHER_TOPICS[topic], {}
    return {"kind": kind, "team": team, "topic": topic, "labels": labels}


def plain(kind, topic=None):
    return {"kind": kind, "team": "other", "topic": topic, "labels": {}}


def tone(rng, mood, kind):
    if kind == "greeting":
        return {
            "frustration": 0,
            "sentiment": rng.choice([2, 3]),
            "politeness": rng.choice([2, 3]),
        }
    if kind != "feeling" and rng.random() < 0.3:
        mood = "calm"
    f, s, p = MOODS[mood]
    return {
        "frustration": rng.choice(f),
        "sentiment": rng.choice(s),
        "politeness": rng.choice(p),
    }


def sample_spec(rng, tid):
    """Every label for one ticket, sampled before any text is written."""
    mood = pick(rng, MOOD_WEIGHTS)
    upset = mood in ("annoyed", "angry")
    # No two issues share a team and kind, so a ticket never repeats one request in other words
    n_issues, issues = pick(rng, {1: 0.3, 2: 0.4, 3: 0.25, 4: 0.05}), []
    while len(issues) < n_issues:
        issue = sample_issue(rng)
        if all(
            (i["team"], i["kind"]) != (issue["team"], issue["kind"]) for i in issues
        ):
            issues.append(issue)
    feeling = plain("feeling") if rng.random() < (0.7 if upset else 0.35) else None
    feeling_first = feeling is not None and rng.random() < 0.25
    pieces = [plain("greeting")] if rng.random() < 0.45 else []
    pieces += [feeling] if feeling_first else []
    pieces += issues
    if rng.random() < 0.12:
        pieces.append(plain("request", HUMAN_TOPIC))
    pieces += [feeling] if feeling and not feeling_first else []
    if rng.random() < 0.3:
        pieces.append(plain("greeting"))
    for p in pieces:
        p["tone"] = tone(rng, mood, p["kind"]) | dict.fromkeys(TONE_FLAGS, False)
        p["entities"] = {}

    issue_idx = [
        i
        for i, p in enumerate(pieces)
        if is_issue(p["kind"]) and p["topic"] != HUMAN_TOPIC
    ]
    feel_idx = [i for i, p in enumerate(pieces) if p["kind"] == "feeling"]
    for p in pieces:
        p["tone"]["asks_for_human"] = p["topic"] == HUMAN_TOPIC
    if rng.random() < 0.2:
        pieces[rng.choice(issue_idx)]["tone"]["time_pressure"] = True
    if feel_idx and upset:
        pieces[feel_idx[0]]["tone"]["threatens_to_leave"] = rng.random() < 0.35
        pieces[feel_idx[0]]["tone"]["frustration_rising"] = rng.random() < 0.4
    if rng.random() < 0.15:
        pieces[rng.choice(feel_idx or issue_idx)]["tone"]["prior_contact"] = True

    link = None
    if len(issue_idx) >= 2 and rng.random() < 0.35:
        link = {"pieces": sorted(rng.sample(issue_idx, 2)), "cause": rng.choice(CAUSES)}
        for i in link["pieces"]:
            labels = pieces[i]["labels"]
            if "problem_after_update" in labels:
                by_update = link["cause"] in UPDATE_CAUSES
                labels.update(problem_after_update=by_update, blames_update=by_update)
                labels["possible_widespread_issue"] |= by_update
    # Only one unlinked piece may mention an update, or unrelated issues would seem to share a cause
    update_taken = link is not None and link["cause"] in UPDATE_CAUSES
    for i, p in enumerate(pieces):
        if p["labels"].get("problem_after_update") and (
            link is None or i not in link["pieces"]
        ):
            if update_taken:
                p["labels"].update(problem_after_update=False, blames_update=False)
            update_taken = True

    def place(kind, idx, chance, hint=None):
        if idx and rng.random() < chance:
            i = rng.choice(idx)
            pieces[i]["entities"][kind] = hint or ENTITY_HINTS[kind]
            return i

    billing = [i for i in issue_idx if pieces[i]["team"] == "billing_and_payments"]
    buggy = [i for i in issue_idx if "bug_severity" in pieces[i]["labels"]]
    deliveries = [
        i
        for i in issue_idx
        if pieces[i]["team"] == "other" and pieces[i]["kind"] == "problem"
    ]
    place("order_number", billing + deliveries, 0.45)
    place("amount", billing, 0.45)
    place("order_date", billing, 0.4)
    place("error_text", buggy, 0.35)
    place("email", issue_idx, 0.12)
    place("address", deliveries, 0.3)
    card = place("card", billing, 0.05)
    if card is not None:
        pieces[card]["labels"]["payment_method_used"] = "card"
    device = rng.choice(DEVICES)
    if (
        place("device", buggy, 0.45, f"the device or browser they use: {device}")
        is None
    ):
        device = None
    if len(pieces) > 1 and pieces[-1]["kind"] == "greeting" and rng.random() < 0.4:
        pieces[-1]["entities"]["name"] = ENTITY_HINTS["name"]

    found = {k for p in pieces for k in p["entities"]}
    apple = device in APPLE_DEVICES or any(
        p["labels"].get("payment_method_used") == "apple_pay"
        or "Apple Pay" in (p["topic"] or "")
        for p in pieces
    )
    return {
        "id": tid,
        "product": rng.choice(PRODUCTS),
        "persona": rng.choice(PERSONAS),
        "style": rng.choice(STYLES),
        "mood": mood,
        "pieces": pieces,
        "link": link,
        "truth": {
            "charge_amount_given": "amount" in found,
            "order_date_given": "order_date" in found,
            "error_message_given": "error_text" in found,
            "device_or_browser_given": "device" in found,
            "identity_details_given": bool(found & {"email", "name"}),
            "sensitive_data_present": bool(found & {"card", "address"}),
            "likely_apple_user": apple,
        },
    }


def level(q, v):
    return bank()[q]["criteria"][v].lower()


def shows(q, v):
    """What one label asks the text to show, or None when it asks for nothing."""
    b = bank()[q]
    if qtype(q) == "noul":
        return b["instructions"] if v else None
    if v == "not_stated":
        return None
    # "Another payment method" alone came back as "paid by another method", too vague to learn from
    if q == "payment_method_used" and v == "other":
        return f"{b['instructions']}: name one method that isn't a card, PayPal or Apple Pay, such as {OTHER_METHODS}"
    return f"{b['instructions']}: {b['criteria'][v]}"


def render(spec):
    """The spec as short lines a writer can follow: what each piece shows and what it avoids."""
    lines = [
        f"### {spec['id']}: a {spec['persona']} writing to a {spec['product']}. Style: {spec['style']}."
    ]
    for n, p in enumerate(spec["pieces"], 1):
        if p["kind"] == "greeting":
            lines.append(f"{n}. greeting: a {'hello' if n == 1 else 'sign-off'}")
        elif p["kind"] == "feeling":
            lines.append(f"{n}. feeling: how they feel about the situation")
        else:
            lines.append(f"{n}. {p['kind']} about {p['topic']}")
        t = p["tone"]
        show = [shows(q, v) for q, v in p["labels"].items()]
        show += [shows(flag, True) for flag in TONE_FLAGS if t[flag]]
        avoid = [q.replace("_", " ") for q, v in p["labels"].items() if v is False]
        if any(show):
            lines.append("- Show: " + "; ".join(s for s in show if s))
        if avoid:
            lines.append("- Avoid: " + ", ".join(avoid))
        lines += [f"- Include {hint}" for hint in p["entities"].values()]
        lines.append(
            f"- Tone: {level('frustration', t['frustration'])}, "
            f"{level('sentiment', t['sentiment'])} overall, {level('politeness', t['politeness'])}"
        )
    if spec["link"]:
        a, b = (i + 1 for i in spec["link"]["pieces"])
        lines.append(
            f"Pieces {a} and {b} share one cause: {spec['link']['cause']}. Mention it in both pieces."
        )
    elif sum(is_issue(p["kind"]) for p in spec["pieces"]) > 1:
        lines.append("The issues are unrelated. Don't suggest a shared cause.")
    return "\n".join(lines)


def parse(reply):
    """The numbered pieces under each '### id' header in Claude's reply."""
    written, current = {}, None
    for line in reply.splitlines():
        if m := HEADER.match(line.strip()):
            current = written.setdefault(m.group(1), [])
        elif current is not None and (m := NUMBERED.match(line)):
            current.append(m.group(2))
    return written


def call_claude(prompt, retries=2):
    cmd = [
        "claude", "-p",
        "--model", GENERATOR_MODEL,
        "--effort", "low",
        "--output-format", "json",
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--setting-sources", "",
        "--system-prompt", SYSTEM,
    ]  # fmt: skip
    for _ in range(retries + 1):
        done = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=tempfile.gettempdir(),
        )
        try:
            out = json.loads(done.stdout)
            if parse(out["result"]):
                return out
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    raise RuntimeError(f"Claude CLI failed: {done.stderr[-500:] or done.stdout[-500:]}")


def run_batch(index, size, seed, prefix):
    path = BATCHES_DIR / f"{prefix}_{index:04d}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(seed * 100_003 + index)
    specs = [sample_spec(rng, f"{prefix}{index:04d}_{k}") for k in range(size)]
    started = time.monotonic()
    out = call_claude(RULES + "\n\n" + "\n\n".join(render(s) for s in specs))
    usage = out.get("usage", {}).get("output_tokens_details", {})
    batch = {
        "index": index,
        "specs": specs,
        "written": parse(out["result"]),
        "cost_usd": out.get("total_cost_usd"),
        "seconds": round(time.monotonic() - started, 1),
        "thinking_tokens": usage.get("thinking_tokens"),
    }
    path.write_text(json.dumps(batch))
    return batch


def valid(spec, texts):
    """True when Claude wrote one usable piece per spec piece, with the details asked for and no others."""
    if len(texts) != len(spec["pieces"]):
        return False
    for text, p in zip(texts, spec["pieces"]):
        if not (1 if p["kind"] == "greeting" else 3) <= len(text.split()) <= 45:
            return False
        if any(
            not ENTITY_CHECKS[k].search(text)
            for k in p["entities"]
            if k in ENTITY_CHECKS
        ):
            return False
        if "@" in text and "email" not in p["entities"]:
            return False
        if re.search(r"\d{5,}", text) and not {"order_number", "card"} & set(
            p["entities"]
        ):
            return False
    return True


def lower_first(text):
    """Lowercases a sentence's first letter so it can follow ', and', unless the word is 'I' or a name."""
    word = text.split()[0]
    if re.fullmatch(r"I(?:['’]\w+)?\W*", word) or any(c.isupper() for c in word[1:]):
        return text
    return text[0].lower() + text[1:]


def assemble(spec, texts):
    """Joins the pieces into one message and returns it with each piece's span.

    A quarter of neighbouring issue pieces become one sentence joined by ', and', ', but' or ', plus'.
    """
    rng = random.Random(spec["id"])
    texts, seps = [t.strip() for t in texts], [""]
    for i in range(1, len(texts)):
        both = is_issue(spec["pieces"][i - 1]["kind"]) and is_issue(
            spec["pieces"][i]["kind"]
        )
        if both and texts[i - 1].endswith(".") and rng.random() < 0.25:
            texts[i - 1], texts[i] = texts[i - 1][:-1], lower_first(texts[i])
            seps.append(pick(rng, JOINERS))
        else:
            seps.append(" ")
    message, spans = "", []
    for sep, text in zip(seps, texts):
        message += sep
        spans.append((len(message), len(message) + len(text)))
        message += text
    return message, spans


def to_record(spec, texts):
    """Joins a spec's labels to the pieces Claude wrote, or returns None when the pieces break the rules."""
    if not valid(spec, texts):
        return None
    message, spans = assemble(spec, texts)
    pieces = [
        {
            "text": message[start:end],
            "start": start,
            "end": end,
            "topic": p["topic"],
            "labels": {
                "piece_kind": p["kind"],
                "primary_team": p["team"],
                **p["tone"],
                **p["labels"],
            },
            "entities": sorted(p["entities"]),
        }
        for (start, end), p in zip(spans, spec["pieces"])
    ]
    issues = [i for i, p in enumerate(spec["pieces"]) if is_issue(p["kind"])]
    linked = set(spec["link"]["pieces"]) if spec["link"] else set()
    return {
        "id": spec["id"],
        "text": message,
        **{
            k: spec[k] for k in ("product", "persona", "style", "mood", "link", "truth")
        },
        "pieces": pieces,
        "pairs": [
            {"a": a, "b": b, "issues_related": {a, b} == linked}
            for a, b in combinations(issues, 2)
        ],
    }


def write_tickets(prefix):
    records, dropped, cost, seconds = [], 0, 0.0, []
    for path in sorted(BATCHES_DIR.glob(f"{prefix}_*.json")):
        batch = json.loads(path.read_text())
        written = batch["written"]
        for spec in batch["specs"]:
            record = (
                to_record(spec, written[spec["id"]]) if spec["id"] in written else None
            )
            records.append(record) if record else None
            dropped += record is None
        cost += batch["cost_usd"] or 0
        seconds.append(batch["seconds"])
    out = (
        TICKETS_FILE
        if prefix == "batch"
        else TICKETS_FILE.with_name(f"{prefix}_tickets.jsonl")
    )
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(
        f"{len(records)} tickets kept, {dropped} dropped, ${cost:.2f}, median {sorted(seconds)[len(seconds) // 2]} s per call → {out}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", type=int, default=480)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--prefix",
        default="batch",
        help="batch file prefix; anything but 'batch' stays out of training",
    )
    args = ap.parse_args()
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    n_batches = -(-args.tickets // args.batch)
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(run_batch, i, args.batch, args.seed, args.prefix): i
            for i in range(n_batches)
        }
        for future in as_completed(futures):
            try:
                b = future.result()
                print(
                    f"batch {b['index']}: {len(b['written'])} tickets, {b['seconds']} s, ${b['cost_usd']}",
                    flush=True,
                )
            except Exception as e:
                print(f"batch {futures[future]} failed: {e}", flush=True)
    write_tickets(args.prefix)


if __name__ == "__main__":
    main()
