"""Draws the sample ticket's path through the 9 models as one HTML page.

    python3 -I scripts/pipeline_diagram.py <repo> docs/pipeline.html

Each of the 47 questions can be clicked to light up the pieces, models and rule behind its answer.
Numbers come from a fresh run of the sample ticket and from models/eval.json.
"""

import html
import json
import re
import sys
from pathlib import Path

repo, out_file = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from micro_llm import entities, splitter  # noqa: E402
from micro_llm.config import SAMPLE_TICKET  # noqa: E402
from micro_llm.pipeline import ROUTE_MIN, Pipeline  # noqa: E402
from micro_llm.questions import bank, qtype  # noqa: E402
from micro_llm.routing import MODEL_QUESTIONS, TEAM_SPECIALIST  # noqa: E402
from micro_llm.rules import NEEDS, TICKET_QUESTIONS, issue_p, kind_p, team_p  # noqa: E402

ev = json.loads((repo / "models" / "eval.json").read_text())
ticket = json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
predicted = json.loads(SAMPLE_TICKET.with_name("my_predictions.json").read_text())[
    "predictions"
]
run = Pipeline().run(ticket)
ans, outs, pieces, pairs = run["answers"], run["outputs"], run["pieces"], run["pairs"]
spans = splitter.split(ticket)
pids = [p["id"] for p in pieces]
ents = [entities.Entity(**e) for e in run["entities"]]
kinds = [{e.kind for e in ents if e.start < s.end and s.start < e.end} for s in spans]
ms = ev["sample"]["ms"]
N = len(pieces)
greeting = ticket[: spans[0].start].strip()
SHORT = {
    "billing_and_payments": "billing",
    "account_support": "account",
    "engineering": "engineering",
    "product_management": "product",
    "other": "other",
}

# ---------------------------------------------------------------- where each answer comes from

ROUTER_QS = [
    "primary_team",
    "category",
    "category_reordered",
    "mentions_billing_issue",
    "mentions_account_issue",
    "mentions_bug",
    "mentions_feature_request",
    "issue_count",
    "message_type",
]
# Entity question -> (what the regexes must find, the entity kinds that count; None = an Apple name)
ENTITY_QS = {
    "likely_apple_user": ("an Apple product or Apple Pay", None),
    "charge_amount_given": ("a money amount", {"money"}),
    "order_date_given": (
        "a date in a billing piece or beside an order number",
        {"date"},
    ),
    "error_message_given": ("quoted error text", {"error_text"}),
    "device_or_browser_given": ("a device or browser", {"device"}),
    "identity_details_given": (
        "an email, name or username",
        {"email", "name", "username"},
    ),
    "sensitive_data_present": (
        "a card number, password or address",
        {"card_number", "password", "address"},
    ),
}
# Code-rule question -> the answers it is built from
COMBINED = {
    "existing_customer": ["mentions_billing_issue"],
    "journey_stage": ["existing_customer"],
    "has_account": ["mentions_account_issue", "cannot_log_in", "existing_customer"],
    "churn_risk": ["threatens_to_leave", "frustration"],
    "priority": [
        "possible_widespread_issue",
        "bug_severity",
        "duplicate_charge",
        "refund_implied",
        "chargeback_mentioned",
        "cannot_log_in",
        "security_related",
        "time_pressure",
    ],
    "needs_staff_action": [
        "refund_requested",
        "refund_implied",
        "duplicate_charge",
        "cannot_log_in",
        "security_related",
    ],
    "needs_more_info": [],
}
GROUPS = {
    "billing": ("Billing", "strongest piece it read", MODEL_QUESTIONS["billing"]),
    "account": ("Account", "strongest piece it read", MODEL_QUESTIONS["account"]),
    "bug": ("Bug", "strongest piece it read", MODEL_QUESTIONS["bug"]),
    "feature": ("Feature", "strongest piece it read", MODEL_QUESTIONS["feature"]),
    "tone": ("Tone", "angriest, average, rudest, any", MODEL_QUESTIONS["tone"]),
    "router": ("Router", "kind and team of each piece", ROUTER_QS),
    "linker": ("Linker", "most related pair", MODEL_QUESTIONS["linker"]),
    "entities": ("Entities", "yes if a regex found it", list(ENTITY_QS)),
    "combined": ("Combined", "built from other answers", list(COMBINED)),
}
COLUMNS = [
    ["billing", "account", "bug", "feature"],
    ["tone", "router"],
    ["linker", "entities", "combined"],
]
listed = [q for g in GROUPS.values() for q in g[2]]
assert sorted(listed) == sorted(TICKET_QUESTIONS), set(listed) ^ set(TICKET_QUESTIONS)
GROUP_OF = {q: k for k, g in GROUPS.items() for q in g[2]}

# ---------------------------------------------------------------- answer text


def levels(q):
    c = bank()[q]["criteria"]
    return list(c.values()) if isinstance(c, dict) else list(c)


def val(q, a):
    """A short answer: a chance, a level or an option."""
    if qtype(q) == "noul":
        return f"{a:.2f}"
    if qtype(q) == "score":
        return f"{a['level']:.2f}"
    return a["answer"]


def long_val(q, a):
    if qtype(q) == "noul":
        return f"{a:.2f} chance of yes"
    if qtype(q) == "score":
        lv = a["level"]
        return f"level {lv:.2f} of {len(levels(q)) - 1} ({levels(q)[round(lv)]})"
    return f"{a['answer']} ({a['p'][a['answer']]:.2f})"


def expect(q):
    e = predicted[q]["expect"]
    if qtype(q) == "noul":
        return f"{e:.2f}"
    if qtype(q) == "score":
        return f"level {e}"
    return e


def either(words):
    words = sorted(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


IDS = re.compile(r"\b(" + "|".join(sorted(bank(), key=len, reverse=True)) + r")\b")


def code(s):
    """Escapes a step and sets question ids in code type."""
    return IDS.sub(r"<code>\1</code>", html.escape(s, quote=False))


# ---------------------------------------------------------------- every answer's path

SPLIT_STEP = f"The splitter cuts the ticket into {N} pieces" + (
    f" and drops the greeting “{greeting}”." if greeting else "."
)
SPLIT = ["ticket", "rail-a", "rail-split", "splitter", "rules", "rules-out"]
ENT = [
    "ticket",
    "rail-a",
    "rail-b",
    "efind",
    "efind-out",
    "ents",
    "lane-ent",
    "ent-note",
    "rules",
    "rules-out",
]


def lane(i):
    return [f"cut-{pids[i]}", f"piece-{pids[i]}", f"lane-{pids[i]}"]


def router(i):
    return lane(i) + [f"read-router-{pids[i]}", f"router-{pids[i]}"]


def tone(i):
    return lane(i) + [f"read-tone-{pids[i]}", f"tone-{pids[i]}"]


def spec(i, m):
    return router(i) + [f"route-{pids[i]}-{m}", f"spec-{m}-{pids[i]}"]


Q = {}


def add(q, headline, steps, nodes, feeds=()):
    t = ev["ticket_answers"][q]
    Q[q] = {
        "group": GROUP_OF[q],
        "text": html.escape(bank()[q]["instructions"]),
        "headline": code(headline),
        "steps": [code(s) for s in steps],
        "nodes": sorted(set(nodes) | {f"q-{q}", f"g-{GROUP_OF[q]}"}),
        "feeds": list(feeds),
        "got": html.escape(long_val(q, ans[q])),
        "expect": html.escape(expect(q)),
        "acc": f"{t['accuracy']:.0%}",
        "base": f"{t['baseline']:.0%}",
    }


def team_for(i, m):
    """The team that routed piece i to model m, or None when a routing rule did."""
    teams = outs[i]["router"]["primary_team"]["p"]
    for team, name in TEAM_SPECIALIST.items():
        if name == m and teams.get(team, 0) >= ROUTE_MIN:
            return team, teams[team]
    return None, None


def reason(i, m):
    team, p = team_for(i, m)
    if team:
        return f"it is a {pieces[i]['kind']} for {team} ({p:.2f}, at least {ROUTE_MIN})"
    return "an account problem also goes to bug"


def edge_label(i, m):
    team, p = team_for(i, m)
    return f"{pieces[i]['kind']} · {SHORT[team]} {p:.2f}" if team else "account problem"


SPEC_RULE = {
    "noul": "take the highest chance across those pieces",
    "score": "take the strongest level across those pieces",
    "choice": "take the answer most likely to name a method",
}
for m in ["billing", "account", "bug", "feature"]:
    read = [i for i in range(N) if m in pieces[i]["models"]]
    names = " and ".join(pids[i] for i in read) or "no piece"
    for q in MODEL_QUESTIONS[m]:
        steps = [SPLIT_STEP]
        steps += [f"The router sends {pids[i]} to {m}: {reason(i, m)}." for i in read]
        steps.append(
            f"The {m} model reads "
            + ", ".join(f"{pids[i]} → {val(q, outs[i][m][q])}" for i in read)
            + "."
        )
        steps.append(f"Rule: {SPEC_RULE[qtype(q)]} → {val(q, ans[q])}.")
        add(
            q,
            f"comes from the {m} model reading {names}",
            steps,
            SPLIT + [n for i in read for n in spec(i, m)],
        )

TONE_RULE = {
    "frustration": "take the angriest piece",
    "sentiment": "average the pieces' mood",
    "politeness": "take the rudest piece",
}
for q in MODEL_QUESTIONS["tone"]:
    reads = ", ".join(f"{pids[i]} → {val(q, outs[i]['tone'][q])}" for i in range(N))
    rule = TONE_RULE.get(q, "take the highest chance, so any one piece can raise it")
    add(
        q,
        f"comes from the tone model reading all {N} pieces",
        [
            SPLIT_STEP,
            f"The tone model reads every piece: {reads}.",
            f"Rule: {rule} → {val(q, ans[q])}.",
        ],
        SPLIT + [n for i in range(N) for n in tone(i)],
    )

issue_idx = [i for i in range(N) if issue_p(outs[i]) >= 0.5]
add(
    "issues_related",
    f"comes from the linker reading {len(pairs)} pairs of issue pieces",
    [
        SPLIT_STEP,
        f"The router marks {either(pids[i] for i in issue_idx)} as issues: a problem or request chance of 0.5 or more.",
        "The linker reads each pair: "
        + ", ".join(f"{p['a']}+{p['b']} → {p['issues_related']:.2f}" for p in pairs)
        + ".",
        f"Rule: take the most related pair → {ans['issues_related']:.2f}.",
    ],
    SPLIT
    + [n for i in issue_idx for n in router(i)]
    + ["pairs", "pairs-edge", "linker", "lane-pairs"],
)

first = issue_idx[0] if issue_idx else 0
ALL_R = SPLIT + [n for i in range(N) for n in router(i)]
pt, cat = ans["primary_team"], ans["category"]
add(
    "primary_team",
    f"comes from the router reading {pids[first]}, the first issue piece",
    [
        SPLIT_STEP,
        "The router reads every piece.",
        f"{pids[first]} is the first issue piece, so its team wins → {pt['answer']} ({pt['p'][pt['answer']]:.2f}).",
    ],
    SPLIT + router(first),
)
for q in ("category", "category_reordered"):
    last = (
        "Code has no position bias, so the reordered question copies category"
        if q == "category_reordered"
        else "Rule: the largest share of the votes wins"
    )
    add(
        q,
        "comes from team votes by the issue pieces",
        [
            SPLIT_STEP,
            "The router reads every piece.",
            f"Each issue piece votes with its team chances, and {pids[first]} votes double because it comes first.",
            f"{last} → {cat['answer']} ({cat['p'][cat['answer']]:.2f}).",
        ],
        SPLIT + [n for i in issue_idx for n in router(i)],
        ["category"] if q == "category_reordered" else [],
    )

MENTIONS = {
    "mentions_billing_issue": (
        "issue chance × billing team chance",
        lambda o: issue_p(o) * team_p(o, "billing_and_payments"),
    ),
    "mentions_account_issue": (
        "issue chance × account team chance",
        lambda o: issue_p(o) * team_p(o, "account_support"),
    ),
    "mentions_bug": (
        "problem chance × (engineering + account team chance)",
        lambda o: kind_p(o, "problem")
        * (team_p(o, "engineering") + team_p(o, "account_support")),
    ),
    "mentions_feature_request": (
        "request chance × product team chance",
        lambda o: kind_p(o, "request") * team_p(o, "product_management"),
    ),
}
for q, (formula, f) in MENTIONS.items():
    vals = [f(o) for o in outs]
    w = max(range(N), key=vals.__getitem__)
    add(
        q,
        "comes from the router's kind and team chances for every piece",
        [
            SPLIT_STEP,
            "The router reads every piece.",
            f"Rule: take the highest {formula} → {ans[q]:.2f}, from {pids[w]}.",
        ],
        ALL_R,
    )

ic = ans["issue_count"]
add(
    "issue_count",
    "comes from the router's issue chance for every piece",
    [
        SPLIT_STEP,
        "The router gives each piece an issue chance: "
        + ", ".join(f"{pids[i]} {issue_p(o):.2f}" for i, o in enumerate(outs))
        + ".",
        f"Rule: count them as independent chances → {ic['answer']} ({ic['p'][ic['answer']]:.2f}).",
    ],
    ALL_R,
)
problem = max(kind_p(o, "problem") for o in outs)
add(
    "message_type",
    "comes from the router's kind chances for every piece",
    [
        SPLIT_STEP,
        "The router reads every piece.",
        f"Rule: problem_report takes the highest problem chance ({problem:.2f}), and request and other share the rest → {ans['message_type']['answer']}.",
    ],
    ALL_R,
)

for q, (what, wanted) in ENTITY_QS.items():
    hits = [
        e
        for e in ents
        if (entities.is_apple(e) if wanted is None else e.kind in wanted)
    ]
    steps = [
        "The entity finder scans the whole ticket with regexes.",
        (
            "It finds " + ", ".join(f"“{e.text}” ({e.kind})" for e in hits) + "."
            if hits
            else f"It finds no {what}."
        ),
    ]
    nodes = ENT + [f"ent-{e.start}" for e in hits]
    if q == "order_date_given" and hits:
        i = next(k for k, s in enumerate(spans) if s.start <= hits[0].start < s.end)
        holds = " and holds an order number" if "order_number" in kinds[i] else ""
        steps.append(
            f"The date sits in {pids[i]}, which the router gives to billing ({team_p(outs[i], 'billing_and_payments'):.2f}){holds}."
        )
        nodes += SPLIT + router(i)
    steps.append(f"Rule: yes when it finds {what} → {ans[q]:.2f}.")
    add(q, "comes from the entity finder's regexes", steps, nodes)


def feed_nodes(qs):
    return [n for f in qs for n in Q[f]["nodes"]]


def fx(q):
    return f"{ans[q]:.2f}"


def lv(q):
    return f"{ans[q]['level']:.2f}"


order = [e for e in ents if e.kind == "order_number"]
add(
    "existing_customer",
    "comes from an order number, or else mentions_billing_issue",
    (
        [
            f"The entity finder finds order number “{order[0].text}”, so the answer is 1.00.",
            f"Without one it would copy mentions_billing_issue ({fx('mentions_billing_issue')}).",
        ]
        if order
        else [
            f"No order number appears, so it copies mentions_billing_issue → {fx('existing_customer')}."
        ]
    ),
    feed_nodes(["mentions_billing_issue"]) + ENT + [f"ent-{e.start}" for e in order],
    ["mentions_billing_issue"],
)
add(
    "journey_stage",
    "is computed from existing_customer",
    [
        f"post_purchase takes existing_customer's chance ({fx('existing_customer')}), and unclear takes the rest → {ans['journey_stage']['answer']}."
    ],
    feed_nodes(["existing_customer"]),
    ["existing_customer"],
)
add(
    "has_account",
    "is computed from 3 other answers",
    [
        "Rule: take the highest of "
        + ", ".join(f"{f} {fx(f)}" for f in COMBINED["has_account"])
        + f" → {fx('has_account')}."
    ],
    feed_nodes(COMBINED["has_account"]),
    COMBINED["has_account"],
)
add(
    "churn_risk",
    "is computed from threatens_to_leave and frustration",
    [
        f"Level 2 takes threatens_to_leave's chance ({fx('threatens_to_leave')}).",
        f"Level 1 takes the rest × frustration ÷ 2 ({lv('frustration')} ÷ 2), and level 0 takes what is left → {lv('churn_risk')}.",
    ],
    feed_nodes(COMBINED["churn_risk"]),
    COMBINED["churn_risk"],
)
blocking = ans["bug_severity"]["p"][2]
urgent = ans["possible_widespread_issue"] * blocking
largest = max(
    ans[f]
    for f in COMBINED["priority"]
    if f not in ("possible_widespread_issue", "bug_severity")
)
high = (1 - urgent) * largest
add(
    "priority",
    f"is computed from {len(COMBINED['priority'])} other answers",
    [
        f"Urgent: possible_widespread_issue {fx('possible_widespread_issue')} × the chance bug_severity is blocking ({blocking:.2f}) = {urgent:.2f}.",
        f"High: the rest × the largest of duplicate_charge, refund_implied, chargeback_mentioned, cannot_log_in, security_related and time_pressure ({largest:.2f}) = {high:.2f}.",
        f"Normal takes the rest × the top problem chance ({problem:.2f}), and low takes what is left → {lv('priority')}.",
    ],
    feed_nodes(COMBINED["priority"]) + ALL_R,
    COMBINED["priority"],
)
add(
    "needs_staff_action",
    f"is computed from {len(COMBINED['needs_staff_action'])} other answers",
    [
        "Rule: take the highest of "
        + ", ".join(f"{f} {fx(f)}" for f in COMBINED["needs_staff_action"])
        + f" → {fx('needs_staff_action')}."
    ],
    feed_nodes(COMBINED["needs_staff_action"]),
    COMBINED["needs_staff_action"],
)
gap, gi, gteam = max(
    (
        (kind_p(o, "problem") * team_p(o, t), i, t)
        for i, (o, k) in enumerate(zip(outs, kinds))
        for t, need in NEEDS.items()
        if not need & k
    ),
    default=(0.0, None, None),
)
add(
    "needs_more_info",
    "comes from the router and the entity finder",
    [
        "The router reads every piece, and the entity finder lists the details in each one.",
        (
            f"{pids[gi]} is a {SHORT[gteam]} problem with no {either(k.replace('_', ' ') for k in NEEDS[gteam])}, so support must ask ({gap:.2f})."
            if gteam
            else "Every problem piece has the details its team needs."
        ),
        f"Rule: take the highest such chance → {fx('needs_more_info')}.",
    ],
    ALL_R + ENT + [f"ent-{e.start}" for e in ents],
)
assert sorted(Q) == sorted(TICKET_QUESTIONS)

# ---------------------------------------------------------------- the drawing

W = 1336
Y0, LH, GAP = 108, 76, 10
PY = Y0 + N * (LH + GAP)  # pairs lane
EY = PY + 52 + GAP  # entities lane
BOTTOM = EY + 64
H = BOTTOM + 16


def ly(i):
    return Y0 + i * (LH + GAP)


def esc(s):
    return html.escape(str(s))


def T(x, y, s, cls="t", anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x}" y="{y}" class="{cls}"{a}>{esc(s)}</text>'


def R(x, y, w, h, cls, rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def P(d, cls="edge"):
    return f'<path d="{d}" class="{cls}"/>'


def G(n, inner, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<g data-n="{n}"{c}>{inner}</g>'


def wrap(s, width):
    lines, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + [cur]


def ticket_text():
    cuts = sorted(
        {0, len(ticket)}
        | {s.start for s in spans}
        | {s.end for s in spans}
        | {e.start for e in ents}
        | {e.end for e in ents}
    )
    out, dy = ['<tspan class="lab">TICKET   </tspan>'], 0
    for a, b in zip(cuts, cuts[1:]):
        for s in spans:
            if s.start == a:
                out.append(f'<tspan class="pm" dy="-5">{s.id} </tspan>')
                dy = 5
        inside = any(s.start <= a and b <= s.end for s in spans)
        named = any(e.start <= a and b <= e.end for e in ents)
        cls = (
            "ent"
            if named
            else ""
            if inside
            else "drop"
            if a < spans[0].start
            else "gap"
        )
        shift = f' dy="{dy}"' if dy else ""
        out.append(f'<tspan class="{cls}"{shift}>{esc(ticket[a:b])}</tspan>')
        dy = 0
    return f'<text x="30" y="39" class="tk">{"".join(out)}</text>'


wave3 = sum(len(p["models"]) - 2 for p in pieces) + len(pairs)
S = []
# Lanes first, so everything else draws on top of them
for i, pid in enumerate(pids):
    y = ly(i)
    S.append(
        G(f"lane-{pid}", R(144, y, 592, LH, "lane", 10) + P(f"M736,{y + 38} H755"))
    )
S.append(G("lane-pairs", R(144, PY, 592, 52, "lane", 10) + P(f"M736,{PY + 26} H755")))
S.append(G("lane-ent", R(144, EY, 592, 64, "lane", 10) + P(f"M736,{EY + 32} H755")))

S.append(G("ticket", R(16, 12, W - 32, 44, "box", 8) + ticket_text()))
for x, title, sub, anchor in [
    (82, "WAVE 1", f"rules · {ms['split_and_entities']:.2f} ms", "middle"),
    (
        266,
        f"{N} PIECES",
        "greeting dropped" if greeting else "one per sentence",
        "middle",
    ),
    (450, "WAVE 2", f"{2 * N} runs · {ms['router_and_tone']:.2f} ms", "middle"),
    (670, "WAVE 3", f"{wave3} runs · {ms['specialists_and_linker']:.2f} ms", "middle"),
    (774, "WAVE 4", f"{ms['rules']:.2f} ms", "middle"),
    (816, f"{len(TICKET_QUESTIONS)} ANSWERS", "click one to trace its path", "start"),
]:
    S.append(T(x, 80, title, "hd", anchor) + T(x, 95, sub, "sub", anchor))

# Wave 1: the ticket feeds the splitter and the entity finder
S.append(G("rail-a", P("M20,56 V129", "edge plain")))
S.append(G("rail-split", P("M20,129 H35")))
S.append(G("rail-b", P(f"M20,129 V{EY + 32} H35")))
top, bottom = ly(0) + 6, ly(N - 1) + LH - 6
S.append(
    G(
        "splitter",
        R(36, top, 92, bottom - top, "box", 8)
        + T(82, top + 22, "splitter", "t b", "middle")
        + T(82, top + 42, "cuts at . ! ?", "sub", "middle")
        + T(82, top + 57, "and at “, and”", "sub", "middle")
        + (T(82, top + 72, f"drops “{greeting}”", "sub", "middle") if greeting else ""),
    )
)

# Waves 2 and 3: one lane per piece
for i, pid in enumerate(pids):
    y, mid = ly(i), ly(i) + 38
    lines = wrap(pieces[i]["text"], 29)
    base = mid - (len(lines) - 1) * 8 + 4
    S.append(G(f"cut-{pid}", P(f"M128,{mid} H153")))
    S.append(
        G(
            f"piece-{pid}",
            R(154, y + 6, 224, 64, "box", 8)
            + T(165, base, pid, "t b")
            + "".join(T(190, base + k * 16, line) for k, line in enumerate(lines)),
        )
    )
    S.append(
        G(f"read-router-{pid}", P(f"M378,{mid} C391,{mid} 391,{y + 21} 403,{y + 21}"))
    )
    S.append(
        G(f"read-tone-{pid}", P(f"M378,{mid} C391,{mid} 391,{y + 55} 403,{y + 55}"))
    )
    S.append(
        G(
            f"router-{pid}",
            R(404, y + 8, 92, 26, "chip") + T(450, y + 25, "router", "t", "middle"),
            "m-router",
        )
    )
    S.append(
        G(
            f"tone-{pid}",
            R(404, y + 42, 92, 26, "chip") + T(450, y + 59, "tone", "t", "middle"),
            "m-tone",
        )
    )
    specs = [m for m in pieces[i]["models"] if m not in ("router", "tone")]
    for j, m in enumerate(specs):
        cy = y + 21 + j * 34
        d = (
            f"M496,{y + 21} H619"
            if j == 0
            else f"M496,{y + 21} C556,{y + 21} 560,{cy} 619,{cy}"
        )
        label = (
            T(558, y + 16, edge_label(i, m), "lbl", "middle")
            if j == 0
            else T(614, cy + 17, edge_label(i, m), "lbl", "end")
        )
        S.append(G(f"route-{pid}-{m}", P(d) + label))
        S.append(
            G(
                f"spec-{m}-{pid}",
                R(620, cy - 13, 100, 26, "chip") + T(670, cy + 4, m, "t", "middle"),
                f"m-{m}",
            )
        )
    if not specs:
        k = pieces[i]["kind"]
        kp = outs[i]["router"]["piece_kind"]["p"][k]
        S.append(
            G(
                f"route-{pid}-none",
                P(f"M496,{y + 21} H619", "edge dash")
                + T(558, y + 16, f"{k} {kp:.2f}", "lbl", "middle"),
            )
        )
        S.append(
            G(
                f"spec-none-{pid}",
                R(620, y + 8, 100, 26, "ghost")
                + T(670, y + 25, "no specialist", "sub", "middle"),
            )
        )

# Wave 3: the linker reads each pair of issue pieces
S.append(
    G(
        "pairs",
        R(154, PY + 6, 224, 40, "box", 8)
        + T(165, PY + 21, "pairs of issue pieces", "sub")
        + T(165, PY + 37, "  ·  ".join(f"{p['a']}+{p['b']}" for p in pairs), "t b"),
    )
)
S.append(
    G(
        "pairs-edge",
        P(f"M378,{PY + 26} H619")
        + T(499, PY + 21, "issue = problem or request ≥ 0.5", "lbl", "middle"),
    )
)
S.append(
    G(
        "linker",
        R(620, PY + 13, 100, 26, "chip")
        + T(670, PY + 30, f"linker ×{len(pairs)}", "t", "middle"),
        "m-linker",
    )
)

# Wave 1, second rule: entities are exact spans of the whole ticket
assert len(ents) <= 3, "the entity card holds three lines"
S.append(
    G(
        "efind",
        R(36, EY + 6, 92, 52, "box", 8)
        + T(82, EY + 28, "entity finder", "t b", "middle")
        + T(82, EY + 44, "regexes", "sub", "middle"),
    )
)
S.append(G("efind-out", P(f"M128,{EY + 32} H153")))
S.append(G("ents", R(154, EY + 6, 224, 52, "box", 8)))
for k, e in enumerate(ents):
    S.append(
        G(
            f"ent-{e.start}",
            T(165, EY + 22 + k * 15, e.text, "t b")
            + T(368, EY + 22 + k * 15, e.kind, "sub", "end"),
        )
    )
S.append(
    G("ent-note", T(560, EY + 36, "no model runs: entities are exact text spans", "sub", "middle"))
)

# Wave 4: rules merge piece answers into ticket answers
S.append(
    G(
        "rules",
        R(756, Y0, 36, BOTTOM - Y0, "bar", 8)
        + f'<text class="bartxt" text-anchor="middle" transform="translate(778,{(Y0 + BOTTOM) / 2}) rotate(-90)">rules merge piece answers into ticket answers</text>',
    )
)
S.append(G("rules-out", P(f"M792,{(Y0 + BOTTOM) / 2} H811")))

for c, keys in enumerate(COLUMNS):
    x, y = 816 + c * 172, Y0
    for key in keys:
        title, rule, qs = GROUPS[key]
        S.append(
            G(
                f"g-{key}",
                f'<circle cx="{x + 5}" cy="{y + 9}" r="4" class="dot"/>'
                + T(x + 14, y + 13, f"{title} · {len(qs)}", "gt")
                + T(x + 14, y + 27, rule, "sub"),
                f"m-{key}",
            )
        )
        y += 34
        for q in qs:
            S.append(
                f'<g data-n="q-{q}" data-q="{q}" class="qrow m-{key}" tabindex="0" role="button" aria-label="Trace {q}">'
                + R(x, y, 160, 17, "qbg", 4)
                + T(x + 14, y + 12.5, q, "qt")
                + "</g>"
            )
            y += 17
        y += 12
    assert y - 12 <= BOTTOM, f"answer column {c} runs past the lanes"

CLAIM = (
    f"How the sample ticket moves: rules cut it into {N} pieces, the router and tone models read every piece, "
    f"the router sends each issue piece to its specialists, the linker reads each pair of issues, "
    f"and rules merge all {run['model_runs']} model runs into {len(TICKET_QUESTIONS)} answers."
)
SVG = (
    f'<svg id="flow" viewBox="0 0 {W} {H}" role="group" aria-label="{esc(CLAIM)}">'
    "<defs>"
    '<marker id="ah" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L10,5 L0,9 z" class="ahead"/></marker>'
    '<marker id="ah-on" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,1 L10,5 L0,9 z" class="ahead-on"/></marker>'
    "</defs>" + "".join(S) + "</svg>"
)

# ---------------------------------------------------------------- the page

TRY = ["duplicate_charge", "frustration", "category", "priority", "likely_apple_user"]
total_ms = ms["total"]
page = (
    (Path(__file__).with_name("pipeline_template.html").read_text())
    .replace(
        "__H1__",
        f"The router sends each ticket piece only to the specialists it needs, so {len(TICKET_QUESTIONS)} answers take {run['model_runs']} model runs and {total_ms:.1f} ms.",
    )
    .replace("__SVG__", SVG)
    .replace(
        "__CAPTION__",
        esc(CLAIM) + " Colors mark the model or rule that answers each group.",
    )
    .replace(
        "__TRY__",
        " ".join(f'<button type="button" data-pick="{q}">{q}</button>' for q in TRY),
    )
    .replace("__N__", str(len(TICKET_QUESTIONS)))
    .replace("__TEST__", str(ev["test_tickets"]))
    .replace("__DATA__", json.dumps(Q, ensure_ascii=False))
)
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(page)
print(out_file, f"{len(page) / 1024:.0f} KB")
