"""Claude vs micro_llm on a random sample of ABCD test chats, reading only the customer's opening.

Three answerers, same chats, same questions:
- micro_syn: our 9 models, trained only on synthetic tickets (our questions only)
- micro_abcd: our model type retrained on ABCD's 8,034 training chats, one head per question
- claude: Claude through the CLI, zero-shot, given ABCD's label names and our question specs

python3 -I compare.py <repo> <out_dir> [n=200] [workers=12] [model=haiku]
n <= 5 is a pilot: it prints the prompt and Claude's replies and skips the micro models.
Claude's replies are cached in <out_dir>/claude_<model>.jsonl, so a rerun only asks for new chats.
"""

import gzip
import json
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
MODEL = sys.argv[5] if len(sys.argv) > 5 else "haiku"
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from micro_llm import abcd  # noqa: E402
from micro_llm.evaluate import hard  # noqa: E402
from micro_llm.models import text_features  # noqa: E402
from micro_llm.pipeline import Pipeline  # noqa: E402

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
OURS = CHOICE + YES_NO
ABCD_QS = ["flow", "intent"]

data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
bank = json.loads((REPO / "micro_llm" / "questions.json").read_text())
train, test = data["train"], data["test"]
sample = random.Random(0).sample(test, N)


def truth(c):
    t = abcd.our_truth(c, "opening")
    return {
        "flow": c["scenario"]["flow"],
        "intent": abcd.intent(c),
        **{q: t[q] for q in OURS},
    }


flows = {}
for c in train:
    flows.setdefault(c["scenario"]["flow"], set()).add(abcd.intent(c))

SYSTEM = (
    "You triage support chats for an online clothing store. "
    "Reply only with the JSON object asked for, with no other text."
)
QUESTIONS = "\n".join(
    [
        "Answer every key below about the customer's opening message. Reply with one JSON object.",
        "",
        '"flow": the topic, one of: ' + ", ".join(f'"{f}"' for f in sorted(flows)),
        '"intent": the specific intent, one of these, listed under their topic:',
        *[
            f"    {f}: " + ", ".join(f'"{i}"' for i in sorted(s))
            for f, s in sorted(flows.items())
        ],
        *[
            f'"{q}": {bank[q]["instructions"]}. One of:\n'
            + "\n".join(f'    "{k}": {v}' for k, v in bank[q]["criteria"].items())
            for q in CHOICE
        ],
        "Answer each of these with true or false:",
        *[f'"{q}": {bank[q]["instructions"]}' for q in YES_NO],
    ]
)


def prompt(c):
    text = abcd.customer_text(c, "opening")
    return f"{QUESTIONS}\n\nThe customer's opening message:\n<<<\n{text}\n>>>"


CMD = [
    "claude", "-p",
    "--model", MODEL,
    "--effort", "low",
    "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--no-session-persistence",
    "--setting-sources", "",
    "--system-prompt", SYSTEM,
]  # fmt: skip


def ask(c, retries=2):
    last = ""
    for _ in range(retries + 1):
        start = time.perf_counter()
        done = subprocess.run(
            CMD, input=prompt(c), capture_output=True, text=True, timeout=900,
            cwd=tempfile.gettempdir(),
        )  # fmt: skip
        wall = round(time.perf_counter() - start, 2)
        try:
            out = json.loads(done.stdout)
            reply = out["result"]
            answers = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])
            return {
                "id": c["convo_id"],
                "answers": answers,
                "wall_s": wall,
                "api_ms": out.get("duration_api_ms"),
                "cost": out.get("total_cost_usd"),
                "usage": out.get("usage"),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            last = (done.stderr or done.stdout)[-300:]
    return {"id": c["convo_id"], "answers": None, "error": last}


cache_file = OUT / f"claude_{MODEL}.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
cached = {}
if cache_file.exists():
    for line in cache_file.read_text().splitlines():
        r = json.loads(line)
        if r.get("answers") is not None:
            cached[r["id"]] = r
todo = [c for c in sample if c["convo_id"] not in cached]
lock = threading.Lock()
started = time.perf_counter()


def ask_and_save(c):
    r = ask(c)
    with lock:
        with cache_file.open("a") as f:
            f.write(json.dumps(r) + "\n")
        if r.get("answers") is not None:
            cached[r["id"]] = r
        n = len(cached)
        if n % 20 == 0:
            print(
                f"  {n}/{N} chats answered, {time.perf_counter() - started:.0f} s",
                flush=True,
            )
    return r


if N <= 5:
    print(prompt(sample[0]), "\n")
print(
    f"asking {MODEL} about {len(todo)} chats ({len(cached)} cached), {WORKERS} at a time",
    flush=True,
)
with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(ask_and_save, todo))
claude = [cached.get(c["convo_id"]) for c in sample]

if N <= 5:
    for c, r in zip(sample, claude):
        print(json.dumps(truth(c)))
        print(json.dumps(r))
    sys.exit()

# Our synthetic-trained pipeline
pipe = Pipeline()
syn = []
for c in sample:
    a = pipe.run(abcd.customer_text(c, "opening"))["answers"]
    syn.append({q: hard(q, a[q]) for q in OURS})

# Our model type retrained on ABCD's training chats
train_truth = [truth(c) for c in train]
with tempfile.TemporaryDirectory() as tmp:
    f = text_features()
    paths = (f"{tmp}/X.pkl", f"{tmp}/Xs.pkl")
    joblib.dump(
        f.fit_transform([abcd.customer_text(c, "opening") for c in train]), paths[0]
    )
    joblib.dump(
        f.transform([abcd.customer_text(c, "opening") for c in sample]), paths[1]
    )

    def head(q):
        y = [t[q] for t in train_truth]
        rows = [i for i, v in enumerate(y) if v is not None]
        X, Xs = (
            joblib.load(paths[0], mmap_mode="r"),
            joblib.load(paths[1], mmap_mode="r"),
        )
        model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[rows], [y[i] for i in rows])
        return model.predict(Xs).tolist()

    heads = joblib.Parallel(n_jobs=-1)(joblib.delayed(head)(q) for q in ABCD_QS + OURS)
retrained = [dict(zip(ABCD_QS + OURS, row)) for row in zip(*heads)]


def norm(q, v):
    """One comparable string per answer; anything malformed counts as wrong."""
    if q in YES_NO:
        if isinstance(v, str):
            v = {"true": True, "false": False}.get(v.strip().lower(), v)
        return (
            {True: "yes", False: "no"}.get(v, "malformed")
            if isinstance(v, bool)
            else "malformed"
        )
    return v.strip() if isinstance(v, str) else "malformed"


def grade(q, truths, preds):
    pairs = [(norm(q, t), norm(q, p)) for t, p in zip(truths, preds) if t is not None]
    tt, pp = [t for t, _ in pairs], [p for _, p in pairs]
    row = {
        "n": len(pairs),
        "accuracy": sum(t == p for t, p in pairs) / len(pairs),
        "baseline": Counter(tt).most_common(1)[0][1] / len(pairs),
    }
    if len(set(tt)) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["balanced"] = balanced_accuracy_score(tt, pp)
    return row


truths = [truth(c) for c in sample]
answered = [r is not None for r in claude]
results = {}
for q in ABCD_QS + OURS:
    t = [x[q] for x in truths]
    results[q] = {
        "micro_syn": grade(q, t, [s[q] for s in syn]) if q in OURS else None,
        "micro_abcd": grade(q, t, [r[q] for r in retrained]),
        "claude": grade(q, t, [r["answers"].get(q) if r else None for r in claude]),
    }

ok = [r for r in claude if r]
usage = [r.get("usage") or {} for r in ok]
summary = {
    "model": MODEL,
    "chats": N,
    "claude_answered": sum(answered),
    "claude_wall_s": statistics.median(r["wall_s"] for r in ok),
    "claude_api_s": statistics.median((r.get("api_ms") or 0) / 1000 for r in ok),
    "claude_output_tokens": statistics.median(u.get("output_tokens", 0) for u in usage),
    "claude_input_tokens": statistics.median(
        u.get("input_tokens", 0)
        + u.get("cache_read_input_tokens", 0)
        + u.get("cache_creation_input_tokens", 0)
        for u in usage
    ),
    "claude_cost_usd": round(sum(r.get("cost") or 0 for r in ok), 4),
}
(OUT / f"results_{MODEL}.json").write_text(
    json.dumps({"summary": summary, "questions": results}, indent=1)
)
(OUT / f"rows_{MODEL}.json").write_text(
    json.dumps(
        [
            {
                "id": c["convo_id"],
                "truth": t,
                "micro_syn": s,
                "micro_abcd": r,
                "claude": k and k["answers"],
            }
            for c, t, s, r, k in zip(sample, truths, syn, retrained, claude)
        ],
        default=str,
    )
)

print(json.dumps(summary, indent=1))
pct = lambda x: "  —  " if x is None else f"{100 * x:4.0f}%"  # noqa: E731
print(
    f"\n{'question':24} {'n':>4} {'common':>6} {'syn':>6} {'abcd':>6} {'claude':>6}   (balanced; accuracy for our 3 choice questions)"
)
for q, r in results.items():
    key = "accuracy" if q in CHOICE else "balanced"
    cell = lambda s: pct(s.get(key)) if s else pct(None)  # noqa: E731
    base = (
        r["claude"]["baseline"]
        if q in CHOICE
        else (1 / len({t[q] for t in truths if t[q] is not None}))
    )
    print(
        f"{q:24} {r['claude']['n']:4} {pct(base)} {cell(r['micro_syn'])} {cell(r['micro_abcd'])} {cell(r['claude'])}"
    )
