"""Training time, latency and size of the ABCD-trained heads that compare.py scored against Claude.

Fits the same 15 heads as compare.py (one shared feature set, then one logistic regression per
question) on ABCD's 8,034 training openings, one head after another on one process. It then
answers the same 300 sampled test openings one at a time, as a live triage step would, and
times the 47-question synthetic-trained pipeline on them too.

python3 -I timing.py <repo> <out_dir>

Writes <out_dir>/timing_micro.json. No model calls; it costs $0.
"""

import gzip
import json
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from micro_llm import abcd  # noqa: E402
from micro_llm.models import text_features  # noqa: E402
from micro_llm.pipeline import Pipeline  # noqa: E402

CHOICE = ["primary_team", "journey_stage", "message_type"]
YES_NO = [
    "cannot_log_in", "security_related", "mentions_account_issue", "mentions_billing_issue",
    "mentions_bug", "refund_requested", "refund_implied", "needs_staff_action",
    "identity_details_given", "charge_amount_given",
]  # fmt: skip
QUESTIONS = ["flow", "intent"] + CHOICE + YES_NO

data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
train, test = data["train"], data["test"]
sample = random.Random(0).sample(test, 300)


def truth(c):
    t = abcd.our_truth(c, "opening")
    return {
        "flow": c["scenario"]["flow"],
        "intent": abcd.intent(c),
        **{q: t[q] for q in CHOICE + YES_NO},
    }


train_text = [abcd.customer_text(c, "opening") for c in train]
train_truth = [truth(c) for c in train]
sample_text = [abcd.customer_text(c, "opening") for c in sample]

# Training, one process, one head at a time
start = time.perf_counter()
features = text_features()
X = features.fit_transform(train_text)
fit_features_s = time.perf_counter() - start
heads, head_fit_s = {}, {}
for q in QUESTIONS:
    y = [t[q] for t in train_truth]
    rows = [i for i, v in enumerate(y) if v is not None]
    t0 = time.perf_counter()
    model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[rows], [y[i] for i in rows])
    head_fit_s[q] = round(time.perf_counter() - t0, 2)
    heads[q] = model
train_s = time.perf_counter() - start

# Size on disk
with tempfile.TemporaryDirectory() as tmp:
    joblib.dump({"features": features, "heads": heads}, f"{tmp}/m.pkl")
    size_mb = Path(f"{tmp}/m.pkl").stat().st_size / 1e6


def ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


# Latency: one opening at a time, after a warm-up
for text in sample_text[:20]:
    heads["intent"].predict(features.transform([text]))
feat_ms, heads_ms, total_ms = [], [], []
for text in sample_text:
    t0 = time.perf_counter()
    x = features.transform([text])
    t1 = time.perf_counter()
    for q in QUESTIONS:
        heads[q].predict(x)
    t2 = time.perf_counter()
    feat_ms.append((t1 - t0) * 1000)
    heads_ms.append((t2 - t1) * 1000)
    total_ms.append((t2 - t0) * 1000)
per_head_ms = {}
x = features.transform([sample_text[0]])
for q in QUESTIONS:
    per_head_ms[q] = round(
        statistics.median(ms(lambda: heads[q].predict(x)) for _ in range(200)), 4
    )

pipe = Pipeline()
for text in sample_text[:20]:
    pipe.run(text)
pipe_ms = [ms(lambda: pipe.run(text)) for text in sample_text]


def pcts(v):
    p50, p90, p99 = np.percentile(v, [50, 90, 99])
    return {
        "p50": round(float(p50), 3),
        "p90": round(float(p90), 3),
        "p99": round(float(p99), 3),
    }


out = {
    "machine": {
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "cores": subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "processes": 1,
    },
    "train": {
        "chats": len(train),
        "seconds_total": round(train_s, 1),
        "seconds_features": round(fit_features_s, 1),
        "seconds_per_head": head_fit_s,
        "feature_count": int(X.shape[1]),
    },
    "size_mb": round(size_mb, 1),
    "latency_ms_15_questions": {
        "openings": len(sample_text),
        "features": pcts(feat_ms),
        "heads": pcts(heads_ms),
        "total": pcts(total_ms),
        "per_head_p50": per_head_ms,
    },
    "latency_ms_47_question_pipeline": pcts(pipe_ms),
}  # fmt: skip
(OUT / "timing_micro.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
