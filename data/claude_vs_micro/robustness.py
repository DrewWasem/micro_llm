"""Does the micro-vs-Claude gap survive once Claude's abstentions are set aside?

Claude answered null for the topic or intent on some openings, and on our choice questions
it sometimes chose an answer ABCD's mapping never uses ("unclear", "other"). This scores both
systems on only the chats where Claude committed to an answer the truth can take, and checks
whether the chats Claude abstained on are short, information-poor openings.

python3 -I robustness.py <repo> <out_dir> [model=haiku]

Writes <out_dir>/robustness_<model>.json. No model calls; it costs $0.
"""

import gzip
import json
import statistics
import sys
from pathlib import Path

from scipy.stats import binomtest

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "haiku"
sys.path.insert(0, str(REPO))

from micro_llm import abcd  # noqa: E402

data = json.loads(gzip.decompress((abcd.ABCD_DIR / "abcd_v1.1.json.gz").read_bytes()))
by_id = {c["convo_id"]: c for c in data["test"]}
rows = json.loads((OUT / f"rows_{MODEL}.json").read_text())
words = {
    r["id"]: len(abcd.customer_text(by_id[r["id"]], "opening").split()) for r in rows
}


def committed(q, r):
    """Claude gave an answer that the truth for this question can take."""
    a = (r["claude"] or {}).get(q)
    truth_values = {x["truth"][q] for x in rows if x["truth"].get(q) is not None}
    return isinstance(a, str) and a.strip() in truth_values


out = {}
for q in ["flow", "intent", "primary_team", "journey_stage", "message_type"]:
    graded = [r for r in rows if r["truth"].get(q) is not None]
    kept = [r for r in graded if committed(q, r)]
    dropped = [r for r in graded if not committed(q, r)]
    micro_kept = [r["micro_abcd"][q] == r["truth"][q] for r in kept]
    claude_kept = [r["claude"][q].strip() == r["truth"][q] for r in kept]
    only_m = sum(m and not c for m, c in zip(micro_kept, claude_kept))
    only_c = sum(c and not m for m, c in zip(micro_kept, claude_kept))
    out[q] = {
        "graded": len(graded),
        "claude_committed": len(kept),
        "claude_abstained_or_unused_label": len(dropped),
        "claude_answers_dropped": dict(
            sorted(
                {
                    str((r["claude"] or {}).get(q)): sum(
                        str((x["claude"] or {}).get(q))
                        == str((r["claude"] or {}).get(q))
                        for x in dropped
                    )
                    for r in dropped
                }.items()
            )
        ),
        "accuracy_on_committed": {
            "micro_abcd": round(statistics.mean(micro_kept), 4),
            "claude": round(statistics.mean(claude_kept), 4),
            "mcnemar_p": round(
                binomtest(min(only_m, only_c), only_m + only_c, 0.5).pvalue, 4
            )
            if only_m + only_c
            else 1.0,
            "only_micro_right": only_m,
            "only_claude_right": only_c,
        },
        "micro_accuracy_on_dropped": round(
            statistics.mean(r["micro_abcd"][q] == r["truth"][q] for r in dropped), 4
        )
        if dropped
        else None,
        "opening_words_median": {
            "committed": statistics.median(words[r["id"]] for r in kept),
            "dropped": statistics.median(words[r["id"]] for r in dropped)
            if dropped
            else None,
        },
    }

(OUT / f"robustness_{MODEL}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
