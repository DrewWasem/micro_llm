"""Write docs/paper/evidence_ledger.md from the result files: python3 -I scripts/make_ledger.py <repo> <out.md>"""

import json
import sys
from collections import Counter
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
cvm = repo / "data/claude_vs_micro"
S = json.loads((cvm / "stats_haiku.json").read_text())
R = json.loads((cvm / "results_haiku.json").read_text())
B = json.loads((cvm / "robustness_haiku.json").read_text())
M = json.loads((cvm / "metrics_haiku.json").read_text())
T = json.loads((cvm / "timing_micro.json").read_text())
A = json.loads((repo / "data/abcd_analysis.json").read_text())
E = json.loads((repo / "models/eval.json").read_text())
rows = json.loads((cvm / "rows_haiku.json").read_text())

lines, n = [], [0]


def L(value, what, source):
    n[0] += 1
    lines.append(f"L{n[0]} | {value} | {what} | {source}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


L("8,034", "ABCD training chats", "data/abcd_analysis.json profile.chats.train = 8034")
L(
    "1,004",
    "ABCD dev chats; also test chats",
    "data/abcd_analysis.json profile.chats.dev = 1004, .test = 1004",
)
L("10,042", "ABCD chats in all", "derived: 8,034 + 1,004 + 1,004")
L("10", "ABCD flows (topics)", "data/abcd_analysis.json len(profile.flows) = 10")
L("55", "ABCD intents", "data/abcd_analysis.json len(profile.intents) = 55")
L("6b8700c", "ABCD commit", "data/abcd_analysis.json source.commit = 6b8700ce67c6...")
w = A["profile"]["words"]
L(
    "16",
    "median words in an ABCD opening; p10 6, p90 28",
    f"data/abcd_analysis.json profile.words.abcd_opening = {w['abcd_opening']}",
)
L(
    "49",
    "median words in all a customer's ABCD messages",
    f"profile.words.abcd_all_customer.median = {w['abcd_all_customer']['median']}",
)
L(
    "55",
    "median words in one synthetic ticket",
    f"profile.words.our_tickets.median = {w['our_tickets']['median']}",
)
L(
    "12.5%",
    "opening words never seen in the synthetic tickets",
    "profile.words_unseen_in_our_tickets.opening = 0.125",
)
L("86.8%", "chats where the agent speaks first", "profile.agent_speaks_first = 0.868")
L(
    "300",
    "ABCD test chats sampled, random.Random(0).sample",
    "data/claude_vs_micro/results_haiku.json summary.chats = 300",
)
L(
    "53",
    "intents present among the 300 sampled chats",
    "rows_haiku.json, distinct truth.intent = 53",
)
L("15", "questions compared", "stats_haiku.json summary.questions = 15")
L(
    "10,000",
    "bootstrap resamples and permutations",
    "stats_haiku.json summary.reps = 10000",
)
L(
    "38,990",
    "TF-IDF features",
    "data/claude_vs_micro/timing_micro.json train.feature_count = 38990",
)
L(
    "4.0",
    "logistic regression C; class_weight balanced; max_iter 3,000",
    "micro_llm/abcd.py:666 and compare.py LogisticRegression(C=4.0, ...)",
)

# Main results
ms = S["summary"]
L(
    pct(ms["mean_score"]["micro_abcd"]),
    "heads' mean score over 15 questions",
    f"stats_haiku.json summary.mean_score.micro_abcd = {ms['mean_score']['micro_abcd']}",
)
L(
    f"{ms['mean_ci95']['micro_abcd']}",
    "heads' mean score, 95% CI",
    "summary.mean_ci95.micro_abcd",
)
L(
    pct(ms["mean_score"]["claude"]),
    "Claude's mean score over 15 questions",
    f"summary.mean_score.claude = {ms['mean_score']['claude']}",
)
L(
    f"{ms['mean_ci95']['claude']}",
    "Claude's mean score, 95% CI",
    "summary.mean_ci95.claude",
)
L(
    f"{ms['mean_diff_points']} points",
    "mean difference, heads minus Claude",
    "summary.mean_diff_points",
)
L(
    f"{ms['mean_diff_ci95_points']}",
    "mean difference, 95% CI in points",
    "summary.mean_diff_ci95_points",
)
L(
    f"{len(ms['micro_ahead_holm_005'])}",
    "questions where the heads lead at Holm 0.05",
    f"summary.micro_ahead_holm_005 = {ms['micro_ahead_holm_005']}",
)
L(
    f"{len(ms['claude_ahead_holm_005'])}",
    "questions where Claude leads at Holm 0.05",
    f"summary.claude_ahead_holm_005 = {ms['claude_ahead_holm_005']}",
)

for q, v in S["questions"].items():
    m, c, d = (
        v["systems"]["micro_abcd"],
        v["systems"]["claude"],
        v["micro_abcd_minus_claude"],
    )
    mc = d["mcnemar"]
    full = R["questions"][q]
    L(
        f"n {v['n']}; heads {pct(full['micro_abcd'][v['metric']])} {m['ci95']}; Claude {pct(full['claude'][v['metric']])} {c['ci95']}",
        f"{q}: {v['metric']} score, {v['classes']} classes; Claude answers the truth can't take {c['answers_outside_labels']}; heads {m['answers_outside_labels']}",
        f"point estimates: results_haiku.json questions.{q}.*.{v['metric']}; intervals: stats_haiku.json questions.{q}.systems",
    )
    L(
        f"{d['points']} points {d['ci95_points']}; permutation p {d['permutation_p']}, Holm p {d['permutation_p_holm']}",
        f"{q}: difference, heads minus Claude",
        f"questions.{q}.micro_abcd_minus_claude",
    )
    L(
        f"only heads right {mc['only_micro_right']}; only Claude right {mc['only_claude_right']}; McNemar p {mc['p']}",
        f"{q}: discordant chats",
        f"questions.{q}.micro_abcd_minus_claude.mcnemar",
    )

for q in ("primary_team", "journey_stage", "message_type"):
    x = R["questions"][q]
    L(
        f"heads {pct(x['micro_abcd']['balanced'])}; Claude {pct(x['claude']['balanced'])}; majority baseline {pct(x['claude']['baseline'])}",
        f"{q}: balanced accuracy, and the majority-class accuracy",
        f"results_haiku.json questions.{q}.*.balanced, .baseline",
    )

YN_PLAIN = ["cannot_log_in", "mentions_bug", "refund_implied", "needs_staff_action"]
yn = [
    "cannot_log_in",
    "security_related",
    "mentions_account_issue",
    "mentions_billing_issue",
    "mentions_bug",
    "refund_requested",
    "refund_implied",
    "needs_staff_action",
    "identity_details_given",
    "charge_amount_given",
]
for q in yn:
    cnt = Counter(r["truth"][q] for r in rows)
    L(
        f"yes {cnt[True]}, no {cnt[False]}",
        f"{q}: truth classes among the 300; {cnt[None]} chats have no truth",
        f"rows_haiku.json truth.{q}",
    )

# Scoring rule
for name, label in (
    ("plain_throughout", "plain accuracy on all 15 questions"),
    ("balanced_throughout", "balanced accuracy on all 15 questions"),
):
    x = M["means"][name]
    L(
        f"heads {pct(x['micro_abcd'])} {x['ci95']['micro_abcd']}; Claude {pct(x['claude'])} {x['ci95']['claude']}; difference {x['diff_points']} points {x['diff_ci95_points']}",
        f"mean of 15 scored by {label}; joint bootstrap over chats, 10,000 resamples",
        f"data/claude_vs_micro/metrics_haiku.json means.{name}",
    )
py = M["yes_no_pooled_recall"]
L(
    f"heads {py['yes']['micro_abcd'][0]} of {py['yes']['micro_abcd'][1]} ({pct(py['yes']['micro_abcd'][0] / py['yes']['micro_abcd'][1])}); Claude {py['yes']['claude'][0]} of {py['yes']['claude'][1]} ({pct(py['yes']['claude'][0] / py['yes']['claude'][1])})",
    "'yes' cases answered yes, pooled over the 10 yes/no questions",
    "metrics_haiku.json yes_no_pooled_recall.yes",
)
L(
    f"heads {py['no']['micro_abcd'][0]:,} of {py['no']['micro_abcd'][1]:,} ({pct(py['no']['micro_abcd'][0] / py['no']['micro_abcd'][1])}); Claude {py['no']['claude'][0]:,} of {py['no']['claude'][1]:,} ({pct(py['no']['claude'][0] / py['no']['claude'][1])})",
    "'no' cases answered no, pooled over the 10 yes/no questions",
    "metrics_haiku.json yes_no_pooled_recall.no",
)
ahead = M["yes_no_claude_ahead_on_plain_accuracy"]
gaps = [
    100
    * (
        M["questions"][q]["accuracy"]["claude"]
        - M["questions"][q]["accuracy"]["micro_abcd"]
    )
    for q in ahead
]
L(
    f"{len(ahead)} of 10; by {min(gaps):.1f} to {max(gaps):.1f} points",
    f"yes/no questions where Claude's plain accuracy is higher: {', '.join(ahead)}",
    "metrics_haiku.json yes_no_claude_ahead_on_plain_accuracy, questions.*.accuracy",
)
for q in YN_PLAIN:
    x = M["questions"][q]
    L(
        f"heads {pct(x['accuracy']['micro_abcd'])}; Claude {pct(x['accuracy']['claude'])}; yes found: heads {x['recall_yes']['micro_abcd'][0]}, Claude {x['recall_yes']['claude'][0]} of {x['recall_yes']['claude'][1]}; no kept: heads {x['recall_no']['micro_abcd'][0]}, Claude {x['recall_no']['claude'][0]} of {x['recall_no']['claude'][1]}",
        f"{q}: plain accuracy and recall on each class",
        f"metrics_haiku.json questions.{q}",
    )
mh = M["mcnemar_holm_005"]
L(
    f"{len(mh['micro_ahead'])}; Claude {len(mh['claude_ahead'])}",
    f"questions where the heads are right on more chats at Holm-corrected McNemar 0.05: {', '.join(mh['micro_ahead'])}",
    "metrics_haiku.json mcnemar_holm_005",
)

# Robustness
for q, v in B.items():
    a = v["accuracy_on_committed"]
    L(
        f"committed {v['claude_committed']} of {v['graded']}; heads {pct(M['committed'][q]['micro_correct'] / v['claude_committed'])}; Claude {pct(M['committed'][q]['claude_correct'] / v['claude_committed'])}; difference {M['committed'][q]['diff_points']} points; McNemar p {a['mcnemar_p']}",
        f"{q}: accuracy on chats where Claude committed to an answer the truth can take; only heads right {a['only_micro_right']}, only Claude right {a['only_claude_right']}",
        f"robustness_haiku.json {q}.accuracy_on_committed; counts in metrics_haiku.json committed.{q} = {M['committed'][q]['micro_correct']} and {M['committed'][q]['claude_correct']} right of {v['claude_committed']}",
    )
    if v["claude_answers_dropped"]:
        L(
            f"{v['claude_answers_dropped']}; heads right on {pct(v['micro_accuracy_on_dropped'])} of them",
            f"{q}: Claude's dropped answers",
            f"robustness_haiku.json {q}.claude_answers_dropped, .micro_accuracy_on_dropped",
        )
        L(
            f"committed {v['opening_words_median']['committed']}; dropped {v['opening_words_median']['dropped']}",
            f"{q}: median opening words",
            f"robustness_haiku.json {q}.opening_words_median",
        )

# Speed and cost
lt = T["latency_ms_15_questions"]
L(
    f"p50 {lt['total']['p50']} ms; p90 {lt['total']['p90']}; p99 {lt['total']['p99']}",
    "heads: all 15 questions for one opening",
    "timing_micro.json latency_ms_15_questions.total",
)
L(
    f"p50 {lt['features']['p50']} ms; heads p50 {lt['heads']['p50']}",
    "heads: feature step and the 15 heads",
    "latency_ms_15_questions.features, .heads",
)
L(
    f"intent {lt['per_head_p50']['intent']} ms; flow {lt['per_head_p50']['flow']}; yes/no about {lt['per_head_p50']['mentions_bug']}",
    "per-head median",
    "latency_ms_15_questions.per_head_p50",
)
L(
    f"{T['train']['seconds_total']} s; features {T['train']['seconds_features']} s; intent head {T['train']['seconds_per_head']['intent']} s",
    "training all 15 heads on 8,034 openings",
    "timing_micro.json train",
)
L(
    f"{T['size_mb']} MB",
    "saved size of the features and 15 heads",
    "timing_micro.json size_mb",
)
L(
    "Apple M1 Max, 10 cores; Python 3.13.9; scikit-learn 1.7.2; 1 process",
    "timing machine",
    "timing_micro.json machine",
)
cc = S["claude_cost"]
L(
    f"p50 {cc['wall_s']['p50']} s; p90 {cc['wall_s']['p90']}; p99 {cc['wall_s']['p99']}; max {cc['wall_s']['max']}",
    "Claude wall time per chat",
    "stats_haiku.json claude_cost.wall_s",
)
L(
    f"p50 {cc['api_s']['p50']} s; p90 {cc['api_s']['p90']}",
    "Claude API time per chat",
    "claude_cost.api_s",
)
L(
    f"p50 {cc['output_tokens']['p50']:,.0f}; p90 {cc['output_tokens']['p90']:,.0f}; max {cc['output_tokens']['max']:,.0f}",
    "Claude output tokens per chat",
    "claude_cost.output_tokens",
)
L(
    f"{pct(cc['thinking_share_of_output'])}",
    "thinking share of Claude's output tokens",
    "claude_cost.thinking_share_of_output = 0.9644",
)
L(
    f"p50 {cc['input_tokens']['p50']:,.0f}",
    "Claude input tokens per chat",
    "claude_cost.input_tokens",
)
L(
    f"${cc['cost_usd']['total']}; mean ${cc['cost_usd']['mean']}; p50 ${cc['cost_usd']['p50']}",
    "Claude cost, 300 chats",
    "claude_cost.cost_usd",
)
L(
    "300",
    "Claude calls that returned a parseable answer",
    "results_haiku.json summary.claude_answered = 300",
)
L(
    "$0.10, $0.20, $0.50",
    "price per million tokens: input, 1-hour cache write, output",
    "derived from claude_haiku.jsonl id 8833: 2 x 0.10e-6 + 1,820 x 0.20e-6 + 2,022 x 0.50e-6 = $0.0013752, its cost field",
)
L(
    "4,679",
    "Claude's median wall time over the heads' median latency",
    "18,610 ms / 3.977 ms",
)
L(
    "4,189",
    "Claude's median API time over the heads' median latency",
    "16,660 ms / 3.977 ms",
)
OV = json.loads((cvm / "checks_haiku.json").read_text())["overhead_s"]
L(
    f"p50 {OV['p50']} s; p10 {OV['p10']}; p90 {OV['p90']}",
    "CLI overhead per call, wall time minus API time",
    "data/claude_vs_micro/checks_haiku.json overhead_s",
)
L(
    f"${10000 * cc['cost_usd']['mean']:.2f}",
    "Claude cost per 10,000 chats at the mean",
    f"10,000 x ${cc['cost_usd']['mean']}",
)
L(
    f"{T['latency_ms_47_question_pipeline']['p50']} ms",
    "the 47-question synthetic pipeline, median per ABCD opening",
    "timing_micro.json latency_ms_47_question_pipeline.p50",
)

# Full test split
lr = A["learn"]
L(
    f"flow {pct(lr['flow']['opening']['balanced'])}; intent {pct(lr['intent']['opening']['balanced'])}",
    "heads on all 1,004 test openings, balanced",
    "data/abcd_analysis.json learn.flow.opening, learn.intent.opening",
)

# Synthetic
L(
    "480; 479 kept",
    "synthetic tickets written by Claude",
    "docs/how_it_works.md line 118",
)
L(
    "383; 96",
    "synthetic tickets for training; held out by ticket",
    "docs/how_it_works.md line 129; models/eval.json test_tickets = 96",
)
L(
    "$6.79; about 79 minutes",
    "cost and time to write the synthetic tickets with Haiku",
    "docs/how_it_works.md line 118",
)
L(
    "93.2%; baseline 69.0%",
    "47 answers on 96 held-out synthetic tickets, mean accuracy",
    "models/eval.json ticket_answers: mean accuracy 0.9322, mean baseline 0.6902 over 47",
)
L(
    "99.4%",
    "synthetic router's team accuracy on 333 held-out synthetic pieces",
    "models/eval.json models.router.primary_team.accuracy = 0.994",
)
for q in ["primary_team", "journey_stage", "message_type"] + yn:
    x = R["questions"][q]
    key = "accuracy" if S["questions"][q]["metric"] == "accuracy" else "balanced"
    L(
        f"synthetic system {pct(x['micro_syn'][key])}; Claude {pct(x['claude'][key])}",
        f"{q}: synthetic system vs Claude on the 300 openings, {key}",
        f"results_haiku.json questions.{q}.micro_syn.{key}, .claude.{key}",
    )
L(
    "12 of 13",
    "shared questions where the synthetic system scores below Claude",
    "derived from the 13 lines above",
)
L("1.1", "ABCD release used", "data/abcd/abcd_v1.1.json.gz")
L(
    "25 points",
    "change in balanced accuracy when the answer on one of 2 positive chats changes",
    "derived: recall on yes moves by 1/2, and balanced accuracy averages 2 recalls: 100 x 0.5 / 2",
)
L(
    "26",
    "message types Claude answered with a value the truth never takes",
    "robustness_haiku.json message_type.claude_answers_dropped: other 25 + feedback 1",
)
L(
    "42",
    "intents Claude answered with null or an intent absent from the sample",
    "robustness_haiku.json intent.claude_answers_dropped: None 30 + status_questions 12",
)
L(
    "333",
    "held-out synthetic pieces behind the router's 99.4%",
    "models/eval.json models.router.primary_team.n",
)
L(
    "50%",
    "balanced accuracy of a constant answer on a yes/no question",
    "derived: recall 1 on one class and 0 on the other, averaged: (100 + 0) / 2",
)
L(
    "12",
    "Claude calls running at once",
    "data/claude_vs_micro/compare.py WORKERS; the run passed workers = 12",
)
L(
    "11",
    "shared questions whose synthetic-system answers come from learned models; identity_details_given and charge_amount_given come from its regex entity finder",
    "micro_llm/rules.py:207 and :217",
)
L(
    "47",
    "questions the synthetic system answers per ticket",
    "models/eval.json ticket_answers, 47 questions; examples/support_ticket_98423/playground_questions.json",
)
L(
    "2.1.310",
    "Claude Code CLI version for the alias check",
    "claude --version, 2026-10-26",
)
L(
    "2026-10-26",
    "date the comparison ran",
    "file dates of data/claude_vs_micro/claude_haiku.jsonl and results_haiku.json",
)
L(
    "5.5",
    "Claude Haiku version in the paper's model name",
    "model ID claude-haiku-5-5, the modelUsage key from claude -p --model haiku, CLI 2.1.310",
)
L(
    "claude-haiku-5-5",
    "model the CLI's haiku alias resolves to",
    "claude -p --model haiku --output-format json: modelUsage key, CLI 2.1.310",
)

# Learning curve (learning_curve.py): heads retrained on nested subsets of the 8,034 chats
LC = json.loads((cvm / "learning_curve_haiku.json").read_text())
ref = LC["reference"]
L(
    "50, 100, 200, 383, 1,000, 2,000, 4,000, 8,034",
    "training sizes in the learning curve; 5 nested random subsets per size below 8,034, 1 run at 8,034",
    f"learning_curve_haiku.json curve[].size; seeds = {LC['seeds']}",
)
cur = {c["size"]: c for c in LC["curve"]}
for k, c in cur.items():
    m = c["mean_15"]
    r = m["as_reported"]
    L(
        f"as reported {pct(r['mean'])} (SD {100 * r['sd']:.1f}; runs {pct(r['min'])} to {pct(r['max'])}); plain {pct(m['plain_throughout']['mean'])}; balanced {pct(m['balanced_throughout']['mean'])}; 13 shared {pct(c['mean_13_shared']['as_reported']['mean'])}",
        f"heads trained on {k:,} ABCD chats, mean of 15 questions over {c['runs']} run(s)",
        f"learning_curve_haiku.json curve, size {k}",
    )
L(
    f"as reported {pct(ref['claude_15']['as_reported'])}; plain {pct(ref['claude_15']['plain_throughout'])}; balanced {pct(ref['claude_15']['balanced_throughout'])}; 13 shared {pct(ref['claude_13_shared']['as_reported'])}",
    "Claude's means, recomputed by learning_curve.py",
    "learning_curve_haiku.json reference.claude_15, .claude_13_shared",
)
L(
    f"{pct(ref['micro_syn_13_shared']['as_reported'])}",
    "synthetic system on the 13 shared questions, as reported",
    "learning_curve_haiku.json reference.micro_syn_13_shared.as_reported",
)
cu = LC["smallest_size_whose_seed_mean_beats_claude"]
L(
    f"{cu['as_reported']}; {cu['plain_throughout']}; {cu['balanced_throughout']}",
    "smallest size whose mean over runs beats Claude: as reported; plain; balanced",
    "learning_curve_haiku.json smallest_size_whose_seed_mean_beats_claude",
)


def above(k, rule, key="mean_15", ref_key="claude_15"):
    runs = [r for r in LC["runs"] if r["size"] == k]
    return f"{sum(r[key][rule] > ref[ref_key][rule] for r in runs)} of {len(runs)}"


L(
    f"{above(383, 'as_reported')}; plain {above(383, 'plain_throughout')}",
    "runs at 383 chats above Claude's mean, as reported; plain",
    "derived from learning_curve_haiku.json runs, size 383",
)
L(
    f"{above(200, 'as_reported')}; plain {above(200, 'plain_throughout')}",
    "runs at 200 chats above Claude's mean, as reported; plain",
    "derived from learning_curve_haiku.json runs, size 200",
)
L(
    f"{above(1000, 'as_reported')}",
    "runs at 1,000 chats above Claude's mean, as reported",
    "derived from learning_curve_haiku.json runs, size 1000",
)
L(
    f"{above(383, 'as_reported', 'mean_13_shared', 'claude_13_shared')}",
    "runs at 383 chats above Claude on the 13 shared questions",
    "derived from learning_curve_haiku.json runs, size 383",
)
s383 = cur[383]["mean_13_shared"]["as_reported"]["mean"]
L(
    f"{100 * (s383 - ref['micro_syn_13_shared']['as_reported']):.1f} points",
    "heads on 383 ABCD chats minus the synthetic system on 383 synthetic tickets, 13 shared questions",
    f"{s383} - {ref['micro_syn_13_shared']['as_reported']}",
)
full, k2, k1 = (cur[k]["mean_15"]["as_reported"]["mean"] for k in (8034, 2000, 1000))
L(
    f"{100 * (full - k2):.1f} points",
    "mean gain from 2,000 chats to all 8,034, as reported",
    f"{full} - {k2}",
)
L(
    f"{100 * (full - k1):.1f} points",
    "mean gain from 1,000 chats to all 8,034, as reported",
    f"{full} - {k1}",
)
L(
    f"{LC['full_size_mismatches_with_compare_py']}",
    "full-size answers that differ from compare.py's heads",
    "learning_curve_haiku.json full_size_mismatches_with_compare_py",
)
L(
    f"{LC['seconds']} s",
    "learning-curve run time, 36 trainings on 10 cores",
    "learning_curve_haiku.json seconds",
)
L(
    "NumPy 2.3.5; SciPy 1.16.3; joblib 1.5.2",
    "library versions behind the statistics and the curve",
    "python3 -I -c 'import numpy, scipy, joblib' on the timing machine, 2026-10-26",
)

# Reviewer checks (checks.py), and the synthetic data with the heads' design (synthetic_heads.py)
C = json.loads((cvm / "checks_haiku.json").read_text())
rt = C["routed"]
L(
    f"{rt['claude_intents_usable']} of {C['chats']}",
    "chats where Claude's intent answer names an intent the mapping knows",
    "checks_haiku.json routed.claude_intents_usable",
)
for q, v in rt["per_question"].items():
    L(
        f"heads {pct(v['micro_abcd_direct'])} direct, {pct(v['micro_abcd_routed'])} routed; Claude {pct(v['claude_direct'])} direct, {pct(v['claude_routed'])} routed",
        f"{q}: each system's own answer, and the answer its own intent maps to",
        f"checks_haiku.json routed.per_question.{q}",
    )
rm = rt["mean"]
L(
    f"heads {pct(rm['micro_abcd_direct'])} direct, {pct(rm['micro_abcd_routed'])} routed; Claude {pct(rm['claude_direct'])} direct, {pct(rm['claude_routed'])} routed",
    "mean over the 10 questions whose truth depends on the intent alone",
    "checks_haiku.json routed.mean",
)
L(
    f"{100 * (rm['micro_abcd_direct'] - rm['claude_direct']):.1f} points direct; {100 * (rm['micro_abcd_direct'] - rm['claude_routed']):.1f} against Claude routed; {100 * (rm['micro_abcd_routed'] - rm['claude_routed']):.1f} both routed",
    "heads minus Claude on those 10 questions",
    "derived from checks_haiku.json routed.mean",
)
ex = C["exclusions"]
L(
    ex["cannot_log_in_equals_security_related_on"],
    "chats where cannot_log_in and security_related have the same truth",
    "checks_haiku.json exclusions",
)
for name, v in ex["means"].items():
    L(
        f"heads {pct(v['heads'])}; Claude {pct(v['claude'])}; gap {v['gap_points']} points; {v['questions']} questions",
        f"mean without {name}",
        "checks_haiku.json exclusions.means",
    )
L(
    f"{ex['gap_range_points'][0]} to {ex['gap_range_points'][1]} points",
    "range of the gap over the exclusions",
    "checks_haiku.json exclusions.gap_range_points",
)
ry = C["rare_yes"]["recall"]
L(
    f"heads {ry['yes_micro_abcd']['right']} of {ry['yes_micro_abcd']['of']} ({pct(ry['yes_micro_abcd']['share'])}); Claude {ry['yes_claude']['right']} of {ry['yes_claude']['of']} ({pct(ry['yes_claude']['share'])})",
    "'yes' cases answered yes, pooled over the 9 yes/no questions whose 'yes' is the minority",
    "checks_haiku.json rare_yes.recall",
)
L(
    f"heads {ry['no_micro_abcd']['right']:,} of {ry['no_micro_abcd']['of']:,} ({pct(ry['no_micro_abcd']['share'])}); Claude {ry['no_claude']['right']:,} of {ry['no_claude']['of']:,} ({pct(ry['no_claude']['share'])})",
    "'no' cases answered no, pooled over the same 9 questions",
    "checks_haiku.json rare_yes.recall",
)
ns = C["rare_yes"]["needs_staff_action_yes"]
L(
    f"heads {ns['micro_abcd'][0]} of {ns['micro_abcd'][1]}; Claude {ns['claude'][0]} of {ns['claude'][1]}",
    "needs_staff_action: 'yes' cases answered yes",
    "checks_haiku.json rare_yes.needs_staff_action_yes",
)
sh = C["shrink"]
L(
    f"{sh['total_points']} points",
    "shrink from the reported gap (9.2) to the plain-accuracy gap (5.2), summed over questions",
    "checks_haiku.json shrink.total_points",
)
L(
    f"charge_amount_given {sh['per_question_points']['charge_amount_given']} ({pct(sh['largest_share'])}); refund_implied {sh['per_question_points']['refund_implied']}; cannot_log_in {sh['per_question_points']['cannot_log_in']}; the 5 choice questions {sum(sh['per_question_points'][q] for q in ('flow', 'intent', 'primary_team', 'journey_stage', 'message_type')):.2f}",
    "each question's share of that shrink, in points",
    "checks_haiku.json shrink.per_question_points, .largest_share",
)
for q, v in C["false_alarms_per_missed_yes"].items():
    L(
        f"{v['no_per_yes']} (no {v['no']}, yes {v['yes']})",
        f"{q}: false alarms that cost as much as one missed 'yes' under balanced accuracy",
        f"checks_haiku.json false_alarms_per_missed_yes.{q}",
    )
for q, v in C["baselines"]["per_question"].items():
    L(
        f"reported metric {pct(v['reported_metric'])}; plain accuracy {pct(v['plain_accuracy'])}; {v['classes']} classes",
        f"{q}: constant-answer baseline",
        f"checks_haiku.json baselines.per_question.{q}",
    )
for f_ in C["baselines"]["interval_or_score_not_above_baseline"]:
    if "ci95" in f_:
        L(
            f"{pct(f_['score'])} (CI {f_['ci95']}) against {pct(f_['baseline'])}",
            f"{f_['question']}, {f_['system']}: interval reaches the constant-answer baseline",
            "checks_haiku.json baselines.interval_or_score_not_above_baseline",
        )
    else:
        L(
            f"{pct(f_['plain_accuracy'])} against always-majority {pct(f_['always_majority'])}",
            f"{f_['question']}, {f_['system']}: plain accuracy not above the majority answer",
            "checks_haiku.json baselines.interval_or_score_not_above_baseline",
        )
for q, v in C["answer_kinds"].items():
    L(
        f"Claude {v['claude']}; heads {v['micro_abcd']}",
        f"{q}: answers by kind (possible, empty, offered but in no sampled chat's truth)",
        f"checks_haiku.json answer_kinds.{q}",
    )
    if v["claude_answers_no_chat_has"] or v["micro_abcd_answers_no_chat_has"]:
        L(
            f"Claude {v['claude_answers_no_chat_has']}; heads {v['micro_abcd_answers_no_chat_has']}",
            f"{q}: answers no sampled chat's truth holds",
            f"checks_haiku.json answer_kinds.{q}",
        )
ov = C["overlap"]
L(
    f"{ov['exact_match_after_lowercasing']} exact (median {ov['words_in_exact_matches_median']:.0f} words); {ov['cosine_at_least_0.9']} at cosine 0.9 or more; median best cosine {ov['median_best_cosine']}",
    "test openings with a near-duplicate among the 8,034 training openings",
    "checks_haiku.json overlap",
)
wn = ov["without_near_duplicates"]
L(
    f"{wn['chats']} chats; heads {pct(wn['heads'])}; Claude {pct(wn['claude'])}; gap {wn['gap_points']} points",
    "mean of 15 without the near-duplicate openings",
    "checks_haiku.json overlap.without_near_duplicates",
)
ft = C["full_test"]
L(
    f"{pct(ft['heads_mean'])} on {ft['openings']:,}; {pct(ft['heads_sample_mean'])} on the 300",
    "heads' mean of 15 on all test openings, as reported",
    "checks_haiku.json full_test",
)
L(
    "; ".join(f"{q} {v}" for q, v in ft["largest_moves_points"].items()),
    "largest moves from the 300 to all 1,004 test openings, points",
    "checks_haiku.json full_test.largest_moves_points",
)
so = C["synthetic_own_tickets"]
L(
    f"{pct(so['mean_13_shared'])} on the 13 shared; primary_team {pct(so['primary_team_per_ticket'])} per ticket; {so['tickets']} tickets",
    "synthetic system on its own held-out tickets",
    "checks_haiku.json synthetic_own_tickets",
)
mr = C["misroutes_per_1000"]
L(
    f"heads {mr['micro_abcd']}; Claude {mr['claude']}",
    "chats per 1,000 sent to the wrong team, from primary_team accuracy",
    "checks_haiku.json misroutes_per_1000",
)
for q, v in C["confusions"].items():
    for s_, x in v.items():
        top = "; ".join(
            f"{p_} {y['count']} ({', '.join(f'{i} {k}' for i, k in y['intents'].items())})"
            for p_, y in x["pairs"].items()
        )
        L(
            f"{x['errors']} errors; {top}",
            f"{q}, {s_}: errors by truth -> answer, with the true intents behind each",
            f"checks_haiku.json confusions.{q}.{s_}",
        )
L(
    "0.0015",
    "smallest possible Holm p over 15 questions with 10,000 permutations",
    "derived: 15 x 1 / 10,001; stats.py adds one to both counts",
)
for q in ("flow", "intent"):
    a = M["questions"][q]["accuracy"]
    L(
        f"heads {pct(a['micro_abcd'])}; Claude {pct(a['claude'])}; difference {100 * (a['micro_abcd'] - a['claude']):.1f} points",
        f"{q}: plain accuracy",
        f"metrics_haiku.json questions.{q}.accuracy",
    )
SH = json.loads((cvm / "synthetic_heads_haiku.json").read_text())
mm, mc = SH["mean_13_shared"], SH["mean_covered"]
L(
    f"synthetic heads {pct(mm['synthetic_heads_on_abcd'])} on ABCD, {pct(mm['synthetic_heads_on_held_out_synthetic'])} on held-out synthetic; synthetic system {pct(mm['synthetic_system_on_abcd'])}; heads on 383 ABCD chats {pct(mm['abcd_heads_383_on_abcd'])}; on 8,034 {pct(mm['abcd_heads_8034_on_abcd'])}; Claude {pct(mm['claude_on_abcd'])}",
    f"mean of the 13 shared questions; synthetic heads trained on {SH['synthetic_training_tickets']} synthetic tickets, held out {SH['synthetic_held_out_tickets']}",
    "data/claude_vs_micro/synthetic_heads_haiku.json mean_13_shared",
)
L(
    f"synthetic heads {pct(mc['synthetic_heads_on_abcd'])}; synthetic system {pct(mc['synthetic_system_on_abcd'])}; heads on 383 ABCD chats {pct(mc['abcd_heads_383_on_abcd'])}; Claude {pct(mc['claude_on_abcd'])}",
    f"mean of the {len(SH['covered_questions'])} shared questions whose ABCD answers all appear in the synthetic training labels",
    "synthetic_heads_haiku.json mean_covered",
)
L(
    f"{100 * (mm['abcd_heads_383_on_abcd'] - mm['synthetic_heads_on_abcd']):.1f} points; {100 * (mc['abcd_heads_383_on_abcd'] - mc['synthetic_heads_on_abcd']):.1f} on the covered questions",
    "same heads, same size: 383 ABCD chats minus 383 synthetic tickets",
    "derived from synthetic_heads_haiku.json",
)
L(
    f"{100 * (mm['synthetic_system_on_abcd'] - mm['synthetic_heads_on_abcd']):.1f} points",
    "synthetic system minus the heads trained on its tickets, 13 shared",
    "derived from synthetic_heads_haiku.json mean_13_shared",
)
for q in ("journey_stage", "message_type"):
    miss = SH["questions"][q]["answers_in_abcd_truth_never_in_synthetic_training"]
    tc = SH["abcd_truth_classes"][q]
    L(
        f"{', '.join(miss)}: {sum(tc[m] for m in miss)} of {sum(tc.values())} chats",
        f"{q}: ABCD answers absent from the synthetic training labels",
        f"synthetic_heads_haiku.json questions.{q}, abcd_truth_classes.{q}",
    )
for q, v in SH["questions"].items():
    L(
        f"synthetic heads {pct(v['synthetic_heads_on_abcd'])}; synthetic system {pct(v['synthetic_system_on_abcd'])}; heads on 383 ABCD chats {pct(v['abcd_heads_383_on_abcd'])}",
        f"{q}: same design on synthetic vs ABCD data, scored on the 300",
        f"synthetic_heads_haiku.json questions.{q}",
    )
L(
    "MAX_THINKING_TOKENS",
    "environment variable the CLI binary reads; claude --help lists only --effort (low to max) and no option to turn thinking off",
    "grep -a on ~/.local/share/claude/versions/2.1.310 (5 matches, one 'MAX_THINKING_TOKENS=0'); claude --help, 2026-10-26",
)
L(
    "send-link, promo-code",
    "2 of the 8 CHANGES actions that only inform or discount; the others change an order, account or password, or pass the case on",
    "micro_llm/abcd.py:59-68 CHANGES",
)
L(
    "status_delivery_date -> status_delivery_time; FAQ subflows -> their first word",
    "intent normalization",
    "micro_llm/abcd.py:281-288 intent()",
)


# Raw ABCD subflows behind the 55 intents, and the chats some questions can't score
import gzip as _gz  # noqa: E402
from collections import Counter as _Counter  # noqa: E402

_abcd = json.loads(_gz.decompress((repo / "data/abcd/abcd_v1.1.json.gz").read_bytes()))
_raw = _Counter(
    (c["scenario"]["flow"], c["scenario"]["subflow"])
    for split in ("train", "dev", "test")
    for c in _abcd[split]
)
_faq = {s for f, s in _raw if f in ("single_item_query", "storewide_query")}
_dd = sum(v for (f, s), v in _raw.items() if s == "status_delivery_date")
L(
    f"{len(_raw)} raw subflows; {len(_faq)} FAQ subflows; status_delivery_date {_dd} chats",
    "ABCD subflows before normalization: the FAQ subflows name single FAQ entries such as boots_how_1",
    "data/abcd/abcd_v1.1.json.gz scenario.subflow over train, dev and test",
)
_rows = json.loads((cvm / "rows_haiku.json").read_text())
_no_mt = _Counter(
    r["truth"]["intent"] for r in _rows if r["truth"].get("message_type") is None
)
L(
    f"{sum(_no_mt.values())}; "
    + ", ".join(f"{k} {v}" for k, v in _no_mt.most_common()),
    "chats among the 300 with no message_type truth, by intent",
    "data/claude_vs_micro/rows_haiku.json truth.message_type is None",
)
_rest = 1004 - len(_rows)
L(
    f"{_rest}; ${_rest * 0.002762:.2f}",
    "test openings outside the 300; Claude's cost on them at its mean per chat",
    f"1,004 - 300; {_rest} x $0.002762 = ${_rest * 0.002762:.4f}",
)

# Numbers cited from related work, as each paper reports them (checked 2026-10-26)
for value, what, source in [
    (
        "135M 82%; Claude 3.7 Sonnet 78%; 70B 88%; 260 tasks",
        "Lawma: accuracy on CaselawQA's legal annotation tasks",
        "arXiv:2407.16615",
    ),
    (
        "a few hundred to a thousand",
        "Lawma: labeled examples usually enough for higher accuracy",
        "arXiv:2407.16615",
    ),
    (
        "SetFit 10-shot 88.1; best LLM 87.7; full data 94.1; 77 intents",
        "Loukas et al.: Banking77 micro-F1",
        "arXiv:2311.06102",
    ),
    (
        "Claude 3 Haiku F1 0.736; SetFit 0.600; within 2%; about 50% less latency; 6 datasets",
        "Arora et al.: intent detection, and a router that sends uncertain queries to the LLM",
        "arXiv:2410.01627",
    ),
    (
        "RoBERTa 0.62; Claude Haiku 4.5 0.50; 0.007 s vs 10.48 s per item",
        "Lee et al. 2026: macro-F1 and time per item",
        "arXiv:2606.04274",
    ),
    (
        "1.5B; 93.17 vs Claude 3.7 Sonnet 92.79; 51 ms vs 1,450 ms",
        "Arch-Router: routing accuracy and latency",
        "arXiv:2506.16655",
    ),
    (
        "stance 0.94 vs Claude 3 Opus 0.61",
        "Bucher and Martini: accuracy, fine-tuned vs zero-shot",
        "arXiv:2406.08660",
    ),
    (
        "AG News RoBERTa 94.63 vs Claude Sonnet 4.5 91.35; IMDB Claude few-shot 96.48 vs RoBERTa 94.84",
        "Valdes Gonzalez: accuracy",
        "arXiv:2602.06370",
    ),
    (
        "770M T5; PaLM 540B; 80% of the data",
        "Hsieh et al.: distilling step-by-step",
        "arXiv:2305.02301",
    ),
    ("25 benchmarks; 13 zero-shot models", "Ziems et al.", "arXiv:2305.03514"),
    (
        "25 tasks; about 25% lower quality",
        "Kocon et al.: ChatGPT against task state of the art",
        "arXiv:2302.10724",
    ),
    ("16 datasets", "Edwards and Camacho-Collados", "arXiv:2403.17661"),
    ("32 datasets; 8 languages", "Vajjala and Shimangaud", "arXiv:2502.11830"),
    (
        "up to 98% lower cost",
        "FrugalGPT: cascades that match GPT-4",
        "arXiv:2305.05176",
    ),
    (
        "17 tasks; Llama 2 7B",
        "Buckmann and Hill: penalized logistic regression on embeddings",
        "arXiv:2408.03414",
    ),
]:
    L(value, what, source)

# Checks added after the second cold read (data/claude_vs_micro/revision_checks.py)
V = json.loads((cvm / "revision_checks_haiku.json").read_text())
_src = "data/claude_vs_micro/revision_checks_haiku.json"
_rs, _rg = V["routing"]["systems"], V["routing"]["gaps_points"]
for _s, _name in (("claude", "Claude's"), ("micro_abcd", "heads'")):
    _m = _rs[_s]["mean_10"]
    L(
        f"{100 * _m['routed_strict']:.1f}%",
        f"{_name} routed mean of 10 under the strict rule: a known intent that settles nothing counts wrong",
        f"{_src} routing.systems.{_s}.mean_10.routed_strict = {_m['routed_strict']}",
    )
L(
    f"{_rg['direct']:.1f}; Claude routed vs heads direct {_rg['claude_routed_vs_heads_direct']['fallback']:.1f} (fallback), {_rg['claude_routed_vs_heads_direct']['strict']:.1f} (strict); both routed {_rg['both_routed']['fallback']:.1f} (fallback), {_rg['both_routed']['strict']:.1f} (strict)",
    "gaps on the 10 intent-derived questions, heads minus Claude, points",
    f"{_src} routing.gaps_points = {json.dumps(_rg)}",
)
_sh = _rg["share_of_direct_gap_closed_both_routed"]
L(
    f"{100 * _sh['fallback']:.0f}%; {100 * _sh['strict']:.0f}%",
    "share of the 10-question direct gap that routing both systems closes: fallback; strict",
    f"{_src} share_of_direct_gap_closed_both_routed = {_sh}",
)
_fc, _fh = (
    _rs["claude"]["graded_pairs_falling_back"],
    _rs["micro_abcd"]["graded_pairs_falling_back"],
)
L(
    f"{_fc['known_intent_routes_nowhere']}; {_fh['known_intent_routes_nowhere']}",
    "graded chat-question pairs where a known intent settles nothing and the direct answer stands: Claude; heads",
    f"{_src} graded_pairs_falling_back.known_intent_routes_nowhere: claude, micro_abcd",
)
L(
    f"{_fc['unknown_intent']}; {_fh['unknown_intent']}",
    "graded chat-question pairs where the intent answer is one the mapping does not know: Claude; heads",
    f"{_src} graded_pairs_falling_back.unknown_intent: claude, micro_abcd",
)
_kc, _hc = (
    _rs["claude"]["intent_known_to_mapping"],
    _rs["claude"]["intent_a_sampled_truth_holds"],
)
L(
    f"{_kc}; {_hc}; {300 - _kc}; {_kc - _hc}",
    "Claude intent answers the mapping knows; that a sampled truth holds; not known (300 - 270); known but held by no sampled truth (270 - 258)",
    f"{_src} routing.systems.claude.intent_known_to_mapping, intent_a_sampled_truth_holds",
)
L(
    f"{_rs['micro_abcd']['intent_known_to_mapping']}; {_rs['micro_abcd']['intent_a_sampled_truth_holds']}",
    "heads' intent answers the mapping knows; that a sampled truth holds",
    f"{_src} routing.systems.micro_abcd",
)
for _x in V["learning_curve_leads"]["curve"]:
    _a = _x["as_reported"]
    _ld, _ci = _a["lead_points_by_seed"], _a["lead_ci95_points_by_seed"]
    L(
        f"{min(_ld):+.1f} to {max(_ld):+.1f}; lower bounds {min(c[0] for c in _ci):.1f} to {max(c[0] for c in _ci):.1f}; upper bounds {min(c[1] for c in _ci):.1f} to {max(c[1] for c in _ci):.1f}; above zero in {_a['runs_whose_interval_is_above_zero']} of 5, below in {_a['runs_whose_interval_is_below_zero']}",
        f"heads minus Claude on the mean of 15 (as reported) at {_x['size']:,} training chats: the 5 runs' leads and 95% paired intervals, points",
        f"{_src} learning_curve_leads.curve[size={_x['size']}].as_reported = {json.dumps(_ld)}; {json.dumps(_ci)}",
    )
for _rule, _what in (
    ("as_reported", "as reported"),
    ("plain_throughout", "plain accuracy throughout"),
    ("balanced_throughout", "balanced accuracy throughout"),
):
    _u = V["unweighted_heads"]["lead"][_rule]
    L(
        f"{100 * _u['heads']:.1f}% vs {100 * _u['claude']:.1f}%; lead {_u['lead_points']:.1f} ({_u['lead_ci95_points'][0]:.1f} to {_u['lead_ci95_points'][1]:.1f})",
        f"heads trained without class weights against Claude, mean of 15, {_what}",
        f"{_src} unweighted_heads.lead.{_rule} = {json.dumps(_u)}",
    )
_w = V["unweighted_heads"]["yes_no_where_claude_plain_is_higher"]
L(
    f"{_w['unweighted']} of 10; weighted {_w['weighted']} of 10",
    "yes/no questions where Claude's plain accuracy is higher than the heads': unweighted heads; weighted heads",
    f"{_src} unweighted_heads.yes_no_where_claude_plain_is_higher = {_w}",
)
_p = V["synthetic_system_minus_synthetic_heads"]
L(
    f"{_p['total_points']:.1f}; regex {_p['regex_part_points']:.1f}; learned {_p['learned_part_points']:.1f}",
    "synthetic system minus synthetic-trained heads on the mean of 13: total; its 2 regular-expression answers; its 11 learned answers, points",
    f"{_src} synthetic_system_minus_synthetic_heads = total {_p['total_points']}, regex {_p['regex_part_points']}, learned {_p['learned_part_points']}",
)
_eh, _ec = (
    V["rare_yes_errors"]["systems"]["micro_abcd"],
    V["rare_yes_errors"]["systems"]["claude"],
)
L(
    f"heads {_eh['missed_yes']} missed, {_eh['false_flags']} false flags, {_eh['errors']} errors; Claude {_ec['missed_yes']} missed, {_ec['false_flags']} false flags, {_ec['errors']} errors",
    "errors on the 9 rare-yes questions",
    f"{_src} rare_yes_errors.systems",
)
L(
    f"{_ec['missed_yes'] - _eh['missed_yes']}; {_eh['false_flags'] - _ec['false_flags']}",
    "Claude's extra missed yes cases; the heads' extra false flags",
    f"derived: {_ec['missed_yes']} - {_eh['missed_yes']}; {_eh['false_flags']} - {_ec['false_flags']}",
)
L(
    "2.8",
    "points of balanced accuracy that one of the 18 yes chats moves on mentions_bug",
    "derived: 100 / (2 x 18) = 2.78",
)
_imp = {
    "primary_team": {"product_management"},
    "journey_stage": {"unclear"},
    "message_type": {"feedback", "other"},
}
_imp_n = {
    q: sum(
        r["truth"].get(q) is not None and (r["claude"] or {}).get(q) in v for r in rows
    )
    for q, v in _imp.items()
}
L(
    f"{sum(_imp_n.values())}; " + ", ".join(f"{q} {k}" for q, k in _imp_n.items()),
    "Claude answers, on chats with a truth, that the prompt offered and no ABCD truth can take",
    "data/claude_vs_micro/rows_haiku.json; the mapping in micro_llm/abcd.py never yields product_management, unclear, feedback or other",
)
for _x in V["learning_curve_leads"]["curve"]:
    _a = _x["plain_throughout"]
    _ld, _ci = _a["lead_points_by_seed"], _a["lead_ci95_points_by_seed"]
    L(
        f"{min(_ld):+.1f} to {max(_ld):+.1f}; lower bounds {min(c[0] for c in _ci):.1f} to {max(c[0] for c in _ci):.1f}; upper bounds {min(c[1] for c in _ci):.1f} to {max(c[1] for c in _ci):.1f}; above zero in {_a['runs_whose_interval_is_above_zero']} of 5, below in {_a['runs_whose_interval_is_below_zero']}",
        f"heads minus Claude on the mean of 15 (plain accuracy throughout) at {_x['size']:,} training chats: the 5 runs' leads and 95% paired intervals, points",
        f"{_src} learning_curve_leads.curve[size={_x['size']}].plain_throughout = {json.dumps(_ld)}; {json.dumps(_ci)}",
    )
_yn_s = [T["train"]["seconds_per_head"][q] for q in yn]
L(
    f"{min(_yn_s)} to {max(_yn_s)} s; features {T['train']['seconds_features']} s",
    "training time of one yes/no head on 8,034 openings, over the 10 yes/no heads; the shared features take extra",
    "timing_micro.json train.seconds_per_head (yes/no questions), train.seconds_features",
)

out.write_text(
    "# Evidence ledger: small heads vs zero-shot Claude on ABCD\n\nEvery number in `small_heads_vs_claude.md` traces to a line here. Paths are relative to the micro_llm repo root.\n\n## Evidence ledger\n\n"
    + "\n".join(lines)
    + "\n"
)
print(len(lines), "lines")
