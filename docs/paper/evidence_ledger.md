# Evidence ledger: small heads vs zero-shot Claude on ABCD

Every number in `small_heads_vs_claude.md` traces to a line here. Paths are relative to the micro_llm repo root.

## Evidence ledger

L1 | 8,034 | ABCD training chats | data/abcd_analysis.json profile.chats.train = 8034
L2 | 1,004 | ABCD dev chats; also test chats | data/abcd_analysis.json profile.chats.dev = 1004, .test = 1004
L3 | 10,042 | ABCD chats in all | derived: 8,034 + 1,004 + 1,004
L4 | 10 | ABCD flows (topics) | data/abcd_analysis.json len(profile.flows) = 10
L5 | 55 | ABCD intents | data/abcd_analysis.json len(profile.intents) = 55
L6 | 6b8700c | ABCD commit | data/abcd_analysis.json source.commit = 6b8700ce67c6...
L7 | 16 | median words in an ABCD opening; p10 6, p90 28 | data/abcd_analysis.json profile.words.abcd_opening = {'median': 16.0, 'p10': 6, 'p90': 28}
L8 | 49 | median words in all a customer's ABCD messages | profile.words.abcd_all_customer.median = 49.0
L9 | 55 | median words in one synthetic ticket | profile.words.our_tickets.median = 55
L10 | 12.5% | opening words never seen in the synthetic tickets | profile.words_unseen_in_our_tickets.opening = 0.125
L11 | 86.8% | chats where the agent speaks first | profile.agent_speaks_first = 0.868
L12 | 300 | ABCD test chats sampled, random.Random(0).sample | data/claude_vs_micro/results_haiku.json summary.chats = 300
L13 | 53 | intents present among the 300 sampled chats | rows_haiku.json, distinct truth.intent = 53
L14 | 15 | questions compared | stats_haiku.json summary.questions = 15
L15 | 10,000 | bootstrap resamples and permutations | stats_haiku.json summary.reps = 10000
L16 | 38,990 | TF-IDF features | data/claude_vs_micro/timing_micro.json train.feature_count = 38990
L17 | 4.0 | logistic regression C; class_weight balanced; max_iter 3,000 | micro_llm/abcd.py:666 and compare.py LogisticRegression(C=4.0, ...)
L18 | 90.7% | heads' mean score over 15 questions | stats_haiku.json summary.mean_score.micro_abcd = 0.9066
L19 | [0.8833, 0.9279] | heads' mean score, 95% CI | summary.mean_ci95.micro_abcd
L20 | 81.5% | Claude's mean score over 15 questions | summary.mean_score.claude = 0.8145
L21 | [0.782, 0.8472] | Claude's mean score, 95% CI | summary.mean_ci95.claude
L22 | 9.2 points | mean difference, heads minus Claude | summary.mean_diff_points
L23 | [6.3, 12.1] | mean difference, 95% CI in points | summary.mean_diff_ci95_points
L24 | 6 | questions where the heads lead at Holm 0.05 | summary.micro_ahead_holm_005 = ['flow', 'intent', 'journey_stage', 'message_type', 'refund_implied', 'needs_staff_action']
L25 | 0 | questions where Claude leads at Holm 0.05 | summary.claude_ahead_holm_005 = []
L26 | n 300; heads 85.9% [0.8176, 0.8991]; Claude 75.8% [0.7082, 0.8062] | flow: balanced score, 10 classes; Claude answers the truth can't take 25; heads 0 | point estimates: results_haiku.json questions.flow.*.balanced; intervals: stats_haiku.json questions.flow.systems
L27 | 10.1 points [6.2, 14.2]; permutation p 0.0001, Holm p 0.0015 | flow: difference, heads minus Claude | questions.flow.micro_abcd_minus_claude
L28 | only heads right 33; only Claude right 5; McNemar p 0.0 | flow: discordant chats | questions.flow.micro_abcd_minus_claude.mcnemar
L29 | n 300; heads 76.9% [0.7075, 0.8147]; Claude 66.7% [0.6023, 0.7079] | intent: balanced score, 53 classes; Claude answers the truth can't take 42; heads 7 | point estimates: results_haiku.json questions.intent.*.balanced; intervals: stats_haiku.json questions.intent.systems
L30 | 10.2 points [5.9, 15.3]; permutation p 0.0002, Holm p 0.0024 | intent: difference, heads minus Claude | questions.intent.micro_abcd_minus_claude
L31 | only heads right 43; only Claude right 12; McNemar p 0.0 | intent: discordant chats | questions.intent.micro_abcd_minus_claude.mcnemar
L32 | n 267; heads 88.4% [0.8427, 0.9213]; Claude 83.1% [0.7865, 0.8764] | primary_team: accuracy score, 4 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.primary_team.*.accuracy; intervals: stats_haiku.json questions.primary_team.systems
L33 | 5.2 points [0.4, 10.1]; permutation p 0.0475, Holm p 0.38 | primary_team: difference, heads minus Claude | questions.primary_team.micro_abcd_minus_claude
L34 | only heads right 29; only Claude right 15; McNemar p 0.0488 | primary_team: discordant chats | questions.primary_team.micro_abcd_minus_claude.mcnemar
L35 | n 145; heads 93.1% [0.8897, 0.9657]; Claude 72.4% [0.6483, 0.8] | journey_stage: accuracy score, 3 classes; Claude answers the truth can't take 27; heads 0 | point estimates: results_haiku.json questions.journey_stage.*.accuracy; intervals: stats_haiku.json questions.journey_stage.systems
L36 | 20.7 points [13.8, 27.6]; permutation p 0.0001, Holm p 0.0015 | journey_stage: difference, heads minus Claude | questions.journey_stage.micro_abcd_minus_claude
L37 | only heads right 31; only Claude right 1; McNemar p 0.0 | journey_stage: discordant chats | questions.journey_stage.micro_abcd_minus_claude.mcnemar
L38 | n 285; heads 88.1% [0.8421, 0.9158]; Claude 67.4% [0.6175, 0.7298] | message_type: accuracy score, 3 classes; Claude answers the truth can't take 26; heads 0 | point estimates: results_haiku.json questions.message_type.*.accuracy; intervals: stats_haiku.json questions.message_type.systems
L39 | 20.7 points [15.4, 26.0]; permutation p 0.0001, Holm p 0.0015 | message_type: difference, heads minus Claude | questions.message_type.micro_abcd_minus_claude
L40 | only heads right 64; only Claude right 5; McNemar p 0.0 | message_type: discordant chats | questions.message_type.micro_abcd_minus_claude.mcnemar
L41 | n 267; heads 92.3% [0.8639, 0.9727]; Claude 88.6% [0.8125, 0.9531] | cannot_log_in: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.cannot_log_in.*.balanced; intervals: stats_haiku.json questions.cannot_log_in.systems
L42 | 3.8 points [-1.1, 9.7]; permutation p 0.1714, Holm p 1.0 | cannot_log_in: difference, heads minus Claude | questions.cannot_log_in.micro_abcd_minus_claude
L43 | only heads right 4; only Claude right 9; McNemar p 0.2668 | cannot_log_in: discordant chats | questions.cannot_log_in.micro_abcd_minus_claude.mcnemar
L44 | n 267; heads 92.3% [0.8638, 0.9723]; Claude 91.8% [0.8537, 0.9733] | security_related: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.security_related.*.balanced; intervals: stats_haiku.json questions.security_related.systems
L45 | 0.6 points [-2.1, 4.2]; permutation p 0.9145, Holm p 1.0 | security_related: difference, heads minus Claude | questions.security_related.micro_abcd_minus_claude
L46 | only heads right 6; only Claude right 9; McNemar p 0.6072 | security_related: discordant chats | questions.security_related.micro_abcd_minus_claude.mcnemar
L47 | n 267; heads 91.4% [0.8551, 0.9644]; Claude 90.1% [0.8364, 0.9593] | mentions_account_issue: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.mentions_account_issue.*.balanced; intervals: stats_haiku.json questions.mentions_account_issue.systems
L48 | 1.2 points [-2.2, 5.4]; permutation p 0.5809, Holm p 1.0 | mentions_account_issue: difference, heads minus Claude | questions.mentions_account_issue.micro_abcd_minus_claude
L49 | only heads right 6; only Claude right 10; McNemar p 0.4545 | mentions_account_issue: discordant chats | questions.mentions_account_issue.micro_abcd_minus_claude.mcnemar
L50 | n 223; heads 93.4% [0.8808, 0.979]; Claude 85.3% [0.7822, 0.9163] | mentions_billing_issue: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.mentions_billing_issue.*.balanced; intervals: stats_haiku.json questions.mentions_billing_issue.systems
L51 | 8.1 points [3.0, 13.9]; permutation p 0.006, Holm p 0.054 | mentions_billing_issue: difference, heads minus Claude | questions.mentions_billing_issue.micro_abcd_minus_claude
L52 | only heads right 13; only Claude right 3; McNemar p 0.0213 | mentions_billing_issue: discordant chats | questions.mentions_billing_issue.micro_abcd_minus_claude.mcnemar
L53 | n 267; heads 98.0% [0.9667, 0.9918]; Claude 97.0% [0.9048, 1.0] | mentions_bug: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.mentions_bug.*.balanced; intervals: stats_haiku.json questions.mentions_bug.systems
L54 | 1.0 points [-2.8, 7.6]; permutation p 0.9999, Holm p 1.0 | mentions_bug: difference, heads minus Claude | questions.mentions_bug.micro_abcd_minus_claude
L55 | only heads right 2; only Claude right 10; McNemar p 0.0386 | mentions_bug: discordant chats | questions.mentions_bug.micro_abcd_minus_claude.mcnemar
L56 | n 186; heads 98.9% [0.9775, 0.9973]; Claude 100.0% [1.0, 1.0] | refund_requested: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.refund_requested.*.balanced; intervals: stats_haiku.json questions.refund_requested.systems
L57 | -1.1 points [-2.2, -0.3]; permutation p 0.1249, Holm p 0.8742 | refund_requested: difference, heads minus Claude | questions.refund_requested.micro_abcd_minus_claude
L58 | only heads right 0; only Claude right 4; McNemar p 0.125 | refund_requested: discordant chats | questions.refund_requested.micro_abcd_minus_claude.mcnemar
L59 | n 221; heads 90.2% [0.8406, 0.9543]; Claude 77.8% [0.6996, 0.8527] | refund_implied: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.refund_implied.*.balanced; intervals: stats_haiku.json questions.refund_implied.systems
L60 | 12.4 points [5.3, 19.8]; permutation p 0.0022, Holm p 0.022 | refund_implied: difference, heads minus Claude | questions.refund_implied.micro_abcd_minus_claude
L61 | only heads right 12; only Claude right 8; McNemar p 0.5034 | refund_implied: discordant chats | questions.refund_implied.micro_abcd_minus_claude.mcnemar
L62 | n 300; heads 73.3% [0.6839, 0.7815]; Claude 58.0% [0.5235, 0.6361] | needs_staff_action: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.needs_staff_action.*.balanced; intervals: stats_haiku.json questions.needs_staff_action.systems
L63 | 15.3 points [8.3, 22.4]; permutation p 0.0002, Holm p 0.0024 | needs_staff_action: difference, heads minus Claude | questions.needs_staff_action.micro_abcd_minus_claude
L64 | only heads right 86; only Claude right 40; McNemar p 0.0001 | needs_staff_action: discordant chats | questions.needs_staff_action.micro_abcd_minus_claude.mcnemar
L65 | n 300; heads 98.2% [0.9494, 1.0]; Claude 92.9% [0.8667, 0.9819] | identity_details_given: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.identity_details_given.*.balanced; intervals: stats_haiku.json questions.identity_details_given.systems
L66 | 5.4 points [-0.8, 12.1]; permutation p 0.1909, Holm p 1.0 | identity_details_given: difference, heads minus Claude | questions.identity_details_given.micro_abcd_minus_claude
L67 | only heads right 6; only Claude right 3; McNemar p 0.5078 | identity_details_given: discordant chats | questions.identity_details_given.micro_abcd_minus_claude.mcnemar
L68 | n 300; heads 99.5% [0.9833, 1.0]; Claude 75.0% [0.5, 1.0] | charge_amount_given: balanced score, 2 classes; Claude answers the truth can't take 0; heads 0 | point estimates: results_haiku.json questions.charge_amount_given.*.balanced; intervals: stats_haiku.json questions.charge_amount_given.systems
L69 | 24.5 points [-1.7, 49.8]; permutation p 1.0, Holm p 1.0 | charge_amount_given: difference, heads minus Claude | questions.charge_amount_given.micro_abcd_minus_claude
L70 | only heads right 1; only Claude right 3; McNemar p 0.625 | charge_amount_given: discordant chats | questions.charge_amount_given.micro_abcd_minus_claude.mcnemar
L71 | heads 90.4%; Claude 85.9%; majority baseline 52.1% | primary_team: balanced accuracy, and the majority-class accuracy | results_haiku.json questions.primary_team.*.balanced, .baseline
L72 | heads 94.9%; Claude 53.2%; majority baseline 69.0% | journey_stage: balanced accuracy, and the majority-class accuracy | results_haiku.json questions.journey_stage.*.balanced, .baseline
L73 | heads 88.9%; Claude 70.0%; majority baseline 40.7% | message_type: balanced accuracy, and the majority-class accuracy | results_haiku.json questions.message_type.*.balanced, .baseline
L74 | yes 35, no 232 | cannot_log_in: truth classes among the 300; 33 chats have no truth | rows_haiku.json truth.cannot_log_in
L75 | yes 35, no 232 | security_related: truth classes among the 300; 33 chats have no truth | rows_haiku.json truth.security_related
L76 | yes 39, no 228 | mentions_account_issue: truth classes among the 300; 33 chats have no truth | rows_haiku.json truth.mentions_account_issue
L77 | yes 43, no 180 | mentions_billing_issue: truth classes among the 300; 77 chats have no truth | rows_haiku.json truth.mentions_billing_issue
L78 | yes 18, no 249 | mentions_bug: truth classes among the 300; 33 chats have no truth | rows_haiku.json truth.mentions_bug
L79 | yes 6, no 180 | refund_requested: truth classes among the 300; 114 chats have no truth | rows_haiku.json truth.refund_requested
L80 | yes 41, no 180 | refund_implied: truth classes among the 300; 79 chats have no truth | rows_haiku.json truth.refund_implied
L81 | yes 150, no 150 | needs_staff_action: truth classes among the 300; 0 chats have no truth | rows_haiku.json truth.needs_staff_action
L82 | yes 36, no 264 | identity_details_given: truth classes among the 300; 0 chats have no truth | rows_haiku.json truth.identity_details_given
L83 | yes 2, no 298 | charge_amount_given: truth classes among the 300; 0 chats have no truth | rows_haiku.json truth.charge_amount_given
L84 | heads 91.4% [0.8956, 0.9322]; Claude 86.2% [0.8457, 0.8777]; difference 5.2 points [3.9, 6.6] | mean of 15 scored by plain accuracy on all 15 questions; joint bootstrap over chats, 10,000 resamples | data/claude_vs_micro/metrics_haiku.json means.plain_throughout
L85 | heads 91.0% [0.8867, 0.9303]; Claude 80.5% [0.7721, 0.8386]; difference 10.4 points [7.5, 13.3] | mean of 15 scored by balanced accuracy on all 15 questions; joint bootstrap over chats, 10,000 resamples | data/claude_vs_micro/metrics_haiku.json means.balanced_throughout
L86 | heads 352 of 405 (86.9%); Claude 281 of 405 (69.4%) | 'yes' cases answered yes, pooled over the 10 yes/no questions | metrics_haiku.json yes_no_pooled_recall.yes
L87 | heads 2,082 of 2,193 (94.9%); Claude 2,116 of 2,193 (96.5%) | 'no' cases answered no, pooled over the 10 yes/no questions | metrics_haiku.json yes_no_pooled_recall.no
L88 | 6 of 10; by 0.7 to 3.0 points | yes/no questions where Claude's plain accuracy is higher: cannot_log_in, security_related, mentions_account_issue, mentions_bug, refund_requested, charge_amount_given | metrics_haiku.json yes_no_claude_ahead_on_plain_accuracy, questions.*.accuracy
L89 | heads 95.1%; Claude 97.0%; yes found: heads 31, Claude 27 of 35; no kept: heads 223, Claude 232 of 232 | cannot_log_in: plain accuracy and recall on each class | metrics_haiku.json questions.cannot_log_in
L90 | heads 96.3%; Claude 99.3%; yes found: heads 18, Claude 17 of 18; no kept: heads 239, Claude 248 of 249 | mentions_bug: plain accuracy and recall on each class | metrics_haiku.json questions.mentions_bug
L91 | heads 93.2%; Claude 91.4%; yes found: heads 35, Claude 23 of 41; no kept: heads 171, Claude 179 of 180 | refund_implied: plain accuracy and recall on each class | metrics_haiku.json questions.refund_implied
L92 | heads 73.3%; Claude 58.0%; yes found: heads 122, Claude 82 of 150; no kept: heads 98, Claude 92 of 150 | needs_staff_action: plain accuracy and recall on each class | metrics_haiku.json questions.needs_staff_action
L93 | 5; Claude 0 | questions where the heads are right on more chats at Holm-corrected McNemar 0.05: flow, intent, journey_stage, message_type, needs_staff_action | metrics_haiku.json mcnemar_holm_005
L94 | committed 275 of 300; heads 93.1%; Claude 83.6%; difference 9.5 points; McNemar p 0.0 | flow: accuracy on chats where Claude committed to an answer the truth can take; only heads right 31, only Claude right 5 | robustness_haiku.json flow.accuracy_on_committed; counts in metrics_haiku.json committed.flow = 256 and 230 right of 275
L95 | {'None': 25}; heads right on 8.0% of them | flow: Claude's dropped answers | robustness_haiku.json flow.claude_answers_dropped, .micro_accuracy_on_dropped
L96 | committed 16; dropped 2 | flow: median opening words | robustness_haiku.json flow.opening_words_median
L97 | committed 258 of 300; heads 85.7%; Claude 76.4%; difference 9.3 points; McNemar p 0.0007 | intent: accuracy on chats where Claude committed to an answer the truth can take; only heads right 36, only Claude right 12 | robustness_haiku.json intent.accuracy_on_committed; counts in metrics_haiku.json committed.intent = 221 and 197 right of 258
L98 | {'None': 30, 'status_questions': 12}; heads right on 16.7% of them | intent: Claude's dropped answers | robustness_haiku.json intent.claude_answers_dropped, .micro_accuracy_on_dropped
L99 | committed 16.5; dropped 4.0 | intent: median opening words | robustness_haiku.json intent.opening_words_median
L100 | committed 267 of 267; heads 88.4%; Claude 83.1%; difference 5.2 points; McNemar p 0.0488 | primary_team: accuracy on chats where Claude committed to an answer the truth can take; only heads right 29, only Claude right 15 | robustness_haiku.json primary_team.accuracy_on_committed; counts in metrics_haiku.json committed.primary_team = 236 and 222 right of 267
L101 | committed 118 of 145; heads 98.3%; Claude 89.0%; difference 9.3 points; McNemar p 0.0034 | journey_stage: accuracy on chats where Claude committed to an answer the truth can take; only heads right 12, only Claude right 1 | robustness_haiku.json journey_stage.accuracy_on_committed; counts in metrics_haiku.json committed.journey_stage = 116 and 105 right of 118
L102 | {'unclear': 27}; heads right on 70.4% of them | journey_stage: Claude's dropped answers | robustness_haiku.json journey_stage.claude_answers_dropped, .micro_accuracy_on_dropped
L103 | committed 18.0; dropped 9 | journey_stage: median opening words | robustness_haiku.json journey_stage.opening_words_median
L104 | committed 259 of 285; heads 92.7%; Claude 74.1%; difference 18.5 points; McNemar p 0.0 | message_type: accuracy on chats where Claude committed to an answer the truth can take; only heads right 53, only Claude right 5 | robustness_haiku.json message_type.accuracy_on_committed; counts in metrics_haiku.json committed.message_type = 240 and 192 right of 259
L105 | {'feedback': 1, 'other': 25}; heads right on 42.3% of them | message_type: Claude's dropped answers | robustness_haiku.json message_type.claude_answers_dropped, .micro_accuracy_on_dropped
L106 | committed 16; dropped 2.0 | message_type: median opening words | robustness_haiku.json message_type.opening_words_median
L107 | p50 3.977 ms; p90 4.327; p99 4.851 | heads: all 15 questions for one opening | timing_micro.json latency_ms_15_questions.total
L108 | p50 0.489 ms; heads p50 3.465 | heads: feature step and the 15 heads | latency_ms_15_questions.features, .heads
L109 | intent 1.9366 ms; flow 0.3923; yes/no about 0.0207 | per-head median | latency_ms_15_questions.per_head_p50
L110 | 26.5 s; features 1.0 s; intent head 13.87 s | training all 15 heads on 8,034 openings | timing_micro.json train
L111 | 27.4 MB | saved size of the features and 15 heads | timing_micro.json size_mb
L112 | Apple M1 Max, 10 cores; Python 3.13.9; scikit-learn 1.7.2; 1 process | timing machine | timing_micro.json machine
L113 | p50 18.61 s; p90 38.93; p99 82.73; max 133.09 | Claude wall time per chat | stats_haiku.json claude_cost.wall_s
L114 | p50 16.66 s; p90 37.04 | Claude API time per chat | claude_cost.api_s
L115 | p50 3,768; p90 8,332; max 28,985 | Claude output tokens per chat | claude_cost.output_tokens
L116 | 96.4% | thinking share of Claude's output tokens | claude_cost.thinking_share_of_output = 0.9644
L117 | p50 1,826 | Claude input tokens per chat | claude_cost.input_tokens
L118 | $0.8287; mean $0.002762; p50 $0.002233 | Claude cost, 300 chats | claude_cost.cost_usd
L119 | 300 | Claude calls that returned a parseable answer | results_haiku.json summary.claude_answered = 300
L120 | $0.10, $0.20, $0.50 | price per million tokens: input, 1-hour cache write, output | derived from claude_haiku.jsonl id 8833: 2 x 0.10e-6 + 1,820 x 0.20e-6 + 2,022 x 0.50e-6 = $0.0013752, its cost field
L121 | 4,679 | Claude's median wall time over the heads' median latency | 18,610 ms / 3.977 ms
L122 | 4,189 | Claude's median API time over the heads' median latency | 16,660 ms / 3.977 ms
L123 | p50 1.46 s; p10 0.81; p90 2.55 | CLI overhead per call, wall time minus API time | data/claude_vs_micro/checks_haiku.json overhead_s
L124 | $27.62 | Claude cost per 10,000 chats at the mean | 10,000 x $0.002762
L125 | 2.237 ms | the 47-question synthetic pipeline, median per ABCD opening | timing_micro.json latency_ms_47_question_pipeline.p50
L126 | flow 87.7%; intent 76.1% | heads on all 1,004 test openings, balanced | data/abcd_analysis.json learn.flow.opening, learn.intent.opening
L127 | 480; 479 kept | synthetic tickets written by Claude | docs/how_it_works.md line 118
L128 | 383; 96 | synthetic tickets for training; held out by ticket | docs/how_it_works.md line 129; models/eval.json test_tickets = 96
L129 | $6.79; about 79 minutes | cost and time to write the synthetic tickets with Haiku | docs/how_it_works.md line 118
L130 | 93.2%; baseline 69.0% | 47 answers on 96 held-out synthetic tickets, mean accuracy | models/eval.json ticket_answers: mean accuracy 0.9322, mean baseline 0.6902 over 47
L131 | 99.4% | synthetic router's team accuracy on 333 held-out synthetic pieces | models/eval.json models.router.primary_team.accuracy = 0.994
L132 | synthetic system 66.3%; Claude 83.1% | primary_team: synthetic system vs Claude on the 300 openings, accuracy | results_haiku.json questions.primary_team.micro_syn.accuracy, .claude.accuracy
L133 | synthetic system 9.7%; Claude 72.4% | journey_stage: synthetic system vs Claude on the 300 openings, accuracy | results_haiku.json questions.journey_stage.micro_syn.accuracy, .claude.accuracy
L134 | synthetic system 26.7%; Claude 67.4% | message_type: synthetic system vs Claude on the 300 openings, accuracy | results_haiku.json questions.message_type.micro_syn.accuracy, .claude.accuracy
L135 | synthetic system 68.6%; Claude 88.6% | cannot_log_in: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.cannot_log_in.micro_syn.balanced, .claude.balanced
L136 | synthetic system 68.6%; Claude 91.8% | security_related: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.security_related.micro_syn.balanced, .claude.balanced
L137 | synthetic system 77.3%; Claude 90.1% | mentions_account_issue: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.mentions_account_issue.micro_syn.balanced, .claude.balanced
L138 | synthetic system 65.5%; Claude 85.3% | mentions_billing_issue: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.mentions_billing_issue.micro_syn.balanced, .claude.balanced
L139 | synthetic system 53.1%; Claude 97.0% | mentions_bug: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.mentions_bug.micro_syn.balanced, .claude.balanced
L140 | synthetic system 66.7%; Claude 100.0% | refund_requested: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.refund_requested.micro_syn.balanced, .claude.balanced
L141 | synthetic system 53.4%; Claude 77.8% | refund_implied: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.refund_implied.micro_syn.balanced, .claude.balanced
L142 | synthetic system 52.3%; Claude 58.0% | needs_staff_action: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.needs_staff_action.micro_syn.balanced, .claude.balanced
L143 | synthetic system 76.4%; Claude 92.9% | identity_details_given: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.identity_details_given.micro_syn.balanced, .claude.balanced
L144 | synthetic system 99.7%; Claude 75.0% | charge_amount_given: synthetic system vs Claude on the 300 openings, balanced | results_haiku.json questions.charge_amount_given.micro_syn.balanced, .claude.balanced
L145 | 12 of 13 | shared questions where the synthetic system scores below Claude | derived from the 13 lines above
L146 | 1.1 | ABCD release used | data/abcd/abcd_v1.1.json.gz
L147 | 25 points | change in balanced accuracy when the answer on one of 2 positive chats changes | derived: recall on yes moves by 1/2, and balanced accuracy averages 2 recalls: 100 x 0.5 / 2
L148 | 26 | message types Claude answered with a value the truth never takes | robustness_haiku.json message_type.claude_answers_dropped: other 25 + feedback 1
L149 | 42 | intents Claude answered with null or an intent absent from the sample | robustness_haiku.json intent.claude_answers_dropped: None 30 + status_questions 12
L150 | 333 | held-out synthetic pieces behind the router's 99.4% | models/eval.json models.router.primary_team.n
L151 | 50% | balanced accuracy of a constant answer on a yes/no question | derived: recall 1 on one class and 0 on the other, averaged: (100 + 0) / 2
L152 | 12 | Claude calls running at once | data/claude_vs_micro/compare.py WORKERS; the run passed workers = 12
L153 | 11 | shared questions whose synthetic-system answers come from learned models; identity_details_given and charge_amount_given come from its regex entity finder | micro_llm/rules.py:207 and :217
L154 | 47 | questions the synthetic system answers per ticket | models/eval.json ticket_answers, 47 questions; examples/support_ticket_98423/playground_questions.json
L155 | 2.1.310 | Claude Code CLI version for the alias check | claude --version, 2026-10-26
L156 | 2026-10-26 | date the comparison ran | file dates of data/claude_vs_micro/claude_haiku.jsonl and results_haiku.json
L157 | 5.5 | Claude Haiku version in the paper's model name | model ID claude-haiku-5-5, the modelUsage key from claude -p --model haiku, CLI 2.1.310
L158 | claude-haiku-5-5 | model the CLI's haiku alias resolves to | claude -p --model haiku --output-format json: modelUsage key, CLI 2.1.310
L159 | 50, 100, 200, 383, 1,000, 2,000, 4,000, 8,034 | training sizes in the learning curve; 5 nested random subsets per size below 8,034, 1 run at 8,034 | learning_curve_haiku.json curve[].size; seeds = 5
L160 | as reported 62.0% (SD 1.6; runs 60.0% to 64.1%); plain 78.9%; balanced 60.1%; 13 shared 65.5% | heads trained on 50 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 50
L161 | as reported 69.8% (SD 2.0; runs 67.9% to 72.1%); plain 83.0%; balanced 68.4%; 13 shared 72.7% | heads trained on 100 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 100
L162 | as reported 78.0% (SD 1.8; runs 76.3% to 80.4%); plain 87.1%; balanced 77.5%; 13 shared 80.2% | heads trained on 200 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 200
L163 | as reported 83.1% (SD 1.0; runs 81.3% to 83.9%); plain 89.6%; balanced 83.0%; 13 shared 84.8% | heads trained on 383 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 383
L164 | as reported 88.1% (SD 0.7; runs 87.5% to 89.2%); plain 91.1%; balanced 88.2%; 13 shared 89.6% | heads trained on 1,000 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 1000
L165 | as reported 89.8% (SD 0.6; runs 88.8% to 90.3%); plain 91.3%; balanced 89.9%; 13 shared 91.4% | heads trained on 2,000 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 2000
L166 | as reported 90.3% (SD 0.7; runs 89.3% to 90.9%); plain 91.9%; balanced 90.6%; 13 shared 91.8% | heads trained on 4,000 ABCD chats, mean of 15 questions over 5 run(s) | learning_curve_haiku.json curve, size 4000
L167 | as reported 90.7% (SD 0.0; runs 90.7% to 90.7%); plain 91.4%; balanced 91.0%; 13 shared 92.1% | heads trained on 8,034 ABCD chats, mean of 15 questions over 1 run(s) | learning_curve_haiku.json curve, size 8034
L168 | as reported 81.5%; plain 86.2%; balanced 80.5%; 13 shared 83.0% | Claude's means, recomputed by learning_curve.py | learning_curve_haiku.json reference.claude_15, .claude_13_shared
L169 | 60.3% | synthetic system on the 13 shared questions, as reported | learning_curve_haiku.json reference.micro_syn_13_shared.as_reported
L170 | 383; 200; 383 | smallest size whose mean over runs beats Claude: as reported; plain; balanced | learning_curve_haiku.json smallest_size_whose_seed_mean_beats_claude
L171 | 4 of 5; plain 5 of 5 | runs at 383 chats above Claude's mean, as reported; plain | derived from learning_curve_haiku.json runs, size 383
L172 | 0 of 5; plain 4 of 5 | runs at 200 chats above Claude's mean, as reported; plain | derived from learning_curve_haiku.json runs, size 200
L173 | 5 of 5 | runs at 1,000 chats above Claude's mean, as reported | derived from learning_curve_haiku.json runs, size 1000
L174 | 4 of 5 | runs at 383 chats above Claude on the 13 shared questions | derived from learning_curve_haiku.json runs, size 383
L175 | 24.5 points | heads on 383 ABCD chats minus the synthetic system on 383 synthetic tickets, 13 shared questions | 0.8477 - 0.6032
L176 | 0.9 points | mean gain from 2,000 chats to all 8,034, as reported | 0.9066 - 0.8976
L177 | 2.6 points | mean gain from 1,000 chats to all 8,034, as reported | 0.9066 - 0.881
L178 | 0 | full-size answers that differ from compare.py's heads | learning_curve_haiku.json full_size_mismatches_with_compare_py
L179 | 38.5 s | learning-curve run time, 36 trainings on 10 cores | learning_curve_haiku.json seconds
L180 | NumPy 2.3.5; SciPy 1.16.3; joblib 1.5.2 | library versions behind the statistics and the curve | python3 -I -c 'import numpy, scipy, joblib' on the timing machine, 2026-10-26
L181 | 270 of 300 | chats where Claude's intent answer names an intent the mapping knows | checks_haiku.json routed.claude_intents_usable
L182 | heads 88.4% direct, 91.0% routed; Claude 83.1% direct, 87.3% routed | primary_team: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.primary_team
L183 | heads 93.1% direct, 93.8% routed; Claude 72.4% direct, 86.2% routed | journey_stage: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.journey_stage
L184 | heads 88.1% direct, 89.5% routed; Claude 67.4% direct, 80.7% routed | message_type: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.message_type
L185 | heads 92.3% direct, 92.9% routed; Claude 88.6% direct, 92.6% routed | cannot_log_in: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.cannot_log_in
L186 | heads 92.3% direct, 92.9% routed; Claude 91.8% direct, 92.6% routed | security_related: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.security_related
L187 | heads 91.4% direct, 92.3% routed; Claude 90.1% direct, 90.8% routed | mentions_account_issue: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.mentions_account_issue
L188 | heads 93.4% direct, 94.0% routed; Claude 85.3% direct, 91.6% routed | mentions_billing_issue: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.mentions_billing_issue
L189 | heads 98.0% direct, 98.6% routed; Claude 97.0% direct, 97.2% routed | mentions_bug: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.mentions_bug
L190 | heads 98.9% direct, 99.7% routed; Claude 100.0% direct, 100.0% routed | refund_requested: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.refund_requested
L191 | heads 90.2% direct, 91.3% routed; Claude 77.8% direct, 90.9% routed | refund_implied: each system's own answer, and the answer its own intent maps to | checks_haiku.json routed.per_question.refund_implied
L192 | heads 92.6% direct, 93.6% routed; Claude 85.3% direct, 91.0% routed | mean over the 10 questions whose truth depends on the intent alone | checks_haiku.json routed.mean
L193 | 7.3 points direct; 1.6 against Claude routed; 2.6 both routed | heads minus Claude on those 10 questions | derived from checks_haiku.json routed.mean
L194 | 300 of 300 chats | chats where cannot_log_in and security_related have the same truth | checks_haiku.json exclusions
L195 | heads 90.7%; Claude 81.5%; gap 9.2 points; 15 questions | mean without none | checks_haiku.json exclusions.means
L196 | heads 90.0%; Claude 81.9%; gap 8.1 points; 14 questions | mean without charge_amount_given (2 positives) | checks_haiku.json exclusions.means
L197 | heads 91.9%; Claude 83.1%; gap 8.8 points; 14 questions | mean without needs_staff_action (follows the agent's later actions) | checks_haiku.json exclusions.means
L198 | heads 90.5%; Claude 80.9%; gap 9.6 points; 14 questions | mean without cannot_log_in (same truth as security_related) | checks_haiku.json exclusions.means
L199 | heads 91.2%; Claude 83.4%; gap 7.9 points; 12 questions | mean without all three | checks_haiku.json exclusions.means
L200 | heads 81.4%; Claude 71.2%; gap 10.2 points; 2 questions | mean without the 13 mapped questions (ABCD-native only) | checks_haiku.json exclusions.means
L201 | heads 92.1%; Claude 83.0%; gap 9.1 points; 13 questions | mean without flow and intent (mapped only) | checks_haiku.json exclusions.means
L202 | 7.9 to 10.2 points | range of the gap over the exclusions | checks_haiku.json exclusions.gap_range_points
L203 | heads 230 of 255 (90.2%); Claude 199 of 255 (78.0%) | 'yes' cases answered yes, pooled over the 9 yes/no questions whose 'yes' is the minority | checks_haiku.json rare_yes.recall
L204 | heads 1,984 of 2,043 (97.1%); Claude 2,024 of 2,043 (99.1%) | 'no' cases answered no, pooled over the same 9 questions | checks_haiku.json rare_yes.recall
L205 | heads 122 of 150; Claude 82 of 150 | needs_staff_action: 'yes' cases answered yes | checks_haiku.json rare_yes.needs_staff_action_yes
L206 | 4.0 points | shrink from the reported gap (9.2) to the plain-accuracy gap (5.2), summed over questions | checks_haiku.json shrink.total_points
L207 | charge_amount_given 1.68 (42.4%); refund_implied 0.71; cannot_log_in 0.38; the 5 choice questions 0.04 | each question's share of that shrink, in points | checks_haiku.json shrink.per_question_points, .largest_share
L208 | 6.6 (no 232, yes 35) | cannot_log_in: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.cannot_log_in
L209 | 6.6 (no 232, yes 35) | security_related: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.security_related
L210 | 5.8 (no 228, yes 39) | mentions_account_issue: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.mentions_account_issue
L211 | 4.2 (no 180, yes 43) | mentions_billing_issue: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.mentions_billing_issue
L212 | 13.8 (no 249, yes 18) | mentions_bug: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.mentions_bug
L213 | 30.0 (no 180, yes 6) | refund_requested: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.refund_requested
L214 | 4.4 (no 180, yes 41) | refund_implied: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.refund_implied
L215 | 1.0 (no 150, yes 150) | needs_staff_action: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.needs_staff_action
L216 | 7.3 (no 264, yes 36) | identity_details_given: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.identity_details_given
L217 | 149.0 (no 298, yes 2) | charge_amount_given: false alarms that cost as much as one missed 'yes' under balanced accuracy | checks_haiku.json false_alarms_per_missed_yes.charge_amount_given
L218 | reported metric 10.0%; plain accuracy 13.7%; 10 classes | flow: constant-answer baseline | checks_haiku.json baselines.per_question.flow
L219 | reported metric 1.9%; plain accuracy 4.0%; 53 classes | intent: constant-answer baseline | checks_haiku.json baselines.per_question.intent
L220 | reported metric 52.1%; plain accuracy 52.1%; 4 classes | primary_team: constant-answer baseline | checks_haiku.json baselines.per_question.primary_team
L221 | reported metric 69.0%; plain accuracy 69.0%; 3 classes | journey_stage: constant-answer baseline | checks_haiku.json baselines.per_question.journey_stage
L222 | reported metric 40.7%; plain accuracy 40.7%; 3 classes | message_type: constant-answer baseline | checks_haiku.json baselines.per_question.message_type
L223 | reported metric 50.0%; plain accuracy 86.9%; 2 classes | cannot_log_in: constant-answer baseline | checks_haiku.json baselines.per_question.cannot_log_in
L224 | reported metric 50.0%; plain accuracy 86.9%; 2 classes | security_related: constant-answer baseline | checks_haiku.json baselines.per_question.security_related
L225 | reported metric 50.0%; plain accuracy 85.4%; 2 classes | mentions_account_issue: constant-answer baseline | checks_haiku.json baselines.per_question.mentions_account_issue
L226 | reported metric 50.0%; plain accuracy 80.7%; 2 classes | mentions_billing_issue: constant-answer baseline | checks_haiku.json baselines.per_question.mentions_billing_issue
L227 | reported metric 50.0%; plain accuracy 93.3%; 2 classes | mentions_bug: constant-answer baseline | checks_haiku.json baselines.per_question.mentions_bug
L228 | reported metric 50.0%; plain accuracy 96.8%; 2 classes | refund_requested: constant-answer baseline | checks_haiku.json baselines.per_question.refund_requested
L229 | reported metric 50.0%; plain accuracy 81.4%; 2 classes | refund_implied: constant-answer baseline | checks_haiku.json baselines.per_question.refund_implied
L230 | reported metric 50.0%; plain accuracy 50.0%; 2 classes | needs_staff_action: constant-answer baseline | checks_haiku.json baselines.per_question.needs_staff_action
L231 | reported metric 50.0%; plain accuracy 88.0%; 2 classes | identity_details_given: constant-answer baseline | checks_haiku.json baselines.per_question.identity_details_given
L232 | reported metric 50.0%; plain accuracy 99.3%; 2 classes | charge_amount_given: constant-answer baseline | checks_haiku.json baselines.per_question.charge_amount_given
L233 | 72.4% (CI [0.6483, 0.8]) against 69.0% | journey_stage, claude: interval reaches the constant-answer baseline | checks_haiku.json baselines.interval_or_score_not_above_baseline
L234 | 99.0% against always-majority 99.3% | charge_amount_given, micro_abcd: plain accuracy not above the majority answer | checks_haiku.json baselines.interval_or_score_not_above_baseline
L235 | 75.0% (CI [0.5, 1.0]) against 50.0% | charge_amount_given, claude: interval reaches the constant-answer baseline | checks_haiku.json baselines.interval_or_score_not_above_baseline
L236 | Claude {'possible': 275, 'empty': 25}; heads {'possible': 300} | flow: answers by kind (possible, empty, offered but in no sampled chat's truth) | checks_haiku.json answer_kinds.flow
L237 | Claude {'null': 25}; heads {} | flow: answers no sampled chat's truth holds | checks_haiku.json answer_kinds.flow
L238 | Claude {'possible': 258, 'empty': 30, 'offered_but_no_chat_has_it': 12}; heads {'possible': 293, 'offered_but_no_chat_has_it': 7} | intent: answers by kind (possible, empty, offered but in no sampled chat's truth) | checks_haiku.json answer_kinds.intent
L239 | Claude {'null': 30, 'status_questions': 12}; heads {'manage_payment_method': 5, 'status_questions': 2} | intent: answers no sampled chat's truth holds | checks_haiku.json answer_kinds.intent
L240 | Claude {'possible': 267}; heads {'possible': 267} | primary_team: answers by kind (possible, empty, offered but in no sampled chat's truth) | checks_haiku.json answer_kinds.primary_team
L241 | Claude {'possible': 118, 'offered_but_no_chat_has_it': 27}; heads {'possible': 145} | journey_stage: answers by kind (possible, empty, offered but in no sampled chat's truth) | checks_haiku.json answer_kinds.journey_stage
L242 | Claude {'unclear': 27}; heads {} | journey_stage: answers no sampled chat's truth holds | checks_haiku.json answer_kinds.journey_stage
L243 | Claude {'possible': 259, 'offered_but_no_chat_has_it': 26}; heads {'possible': 285} | message_type: answers by kind (possible, empty, offered but in no sampled chat's truth) | checks_haiku.json answer_kinds.message_type
L244 | Claude {'other': 25, 'feedback': 1}; heads {} | message_type: answers no sampled chat's truth holds | checks_haiku.json answer_kinds.message_type
L245 | 28 exact (median 2 words); 37 at cosine 0.9 or more; median best cosine 0.659 | test openings with a near-duplicate among the 8,034 training openings | checks_haiku.json overlap
L246 | 263 chats; heads 93.7%; Claude 84.0%; gap 9.7 points | mean of 15 without the near-duplicate openings | checks_haiku.json overlap.without_near_duplicates
L247 | 89.8% on 1,004; 90.7% on the 300 | heads' mean of 15 on all test openings, as reported | checks_haiku.json full_test
L248 | charge_amount_given -10.8; refund_requested -7.7; mentions_bug -4.8 | largest moves from the 300 to all 1,004 test openings, points | checks_haiku.json full_test.largest_moves_points
L249 | 97.3% on the 13 shared; primary_team 92.7% per ticket; 96 tickets | synthetic system on its own held-out tickets | checks_haiku.json synthetic_own_tickets
L250 | heads 116.1; Claude 168.5 | chats per 1,000 sent to the wrong team, from primary_team accuracy | checks_haiku.json misroutes_per_1000
L251 | 31 errors; other -> account_support 7 (manage_downgrade 1, pricing 1, manage_create 1, out_of_stock_one_item 1, status_delivery_time 1); billing_and_payments -> other 5 (refund_update 2, bad_price_competitor 1, manage_extension 1, bad_price_yesterday 1); other -> billing_and_payments 5 (policy 2, return_color 1, status_shipping_question 1, jacket 1); other -> engineering 4 (manage_upgrade 2, manage_cancel 1, policy 1) | primary_team, micro_abcd: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.primary_team.micro_abcd
L252 | 45 errors; other -> billing_and_payments 20 (policy 4, return_color 3, return_size 3, status 3, return_stain 2); billing_and_payments -> other 10 (bad_price_competitor 3, manage_extension 3, bad_price_yesterday 3, refund_update 1); other -> account_support 6 (manage 4, status 2); account_support -> other 5 (reset_2fa 3, manage_change_name 1, recover_username 1) | primary_team, claude: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.primary_team.claude
L253 | 10 errors; post_purchase -> checkout 5 (bad_price_yesterday 2, manage_downgrade 1, manage_upgrade 1, refund_update 1); post_purchase -> browsing 2 (bad_price_competitor 1, bad_price_yesterday 1); browsing -> post_purchase 2 (shirt 1, jacket 1); browsing -> checkout 1 (out_of_stock_one_item 1) | journey_stage, micro_abcd: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.journey_stage.micro_abcd
L254 | 40 errors; post_purchase -> unclear 16 (bad_price_yesterday 4, manage_upgrade 2, manage 2, status 2, manage_downgrade 1); browsing -> unclear 10 (shirt 3, jeans 2, jacket 2, out_of_stock_general 1, out_of_stock_one_item 1); checkout -> browsing 7 (shopping_cart 7); post_purchase -> browsing 5 (bad_price_yesterday 4, bad_price_competitor 1) | journey_stage, claude: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.journey_stage.claude
L255 | 34 errors; request -> problem_report 11 (manage_upgrade 2, manage_extension 2, bad_price_yesterday 2, return_stain 1, manage_change_name 1); question -> request 9 (status_delivery_time 2, refund_status 1, status_due_amount 1, shirt 1, timing 1); request -> question 6 (manage_downgrade 1, refund_update 1, bad_price_competitor 1, bad_price_yesterday 1, manage_pay_bill 1); question -> problem_report 4 (status_quantity 1, refund_status 1, policy 1, status_due_date 1) | message_type, micro_abcd: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.message_type.micro_abcd
L256 | 93 errors; request -> problem_report 21 (reset_2fa 6, recover_password 4, bad_price_yesterday 4, recover_username 3, return_stain 2); question -> problem_report 17 (status_service_added 4, status_delivery_time 3, status_quantity 3, status_service_removed 2, policy 1); request -> other 15 (manage_extension 3, bad_price_yesterday 3, reset_2fa 2, manage_downgrade 1, manage_change_name 1); request -> question 13 (refund_update 3, bad_price_yesterday 2, manage_upgrade 2, bad_price_competitor 1, manage_extension 1) | message_type, claude: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.message_type.claude
L257 | 15 errors; no -> yes 9 (manage_extension 2, reset_2fa 2, manage_change_name 1, manage_upgrade 1, status_delivery_time 1); yes -> no 6 (bad_price_yesterday 3, return_color 2, bad_price_competitor 1) | refund_implied, micro_abcd: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.refund_implied.micro_abcd
L258 | 19 errors; yes -> no 18 (bad_price_yesterday 8, return_color 4, bad_price_competitor 3, return_stain 2, return_size 1); no -> yes 1 (reset_2fa 1) | refund_implied, claude: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.refund_implied.claude
L259 | 80 errors; no -> yes 52 (status_quantity 4, manage_downgrade 3, missing 3, manage_dispute_bill 3, status_service_removed 3); yes -> no 28 (status_delivery_time 3, missing 3, manage_extension 3, manage_create 2, reset_2fa 2) | needs_staff_action, micro_abcd: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.needs_staff_action.micro_abcd
L260 | 126 errors; yes -> no 68 (recover_password 10, out_of_stock_one_item 10, status_due_date 7, bad_price_yesterday 4, status_due_amount 4); no -> yes 58 (shopping_cart 8, search_results 6, status 5, manage_downgrade 4, manage_dispute_bill 3) | needs_staff_action, claude: errors by truth -> answer, with the true intents behind each | checks_haiku.json confusions.needs_staff_action.claude
L261 | 0.0015 | smallest possible Holm p over 15 questions with 10,000 permutations | derived: 15 x 1 / 10,001; stats.py adds one to both counts
L262 | heads 86.0%; Claude 76.7%; difference 9.3 points | flow: plain accuracy | metrics_haiku.json questions.flow.accuracy
L263 | heads 76.0%; Claude 65.7%; difference 10.3 points | intent: plain accuracy | metrics_haiku.json questions.intent.accuracy
L264 | synthetic heads 58.5% on ABCD, 90.5% on held-out synthetic; synthetic system 60.3%; heads on 383 ABCD chats 84.8%; on 8,034 92.1%; Claude 83.0% | mean of the 13 shared questions; synthetic heads trained on 383 synthetic tickets, held out 96 | data/claude_vs_micro/synthetic_heads_haiku.json mean_13_shared
L265 | synthetic heads 64.3%; synthetic system 68.0%; heads on 383 ABCD chats 84.1%; Claude 85.4% | mean of the 11 shared questions whose ABCD answers all appear in the synthetic training labels | synthetic_heads_haiku.json mean_covered
L266 | 26.2 points; 19.9 on the covered questions | same heads, same size: 383 ABCD chats minus 383 synthetic tickets | derived from synthetic_heads_haiku.json
L267 | 1.8 points | synthetic system minus the heads trained on its tickets, 13 shared | derived from synthetic_heads_haiku.json mean_13_shared
L268 | browsing, checkout: 45 of 145 chats | journey_stage: ABCD answers absent from the synthetic training labels | synthetic_heads_haiku.json questions.journey_stage, abcd_truth_classes.journey_stage
L269 | question: 110 of 285 chats | message_type: ABCD answers absent from the synthetic training labels | synthetic_heads_haiku.json questions.message_type, abcd_truth_classes.message_type
L270 | synthetic heads 47.2%; synthetic system 66.3%; heads on 383 ABCD chats 87.3% | primary_team: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.primary_team
L271 | synthetic heads 23.4%; synthetic system 9.7%; heads on 383 ABCD chats 92.1% | journey_stage: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.journey_stage
L272 | synthetic heads 30.5%; synthetic system 26.7%; heads on 383 ABCD chats 84.3% | message_type: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.message_type
L273 | synthetic heads 65.7%; synthetic system 68.6%; heads on 383 ABCD chats 92.7% | cannot_log_in: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.cannot_log_in
L274 | synthetic heads 71.4%; synthetic system 68.6%; heads on 383 ABCD chats 92.7% | security_related: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.security_related
L275 | synthetic heads 79.9%; synthetic system 77.3%; heads on 383 ABCD chats 91.8% | mentions_account_issue: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.mentions_account_issue
L276 | synthetic heads 68.4%; synthetic system 65.5%; heads on 383 ABCD chats 88.2% | mentions_billing_issue: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.mentions_billing_issue
L277 | synthetic heads 78.9%; synthetic system 53.1%; heads on 383 ABCD chats 95.6% | mentions_bug: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.mentions_bug
L278 | synthetic heads 66.4%; synthetic system 66.7%; heads on 383 ABCD chats 81.2% | refund_requested: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.refund_requested
L279 | synthetic heads 54.0%; synthetic system 53.4%; heads on 383 ABCD chats 83.9% | refund_implied: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.refund_implied
L280 | synthetic heads 50.7%; synthetic system 52.3%; heads on 383 ABCD chats 68.8% | needs_staff_action: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.needs_staff_action
L281 | synthetic heads 51.4%; synthetic system 76.4%; heads on 383 ABCD chats 83.7% | identity_details_given: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.identity_details_given
L282 | synthetic heads 72.8%; synthetic system 99.7%; heads on 383 ABCD chats 59.7% | charge_amount_given: same design on synthetic vs ABCD data, scored on the 300 | synthetic_heads_haiku.json questions.charge_amount_given
L283 | MAX_THINKING_TOKENS | environment variable the CLI binary reads; claude --help lists only --effort (low to max) and no option to turn thinking off | grep -a on ~/.local/share/claude/versions/2.1.310 (5 matches, one 'MAX_THINKING_TOKENS=0'); claude --help, 2026-10-26
L284 | send-link, promo-code | 2 of the 8 CHANGES actions that only inform or discount; the others change an order, account or password, or pass the case on | micro_llm/abcd.py:59-68 CHANGES
L285 | status_delivery_date -> status_delivery_time; FAQ subflows -> their first word | intent normalization | micro_llm/abcd.py:281-288 intent()
L286 | 96 raw subflows; 48 FAQ subflows; status_delivery_date 3 chats | ABCD subflows before normalization: the FAQ subflows name single FAQ entries such as boots_how_1 | data/abcd/abcd_v1.1.json.gz scenario.subflow over train, dev and test
L287 | 15; out_of_stock_one_item 11, out_of_stock_general 4 | chats among the 300 with no message_type truth, by intent | data/claude_vs_micro/rows_haiku.json truth.message_type is None
L288 | 704; $1.94 | test openings outside the 300; Claude's cost on them at its mean per chat | 1,004 - 300; 704 x $0.002762 = $1.9444
L289 | 135M 82%; Claude 3.7 Sonnet 78%; 70B 88%; 260 tasks | Lawma: accuracy on CaselawQA's legal annotation tasks | arXiv:2407.16615
L290 | a few hundred to a thousand | Lawma: labeled examples usually enough for higher accuracy | arXiv:2407.16615
L291 | SetFit 10-shot 88.1; best LLM 87.7; full data 94.1; 77 intents | Loukas et al.: Banking77 micro-F1 | arXiv:2311.06102
L292 | Claude 3 Haiku F1 0.736; SetFit 0.600; within 2%; about 50% less latency; 6 datasets | Arora et al.: intent detection, and a router that sends uncertain queries to the LLM | arXiv:2410.01627
L293 | RoBERTa 0.62; Claude Haiku 4.5 0.50; 0.007 s vs 10.48 s per item | Lee et al. 2026: macro-F1 and time per item | arXiv:2606.04274
L294 | 1.5B; 93.17 vs Claude 3.7 Sonnet 92.79; 51 ms vs 1,450 ms | Arch-Router: routing accuracy and latency | arXiv:2506.16655
L295 | stance 0.94 vs Claude 3 Opus 0.61 | Bucher and Martini: accuracy, fine-tuned vs zero-shot | arXiv:2406.08660
L296 | AG News RoBERTa 94.63 vs Claude Sonnet 4.5 91.35; IMDB Claude few-shot 96.48 vs RoBERTa 94.84 | Valdes Gonzalez: accuracy | arXiv:2602.06370
L297 | 770M T5; PaLM 540B; 80% of the data | Hsieh et al.: distilling step-by-step | arXiv:2305.02301
L298 | 25 benchmarks; 13 zero-shot models | Ziems et al. | arXiv:2305.03514
L299 | 25 tasks; about 25% lower quality | Kocon et al.: ChatGPT against task state of the art | arXiv:2302.10724
L300 | 16 datasets | Edwards and Camacho-Collados | arXiv:2403.17661
L301 | 32 datasets; 8 languages | Vajjala and Shimangaud | arXiv:2502.11830
L302 | up to 98% lower cost | FrugalGPT: cascades that match GPT-4 | arXiv:2305.05176
L303 | 17 tasks; Llama 2 7B | Buckmann and Hill: penalized logistic regression on embeddings | arXiv:2408.03414
L304 | 88.5% | Claude's routed mean of 10 under the strict rule: a known intent that settles nothing counts wrong | data/claude_vs_micro/revision_checks_haiku.json routing.systems.claude.mean_10.routed_strict = 0.885122
L305 | 91.9% | heads' routed mean of 10 under the strict rule: a known intent that settles nothing counts wrong | data/claude_vs_micro/revision_checks_haiku.json routing.systems.micro_abcd.mean_10.routed_strict = 0.919207
L306 | 7.3; Claude routed vs heads direct 1.6 (fallback), 4.1 (strict); both routed 2.6 (fallback), 3.4 (strict) | gaps on the 10 intent-derived questions, heads minus Claude, points | data/claude_vs_micro/revision_checks_haiku.json routing.gaps_points = {"direct": 7.2584, "claude_routed_vs_heads_direct": {"fallback": 1.6074, "strict": 4.0947}, "both_routed": {"fallback": 2.5842, "strict": 3.4085}, "share_of_direct_gap_closed_both_routed": {"fallback": 0.643971, "strict": 0.530406}}
L307 | 64%; 53% | share of the 10-question direct gap that routing both systems closes: fallback; strict | data/claude_vs_micro/revision_checks_haiku.json share_of_direct_gap_closed_both_routed = {'fallback': 0.643971, 'strict': 0.530406}
L308 | 62; 66 | graded chat-question pairs where a known intent settles nothing and the direct answer stands: Claude; heads | data/claude_vs_micro/revision_checks_haiku.json graded_pairs_falling_back.known_intent_routes_nowhere: claude, micro_abcd
L309 | 260; 0 | graded chat-question pairs where the intent answer is one the mapping does not know: Claude; heads | data/claude_vs_micro/revision_checks_haiku.json graded_pairs_falling_back.unknown_intent: claude, micro_abcd
L310 | 270; 258; 30; 12 | Claude intent answers the mapping knows; that a sampled truth holds; not known (300 - 270); known but held by no sampled truth (270 - 258) | data/claude_vs_micro/revision_checks_haiku.json routing.systems.claude.intent_known_to_mapping, intent_a_sampled_truth_holds
L311 | 300; 293 | heads' intent answers the mapping knows; that a sampled truth holds | data/claude_vs_micro/revision_checks_haiku.json routing.systems.micro_abcd
L312 | -21.5 to -17.4; lower bounds -24.8 to -20.9; upper bounds -17.8 to -13.2; above zero in 0 of 5, below in 5 | heads minus Claude on the mean of 15 (as reported) at 50 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=50].as_reported = [-18.687649, -21.489946, -20.328555, -19.147125, -17.394579]; [[-22.344093, -14.498127], [-24.847529, -17.832357], [-23.909473, -16.131983], [-22.48101, -15.274246], [-20.909778, -13.222781]]
L313 | -13.5 to -9.4; lower bounds -16.9 to -12.9; upper bounds -9.8 to -5.1; above zero in 0 of 5, below in 5 | heads minus Claude on the mean of 15 (as reported) at 100 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=100].as_reported = [-13.459234, -9.699354, -9.378357, -13.544881, -11.949744]; [[-16.720644, -9.603419], [-13.097057, -5.851573], [-12.903763, -5.120774], [-16.926704, -9.844843], [-15.460337, -7.835422]]
L314 | -5.1 to -1.1; lower bounds -8.4 to -4.4; upper bounds -1.7 to 2.7; above zero in 0 of 5, below in 3 | heads minus Claude on the mean of 15 (as reported) at 200 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=200].as_reported = [-4.259199, -2.10199, -1.072163, -4.636772, -5.147583]; [[-6.620804, -1.656348], [-5.291069, 1.350073], [-4.363759, 2.74625], [-7.969018, -0.721057], [-8.371381, -1.29232]]
L315 | -0.1 to +2.4; lower bounds -3.4 to 0.1; upper bounds 3.8 to 6.9; above zero in 1 of 5, below in 0 | heads minus Claude on the mean of 15 (as reported) at 383 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=383].as_reported = [2.399374, 2.023653, 1.939427, -0.144585, 2.007194]; [[0.148143, 4.771627], [-1.023608, 5.181537], [-1.248342, 5.417912], [-3.373761, 3.835761], [-2.778383, 6.85397]]
L316 | +6.0 to +7.8; lower bounds 1.3 to 4.5; upper bounds 8.4 to 10.8; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (as reported) at 1,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=1000].as_reported = [7.756384, 6.342959, 5.992773, 6.891205, 6.258781]; [[4.474439, 10.827099], [4.254806, 8.449117], [1.279305, 10.787348], [3.597986, 10.149643], [2.850673, 9.413119]]
L317 | +7.3 to +8.9; lower bounds 3.9 to 6.0; upper bounds 10.5 to 11.7; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (as reported) at 2,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=2000].as_reported = [8.855458, 8.519935, 8.354576, 8.490052, 7.328326]; [[6.036283, 11.615226], [5.207624, 11.574208], [5.184321, 11.378352], [5.154488, 11.672302], [3.851132, 10.508525]]
L318 | +7.9 to +9.5; lower bounds 4.5 to 6.6; upper bounds 11.0 to 12.5; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (as reported) at 4,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=4000].as_reported = [9.404886, 8.728637, 8.856398, 9.472945, 7.850919]; [[6.64341, 12.155748], [5.644045, 11.762611], [5.975785, 11.754802], [6.37702, 12.451094], [4.458903, 10.98502]]
L319 | 90.6% vs 81.5%; lead 9.1 (6.1 to 12.1) | heads trained without class weights against Claude, mean of 15, as reported | data/claude_vs_micro/revision_checks_haiku.json unweighted_heads.lead.as_reported = {"heads": 0.905658, "claude": 0.814534, "lead_points": 9.11234, "lead_ci95_points": [6.111745, 12.051398]}
L320 | 93.2% vs 86.2%; lead 7.1 (5.9 to 8.2) | heads trained without class weights against Claude, mean of 15, plain accuracy throughout | data/claude_vs_micro/revision_checks_haiku.json unweighted_heads.lead.plain_throughout = {"heads": 0.932391, "claude": 0.861766, "lead_points": 7.062542, "lead_ci95_points": [5.91882, 8.243894]}
L321 | 90.3% vs 80.5%; lead 9.8 (6.7 to 12.7) | heads trained without class weights against Claude, mean of 15, balanced accuracy throughout | data/claude_vs_micro/revision_checks_haiku.json unweighted_heads.lead.balanced_throughout = {"heads": 0.903129, "claude": 0.805269, "lead_points": 9.785919, "lead_ci95_points": [6.734653, 12.730584]}
L322 | 1 of 10; weighted 6 of 10 | yes/no questions where Claude's plain accuracy is higher than the heads': unweighted heads; weighted heads | data/claude_vs_micro/revision_checks_haiku.json unweighted_heads.yes_no_where_claude_plain_is_higher = {'unweighted': 1, 'weighted': 6}
L323 | 1.8; regex 4.0; learned -2.2 | synthetic system minus synthetic-trained heads on the mean of 13: total; its 2 regular-expression answers; its 11 learned answers, points | data/claude_vs_micro/revision_checks_haiku.json synthetic_system_minus_synthetic_heads = total 1.799446, regex 3.988123, learned -2.188677
L324 | heads 25 missed, 59 false flags, 84 errors; Claude 56 missed, 19 false flags, 75 errors | errors on the 9 rare-yes questions | data/claude_vs_micro/revision_checks_haiku.json rare_yes_errors.systems
L325 | 31; 40 | Claude's extra missed yes cases; the heads' extra false flags | derived: 56 - 25; 59 - 19
L326 | 2.8 | points of balanced accuracy that one of the 18 yes chats moves on mentions_bug | derived: 100 / (2 x 18) = 2.78
L327 | 53; primary_team 0, journey_stage 27, message_type 26 | Claude answers, on chats with a truth, that the prompt offered and no ABCD truth can take | data/claude_vs_micro/rows_haiku.json; the mapping in micro_llm/abcd.py never yields product_management, unclear, feedback or other
L328 | -7.8 to -6.1; lower bounds -9.6 to -7.8; upper bounds -6.0 to -4.3; above zero in 0 of 5, below in 5 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 50 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=50].plain_throughout = [-7.612844, -7.75188, -7.710262, -7.042625, -6.070743]; [[-9.634262, -5.587597], [-9.591375, -5.927461], [-9.427091, -5.956647], [-8.951273, -5.161907], [-7.826518, -4.342428]]
L329 | -5.3 to -1.7; lower bounds -7.2 to -3.3; upper bounds -3.4 to -0.0; above zero in 0 of 5, below in 5 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 100 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=100].plain_throughout = [-5.307505, -2.075924, -1.681796, -4.372137, -2.454861]; [[-7.218429, -3.367329], [-3.605303, -0.523979], [-3.316346, -0.006357], [-6.084316, -2.639618], [-4.047847, -0.853223]]
L330 | -0.8 to +2.2; lower bounds -2.6 to 0.7; upper bounds 0.8 to 3.7; above zero in 2 of 5, below in 0 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 200 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=200].plain_throughout = [-0.846815, 1.5894, 2.197093, 0.987849, 0.813557]; [[-2.570561, 0.848938], [0.241489, 2.938119], [0.678643, 3.722199], [-0.573704, 2.601385], [-0.622627, 2.284454]]
L331 | +3.1 to +3.8; lower bounds 1.7 to 2.6; upper bounds 4.5 to 5.1; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 383 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=383].plain_throughout = [3.09958, 3.82648, 3.173942, 3.341779, 3.754372]; [[1.714823, 4.485061], [2.577328, 5.063558], [1.745834, 4.588319], [1.8411, 4.872188], [2.520358, 5.015732]]
L332 | +4.5 to +5.2; lower bounds 3.3 to 3.9; upper bounds 5.8 to 6.6; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 1,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=1000].plain_throughout = [4.903969, 5.218949, 4.856826, 4.993339, 4.522966]; [[3.517348, 6.304271], [3.867286, 6.590557], [3.467228, 6.268496], [3.640394, 6.343863], [3.276862, 5.782537]]
L333 | +3.9 to +6.0; lower bounds 2.2 to 4.7; upper bounds 5.5 to 7.2; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 2,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=2000].plain_throughout = [5.192536, 5.609196, 4.925791, 5.962734, 3.889091]; [[3.908598, 6.495136], [4.251062, 6.951118], [3.470282, 6.34027], [4.727583, 7.234719], [2.212578, 5.532848]]
L334 | +4.8 to +6.5; lower bounds 3.4 to 5.3; upper bounds 6.3 to 7.7; above zero in 5 of 5, below in 0 | heads minus Claude on the mean of 15 (plain accuracy throughout) at 4,000 training chats: the 5 runs' leads and 95% paired intervals, points | data/claude_vs_micro/revision_checks_haiku.json learning_curve_leads.curve[size=4000].plain_throughout = [6.055349, 6.115152, 4.925015, 6.501074, 4.823121]; [[4.802229, 7.303932], [4.926752, 7.304416], [3.40963, 6.435819], [5.31422, 7.737151], [3.36975, 6.267911]]
L335 | 0.11 to 0.45 s; features 1.0 s | training time of one yes/no head on 8,034 openings, over the 10 yes/no heads; the shared features take extra | timing_micro.json train.seconds_per_head (yes/no questions), train.seconds_features
