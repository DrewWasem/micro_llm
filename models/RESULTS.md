# micro_llm v1 results: 9 small models on support tickets

**Date:** 2026-10-24 · **Data:** 479 synthetic tickets, 96 held out by ticket


**The 9 models answer all 47 questions at 93% accuracy against a 69% baseline, in 3.9 ms per ticket.**


- **Yardstick:** each accuracy sits beside its majority-class baseline, the score of always giving the most common answer.
- **Truth for ticket answers:** the same rules run on the true labels, so a miss means a model or entity rule erred.
- **Speed:** median 3.9 ms and p95 5.96 ms per ticket on this laptop, against ~100–350 ms estimated for one Jev call.
- **Data cost:** $6.79 of Haiku through the Claude CLI, about 79 minutes with 12 parallel calls.
- **The sample ticket:** 15 model runs in 4.74 ms; 7 of 7 choice answers match `my_predictions.json`, and the other 40 answers sit 0.18 from it on average.

## Ticket answers beat their baselines in all three groups

| Source       | Questions | Accuracy | Baseline |
|--------------|-----------|----------|----------|
| models       | 25        | 0.916    | 0.718    |
| code rules   | 15        | 0.942    | 0.593    |
| entity rules | 7         | 0.969    | 0.798    |

## Each model on the held-out pieces it is meant for

Specialists are scored only on their own team's pieces. `n` counts those pieces.

| Model   | Question                  | n   | Accuracy | Baseline | Balanced | MAE   |
|---------|---------------------------|-----|----------|----------|----------|-------|
| router  | piece_kind                | 333 | 0.976    | 0.396    |          |       |
| router  | primary_team              | 333 | 0.994    | 0.505    |          |       |
| tone    | frustration               | 333 | 0.706    | 0.658    |          | 0.347 |
| tone    | sentiment                 | 333 | 0.577    | 0.462    |          | 0.529 |
| tone    | politeness                | 333 | 0.508    | 0.402    |          | 0.546 |
| tone    | time_pressure             | 333 | 0.979    | 0.94     | 0.825    |       |
| tone    | threatens_to_leave        | 333 | 1.0      | 0.961    | 1.0      |       |
| tone    | asks_for_human            | 333 | 0.997    | 0.973    | 0.944    |       |
| tone    | prior_contact             | 333 | 0.994    | 0.955    | 0.965    |       |
| tone    | frustration_rising        | 333 | 1.0      | 0.949    | 1.0      |       |
| billing | duplicate_charge          | 40  | 0.975    | 0.75     | 0.983    |       |
| billing | refund_requested          | 40  | 0.975    | 0.625    | 0.98     |       |
| billing | refund_implied            | 40  | 1.0      | 0.7      | 1.0      |       |
| billing | payment_method_used       | 40  | 0.825    | 0.4      |          |       |
| billing | chargeback_mentioned      | 40  | 1.0      | 0.9      | 1.0      |       |
| account | cannot_log_in             | 51  | 1.0      | 0.569    | 1.0      |       |
| account | security_related          | 51  | 0.98     | 0.51     | 0.98     |       |
| account | tried_fix_themselves      | 51  | 0.824    | 0.765    | 0.827    |       |
| bug     | bug_severity              | 87  | 0.908    | 0.448    |          | 0.151 |
| bug     | has_reproducible_steps    | 87  | 0.862    | 0.563    | 0.863    |       |
| bug     | problem_after_update      | 87  | 0.966    | 0.609    | 0.956    |       |
| bug     | blames_update             | 87  | 0.897    | 0.77     | 0.88     |       |
| bug     | possible_widespread_issue | 87  | 0.908    | 0.575    | 0.902    |       |
| feature | wants_new_payment_method  | 25  | 1.0      | 0.6      | 1.0      |       |
| feature | feature_request_strength  | 25  | 0.96     | 0.52     |          | 0.131 |
| linker  | issues_related            | 155 | 0.994    | 0.826    | 0.981    |       |

**Splitter:** 91.4% of non-greeting pieces cut exactly; 91.7% of tickets cut exactly in full.

**Every splitter miss is a ticket written without punctuation**, such as "could you let me know which countries you ship to i was also thinking…". Regex has no boundary to find there. The generator's exact spans could train a boundary model.

**The tone levels are the reverse condition firing:** frustration, sentiment and politeness sit 0.05–0.12 above baseline, about half a level off on average (MAE 0.35–0.55). They are the first candidates for a small transformer.

## The 10 weakest ticket answers, by margin over baseline

| Question                 | Accuracy | Baseline | Margin |
|--------------------------|----------|----------|--------|
| sentiment                | 0.573    | 0.604    | -0.031 |
| churn_risk               | 0.75     | 0.771    | -0.021 |
| chargeback_mentioned     | 1.0      | 0.958    | 0.042  |
| tried_fix_themselves     | 0.917    | 0.875    | 0.042  |
| order_date_given         | 0.927    | 0.875    | 0.052  |
| duplicate_charge         | 0.979    | 0.896    | 0.083  |
| asks_for_human           | 0.99     | 0.906    | 0.084  |
| message_type             | 1.0      | 0.896    | 0.104  |
| wants_new_payment_method | 1.0      | 0.896    | 0.104  |
| prior_contact            | 0.958    | 0.844    | 0.114  |

## The sample ticket routes the way the plan said

| Piece | Kind    | Team                 | Models                  | Text                                                            |
|-------|---------|----------------------|-------------------------|-----------------------------------------------------------------|
| P1    | problem | billing_and_payments | router+tone+billing     | I placed an order (#98423) last Thursday and was charged twice. |
| P2    | problem | account_support      | router+tone+account+bug | I also can't log in after the site update                       |
| P3    | request | product_management   | router+tone+feature     | adding Apple Pay would be really helpful.                       |
| P4    | feeling | other                | router+tone             | This is getting frustrating.                                    |

Largest gaps from the predictions written before any model existed:

| Question                  | Predicted | Got   | Gap  |
|---------------------------|-----------|-------|------|
| politeness                | 2.5       | 1.61  | 0.89 |
| sentiment                 | 1.0       | 1.79  | 0.79 |
| churn_risk                | 1.0       | 0.56  | 0.44 |
| tried_fix_themselves      | 0.05      | 0.467 | 0.42 |
| security_related          | 0.55      | 0.951 | 0.4  |
| likely_apple_user         | 0.6       | 1.0   | 0.4  |
| mentions_bug              | 0.6       | 0.953 | 0.35 |
| possible_widespread_issue | 0.65      | 0.318 | 0.33 |


**The 9 models answer all 47 questions at 93% accuracy against a 69% baseline, in 3.9 ms per ticket.**
