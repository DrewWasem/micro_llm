# How micro_llm answers a support ticket


**Each answer comes from a small model reading one piece of the ticket, then a one-line rule. The 7 trained models are logistic regressions that trained in 7.8 s on tickets Claude wrote for $6.79.**


Every number below comes from the sample ticket #98423. [pipeline.html](pipeline.html) draws the same paths; click a question there to light up its route.

> Hi, I placed an order (#98423) last Thursday and was charged twice. I also can't log in after the site update, and adding Apple Pay would be really helpful. This is getting frustrating.

## Five answers show all three ways an answer gets made

| Answer              | Made by                                | Sample answer         | Held-out accuracy vs baseline |
|---------------------|----------------------------------------|-----------------------|-------------------------------|
| `duplicate_charge`  | billing model, on the piece sent to it | 0.91 chance of yes    | 98% vs 90%                    |
| `frustration`       | tone model, on every piece             | level 1.04 of 2       | 68% vs 45%                    |
| `category`          | rule over the router's team chances    | billing, 0.49         | 90% vs 36%                    |
| `priority`          | rule over 8 other answers              | level 2.19 of 3, High | 90% vs 41%                    |
| `likely_apple_user` | regex over the whole ticket            | 1.00 chance of yes    | 100% vs 78%                   |

Held-out accuracy comes from 96 tickets the models never saw. The baseline always gives the most common answer.

All five start the same way. The splitter cuts the ticket at sentence ends and at ", and", ", but" or ", plus". It drops "Hi," because a greeting asks for nothing. Then the router reads each piece:

| Piece | Text                                                            | Router's top kind and team    |
|-------|-----------------------------------------------------------------|-------------------------------|
| P1    | I placed an order (#98423) last Thursday and was charged twice. | problem 0.93, billing 0.92    |
| P2    | I also can't log in after the site update                       | problem 0.98, account 0.95    |
| P3    | adding Apple Pay would be really helpful.                       | request 0.91, product 0.85    |
| P4    | This is getting frustrating.                                    | feeling 0.92, other team 0.90 |

### `duplicate_charge` is 0.91 because only P1 reached the billing model

1. P1 counts as an issue, because its problem and request chances add up to at least 0.5.
2. P1's billing chance is 0.92, above the 0.3 cut-off, so P1 goes to billing. P2 (0.01) and P3 (0.04) stay below it.
3. The billing model reads P1 and gives duplicate_charge a 0.905 chance.
4. The rule takes the highest chance across the pieces billing read. P1 is the only one, so the answer is 0.91.

Predicted before training: 0.97. The billing model reads only P1, so the login problem stays out of billing answers.

### `frustration` is level 1.04 because the angriest piece decides

1. The tone model reads all 4 pieces, because tone can sit in any of them.
2. It gives each piece a chance for each level. The piece's level is the chance-weighted average.
3. The rule takes the piece with the highest level: P4 at 1.04, "Frustrated but civil".

| Piece | 0 Calm | 1 Frustrated but civil | 2 Very angry | Level |
|-------|--------|------------------------|--------------|-------|
| P1    | 0.78   | 0.17                   | 0.05         | 0.27  |
| P2    | 0.51   | 0.39                   | 0.11         | 0.60  |
| P3    | 0.94   | 0.03                   | 0.03         | 0.09  |
| P4    | 0.01   | 0.94                   | 0.05         | 1.04  |

Predicted before training: 1.0. Frustration is the weakest of the five, because word features see "frustrating" but not tone carried by phrasing.

### `category` is billing because P1's vote counts double

1. Only issue pieces vote, so P1, P2 and P3 vote and P4 does not.
2. Each issue adds its team chances to the matching category. Engineering maps to bug_report and product to feature_request.
3. P1 counts double, because the rule treats the first issue as the reason for writing.
4. Each share is the category's votes divided by the 3.824 total. Billing wins with 0.49.

| Category        | P1 × 2 | P2    | P3    | Votes | Share |
|-----------------|--------|-------|-------|-------|-------|
| billing         | 1.836  | 0.007 | 0.043 | 1.886 | 0.49  |
| account         | 0.022  | 0.954 | 0.013 | 0.989 | 0.26  |
| feature_request | 0.022  | 0.008 | 0.852 | 0.882 | 0.23  |
| bug_report      | 0.028  | 0.023 | 0.016 | 0.067 | 0.02  |

Predicted before training: billing. The double vote decides this ticket: without it, account would win 0.978 to 0.968.

### `priority` is High because money and login access are both at stake

1. A rule, not a model, builds priority from 8 other answers. It fills the most urgent level first.
2. **Urgent** is the chance the bug is widespread (0.318) times the chance it blocks the user (0.673), which gives 0.214. The bug model gave both after reading P2.
3. **High** is what is left (0.786) times the largest money, access or time-pressure answer. cannot_log_in is largest at 0.969, so High gets 0.762.
4. **Normal** is what is left (0.024) times the top problem chance (0.975), which gives 0.024.
5. **Low** takes the remaining 0.001.
6. The level is the weighted average: 1 × 0.024 + 2 × 0.762 + 3 × 0.214 = 2.19, which rounds to High.

Predicted before training: 2.0. Either the double charge or the login failure alone would still give High, because High takes the largest input, not the sum.

### `likely_apple_user` is 1.0 because a regex found "Apple Pay"

1. The entity finder scans the whole ticket with regexes. No model runs.
2. It finds "Apple Pay" at characters 122–131 and tags it as a payment method.
3. The rule says yes when an entity names an Apple product or Apple Pay, so the answer is 1.0.

Predicted before training: 0.6, because asking for Apple Pay hints at an Apple device without proving it. The 100% score cannot catch that overconfidence: the generator marks Apple users by the same test.

## The 7 trained models learned from 383 tickets Claude wrote

2 of the 9 models are rules, and 7 are classifiers trained on labelled ticket pieces.

| Model         | Type    | Reads                     | Questions          | Training rows    |
|---------------|---------|---------------------------|--------------------|------------------|
| splitter      | rules   | the whole ticket          | none; makes pieces | none             |
| entity finder | regexes | the whole ticket          | 7                  | none             |
| router        | trained | every piece               | 2                  | 1,295 pieces     |
| tone          | trained | every piece               | 8                  | 1,295 pieces     |
| billing       | trained | issue pieces sent to it   | 5                  | 813 issue pieces |
| account       | trained | issue pieces sent to it   | 3                  | 813 issue pieces |
| bug           | trained | issue pieces sent to it   | 5                  | 813 issue pieces |
| feature       | trained | issue pieces sent to it   | 2                  | 813 issue pieces |
| linker        | trained | each pair of issue pieces | 1                  | 603 pairs        |

The router's piece kind only steers routing, so 25 of the 26 model questions are ticket answers. 15 more, such as category and priority, are rules over those answers. With the 7 regex answers, that makes 47.

### Claude wrote the training tickets from labels picked in code

Code picks the labels first and Claude only supplies the wording, so every label and every piece boundary is exact.

1. Code samples one ticket's labels: mood, 1–4 issues with a team and kind, greetings, a feeling piece, flags and links. The appendix lists the rates.
2. Claude Haiku, called through `claude -p`, writes one sentence of 4–25 words for each piece.
3. Code checks each sentence and its entities, then joins the pieces in order.
4. A quarter of adjacent issue pairs get joined with ", and", ", but" or ", plus", so the splitter must handle run-ons.

The run kept 479 of 480 tickets. It cost $6.79 and took about 79 minutes, with 12 calls running at once and 2 tickets per call.

### Each trained model is word features plus one logistic regression per question

- **Features:** TF-IDF over words and word pairs, plus runs of 2–5 characters. Character runs let "charged" and "charge" share features.
- **One regression per question:** each question gets its own logistic regression on the shared features. Balanced class weights keep rare answers, such as chargebacks, from being ignored.
- **Three answer shapes:** a yes/no question returns a chance of yes. A choice returns each option's chance. A level returns each level's chance and their weighted average.
- **The linker reads two pieces at once.** It stems each content word and marks whether both pieces use it, so a shared word like "update" signals a link.

### Training split by ticket and taught the specialists to say no

- **Split:** the 479 tickets were shuffled with a fixed seed. 96 were held out and 383 trained the models, so no ticket has pieces on both sides.
- **Specialists train on all 813 issue pieces.** Pieces from other teams carry "no", "not stated" or level 0, so a misrouted piece gets "no" answers instead of guesses.
- **Own pieces per specialist:** billing 199, account 149, bug 259 and feature 125.
- **Speed:** all 7 models trained in 7.8 s on a CPU. They are saved to `models/`, which git ignores.

### The 47 answers score 93% on held-out tickets against a 69% baseline

| Answers from   | Questions | Accuracy | Baseline |
|----------------|-----------|----------|----------|
| trained models | 25        | 91.6%    | 71.8%    |
| code rules     | 15        | 94.2%    | 59.3%    |
| entity regexes | 7         | 96.9%    | 79.8%    |
| all            | 47        | 93.2%    | 69.0%    |

- **Scoring:** a yes/no answer counts as yes at 0.5 or more. A level is rounded to the nearest whole level.
- **Splitter:** it cuts 91.4% of pieces exactly where the generator joined them.
- **Speed:** a ticket takes 3.9 ms at the median and 6.0 ms at the 95th percentile.

### The 93% hides three limits

1. **Tone is the weak model.** On single pieces, frustration scores 71% against 66%, sentiment 58% against 46% and politeness 51% against 40%.
2. **Every test ticket is synthetic.** One generator wrote the training and test tickets, so real tickets with typos and no punctuation will score lower.
3. **Rules are graded against themselves.** Test truth for the 15 code rules applies those same rules to the true piece labels. A wrong rule costs no points.


**Each answer is a small model reading one piece plus a one-line rule. Tone is the weak link at 68% on frustration, and every score comes from synthetic tickets.**


## Appendix: the generator picks each label at a fixed rate

| Label                    | Rate                                                              |
|--------------------------|-------------------------------------------------------------------|
| Mood                     | calm 30%, annoyed 32%, angry 16%, happy 10%, anxious 12%          |
| Issues per ticket        | 1: 30%, 2: 40%, 3: 25%, 4: 5%; no two share a team and kind       |
| Issue team               | billing 25%, account 22%, engineering 22%, product 16%, other 15% |
| Payment method (billing) | card 30%, PayPal 15%, Apple Pay 12%, other 8%, not stated 35%     |
| Chargeback               | 25% of wrong charges                                              |
| Feeling piece            | 70% when annoyed or angry, otherwise 35%                          |
| Greetings                | opening 45%, closing 30%                                          |
| Asks for a human         | 12%                                                               |
| Time pressure            | 20%, on one issue piece                                           |
| Threatens to leave       | 35% of annoyed or angry tickets with a feeling piece              |
| Frustration rising       | 40% of annoyed or angry tickets with a feeling piece              |
| Prior contact            | 15%                                                               |
| Two issues linked        | 35% of tickets with 2 or more issues, with a stated cause         |
