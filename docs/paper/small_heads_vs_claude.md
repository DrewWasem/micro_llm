# Four-Millisecond Classifiers Beat Zero-Shot Claude Haiku 5.5 at Triaging ABCD Support Chats

Drew Wasem, Independent

## Abstract

Support teams can triage chats by prompting a large language model or by training small classifiers on labeled chats. I compare the two on 15 triage questions for 300 openings from ABCD, a corpus of role-played store chats. Logistic-regression heads over TF-IDF features, trained on 8,034 labeled chats, average 90.7% (95% CI 88.3 to 92.8). Claude Haiku 5.5, prompted zero-shot with label names, one-line definitions for 13 questions and no examples, scores 81.5% (78.2 to 84.7). On ABCD's own labels, topic and intent, the heads lead by 10.1 and 10.2 points, both significant after Holm correction. Across three scoring rules the mean gap is 5.2 to 10.4 points. The heads draw level with Claude at 383 labeled chats and lead in every run from 1,000. Deriving both systems' answers from their own intent answers shrinks the gap on 10 intent-derived questions from 7.3 to 2.6–3.4 points. At the median the heads answer in 3.98 ms, and Claude, with thinking on through its command-line interface, in 18.61 s at a mean $0.0028 per chat. Trained instead on 383 tickets Claude Haiku wrote, the heads score 58.5% on 13 shared questions, against 84.8% when trained on 383 ABCD chats.

## 1 Introduction

A support desk asks every new chat the same few questions: its topic, its team, whether the customer wants money back, and whether staff must act on an account or order. The answers decide which queue a chat lands in and what an agent sees on opening it. A wrong answer costs a reroute by hand and a slower reply.

A team has two practical ways to get these answers. It can prompt a large language model (LLM) with the label set and the customer's words. That needs no labeled data but costs seconds and a fraction of a cent per chat. Or it can train a small classifier per question on chats it has already labeled, which costs labeling effort and then runs in milliseconds on a laptop. Both must beat a constant answer. Published comparisons of the two mostly report accuracy, and fewer report what each answer costs in time and money (Section 2).

I compare the two on ABCD (Chen et al., 2021), a public corpus of 10,042 role-played customer-service chats for an online clothing store. The input is the customer's opening, which is what a triage step sees. One system is a set of logistic-regression heads over shared TF-IDF features, trained on ABCD's 8,034 training chats. The other is Claude Haiku 5.5, given one prompt per chat that holds all 15 questions and their allowed answers, with no examples. Topics and intents appear by name only, and the other 13 questions carry one-line definitions. Both answer the same 15 questions on the same 300 test chats, and I compare them with paired bootstrap intervals and corrected tests.

The heads are more accurate on this data, and the gap shrinks but holds under every check I ran. On the two questions whose labels ABCD supplies, they lead by about 10 points each. Over all 15 they lead by 9.2 points on the mean, on 6 questions after Holm correction, and by 5.2 to 10.4 points under three scoring rules. They draw level with Claude at 383 labeled chats and lead clearly from 1,000. They answer in 3.98 ms against 18.61 s, with no per-call charge. Claude's time includes thinking; the help text of its command-line interface lists no option to turn it off. Two findings bound the result. On the 10 questions whose truth I derived from the intent, deriving both systems' answers from their own intent answers shrinks the gap from 7.3 points to 2.6–3.4. And the heads needed in-domain labels here: trained on 383 tickets from one synthetic recipe, they lose to Claude.

The contributions are five:

1. **A paired comparison against a constant baseline.** The heads lead by 9.2 points (6.3 to 12.1) on the mean over 15 questions, and on 6 questions after Holm correction. Claude leads on none. On ABCD's own two labels the heads lead by 10.1 and 10.2 points, so the result does not rest on my mapping alone. Three scores fail to clear the constant answer, two of them Claude's (Section 4.1).
2. **Checks on the gap.** Changing the scoring rule, dropping weak questions and removing near-duplicate openings keep the mean gap between 5.2 and 10.4 points. Heads trained without class weights still lead by 7.1 to 9.8 points. Setting aside Claude's unusable answers, the heads still lead on all 5 choice questions (Sections 4.2 to 4.4).
3. **Where the gap comes from.** Deriving answers from each system's own intent cuts the gap on 10 intent-derived questions from 7.3 to 2.6 points, or 3.4 under a stricter rule. Most of the rest comes from Claude's weaker intent answers, and its confusions show which definitions it read differently (Section 5.1, Appendix E).
4. **A learning curve with paired intervals.** Retrained on random subsets, the heads draw level with Claude at 383 labeled chats, where 1 of 5 runs has an interval above zero. From 1,000 chats every run leads, each with an interval above zero (Section 5.2).
5. **Synthetic data, speed, cost and a decision rule.** Trained on 383 Claude-written tickets, the same heads score 58.5% on 13 shared questions, against 84.8% on 383 ABCD chats (Section 5.3). The heads answer about 4,700 times faster than Claude run with thinking on through its command-line interface, with no per-call charge (Section 5.4). Section 5.6 turns these results into a rule for choosing between the two.

Appendix B lists the script behind each result.

## 2 Related Work

**Fine-tuned small models against prompted LLMs.** Broad benchmarks find that prompted LLMs trail fine-tuned models on most classification tasks. Ziems et al. (2024) test 13 zero-shot models on 25 computational social science benchmarks and find that they do not outperform the best fine-tuned models. Kocoń et al. (2023) put ChatGPT's quality about 25% below the task-specific state of the art, on average over 25 tasks. Edwards and Camacho-Collados (2024) reach a similar conclusion over 16 datasets. Vajjala and Shimangaud (2025) do so over 32 datasets in 8 languages, where zero-shot models compete only on sentiment. Liu et al. (2022) find that few-shot parameter-efficient fine-tuning beats few-shot in-context learning on accuracy, at far lower computational cost. Comparisons with Claude point the same way. Bucher and Martini (2024) report stance accuracy of 0.94 for a fine-tuned encoder against 0.61 for Claude 3 Opus. Dominguez-Olmedo et al. (2025) fine-tune models on 260 legal annotation tasks. Their 135M-parameter model averages 82% against 78% for Claude 3.7 Sonnet, and their 70B model 88%. They report that a few hundred to a thousand labeled examples usually suffice for higher accuracy. These studies score against labels native to each benchmark. Here 13 of the 15 truths come from my mapping, so I measure how much of the gap that mapping explains (Section 5.1).

**Intent detection and customer service.** Two studies sit closest to triage. Loukas et al. (2023) classify the 77 intents of Banking77 and price each LLM setting in dollars. SetFit (Tunstall et al., 2022) with 10 labeled examples per class reaches 88.1 micro-F1, above the best LLM setting at 87.7, and full-data fine-tuning reaches 94.1. Arora et al. (2024) compare Claude models and others with SetFit on 6 intent datasets. Averaged over the 6 datasets, Claude 3 Haiku reaches an F1 of 0.736 against 0.600 for SetFit. A router that sends only uncertain queries to the LLM stays within 2% of the LLM's accuracy with about 50% less latency. Their LLMs saw retrieved examples and reasoned step by step, while Claude here saw no examples. Chen et al. (2021) built ABCD to test whether agents follow a company's workflows. I use only its intent labels and the customer's opening, and ask the questions a triage step needs.

**Cost and latency.** Several recent studies price or time both systems. Lee et al. (2026) classify Reddit replies to misinformation. A fine-tuned RoBERTa reaches 0.62 macro-F1 against 0.50 for Claude Haiku 4.5, at 0.007 s per item against 10.48 s. Valdes Gonzalez (2026) weighs accuracy against cost on four benchmarks. A fine-tuned RoBERTa leads Claude Sonnet 4.5 on AG News, 94.63 to 91.35, while few-shot Claude leads on IMDB, 96.48 to 94.84. Tran et al. (2025) train a 1.5B-parameter router that scores 93.17 against 92.79 for Claude 3.7 Sonnet, at 51 ms against 1,450 ms. I add a learning curve beside the same measures, so a team can set the labels it needs against Claude's measured price and time per chat (Section 5.2).

**Cheaper paths through an LLM.** Other work uses a large model to build a small one, or to spend less on the large one. Hsieh et al. (2023) train a 770M-parameter T5 on labels and rationales from a larger model, and beat few-shot PaLM 540B on ANLI with 80% of its training data. Chen et al. (2023) learn a cascade over LLM APIs that matches GPT-4 at up to 98% lower cost. Both keep real inputs. Li et al. (2023) instead train classifiers on text an LLM wrote, and find that such data helps less as the task grows more subjective. In Section 5.3 an LLM likewise writes the inputs from labels sampled in code, and I score heads trained on them against human-written chats.

**Narrow small models.** A further line builds many narrow models in place of one general model. Lee et al. (2021) propose micromodels, small models that each detect one behavior, combined into systems that are explainable and reusable. Buckmann and Hill (2024) fit penalized logistic regression on the embeddings of Llama 2 7B. Across 17 sentence classification tasks, given tens of labeled examples, it performs on par with GPT-4 and sometimes better. The heads here are simpler still: one logistic regression per question over shared TF-IDF features, with no neural network.

## 3 Setup

### 3.1 The data are 300 test openings from ABCD

ABCD (Chen et al., 2021) holds 10,042 chats in which one person plays a customer from a scripted scenario and another plays an agent of an online clothing store. They are split into 8,034 training, 1,004 development and 1,004 test chats. I use release 1.1 at commit `6b8700c`, under its MIT license. Each chat carries a topic, which ABCD calls a flow (10 values), a finer intent, and the actions the agent took, such as offering a refund or updating an order.

ABCD's raw intents are 96 subflows, 48 of which name single FAQ entries such as `boots_how_1`. I collapse each FAQ subflow to its leading word (`boots`) and merge `status_delivery_date`, which has 3 chats, into `status_delivery_time`. That leaves 55 intents. The heads learn these 55, and Claude's prompt lists them.

The input to both systems is the customer's opening: the messages they send until the agent replies to them. The agent opens 86.8% of chats, usually with a greeting, so most openings answer it. Openings are short, with a median of 16 words (p10 6, p90 28), against 49 words for everything a customer says in a chat. I drew 300 test chats with Python's `random.Random(0).sample`, and both systems answered the same 300. They hold 53 of the 55 intents.

### 3.2 Thirteen of the 15 questions have truths I mapped from ABCD's labels

Two questions use ABCD's own labels: the topic (`flow`, 10 classes) and the intent (55 classes). The other 13 come from the question bank of an earlier triage system that I built for synthetic tickets (Section 5.3). Three of them are choice questions, `primary_team`, `journey_stage` and `message_type`, which I call the bank choice questions. With `flow` and `intent` they make the 5 choice questions. The other 10 are yes/no questions, such as `refund_implied` and `needs_staff_action`. Appendix A gives each with the definition Claude saw.

ABCD does not label these 13, so I derived their truth from the chat's intent, the agent's actions and the details the customer typed (Appendix C). For example, `refund_implied` is yes for returns, refund requests, price complaints and bills for items the customer returned or did not buy. It is no for intents where no money is owed back, such as a password reset or a bill payment. A chat whose intent cannot settle a question has no truth for it and drops out of that question's score. That is why the chats per question run from 145 to 300.

Four facts about these truths shape the results:

- `cannot_log_in` and `security_related` have the same truth on all 300 chats. They are one test counted twice, and the 15 questions need only 14 distinct heads.
- `needs_staff_action` is yes when the agent later took one of 8 actions, such as offering a refund, updating an order or notifying a team. Two of them, sending a link and applying a promo code, inform or discount rather than change an account or order. The truth comes from the rest of the chat, which the opening can only predict.
- `charge_amount_given` has 2 positive chats among the 300, so changing the answer on one of them moves its balanced score by 25 points.
- I fixed the mapping and the scoring rules in an earlier analysis that scored the heads on this same test split, and the development split went unused. The mapping file was last edited before Claude's answers existed, so no rule was set after seeing them.

### 3.3 The two systems saw the same opening and different help

**The heads.** One feature step turns an opening into 38,990 TF-IDF features, all with sublinear term frequency. They are word 1- and 2-grams, plus character 2- to 5-grams within word boundaries that occur in at least 2 training openings. Each question gets its own logistic regression (scikit-learn 1.7.2; Pedregosa et al., 2011), with C = 4.0, balanced class weights and up to 3,000 iterations. It trains on the openings of the training chats that have a truth for that question, and returns its most likely class. Balanced class weights push each head toward rare classes, which the balanced accuracy of Section 3.4 rewards, so Section 4.3 also scores heads trained without them. These settings come unchanged from the synthetic-ticket system, and I tuned none of them on ABCD. I kept TF-IDF and logistic regression because they train in seconds on a laptop with no GPU. A fine-tuned encoder or SetFit might score higher; I did not test one.

**Claude.** I called Claude Haiku 5.5 through the Claude Code command-line interface (CLI) 2.1.310 on 2026-10-26. I used the CLI, not the API, because it was the access I had. I chose the Haiku tier because it is Claude's lowest-priced tier, the natural default for high-volume triage. Twelve calls ran at once, with tools, saved settings and session storage off. I wanted thinking off, because a triage step needs a fast label. The CLI's help text lists no option for that, so I set effort to low, its lowest setting, to keep thinking short; the model's default is medium. The CLI's binary reads a `MAX_THINKING_TOKENS` setting, which I did not set or test, so the model still thought before answering. The run's logs do not name the model. A separate call through the same `haiku` alias that day reported `claude-haiku-5-5`.

The prompt holds all 15 questions with their allowed answers (Appendix A). It names the 10 topics and the 55 intents, grouped by topic, with no definitions. It gives a one-line definition for each answer to the bank choice questions and for each yes/no question, and no examples. The answer sets came unchanged from the earlier system's question bank, so no ABCD truth can take four of the answers it offers. They are the product-management team, the "unclear" journey stage, and the "feedback" and "other" message types. Claude chose them 53 times on chats with a truth (Section 4.2). A reply with no JSON object that parsed was retried up to 2 times, and only the last attempt's time and cost were logged. All 300 chats ended with a reply that parsed.

The comparison is unequal by design. It sets what a team can run before labeling anything, a prompt with label names, against what labeled history buys. Section 5.2 measures how much history that takes. Few-shot or retrieved examples, descriptions of each intent, a prompt without the four impossible answers, a larger model or thinking turned off could each narrow the gap. I ran none of them.

### 3.4 Scoring counts an unusable answer as wrong

Any answer the truth cannot take counts as wrong. That covers an empty answer and a malformed one. It also covers an answer the prompt offered but no sampled chat's truth holds, such as a journey stage of "unclear" or the intent `status_questions`. Section 4.2 rescores without them.

For `flow`, `intent` and the 10 yes/no questions I report balanced accuracy, the mean recall over the true classes, because their classes are imbalanced. `mentions_bug`, for example, is yes in 18 of 267 chats. Balanced accuracy weighs all the yes cases together as heavily as all the no cases. On `mentions_bug`, one missed bug therefore costs as much as 13.8 false alarms (249 ÷ 18). Across the yes/no questions that ratio runs from 1.0 to 149.0, so the rule prices a miss differently on each question (Section 4.3). For the 3 bank choice questions I report plain accuracy, as the synthetic system's evaluation did. The headline is the mean over the 15 questions, and Section 4.3 rescores it with one rule throughout.

A constant answer is the baseline each system must beat. Under balanced accuracy it scores 1 divided by the number of classes: 50.0% on a yes/no question, 10.0% on `flow` and 1.9% on the 53 intents in the sample. On the bank choice questions the most common answer scores 52.1%, 69.0% and 40.7%.

Intervals are 95% paired-bootstrap intervals over 10,000 resamples of chats (Berg-Kirkpatrick et al., 2012). The mean's interval resamples chats jointly across the 15 questions. Differences are heads minus Claude, in points, computed before rounding. A few therefore differ by a tenth of a point from the gap between the rounded scores. Each question's p-value comes from a two-sided paired approximate-randomization test (Dror et al., 2018) on the reported metric. It swaps the two systems' answers on random chats 10,000 times, and p is (count + 1) / 10,001. Holm's method (Holm, 1979) corrects the 15 p-values, so the smallest possible corrected p is 0.0015.

I also report McNemar's exact test (McNemar, 1947) on the chats where only one system is right. It tests plain accuracy, so it can disagree with the permutation test. On `mentions_bug` the permutation p is 0.9999, while McNemar gives 0.0386, with 10 chats where only Claude is right against 2. The permutation p is near 1 because each of the 18 yes chats moves balanced accuracy by 2.8 points, more than the 1.0-point gap. Almost every random swap therefore yields a gap at least as large. McNemar's test counts chats instead, and all 10 chats where only Claude is right are no cases that the heads flagged. I designed the checks in Sections 4.2 to 5.5 after seeing the main comparison, some in response to an internal review of an earlier draft. I report each one whatever its result.

## 4 Results

### 4.1 The heads lead by about 10 points on ABCD's own labels and by 9.2 on the mean

The heads score a mean 90.7% (95% CI 88.3 to 92.8) over the 15 questions and Claude 81.5% (78.2 to 84.7), a difference of 9.2 points (6.3 to 12.1). The two questions whose truth is ABCD's own carry the most weight, because my mapping does not shape them. On them the heads lead by 10.1 points on `flow` and 10.2 on `intent`, and both leads survive Holm correction. Table 1 gives each question beside its constant-answer baseline.

**Table 1:** The heads lead on 6 of 15 questions after Holm correction, and Claude leads on none. Scores are percentages with 95% paired-bootstrap intervals. The baseline is a constant answer scored by the question's metric. Differences are heads minus Claude, in points, computed before rounding. p is a paired permutation test, Holm-corrected over 15 questions.

| Question | n | Metric | Baseline | Heads | Claude | Difference (points) | Holm p |
|---|---|---|---|---|---|---|---|
| flow | 300 | balanced | 10.0 | 85.9 (81.8–89.9) | 75.8 (70.8–80.6) | +10.1 (+6.2 to +14.2) | 0.0015 |
| intent | 300 | balanced | 1.9 | 76.9 (70.8–81.5) | 66.7 (60.2–70.8) | +10.2 (+5.9 to +15.3) | 0.0024 |
| primary_team | 267 | accuracy | 52.1 | 88.4 (84.3–92.1) | 83.1 (78.6–87.6) | +5.2 (+0.4 to +10.1) | 0.38 |
| journey_stage | 145 | accuracy | 69.0 | 93.1 (89.0–96.6) | 72.4 (64.8–80.0) | +20.7 (+13.8 to +27.6) | 0.0015 |
| message_type | 285 | accuracy | 40.7 | 88.1 (84.2–91.6) | 67.4 (61.8–73.0) | +20.7 (+15.4 to +26.0) | 0.0015 |
| cannot_log_in | 267 | balanced | 50.0 | 92.3 (86.4–97.3) | 88.6 (81.2–95.3) | +3.8 (−1.1 to +9.7) | 1.0 |
| security_related | 267 | balanced | 50.0 | 92.3 (86.4–97.2) | 91.8 (85.4–97.3) | +0.6 (−2.1 to +4.2) | 1.0 |
| mentions_account_issue | 267 | balanced | 50.0 | 91.4 (85.5–96.4) | 90.1 (83.6–95.9) | +1.2 (−2.2 to +5.4) | 1.0 |
| mentions_billing_issue | 223 | balanced | 50.0 | 93.4 (88.1–97.9) | 85.3 (78.2–91.6) | +8.1 (+3.0 to +13.9) | 0.054 |
| mentions_bug | 267 | balanced | 50.0 | 98.0 (96.7–99.2) | 97.0 (90.5–100) | +1.0 (−2.8 to +7.6) | 1.0 |
| refund_requested | 186 | balanced | 50.0 | 98.9 (97.8–99.7) | 100.0 (100–100) | −1.1 (−2.2 to −0.3) | 0.8742 |
| refund_implied | 221 | balanced | 50.0 | 90.2 (84.1–95.4) | 77.8 (70.0–85.3) | +12.4 (+5.3 to +19.8) | 0.022 |
| needs_staff_action | 300 | balanced | 50.0 | 73.3 (68.4–78.1) | 58.0 (52.3–63.6) | +15.3 (+8.3 to +22.4) | 0.0024 |
| identity_details_given | 300 | balanced | 50.0 | 98.2 (94.9–100) | 92.9 (86.7–98.2) | +5.4 (−0.8 to +12.1) | 1.0 |
| charge_amount_given | 300 | balanced | 50.0 | 99.5 (98.3–100) | 75.0 (50.0–100) | +24.5 (−1.7 to +49.8) | 1.0 |
| **Mean of 15** | | | | **90.7** (88.3–92.8) | 81.5 (78.2–84.7) | **+9.2** (+6.3 to +12.1) | |

Six questions survive correction, and all favor the heads: `flow` and `intent` by about 10 points, `journey_stage` and `message_type` by 20.7 each, `refund_implied` by 12.4 and `needs_staff_action` by 15.3. Four of the six rest on my mapping, so Section 5.1 tests how much of their gap it explains. `mentions_billing_issue` falls just short, at a corrected p of 0.054.

Claude's one lead is `refund_requested`, where it is right on all 186 chats and the heads miss 4. The bootstrap interval for that difference excludes zero. With only 4 discordant chats, though, McNemar's exact test gives p = 0.125, and the corrected permutation p is 0.8742.

Three scores fail to clear the constant answer, two of them in Table 1. Claude's `journey_stage` score of 72.4% has an interval that reaches the 69.0% baseline. On `charge_amount_given`, Claude's interval reaches 50.0%. The third is a plain accuracy that Table 1 omits: the heads' 99.0% on `charge_amount_given` sits below the 99.3% that a constant "no" would score. Two positive chats decide that question, so its 24.5-point gap, with an interval from −1.7 to +49.8 points, supports no conclusion.

### 4.2 Claude's unusable answers explain about half the journey-stage gap and little of the rest

Claude gave answers the truth cannot take on 4 of the 5 choice questions, and they are of two kinds. Some are empty: no topic for 25 chats and no intent for 30. Others are out of set, answers the prompt offered but no sampled chat's truth holds. They are 12 intents naming `status_questions`, 27 journey stages of "unclear", and 25 message types of "other" plus 1 of "feedback". No ABCD truth can take the last three, because the mapping does not yield them (Section 3.3). The heads gave 7 out-of-set intents, `manage_payment_method` 5 times and `status_questions` twice. Table 2 rescores each choice question on only the chats where Claude gave an answer the truth can take.

**Table 2:** On chats where Claude gave a usable answer, the heads still lead on all 5 choice questions, by 5.2 to 18.5 points. Plain accuracy in percent. McNemar's exact test on the discordant chats, before correction. The last two columns describe the chats whose Claude answer was set aside.

| Question | Claude usable | Heads | Claude | Gap (points) | Only heads right | Only Claude right | McNemar p | Heads right where set aside | Median opening words, usable vs set aside |
|---|---|---|---|---|---|---|---|---|---|
| flow | 275 of 300 | 93.1 | 83.6 | 9.5 | 31 | 5 | < 0.001 | 8.0 | 16 vs 2 |
| intent | 258 of 300 | 85.7 | 76.4 | 9.3 | 36 | 12 | 0.0007 | 16.7 | 16.5 vs 4 |
| primary_team | 267 of 267 | 88.4 | 83.1 | 5.2 | 29 | 15 | 0.0488 | n/a | n/a |
| journey_stage | 118 of 145 | 98.3 | 89.0 | 9.3 | 12 | 1 | 0.0034 | 70.4 | 18 vs 9 |
| message_type | 259 of 285 | 92.7 | 74.1 | 18.5 | 53 | 5 | < 0.001 | 42.3 | 16 vs 2 |

Compared plain against plain, setting these answers aside changes little except on `journey_stage`. On all 300 chats the plain-accuracy gap is 9.3 points on `flow` and 10.3 on `intent`; on usable chats it is 9.5 and 9.3. `message_type` narrows from 20.7 to 18.5 points. `journey_stage` narrows from 20.7 to 9.3, so about half its gap comes from the 27 "unclear" answers. Those openings are not near-empty, with a median of 9 words, and the heads are right on 70.4% of them.

The empty topic answers fall on near-empty openings, with a median of 2 words, and the heads are right on only 8.0% of those chats. Declining to guess on such openings may serve a team better than a wrong guess. These data cannot test that, because the scoring marks both as wrong and prices neither.

### 4.3 The scoring rule sets the gap's size, from 5.2 to 10.4 points

Table 3 rescores the mean with one rule for all 15 questions. The heads lead under every rule, by 5.2 points under plain accuracy and by 10.4 under balanced accuracy. All three rows come from the same 10,000 bootstrap resamples.

**Table 3:** The heads lead on the mean under all three scoring rules, and plain accuracy shrinks the gap to 5.2 points. Mean over 15 questions in percent, with 95% intervals from one joint bootstrap over chats; differences in points.

| Scoring rule | Heads | Claude | Difference (points) |
|---|---|---|---|
| As reported: plain accuracy on the 3 bank choice questions, balanced on the rest | 90.7 (88.3–92.8) | 81.5 (78.2–84.7) | +9.2 (+6.3 to +12.1) |
| Plain accuracy on all 15 | 91.4 (89.6–93.2) | 86.2 (84.6–87.8) | +5.2 (+3.9 to +6.6) |
| Balanced accuracy on all 15 | 91.0 (88.7–93.0) | 80.5 (77.2–83.9) | +10.4 (+7.5 to +13.3) |

The shrink under plain accuracy comes from the rare yes cases. On the 9 yes/no questions whose yes is the minority, the heads find 230 of 255 yes cases (90.2%) and Claude 199 (78.0%). Claude keeps 2,024 of 2,043 no cases (99.1%) and the heads 1,984 (97.1%). Plain accuracy counts each chat once, so the no cases dominate it. Claude's plain accuracy is higher on 6 of the 10 yes/no questions, by 0.7 to 3.0 points. One question carries 1.68 of the 4.0-point shrink, or 42.4%: `charge_amount_given`, where Claude finds 1 of the 2 positive chats.

The shrink is not a trade on every question. On `needs_staff_action`, where yes and no are equal in number, the heads lead on both classes. They find 122 of the 150 yes cases against Claude's 82, and keep 98 of the 150 no cases against 92. Under a Holm-corrected McNemar test, which scores plain accuracy, the heads are right on more chats for 5 questions and Claude for none. The 5 are `flow`, `intent`, `journey_stage`, `message_type` and `needs_staff_action`.

Which rule fits depends on what each mistake costs a team, and no rule here prices them. Balanced accuracy implies a different price on every question. One missed yes costs as much as 1.0 false alarm on `needs_staff_action`, 13.8 on `mentions_bug` and 149.0 on `charge_amount_given`. Moving each head's decision threshold would trade one error for the other; I did not test it.

The heads' balanced class weights explain part of this trade, because they favor rare classes (Section 3.3). Retrained without class weights, the heads still lead under every rule. They lead by 9.1 points (6.1 to 12.1) as reported, 7.1 (5.9 to 8.2) under plain accuracy and 9.8 (6.7 to 12.7) under balanced accuracy. Their plain accuracy then trails Claude's on 1 of the 10 yes/no questions, against 6 of 10 with the weights.

### 4.4 The gap holds when weak questions or near-duplicate chats are removed

Two further checks ask whether the headline depends on a few questions or on test openings that repeat training ones. Table 4 lists them, and a third check widens the heads' sample to the full test split.

**Table 4:** The mean gap stays between 7.9 and 10.2 points when questions or chats are removed. Means over the remaining questions in percent, scored as in Table 1; gaps in points.

| Check | Questions | Chats | Heads | Claude | Gap |
|---|---|---|---|---|---|
| None (Table 1) | 15 | 300 | 90.7 | 81.5 | 9.2 |
| Without `charge_amount_given` (2 positive chats) | 14 | 300 | 90.0 | 81.9 | 8.1 |
| Without `needs_staff_action` (truth from later actions) | 14 | 300 | 91.9 | 83.1 | 8.8 |
| Without `cannot_log_in` (same truth as `security_related`) | 14 | 300 | 90.5 | 80.9 | 9.6 |
| Without all three | 12 | 300 | 91.2 | 83.4 | 7.9 |
| ABCD-native questions only | 2 | 300 | 81.4 | 71.2 | 10.2 |
| Mapped questions only | 13 | 300 | 92.1 | 83.0 | 9.1 |
| Without near-duplicate openings | 15 | 263 | 93.7 | 84.0 | 9.7 |

**Near-duplicates.** ABCD's customers follow scripted scenarios, so a test opening could repeat a training one and flatter the heads. Of the 300 test openings, 28 match a training opening exactly after lowercasing and stripping punctuation, with a median of 2 words. Another check uses character n-gram TF-IDF fitted on the training openings: 37 test openings have a training opening at a cosine similarity of 0.9 or more. The median test opening's closest training opening sits at 0.659. Removing those 37 leaves the gap at 9.7 points.

**The full test split.** The heads score a mean 89.8% on all 1,004 test openings, against 90.7% on the 300. The largest moves are on rare-yes questions: `charge_amount_given` falls 10.8 points, `refund_requested` 7.7 and `mentions_bug` 4.8. I did not run Claude on the other 704 openings, which would cost $1.94 at its mean cost per chat.

## 5 Analysis

### 5.1 Derived from each system's own intent answer, the gap on 10 questions falls from 7.3 points to 2.6–3.4

Thirteen truths come from my mapping, and Claude saw only one-line definitions of them. The two ABCD-native questions show that the gap is not only a mapping artifact. The heads lead on them by 10.2 points, against 9.1 on the 13 mapped questions (Table 4). They do not show how much of each mapped gap the mapping explains.

To measure that, I derived each system's answers from its own intent answer. Ten questions take their truth from the intent alone: the 3 bank choice questions and the 7 yes/no questions other than `needs_staff_action`, `identity_details_given` and `charge_amount_given`. A routed answer is the one that the system's intent answer maps to. Claude names an intent the mapping knows on 270 of the 300 chats. On 12 of them it names `status_questions`, which no sampled chat holds, so Table 2 counts 258. Two cases need a rule. When the intent answer is empty or unknown, the direct answer stands, on 260 of Claude's graded chat-question pairs and none of the heads'. A known intent can also settle nothing for a question, as six intents do for `primary_team`. There the fallback rule keeps the direct answer and the strict rule counts it wrong, on 62 of Claude's pairs and 66 of the heads'. Table 5 sets each system's direct answers beside its routed ones.

**Table 5:** Routed through the mapping, Claude's own intent answers lift its mean on the 10 intent-derived questions from 85.3% to 91.0%, or to 88.5% under the strict rule. Percentages, scored as in Table 1. Direct is each system's own answer; routed is the answer its own intent maps to. The rows for single questions use the fallback rule.

| Question | Heads, direct | Heads, routed | Claude, direct | Claude, routed |
|---|---|---|---|---|
| primary_team | 88.4 | 91.0 | 83.1 | 87.3 |
| journey_stage | 93.1 | 93.8 | 72.4 | 86.2 |
| message_type | 88.1 | 89.5 | 67.4 | 80.7 |
| cannot_log_in | 92.3 | 92.9 | 88.6 | 92.6 |
| security_related | 92.3 | 92.9 | 91.8 | 92.6 |
| mentions_account_issue | 91.4 | 92.3 | 90.1 | 90.8 |
| mentions_billing_issue | 93.4 | 94.0 | 85.3 | 91.6 |
| mentions_bug | 98.0 | 98.6 | 97.0 | 97.2 |
| refund_requested | 98.9 | 99.7 | 100.0 | 100.0 |
| refund_implied | 90.2 | 91.3 | 77.8 | 90.9 |
| **Mean of 10, fallback rule** | **92.6** | **93.6** | **85.3** | **91.0** |
| **Mean of 10, strict rule** | **92.6** | **91.9** | **85.3** | **88.5** |

With both systems routed, the gap on these 10 questions falls from 7.3 points to 2.6 under the fallback rule and 3.4 under the strict one. Routing thus closes 64% or 53% of it. Most of what remains comes from Claude's weaker intent answers. It names the exact intent on 65.7% of chats, against 76.0% for the heads, though many of its intent errors still map to the right answer. The part routing closes came from Claude's direct answers, which follow its own reading of the one-line definitions. The confusions show where that reading parts from the mapping (Table 9 in Appendix E):

- **`primary_team`:** 20 of Claude's 45 errors send to billing a chat the mapping gives to "other", led by returns and store-policy questions. The prompt defines the billing team as handling "Charges, refunds, invoices and payment methods".
- **`journey_stage`:** 26 of Claude's 40 errors answer "unclear" where the truth is post-purchase (16) or browsing (10). Another 7 call shopping-cart chats browsing, a stage the prompt defines as "Has not bought anything yet".
- **`message_type`:** 21 of Claude's 93 errors call a request a problem report, and 13 of those are login-recovery chats.
- **`refund_implied`:** 18 of Claude's 19 errors miss a yes, all on price complaints or returns. Whether those customers want money back is a judgment call.
- **`needs_staff_action`:** Claude misses 68 yes cases, led by password-recovery and out-of-stock chats (10 each). Their truth follows actions the agent took later.

A person could defend most of these readings of the definitions. A second annotator labeling 100 openings against the prompt's definitions would measure how often a careful reader agrees with the mapping; I did not run one. The routed rule also suggests a design a team could adopt: ask the LLM for the intent and derive the other answers from it in code. On these 10 questions that design scores 91.0%, or 88.5% under the strict rule, against 92.6% for the heads' direct answers.

### 5.2 The heads draw level with Claude at 383 labeled chats and lead clearly from 1,000

The comparison so far uses all 8,034 training chats, which a new team would have to label. To measure how many it needs, I retrained the heads on nested random subsets of the training chats. For each of 5 seeds, `random.Random(seed).sample` fixes one order of the training chats, and each size takes the leading chats of that order. Sizes run from 50 to 4,000, with one run on all 8,034. Each run scores the same 300 test chats with the settings of Section 3.3. The full-size run reproduces the heads' answers with 0 mismatches, and all 36 trainings took 38.5 s on 10 cores. I included 383 as a size because the synthetic system trained on 383 tickets (Section 5.3), so the two compare at equal size.

**Table 6:** The heads draw level with Claude at 383 labeled chats and lead in every run from 1,000, each with an interval above zero. Mean of 15 questions in percent; ± is the standard deviation over 5 runs, and the range gives the lowest and highest run. "13 shared" is the mean of the 13 questions the synthetic system shares. The last two columns give each run's lead over Claude's reported mean in points, and how many of the runs' 95% paired intervals lie above zero.

| Training chats | As reported | Range of runs | Plain | Balanced | 13 shared | Lead over Claude, runs | Intervals above zero |
|---|---|---|---|---|---|---|---|
| 50 | 62.0 ± 1.6 | 60.0–64.1 | 78.9 | 60.1 | 65.5 | −21.5 to −17.4 | 0 of 5 |
| 100 | 69.8 ± 2.0 | 67.9–72.1 | 83.0 | 68.4 | 72.7 | −13.5 to −9.4 | 0 of 5 |
| 200 | 78.0 ± 1.8 | 76.3–80.4 | 87.1 | 77.5 | 80.2 | −5.1 to −1.1 | 0 of 5 |
| 383 | 83.1 ± 1.0 | 81.3–83.9 | 89.6 | 83.0 | 84.8 | −0.1 to +2.4 | 1 of 5 |
| 1,000 | 88.1 ± 0.7 | 87.5–89.2 | 91.1 | 88.2 | 89.6 | +6.0 to +7.8 | 5 of 5 |
| 2,000 | 89.8 ± 0.6 | 88.8–90.3 | 91.3 | 89.9 | 91.4 | +7.3 to +8.9 | 5 of 5 |
| 4,000 | 90.3 ± 0.7 | 89.3–90.9 | 91.9 | 90.6 | 91.8 | +7.9 to +9.5 | 5 of 5 |
| 8,034 | 90.7 (1 run) | | 91.4 | 91.0 | 92.1 | +9.2 | 1 of 1 |
| Claude, zero-shot | 81.5 | | 86.2 | 80.5 | 83.0 | | |

Run ranges alone do not show whether a lead is beyond noise, so I bootstrapped each run's lead over Claude with the paired resamples of Section 3.4. At 383 chats 4 of 5 runs beat Claude's reported mean, but by at most 2.4 points, and only 1 of the 5 intervals lies above zero. The heads draw level there rather than pass. At 200 chats 3 of the 5 intervals lie below zero. At 1,000 chats every run leads by 6.0 to 7.8 points, with interval lower bounds of 1.3 to 4.5, and every interval at the larger sizes also lies above zero. Under plain accuracy the heads pass sooner: at 200 chats 2 of 5 intervals lie above zero, and at 383 all 5 do, with lower bounds of 1.7 to 2.6. The curve flattens past 2,000 chats: going from 2,000 to all 8,034 adds 0.9 points, and from 1,000 adds 2.6. I did not measure labeling time, so the curve gives the number of chats a team needs, not the hours.

### 5.3 Trained on Claude-written tickets, the same heads lose to Claude

Before ABCD, I built a triage system on synthetic data. A Claude model called through the same `haiku` alias, two days before the comparison, wrote 480 support tickets from labels sampled in code. That took $6.79 and about 79 minutes. I kept 479, trained on 383 and held out 96. The system is a pipeline that runs TF-IDF and logistic-regression models inside hand-written rules. A regular-expression splitter cuts a ticket into pieces, models read each piece, and rules combine their outputs into 47 answers. On its own held-out tickets it scores 97.3% on the 13 questions it shares with this paper, and 92.7% per ticket on `primary_team`.

On ABCD's openings that system scores 60.3% on the 13 shared questions, below Claude on 12 of them (Table 8 in Appendix D). Its team answer falls from 92.7% on synthetic tickets to 66.3% on ABCD openings, against 83.1% for Claude. That result mixes the training data with the pipeline's design, so I separated them.

I trained this paper's heads, with the same features and settings, on the same 383 synthetic tickets and scored them on the 300 openings. They score 58.5% on the 13 shared questions, below Claude on all 13. The same heads trained on 383 ABCD chats score 84.8% (Table 6), and Claude 83.0%. On the 96 held-out synthetic tickets the synthetic-trained heads score 90.5%, so they learned the synthetic task. With the design held fixed, the data account for 26.2 points. The pipeline adds 1.8 points over the plain heads, as the net of two parts. Its regular-expression answers add 4.0 points, and its learned answers trail the plain heads by 2.2.

Part of the loss is labels that the synthetic tickets do not use. Their journey stages hold no browsing or checkout chats, which are 45 of the 145 ABCD chats with a journey truth. Their message types hold no questions, which are 110 of 285. On the 11 shared questions whose ABCD answers all appear in the synthetic labels, the synthetic-trained heads still trail the ABCD-trained ones by 19.9 points. The text differs too. Openings run 16 words at the median against 55 for a synthetic ticket, and 12.5% of opening words are absent from the synthetic tickets.

This is one synthetic recipe: one generator, one prompt, labels sampled in code and 383 training tickets. Other recipes, such as rewriting real openings or having an LLM label real chats, may do better; I did not test them.

### 5.4 The heads answer about 4,700 times faster than Claude with thinking on through its CLI, with no per-call charge

Table 7 sets the two systems' speed and cost side by side. The heads ran in one process on an Apple M1 Max laptop with Python 3.13.9. Claude's times are the wall time of each CLI call.

**Table 7:** The heads answer all 15 questions in 3.98 ms at the median; Claude, with thinking on, takes 18.61 s at the median and $0.0028 per chat on average. Heads: one process on an Apple M1 Max. Claude: Haiku 5.5 through the Claude Code CLI, effort low, the 300 logged calls.

| Measure | Heads | Claude Haiku 5.5 |
|---|---|---|
| Median time per chat | 3.98 ms | 18.61 s |
| p90 time per chat | 4.33 ms | 38.93 s |
| p99 time per chat | 4.85 ms | 82.73 s |
| Cost per chat | no per-call charge | $0.0028 mean, $0.0022 median |
| Cost per 10,000 chats | no per-call charge | $27.62 |
| Training time | 26.5 s for all 15 heads | none |
| Size on disk | 27.4 MB | hosted |
| Output tokens per chat | none | 3,768 at the median |
| Share of output that was thinking | none | 96.4% |

At the median, Claude's wall time is about 4,700 times the heads' latency. Two parts of Claude's time come from how I called it. The CLI adds a median 1.46 s per call (p10 0.81 s, p90 2.55 s). On the API's own time, 16.66 s at the median, the ratio is 4,189. Thinking made up 96.4% of Claude's output tokens. A direct API call with thinking off would be faster and cheaper; I did not measure by how much, or how it would change accuracy.

The heads' costs sit elsewhere. Training all 15 heads took 26.5 s, and the intent head, the one with the most classes, takes 1.94 ms of the median latency. The larger cost is labeling, which Section 5.2 measures in chats. Claude's cost per chat is the cost the CLI reported for each call's last attempt. It matches $0.10 per million input tokens, $0.20 per million tokens written to the one-hour cache and $0.50 per million output tokens.

### 5.5 A team lead routes on the answers, and an agent fixes the misses by hand

Two people use the answers. A team lead writes routing rules on them, such as sending every chat whose team is billing to the billing queue. An agent then opens a chat already marked with its topic, its team and flags such as an implied refund. Neither system changes who reads the chat; both replace the step of reading each opening to pick its queue.

Judged by `primary_team` accuracy, the heads would send 116.1 chats per 1,000 to the wrong team, against 168.5 for Claude. That 5.2-point gap has an interval of 0.4 to 10.1 points and a Holm-corrected p of 0.38, so it suggests a difference rather than shows one. When a chat lands in the wrong queue, the agent moves it by hand, as with no triage at all. On the 9 rare-yes questions the two systems err in opposite directions. Over the 300 chats Claude misses 56 yes cases and raises 19 false flags, while the heads miss 25 and raise 59. Claude's 31 extra misses are flags, such as an implied refund, that the agent does not see. The heads' 40 extra false flags are ones an agent must read and dismiss. Counting both kinds, the heads make 84 errors to Claude's 75.

Two ways to use Claude beside the heads remain untested. A router could send the openings where a head is least confident to Claude, as Arora et al. (2024) do for intent detection. Claude could also label real chats for the heads to learn from. Section 5.3 shows only that tickets Claude wrote from sampled labels fell short of real ones.

### 5.6 Against zero-shot Claude, the number of labeled chats decides which system to deploy

These results give a team a rule with one threshold, against Claude prompted zero-shot as here. With no labeled chats, prompt Claude: it needs no training data, and it clears the constant answer on all but two questions. With about 1,000 labeled chats, train heads. At that size every run led Claude by 6.0 to 7.8 points, each with an interval above zero, and at 383 the two drew level. The same labels could give Claude retrieved examples, which I did not test (Section 3.3). The intervals also leave out Claude's run-to-run variation (Section 6). The labels must cover what the heads answer. ABCD came with topics, intents and the agent's logged actions, and I derived the other 13 truths from those and the details customers typed (Appendix C). A team without such records must label each question, or label intents and derive the rest in code. A new question costs Claude one line of prompt. It costs the heads labels for that question and one new head: on 8,034 chats a yes/no head trained in 0.11 to 0.45 s, on features built in 1.0 s.

Both systems can drift. The heads learn only the traffic they were trained on: trained on synthetic tickets, they scored 90.5% on held-out synthetic tickets and 58.5% on ABCD. A team should therefore rescore its heads on freshly labeled chats as its traffic changes. Claude's answers can change when the model behind an alias changes, so a team should pin a model version and rescore after any change. Between the two sits the untested router of Section 5.5, which would spend Claude's time only on the openings where the heads are least sure.

## 6 Limitations

**One dataset of role-played chats.** ABCD's customers follow a scripted scenario for one fictional store, and their openings run 16 words at the median. Real support traffic may be longer, noisier or raise more than one issue in an opening. I have no measure of how the gap would move there. ABCD has also been public since its release (Chen et al., 2021), so Claude may have seen its chats in training, which would more likely help Claude than hurt it.

**One prompted model, run once, zero-shot, with thinking on.** I tested Claude Haiku 5.5 on one day, with label names, one-line definitions for 13 questions and no examples. Few-shot or retrieved examples, descriptions of each intent, a larger model, or a model fine-tuned on the same chats could narrow or close the gap. The prompt also offered four answers that no ABCD truth can take, and Claude used them 53 times on chats with a truth. The help text listed no way to turn thinking off, and I did not test `MAX_THINKING_TOKENS`, so thinking was on at low effort. That inflates Claude's time and cost, and may change its accuracy in either direction. With thinking on, Claude's answers can vary from run to run. The intervals resample chats, not runs, so they do not cover that variation. Only the last attempt of each call was logged.

**The truth for 13 questions is my mapping, with no second annotator.** The heads learned that mapping from training chats; Claude inferred it from one-line definitions. Section 5.1 shows that routing both systems through the mapping closes 53% to 64% of the gap on the 10 intent-derived questions. Some truths read loosely against their definitions. `status_mystery_fee` counts as a billing problem for the team and billing questions but as a question for `message_type`. `needs_staff_action` counts sending a link as staff action.

**The test split was reused.** I fixed the mapping and scoring rules in an analysis that scored the heads on this test split, and the development split went unused. No setting of the heads was tuned on test chats, and the mapping predates Claude's answers. The rules were still not chosen blind to the test split.

**Mixed scoring rules.** The headline mixes plain accuracy on 3 questions with balanced accuracy on 12. Under plain accuracy throughout, the gap is 5.2 points, and Claude's plain accuracy is higher on 6 of the 10 yes/no questions (Section 4.3). The heads' thresholds were not tuned for either rule. Their balanced class weights favor rare classes, which balanced accuracy rewards; without the weights the heads still lead by 7.1 to 9.8 points.

**Three weak questions.** `cannot_log_in` and `security_related` share a truth, `needs_staff_action` depends on what the agent did later, and `charge_amount_given` has 2 positive chats. Dropping all three leaves a gap of 7.9 points (Table 4).

**One synthetic recipe.** Section 5.3 tests one generator and one prompt at one size, so it shows what that recipe gave, not what synthetic data can give.

**Labeling time not measured.** Section 5.2 counts chats, not hours. Only the topic and intent came with ABCD as labels. The other 13 truths derive from those, the agent's logged actions and the details customers typed, which a team would have to label or log itself.

**A 300-chat sample.** The heads' full-test mean is within a point of the sample's (Section 4.4), but Claude ran on the 300 only.

## 7 Conclusion

On 300 role-played ABCD chats, logistic-regression heads trained on 8,034 labeled chats were more accurate than zero-shot Claude Haiku 5.5 on 15 triage questions. They led by about 10 points on each of ABCD's own two labels and on 6 questions after Holm correction, and Claude led on none. Depending on the scoring rule, the mean gap ran from 5.2 to 10.4 points. On the 10 questions whose truth I derived from the intent, deriving both systems' answers from their own intent answers left a gap of 2.6 to 3.4 points. Most of it came from Claude's weaker intent answers. The heads drew level with Claude at 383 labeled chats and led in every run from 1,000. Trained on 383 Claude-written tickets instead, the same heads scored 58.5% on the 13 shared questions, against 84.8% on 383 ABCD chats. The heads answered in 3.98 ms at the median. Claude, with thinking on through its CLI, took 18.61 s at the median and $0.0028 per chat on average. Whether retrieved examples, intent descriptions or thinking turned off would close the gap remains the open question.

## References

Arora, G., Jain, S., and Merugu, S. 2024. Intent Detection in the Age of LLMs. In *Proceedings of EMNLP 2024: Industry Track*. arXiv:2410.01627.

Berg-Kirkpatrick, T., Burkett, D., and Klein, D. 2012. An Empirical Investigation of Statistical Significance in NLP. In *Proceedings of EMNLP-CoNLL 2012*, pages 995–1005.

Bucher, M. J. J. and Martini, M. 2024. Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification. arXiv:2406.08660.

Buckmann, M. and Hill, E. 2024. Logistic Regression makes small LLMs strong and explainable "tens-of-shot" classifiers. arXiv:2408.03414.

Chen, D., Chen, H., Yang, Y., Lin, A., and Yu, Z. 2021. Action-Based Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented Dialogue Systems. In *Proceedings of NAACL-HLT 2021*, pages 3002–3017. doi:10.18653/v1/2021.naacl-main.239.

Chen, L., Zaharia, M., and Zou, J. 2023. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.

Dominguez-Olmedo, R., Nanda, V., Abebe, R., Bechtold, S., Engel, C., Frankenreiter, J., Gummadi, K., Hardt, M., and Livermore, M. 2025. Lawma: The Power of Specialization for Legal Annotation. In *Proceedings of ICLR 2025*. arXiv:2407.16615.

Dror, R., Baumer, G., Shlomov, S., and Reichart, R. 2018. The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. In *Proceedings of ACL 2018*, pages 1383–1392. doi:10.18653/v1/P18-1128.

Edwards, A. and Camacho-Collados, J. 2024. Language Models for Text Classification: Is In-Context Learning Enough? In *Proceedings of LREC-COLING 2024*.

Holm, S. 1979. A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2):65–70.

Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., and Pfister, T. 2023. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. In *Findings of ACL 2023*. arXiv:2305.02301.

Kocoń, J., Cichecki, I., Kaszyca, O., Kochanek, M., Szydło, D., Baran, J., Bielaniewicz, J., Gruza, M., Janz, A., Kanclerz, K., Kocoń, A., Koptyra, B., Mieleszczenko-Kowszewicz, W., Miłkowski, P., Oleksy, M., Piasecki, M., Radliński, Ł., Wojtasik, K., Woźniak, S., and Kazienko, P. 2023. ChatGPT: Jack of All Trades, Master of None. *Information Fusion*, 101861. doi:10.1016/j.inffus.2023.101861.

Lee, A., Kummerfeld, J. K., An, L. C., and Mihalcea, R. 2021. Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health. In *Findings of EMNLP 2021*. arXiv:2109.13770.

Lee, J., Tian, L., Brillantes, A., Mihăiţă, A.-S., and Rizoiu, M.-A. 2026. Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit. *Social Network Analysis and Mining*. arXiv:2606.04274.

Li, Z., Zhu, H., Lu, Z., and Yin, M. 2023. Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations. In *Proceedings of EMNLP 2023*. arXiv:2310.07849.

Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., and Raffel, C. 2022. Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2205.05638.

Loukas, L., Stogiannidis, I., Diamantopoulos, O., Malakasiotis, P., and Vassos, S. 2023. Making LLMs Worth Every Penny: Resource-Limited Text Classification in Banking. In *Proceedings of ICAIF '23*. doi:10.1145/3604237.3626891.

McNemar, Q. 1947. Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika*, 12(2):153–157.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, É. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12(85):2825–2830.

Tran, C., Paracha, S., Hafeez, A., and Chen, S. 2025. Arch-Router: Aligning LLM Routing with Human Preferences. arXiv:2506.16655.

Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., and Pereg, O. 2022. Efficient Few-Shot Learning Without Prompts. arXiv:2209.11055.

Vajjala, S. and Shimangaud, S. 2025. Text Classification in the LLM Era: Where Do We Stand? arXiv:2502.11830.

Valdes Gonzalez, A. A. 2026. Cost-Aware Model Selection for Text Classification: Multi-Objective Trade-offs Between Fine-Tuned Encoders and LLM Prompting in Production. arXiv:2602.06370.

Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., and Yang, D. 2024. Can Large Language Models Transform Computational Social Science? *Computational Linguistics*, 50(1). doi:10.1162/coli_a_00502.

## Appendix A: The prompt Claude saw

The system prompt was: "You triage support chats for an online clothing store. Reply only with the JSON object asked for, with no other text." The user message for each chat was the text below, with the customer's opening in place of `{opening}`.

```
Answer every key below about the customer's opening message. Reply with one JSON object.

"flow": the topic, one of: "account_access", "manage_account", "order_issue", "product_defect", "purchase_dispute", "shipping_issue", "single_item_query", "storewide_query", "subscription_inquiry", "troubleshoot_site"
"intent": the specific intent, one of these, listed under their topic:
    account_access: "recover_password", "recover_username", "reset_2fa"
    manage_account: "manage_change_address", "manage_change_name", "manage_change_phone", "manage_payment_method", "status_credit_missing", "status_service_added", "status_service_removed", "status_shipping_question"
    order_issue: "manage_cancel", "manage_create", "manage_downgrade", "manage_upgrade", "status_delivery_time", "status_mystery_fee", "status_payment_method", "status_quantity"
    product_defect: "refund_initiate", "refund_status", "refund_update", "return_color", "return_size", "return_stain"
    purchase_dispute: "bad_price_competitor", "bad_price_yesterday", "mistimed_billing_already_returned", "mistimed_billing_never_bought", "out_of_stock_general", "out_of_stock_one_item", "promo_code_invalid", "promo_code_out_of_date"
    shipping_issue: "cost", "manage", "missing", "status"
    single_item_query: "boots", "jacket", "jeans", "shirt"
    storewide_query: "membership", "policy", "pricing", "timing"
    subscription_inquiry: "manage_dispute_bill", "manage_extension", "manage_pay_bill", "status_due_amount", "status_due_date", "status_questions"
    troubleshoot_site: "credit_card", "search_results", "shopping_cart", "slow_speed"
"primary_team": Which team should handle this ticket first. One of:
    "account_support": Login, passwords, profile and access
    "engineering": Site errors and broken features
    "billing_and_payments": Charges, refunds, invoices and payment methods
    "product_management": Ideas and requests for new features
    "other": None of these teams
"journey_stage": Determine where the user is in their relationship with the company. One of:
    "browsing": Has not bought anything yet
    "checkout": Is in the middle of paying
    "post_purchase": Has already bought something
    "unclear": The message doesn't say
"message_type": Determine what kind of message this is overall. One of:
    "question": Mainly asks for information
    "request": Mainly asks for something to be done
    "feedback": Mainly gives an opinion or a suggestion
    "problem_report": Mainly reports something that went wrong
    "other": None of these fit
Answer each of these with true or false:
"cannot_log_in": The user cannot log in to their account
"security_related": The ticket involves account security, such as access, passwords or suspicious activity
"mentions_account_issue": The user raises a problem with logging in, permissions, their profile or account security
"mentions_billing_issue": The user raises a problem with a charge, invoice, refund or subscription
"mentions_bug": The user reports something that is broken or producing errors
"refund_requested": The user is explicitly asking for a refund or credit
"refund_implied": The user wants money back
"needs_staff_action": Resolving this ticket needs a staff member to take an action, not just send an answer
"identity_details_given": The user gives their name, email address or account username
"charge_amount_given": The user states the amount they were charged

The customer's opening message:
<<<
{opening}
>>>
```

The command was `claude -p --model haiku --effort low --output-format json --tools "" --strict-mcp-config --no-session-persistence --setting-sources "" --system-prompt "<system prompt>"`. A separate call on the same day, under Claude Code 2.1.310, showed the `haiku` alias resolving to `claude-haiku-5-5`. A call whose reply held no JSON object that parsed was retried up to 2 times, and only the last attempt's time and cost were logged.

## Appendix B: Reproduction

All paths are relative to the micro_llm repository. None of the code or result files below is public yet. The comparison scripts and their outputs sit in `data/claude_vs_micro/`.

| Step | Command | Writes |
|---|---|---|
| Write the synthetic tickets | `python3 -m micro_llm.generate --tickets 480 --batch 2 --workers 12` | ticket batches in `data/` |
| Train the synthetic system | `python3 -m micro_llm.train` | `models/` |
| Score it on held-out synthetic tickets | `python3 -m micro_llm.evaluate` | `models/eval.json` |
| ABCD analysis and the truth mapping | `python -m micro_llm.abcd` | `data/abcd_analysis.json` |
| Both systems on the 300 openings | `python3 -I data/claude_vs_micro/compare.py . data/claude_vs_micro 300 12 haiku` | `results_haiku.json`, `rows_haiku.json`, `claude_haiku.jsonl` |
| Intervals and tests | `python3 -I data/claude_vs_micro/stats.py data/claude_vs_micro haiku 10000` | `stats_haiku.json` |
| Rescoring on usable answers | `python3 -I data/claude_vs_micro/robustness.py . data/claude_vs_micro haiku` | `robustness_haiku.json` |
| Scoring rules and recall by class | `python3 -I data/claude_vs_micro/metrics.py data/claude_vs_micro haiku 10000` | `metrics_haiku.json` |
| Heads' training and latency | `python3 -I data/claude_vs_micro/timing.py . data/claude_vs_micro` | `timing_micro.json` |
| Learning curve | `python3 -I data/claude_vs_micro/learning_curve.py . data/claude_vs_micro haiku 5` | `learning_curve_haiku.json` |
| Baselines, exclusions, routing, overlap and confusions | `python3 -I data/claude_vs_micro/checks.py . data/claude_vs_micro haiku` | `checks_haiku.json` |
| Heads trained on the synthetic tickets | `python3 -I data/claude_vs_micro/synthetic_heads.py . data/claude_vs_micro haiku` | `synthetic_heads_haiku.json` |
| Routing rules, learning-curve intervals, unweighted heads and error counts | `python3 -I data/claude_vs_micro/revision_checks.py . data/claude_vs_micro haiku 5 10000` | `revision_checks_haiku.json` |

The runs used Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2 and joblib 1.5.2 on an Apple M1 Max with 10 cores. The sample uses seed 0, the bootstraps and permutations use NumPy seed 0, and the learning curve uses seeds 0 to 4. Claude's answers are cached per chat in `claude_haiku.jsonl`, so rerunning the comparison reuses them rather than calling the model again. Only the ticket writer and the comparison call a model; every other step runs offline.

## Appendix C: How the 13 bank questions get their truth from ABCD

`micro_llm/abcd.py` holds the full mapping. Ten truths come from the chat's intent, and three read the chat itself.

- **`primary_team`:** account access and profile changes go to account support. Money problems go to billing and payments: missing credit, an unexplained fee, refunds, price complaints, and disputed or mistimed bills. So do billing tasks that raise no problem, such as payment methods, bill payments, subscription status and extensions. Site problems (shopping cart, search results, slow speed) go to engineering. Returns, orders, shipping, stock and product or store questions go to "other". No ABCD intent maps to the bank's fifth team, product management.
- **Six intents have no team truth.** The intent alone does not settle which team takes `status_service_added`, `status_service_removed`, the two promo-code intents, `credit_card` or `cost`. They hold 33 of the 300 chats, and the yes/no questions that read the intent leave them out too.
- **`journey_stage`:** product questions and the out-of-stock intents are browsing, and the shopping-cart intent is checkout. Intents about an existing order, return, refund or charge are post-purchase. Other intents, such as account access or store policy, have no truth.
- **`message_type`:** product and store questions are questions, as are status checks other than missing credit, refund status, shipping status and shipping cost. Account recovery, account, order, shipping and billing changes, returns, refund starts and updates, and price complaints are requests. Site problems, a rejected credit card, a missing item or credit, disputed or mistimed bills, and failed promo codes are problem reports. The out-of-stock intents have no truth, which leaves out 15 of the 300 chats.
- **One intent reads two ways.** `status_mystery_fee` is a billing problem for `primary_team` and `mentions_billing_issue`, but a question for `message_type`. It covers 1 of the 300 chats.
- **Seven yes/no questions read the intent.** Each has intents that mean yes and intents that mean no; the rest have no truth.
  - `cannot_log_in` and `security_related` are yes for the 3 account-access intents, so their truths match on every chat.
  - `mentions_account_issue` is yes for account access and profile changes.
  - `mentions_billing_issue` is yes for the money problems above. It is no for account, profile, site, order, shipping and product or store intents, and billing tasks and returns have no truth.
  - `mentions_bug` is yes for the 3 site-problem intents.
  - `refund_requested` is yes only when the customer starts a refund.
  - `refund_implied` is yes for returns, refund starts, price complaints and mistimed bills.
  - Both refund questions are no for intents where no money is owed back. Those are account, profile, site, product and store intents, billing tasks, and order and shipping intents other than out-of-stock items, cancellations and missing items.
- **Three questions read the chat, not the intent.** `needs_staff_action` is yes when the agent took one of 8 actions. They are offering a refund, making a purchase, updating an order or an account, creating a password, applying a promo code, sending a link, or notifying a team. Two of these, sending a link and applying a promo code, inform or discount rather than change an account or order. ABCD's delexicalized text marks the details a customer typed, such as `<email>` or `<amount>`. `identity_details_given` is yes when the opening holds a name, email address or username; a name counts when the customer's full or given name appears. `charge_amount_given` is yes when the opening holds an amount.

## Appendix D: The synthetic data against both other systems

**Table 8:** Trained on 383 synthetic tickets, the heads score below Claude on all 13 shared questions; trained on 383 ABCD chats, they score above it on 9. Percentages on the 300 openings, scored as in Table 1. The 383-chat column is the mean of 5 runs. The synthetic system answers `identity_details_given` and `charge_amount_given` with a regular-expression entity finder, and its other 11 answers come from learned models.

| Question | Synthetic system | Heads, 383 synthetic tickets | Heads, 383 ABCD chats | Heads, 8,034 ABCD chats | Claude |
|---|---|---|---|---|---|
| primary_team | 66.3 | 47.2 | 87.3 | 88.4 | 83.1 |
| journey_stage | 9.7 | 23.4 | 92.1 | 93.1 | 72.4 |
| message_type | 26.7 | 30.5 | 84.3 | 88.1 | 67.4 |
| cannot_log_in | 68.6 | 65.7 | 92.7 | 92.3 | 88.6 |
| security_related | 68.6 | 71.4 | 92.7 | 92.3 | 91.8 |
| mentions_account_issue | 77.3 | 79.9 | 91.8 | 91.4 | 90.1 |
| mentions_billing_issue | 65.5 | 68.4 | 88.2 | 93.4 | 85.3 |
| mentions_bug | 53.1 | 78.9 | 95.6 | 98.0 | 97.0 |
| refund_requested | 66.7 | 66.4 | 81.2 | 98.9 | 100.0 |
| refund_implied | 53.4 | 54.0 | 83.9 | 90.2 | 77.8 |
| needs_staff_action | 52.3 | 50.7 | 68.8 | 73.3 | 58.0 |
| identity_details_given | 76.4 | 51.4 | 83.7 | 98.2 | 92.9 |
| charge_amount_given | 99.7 | 72.8 | 59.7 | 99.5 | 75.0 |
| **Mean of 13** | **60.3** | **58.5** | **84.8** | **92.1** | **83.0** |

## Appendix E: Where each system goes wrong

**Table 9:** Claude's errors cluster where its reading of a definition differs from the mapping. Errors on the 300 openings for the five questions with the largest gaps, grouped by truth → answer, with the commonest true intents in brackets. "billing" is `billing_and_payments` and "account" is `account_support`.

| Question | System | Errors | Largest confusions, truth → answer |
|---|---|---|---|
| primary_team | Heads | 31 | other → account 7; billing → other 5 (`refund_update` 2); other → billing 5 (`policy` 2); other → engineering 4 (`manage_upgrade` 2) |
| primary_team | Claude | 45 | other → billing 20 (`policy` 4, `return_color` 3, `return_size` 3, `status` 3); billing → other 10 (`bad_price_competitor` 3, `manage_extension` 3, `bad_price_yesterday` 3); other → account 6 (`manage` 4); account → other 5 (`reset_2fa` 3) |
| journey_stage | Heads | 10 | post_purchase → checkout 5 (`bad_price_yesterday` 2); post_purchase → browsing 2; browsing → post_purchase 2 |
| journey_stage | Claude | 40 | post_purchase → unclear 16 (`bad_price_yesterday` 4); browsing → unclear 10 (`shirt` 3); checkout → browsing 7 (`shopping_cart` 7); post_purchase → browsing 5 (`bad_price_yesterday` 4) |
| message_type | Heads | 34 | request → problem_report 11; question → request 9; request → question 6; question → problem_report 4 |
| message_type | Claude | 93 | request → problem_report 21 (`reset_2fa` 6, `recover_password` 4, `bad_price_yesterday` 4, `recover_username` 3); question → problem_report 17 (`status_service_added` 4); request → other 15 (`manage_extension` 3, `bad_price_yesterday` 3); request → question 13 (`refund_update` 3) |
| refund_implied | Heads | 15 | no → yes 9 (`manage_extension` 2, `reset_2fa` 2); yes → no 6 (`bad_price_yesterday` 3, `return_color` 2) |
| refund_implied | Claude | 19 | yes → no 18 (`bad_price_yesterday` 8, `return_color` 4, `bad_price_competitor` 3, `return_stain` 2, `return_size` 1); no → yes 1 |
| needs_staff_action | Heads | 80 | no → yes 52 (`status_quantity` 4); yes → no 28 (`status_delivery_time` 3, `missing` 3, `manage_extension` 3) |
| needs_staff_action | Claude | 126 | yes → no 68 (`recover_password` 10, `out_of_stock_one_item` 10, `status_due_date` 7); no → yes 58 (`shopping_cart` 8, `search_results` 6, `status` 5) |

## Appendix F: Responsible NLP checklist, in brief

- **Data:** ABCD release 1.1, MIT license, English. Customer names, emails and account details come from each chat's scripted scenario.
- **Settings and runs:** Section 3.3, Appendix A and Appendix B give every setting of both systems. Each system ran once on the 300 chats. The intervals come from resampling chats, so they do not cover Claude's run-to-run variation, and the learning curve uses 5 seeds.
- **Statistics:** paired bootstrap intervals, paired permutation tests with Holm correction, and McNemar's exact test (Section 3.4).
- **Compute:** the heads train in 26.5 s in one laptop process. The learning curve's 36 trainings took 38.5 s on 10 cores, and the synthetic-trained heads made no model calls. Claude's 300 logged calls cost $0.8287 in total, and writing the synthetic tickets cost $6.79.
- **AI assistance:** the scripts and this draft were written with Claude Code under the author's direction.
- **Release:** none of the code or result files behind this paper is public yet.
