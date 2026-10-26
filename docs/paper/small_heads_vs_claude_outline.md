# Outline of the small-heads paper


**The paper's practical rule: prompt Claude while a team has no labeled chats, and train small classifiers once it has about 1,000.**


Source: `docs/paper/small_heads_vs_claude.md`. The *heads* are the paper's system: one logistic-regression classifier per question over shared TF-IDF features. An *opening* is what a customer types before the agent first replies.

## Abstract: heads trained on real labels beat zero-shot Claude by 9.2 points and answer in about 4 ms

- **Question:** should a support team triage chats by prompting an LLM or by training small classifiers on labeled chats?
- **Test:** 15 triage questions on 300 customer openings from ABCD, a corpus of role-played store chats.
- **Accuracy:** the heads average 90.7%, and Claude Haiku 5.5, prompted with no examples, averages 81.5%.
- **ABCD's own labels:** the heads lead by 10.1 points on topic and 10.2 on intent, both significant after Holm correction.
- **Scoring rules:** the mean gap runs from 5.2 to 10.4 points across three rules.
- **Labels needed:** the heads draw level at 383 labeled chats and lead in every run from 1,000.
- **Routing:** deriving answers from each system's own intent answer cuts the gap on 10 questions from 7.3 to 2.6–3.4 points.
- **Speed and cost:** 3.98 ms against 18.61 s per chat, and Claude costs $0.0028 per chat on average.
- **Synthetic data:** trained on 383 Claude-written tickets, the heads score 58.5%, against 84.8% on 383 real chats.

## 1 Introduction: the heads win on accuracy and speed, and two findings bound the win

- **Problem:** a support desk asks every chat the same few questions, and a wrong answer means a reroute by hand.
- **Two options:** prompt an LLM with no labels for seconds and a fraction of a cent per chat, or label chats and train classifiers that answer in milliseconds.
- **Gap in prior work:** published comparisons mostly report accuracy, and fewer report what each answer costs in time and money.
- **Study:** both systems answer 15 questions on the same 300 ABCD test openings, compared with paired intervals and corrected tests.
- **Result:** the heads lead by about 10 points on ABCD's two labels and 9.2 on the mean, and answer in 3.98 ms against 18.61 s.
- **First bound:** routing both systems through their own intent answers shrinks the gap on 10 questions from 7.3 to 2.6–3.4 points.
- **Second bound:** trained on 383 tickets from one synthetic recipe, the heads lose to Claude.
- **Five contributions:** a paired comparison, checks on the gap, the gap's source, a learning curve, and synthetic data with speed, cost and a decision rule.

## 2 Related Work: earlier studies mostly favor fine-tuned models, and this paper adds a learning curve

- **Fine-tuned against prompted:** seven studies, two of them with Claude, find fine-tuned models mostly win.
- **Native labels:** those studies score labels native to each benchmark, while 13 of this paper's truths come from the author's mapping.
- **Intent detection:** on Banking77 SetFit edged the best LLM setting, while Claude 3 Haiku with retrieved examples beat SetFit on 6 datasets.
- **ABCD:** built to test whether agents follow a company's workflows; this paper uses only its intent labels and openings.
- **Cost and latency:** three studies time or price both systems, and in each the small model answers far faster.
- **Cheaper paths through an LLM:** distillation, cascades and LLM-written training data, which helps less as tasks grow more subjective.
- **Narrow small models:** micromodels and logistic regression on LLM embeddings; the heads are simpler still, with no neural network.
- **This paper's addition:** a learning curve beside measured price and time, so a team can see how many labels it needs.

## 3 Setup: both systems answer the same 300 openings, and an unusable answer counts as wrong

### 3.1 The data are 300 test openings from ABCD

- **Corpus:** 10,042 role-played chats for an online clothing store, split 8,034 / 1,004 / 1,004, release 1.1 under the MIT license.
- **Labels:** each chat carries a topic (ABCD's "flow", 10 values), an intent and the actions the agent took.
- **Intents:** the 96 raw subflows collapse to 55 intents.
- **Input:** the customer's opening, with a median of 16 words.
- **Sample:** 300 test chats drawn with seed 0, holding 53 of the 55 intents.

### 3.2 Thirteen of the 15 questions have truths I mapped from ABCD's labels

- **Native:** two questions use ABCD's labels, topic (10 classes) and intent (55).
- **Mapped:** the other 13 come from the earlier synthetic system's question bank, 3 choice questions and 10 yes/no.
- **Mapping:** the author's rules derive their truth from the intent, the agent's actions and typed details (Appendix C).
- **Coverage:** a chat whose intent cannot settle a question drops out, so each question scores 145 to 300 chats.
- **Duplicate:** `cannot_log_in` and `security_related` have the same truth on all 300 chats.
- **Later actions:** `needs_staff_action` depends on what the agent did later in the chat.
- **Two positives:** `charge_amount_given` has 2 yes chats, so one changed answer moves its score 25 points.
- **Timing:** the mapping was fixed before Claude's answers existed, though while scoring the heads on this test split.

### 3.3 The two systems saw the same opening and different help

- **Heads:** 38,990 TF-IDF features from word 1–2-grams and character 2–5-grams, with one logistic regression per question.
- **Settings:** C = 4.0 and balanced class weights, carried over from the synthetic system and never tuned on ABCD.
- **Why TF-IDF:** it trains in seconds on a laptop without a GPU; SetFit and fine-tuned encoders went untested.
- **Claude:** Haiku 5.5 through the Claude Code CLI 2.1.310 on 2026-10-26, at effort low, with 12 calls at once.
- **Thinking:** stayed on, because the CLI's help text lists no option to turn it off.
- **Prompt:** all 15 questions with allowed answers, topics and intents by name only, one-line definitions for the rest, and no examples.
- **Out-of-set answers:** four offered answers match no ABCD truth, and Claude chose them 53 times on chats with a truth.
- **Unequal by design:** the test sets what a team can run before labeling against what labeled history buys.

### 3.4 Scoring counts an unusable answer as wrong

- **Unusable answers:** empty, malformed and out-of-set answers count as wrong, and Section 4.2 rescores without them.
- **Metrics:** balanced accuracy for topic, intent and the 10 yes/no questions, and plain accuracy for the 3 bank choice questions.
- **Hidden prices:** balanced accuracy makes one missed yes cost 1.0 to 149.0 false alarms, depending on the question.
- **Baseline:** a constant answer, which scores 50.0% on yes/no, 10.0% on topic and 1.9% on intent.
- **Statistics:** paired bootstrap intervals, paired permutation tests with Holm correction, and McNemar's exact test.
- **Disclosure:** the checks in Sections 4.2 to 5.5 came after the main comparison, and each is reported whatever its result.

## 4 Results: the heads lead under every check, by 5.2 to 10.4 points on the mean

### 4.1 The heads lead by about 10 points on ABCD's own labels and by 9.2 on the mean

- **Mean of 15:** heads 90.7%, Claude 81.5%, a 9.2-point gap with a 95% interval of 6.3 to 12.1.
- **Native labels:** +10.1 on topic and +10.2 on intent, both significant after Holm correction.
- **Significant questions:** 6 of 15 survive correction, all favoring the heads, and 4 of the 6 rest on the mapping.
- **Claude's one lead:** `refund_requested`, 100% against 98.9%, not significant after correction.
- **Below baseline:** three scores fail to clear the constant answer, two of them Claude's.
- **Table 1:** every question's score, baseline, interval and Holm p-value.

### 4.2 Claude's unusable answers explain about half the journey-stage gap and little of the rest

- **Empty answers:** Claude left the topic blank for 25 chats and the intent for 30.
- **Out-of-set answers:** 12 intents of `status_questions`, 27 journey stages of "unclear" and 26 message types of "other" or "feedback".
- **Usable chats only:** the heads still lead on all 5 choice questions, by 5.2 to 18.5 points (Table 2).
- **Journey stage:** its gap narrows from 20.7 to 9.3 points, so the "unclear" answers carry about half of it.
- **Near-empty openings:** Claude's empty topics fall on 2-word openings, where declining to guess may beat a wrong guess.

### 4.3 The scoring rule sets the gap's size, from 5.2 to 10.4 points

- **Three rules:** +9.2 as reported, +5.2 under plain accuracy and +10.4 under balanced accuracy (Table 3).
- **Rare yes cases:** the heads find 90.2% of them and Claude 78.0%.
- **No cases:** Claude keeps 99.1% of them and the heads 97.1%, so plain accuracy shrinks the gap.
- **Claude's plain wins:** higher plain accuracy on 6 of the 10 yes/no questions, by 0.7 to 3.0 points.
- **One question:** `charge_amount_given` carries 42.4% of the shrink.
- **Without class weights:** the heads still lead by 7.1 to 9.8 points.

### 4.4 The gap holds when weak questions or near-duplicate chats are removed

- **Weak questions:** dropping all three leaves a 7.9-point gap (Table 4).
- **Native against mapped:** the gap is 10.2 points on ABCD's two labels and 9.1 on the 13 mapped questions.
- **Near-duplicates:** 37 test openings have a training opening at cosine 0.9 or more, and without them the gap is 9.7.
- **Full test split:** the heads score 89.8% on all 1,004 openings against 90.7% on the 300.
- **Not run:** Claude on the other 704 openings, which would cost $1.94.

## 5 Analysis: Claude's weaker intent answers drive most of the gap, and label count decides the winner

### 5.1 Derived from each system's own intent answer, the gap on 10 questions falls from 7.3 points to 2.6–3.4

- **Why:** 13 truths come from the mapping, and Claude saw only one-line definitions of them.
- **Method:** answer the 10 intent-derived questions from each system's own intent answer.
- **Two rules:** when an intent settles nothing, the fallback rule keeps the direct answer and the strict rule counts it wrong.
- **Result:** Claude rises from 85.3% to 91.0%, or 88.5% under the strict rule, and routing closes 64% or 53% of the gap.
- **What remains:** Claude's weaker intent answers, exactly right on 65.7% of chats against 76.0% for the heads.
- **Where readings differ:** Claude sends 20 return and policy chats to billing, and answers "unclear" on 26 journey stages.
- **Untested:** a second annotator on 100 openings.
- **A design to try:** ask the LLM only for the intent and derive the rest in code, which scores 91.0% here.

### 5.2 The heads draw level with Claude at 383 labeled chats and lead clearly from 1,000

- **Method:** retrain the heads on nested random subsets of 50 to 8,034 chats, with 5 seeds (Table 6).
- **Why 383:** the synthetic system trained on 383 tickets, so the two compare at equal size.
- **At 383 chats:** 4 of 5 runs beat Claude by at most 2.4 points, and 1 of 5 intervals lies above zero.
- **At 1,000 chats:** every run leads by 6.0 to 7.8 points, each with an interval above zero.
- **Plain accuracy:** the heads pass sooner, with all 5 intervals above zero at 383.
- **Flattening:** going from 2,000 chats to all 8,034 adds 0.9 points.
- **Caveat:** labeling time was not measured, so the curve counts chats rather than hours.

### 5.3 Trained on Claude-written tickets, the same heads lose to Claude

- **Earlier system:** Claude wrote 480 tickets for $6.79, and the system scored 97.3% on its own held-out tickets.
- **On ABCD:** that system scores 60.3% on the 13 shared questions, below Claude on 12 of them.
- **Same heads, synthetic data:** 58.5%, against 84.8% on 383 ABCD chats, so the data account for 26.2 points.
- **They learned the synthetic task:** the synthetic-trained heads score 90.5% on held-out synthetic tickets.
- **Missing labels:** the synthetic tickets lack browsing and checkout stages, and they hold no questions.
- **Different text:** openings run 16 words at the median against 55 for a synthetic ticket.
- **Scope:** one recipe only; rewriting real openings or having an LLM label real chats was not tested.

### 5.4 The heads answer about 4,700 times faster than Claude with thinking on through its CLI, with no per-call charge

- **Heads:** 3.98 ms at the median and 4.85 ms at p99, in one process on an Apple M1 Max laptop (Table 7).
- **Claude:** 18.61 s at the median and 82.73 s at p99.
- **Claude's cost:** $0.0028 per chat on average, or $27.62 per 10,000 chats.
- **Thinking:** 96.4% of Claude's output tokens.
- **The CLI's share:** about 1.46 s per call; on the API's own time the ratio is 4,189.
- **The heads' real cost:** labeling, since training all 15 heads takes 26.5 s.

### 5.5 A team lead routes on the answers, and an agent fixes the misses by hand

- **Users:** a team lead writes routing rules on the answers, and an agent opens each chat already marked.
- **Misroutes:** 116.1 per 1,000 chats for the heads against 168.5 for Claude, a gap that suggests a difference without showing one.
- **Missed flags:** Claude misses 56 rare yes cases and the heads 25.
- **False flags:** the heads raise 59 and Claude 19, which an agent must read and dismiss.
- **Total errors:** the heads make 84 to Claude's 75.
- **Untested:** sending only the least-confident openings to Claude, or having Claude label real chats.

### 5.6 Against zero-shot Claude, the number of labeled chats decides which system to deploy

- **The rule:** with no labeled chats, prompt Claude; with about 1,000, train heads.
- **Caveats:** Claude with retrieved examples was not tested, and the intervals omit Claude's run-to-run variation.
- **Label coverage:** the labels must cover every question; ABCD supplied topics, intents and the agent's logged actions.
- **A new question:** costs Claude one line of prompt, and costs the heads labels plus one head that trains in 0.11 to 0.45 s.
- **Drift:** rescore the heads on fresh labels as traffic changes, and pin Claude's model version.

## 6 Limitations: one dataset, one zero-shot run, and a mapping no second person checked

- **One dataset:** short, role-played chats for one store, which Claude may have seen in training.
- **One Claude run:** zero-shot with thinking on, and the intervals do not cover its run-to-run variation.
- **Four out-of-set answers:** the prompt offered them, and Claude used them 53 times.
- **The mapping:** 13 truths come from the author's rules, with no second annotator.
- **Reused test split:** the mapping and scoring rules were fixed while scoring the heads on this split.
- **Mixed scoring rules:** under plain accuracy throughout, the gap is 5.2 points.
- **Three weak questions:** dropping them leaves a 7.9-point gap.
- **One synthetic recipe:** Section 5.3 shows what that recipe gave, not what synthetic data can give.
- **Labeling time:** not measured.
- **Sample:** Claude ran on 300 of the 1,004 test chats.

## 7 Conclusion: the heads won on these chats, and richer prompting is the open question

- **Accuracy:** the heads led on 6 questions after correction, and Claude led on none.
- **Range:** the mean gap ran from 5.2 to 10.4 points, depending on the scoring rule.
- **Source:** with both systems routed through intent, 2.6 to 3.4 points remained, mostly from Claude's weaker intent answers.
- **Labels:** level at 383 labeled chats, and ahead in every run from 1,000.
- **Synthetic data:** heads trained on Claude-written tickets fell to 58.5%.
- **Speed and cost:** 3.98 ms against 18.61 s, and $0.0028 per Claude call.
- **Open question:** whether retrieved examples, intent descriptions or thinking turned off would close the gap.

## References: 24 works, each summarized in the companion file

- **Mix:** 18 related studies, the ABCD paper, 4 statistics papers and the scikit-learn paper.
- **Summaries:** `docs/paper/small_heads_vs_claude_references.md` gives each exact title and a summary.

## Appendix A: Claude saw one prompt with all 15 questions and no examples

- **System prompt:** "You triage support chats for an online clothing store," with replies limited to JSON.
- **User message:** every question with its allowed answers, then the customer's opening between markers.
- **Command:** `claude -p --model haiku --effort low --output-format json`, with tools, MCP servers, sessions and saved settings off.
- **Model check:** a separate call that day showed the `haiku` alias resolving to `claude-haiku-5-5`.
- **Retries:** up to 2 for a reply with no parsable JSON, and only the last attempt was logged.

## Appendix B: 13 commands rebuild every result, and none of the code is public yet

- **Table:** each step's command and the file it writes, from writing tickets to the revision checks.
- **Environment:** Python 3.13.9 and scikit-learn 1.7.2 on an Apple M1 Max with 10 cores.
- **Seeds:** 0 for the sample, bootstraps and permutations, and 0 to 4 for the learning curve.
- **Caching:** Claude's answers are cached per chat, so a rerun does not call the model again.
- **Model calls:** only the ticket writer and the comparison call a model.

## Appendix C: 10 truths come from the chat's intent, and 3 read the chat itself

- **Team:** account access goes to account support, money problems and billing tasks to billing, site problems to engineering, and the rest to "other".
- **Unsettled teams:** six intents leave the team open, and they hold 33 of the 300 chats.
- **Stage and type:** intents map to journey stages and message kinds, and some intents have no truth.
- **One intent, two readings:** `status_mystery_fee` is a billing problem for the team but a question for message type.
- **Seven yes/no questions:** each reads the intent, and `cannot_log_in` and `security_related` match on every chat.
- **Three read the chat:** `needs_staff_action` uses 8 agent actions, and the two "given" questions use typed-detail markers.

## Appendix D: synthetic-trained heads trail Claude on all 13 shared questions

- **Table 8:** five systems on the 13 shared questions.
- **Synthetic training:** heads on 383 synthetic tickets average 58.5% against Claude's 83.0%.
- **Real training:** heads on 383 ABCD chats average 84.8% and beat Claude on 9 of 13.
- **Synthetic system:** 60.3% overall, though its regular-expression entity finder scores 99.7% on `charge_amount_given`.
- **Full data:** heads on all 8,034 chats average 92.1%.

## Appendix E: Claude's errors cluster where it read a definition differently from the mapping

- **Table 9:** errors grouped by truth and answer for the five questions with the largest gaps.
- **Team:** Claude sends 20 "other" chats to billing, led by returns and store policy.
- **Journey stage:** 26 of Claude's 40 errors answer "unclear".
- **Message type:** Claude makes 93 errors to the heads' 34, and 21 of them call a request a problem report.
- **Refund implied:** 18 of Claude's 19 errors miss a yes, on price complaints and returns.
- **Staff action:** both systems err most here, the heads 80 times and Claude 126.

## Appendix F: the checklist reports one run per system and every cost

- **Data:** ABCD 1.1, MIT license, English, with names and emails from scripted scenarios.
- **Runs:** each system ran once on the 300 chats, and the learning curve uses 5 seeds.
- **Statistics:** bootstrap intervals, permutation tests with Holm correction, and McNemar's exact test.
- **Compute:** the heads train in 26.5 s in one laptop process.
- **Model spend:** Claude's 300 logged calls cost $0.8287, and the synthetic tickets $6.79.
- **AI assistance:** the scripts and the draft were written with Claude Code under the author's direction.
- **Release:** none of the code or result files is public yet.


**The paper's one claim: trained on 8,034 labeled ABCD chats, four-millisecond classifiers beat zero-shot Claude Haiku 5.5 by 9.2 points on 15 triage questions.**


