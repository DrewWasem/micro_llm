# ABCD grows micro_llm from 47 to 169 questions per ticket


**96 of the 122 added questions score at least 80% from the customer's first messages, so most can run at triage.**


ABCD (ASAPP, MIT license) holds 10,042 store support chats that crowd workers wrote by role-play. Each chat carries its intent and every action the agent took. I downloaded it at commit `6b8700c` into `data/abcd/`, which git ignores, and `python -m micro_llm.abcd` reruns the whole analysis in 109 s.

Two terms recur below:

- **The opening** is the customer's first messages, up to the agent's first reply. It is all a triage step sees.
- **Balanced accuracy** averages the hit rate over each possible answer, so guessing scores 50% on a yes/no question. Every score below is balanced unless it says accuracy.

## This analysis answers 10 questions

| Question                                            | Answer                                                                                                    |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| How many questions can ABCD add to our 47?          | 122, so each ticket gets 169 answers. None of the 47 is dropped.                                          |
| Which new ones work from the opening alone?         | 96 score at least 80%. 14 more need the whole chat, and 11 stay weak.                                     |
| Do our synthetic-trained models work on real chats? | Weakly. Team routing scores 67% accuracy against 54% for always guessing; retrained on ABCD, it scores 89%. |
| Which of our 47 can ABCD grade?                     | 23. The other 24, such as sentiment, priority and churn risk, have no ABCD label.                         |
| Do agents follow the intent's playbook?             | Mostly: 78% of chats use only playbook actions, and 72% also keep its order.                              |
| Does our account gate match what agents do?         | Yes: 94% of chats that change something check identity first.                                            |
| Do customers give identity details up front?        | Rarely: 9% of openings hold one.                                                                          |
| How well does our entity finder read real chats?    | It catches every email and phone number, but only 17% of names and 41% of order IDs.                      |
| How different is ABCD's text from our tickets?      | Openings run 16 words against 55 in our tickets, and 12.5% of their words never appear in ours.           |
| How fast is the pipeline on real chats?             | 2.4 ms median per opening and 8.4 ms per whole chat.                                                      |

## Keep all 47 questions and add 122 that ABCD can grade

| What the new questions ask                                                     | New     | From the opening | Need the whole chat | Weak   |
|--------------------------------------------------------------------------------|---------|------------------|---------------------|--------|
| The chat's topic: which of 10 topics, 55 intents or 48 FAQ entries, plus a yes/no for each topic and intent | 68      | 63               | 4                   | 0      |
| The agent's actions: a yes/no per action, plus the first action and the identity check | 34      | 26               | 6                   | 2      |
| Details the agent will need: a yes/no each for name, order ID and 8 more       | 10      | 7                | 3                   | 0      |
| Values the agent enters: escalation team, refund method, product and 4 more    | 7       | 0                | 1                   | 6      |
| Facts about the customer: membership level, payment method on file            | 2       | 0                | 0                   | 2      |
| The chat's length in turns                                                     | 1       | 0                | 0                   | 1      |
| **Total**                                                                      | **122** | **96**           | **14**              | **11** |

A column counts the questions that score at least 80% from the opening, or only from the whole chat. The intent `status_active` has no chats, so 121 are graded. Appendix A scores each one.

- **The topic is clear from the opening:** the 10-way topic question scores 88% and the 55-way intent question 76%, where guessing scores 10% and 2%.
- **What the agent will do is mostly clear too:** 26 of the 30 action questions reach 80% from the opening, with a median of 84%.
- **Answer the 2 customer facts by account lookup, not by a model,** because the text barely shows them. Membership level scores 26% from the opening, against 25% chance.
- **The 7 values the agent enters stay weak:** 6 score under 80% even with the whole chat. The product scores 11% from the opening and 62% from the whole chat, because customers name the item later.
- **The 24 of our 47 that ABCD can't label stay in the set:** sentiment, priority, churn risk and 21 more need labels from another source.

## Our synthetic-trained models miss on real chats, and ABCD's labels close the gap

- **Team routing scores 67% accuracy,** against 54% for always guessing "other". Retrained on ABCD's 8,034 training chats, the same model type scores 89% from the opening and 97% from the whole chat.
- **Misses fall into "other":** from the opening, billing gets 115 of 228 chats right, account support 67 of 117 and engineering 26 of 79.
- **Our yes/no questions sit near chance:** across the 10 that ABCD labels both ways, the median is 68% from the opening. Retrained, it is 93%. `mentions_bug` and `refund_implied` score 53%.
- **`message_type` can't give ABCD's most common answer:** it never answers "question", which 41% of chats are, so it scores 27% accuracy.
- **`journey_stage` rarely answers post_purchase,** because it needs an order number or a billing issue. Only 1.3% of openings hold an order number, so it mostly answers "unclear" and scores 10% accuracy.
- **The test mixes two gaps:** a clothing store against our SaaS tickets, and real text against synthetic text. The retrained models were tested on the same store, so 89% is a ceiling for a new domain, not a forecast.

## Agents mostly follow the intent's playbook, and verify before they change anything

ABCD's knowledge base lists each intent's playbook: the actions an agent may take, in order.

- **78% of chats use only playbook actions,** and 72% also keep the playbook's order. Only 43% run every step, because playbooks include conditional steps and chats end once the problem is solved.
- **Site problems break the order most:** only 48% of `troubleshoot_site` chats keep it. For a slow site, agents often run log-out-in before try-again, the reverse of the playbook.
- **The account gate matches practice:** 51.5% of chats change something, such as a refund or an order, and 93.9% of those check identity first.
- **Agents escalate in 5.6% of chats:** 274 times to purchasing, 187 to the website team and 86 to a manager.

## Customers hold back identity details, and our entity finder misses bare answers

- **Only 9.3% of openings hold an identity detail,** so the agent must ask before verifying in 9 of 10 chats. Over the whole chat, 77.5% of customers give their name and 44.4% an order ID.
- **The entity finder misses bare answers,** though it catches details with a fixed shape. Recall is the share of typed details it finds. Precision is the share of its finds that are real.

| Detail   | Recall | Precision | What it misses, and why                                     |
|----------|--------|-----------|-------------------------------------------------------------|
| Email    | 100%   | 90%       | Nothing                                                     |
| Phone    | 100%   | 73%       | Nothing                                                     |
| Amount   | 84%    | 76%       | "59" and "94$": it needs a currency sign first or a word after |
| Address  | 76%    | 96%       | "1st Ave" and lowercase streets: it wants capitalised words before "Ave" |
| Order ID | 41%    | 99%       | "my order id is 4870952797": "is" breaks the pattern        |
| Username | 28%    | 98%       | "my username is josephb9": it needs "@" or ":"              |
| Name     | 17%    | 97%       | "Crystal Minh" typed alone: it needs "my name is" or a sign-off |

- **Most false alarms are ABCD's own gaps:** phone numbers, emails and dollar amounts that its masking missed.

## ABCD's openings are short, and the pipeline reads one in 2.4 ms

> Hi. I am trying to make a purchase but my credit card keeps declining.

- **Openings run 16 words at the median,** against 49 for all the customer's messages and 55 for one of our synthetic tickets.
- **12.5% of opening words never appear in our tickets,** and 19.6% across the whole chat, so the store's vocabulary is new to our models.
- **The pipeline takes a median 2.4 ms per opening,** and 8.4 ms per whole chat, because a whole chat splits into 10 pieces, not 2.

## ABCD's limits are one issue per chat, one store and 10 customers

- **Every chat holds one issue,** so ABCD can't test `issue_count` or `issues_related`. On the whole chat our pipeline finds several issues, so it scores 1% on `issue_count`.
- **It is one clothing store,** so bugs appear only as 3 site intents and feature requests never appear.
- **Crowd workers played 10 customers,** so a name model trained on ABCD would learn those 10 names.
- **The mapping from ABCD's intents to our answers is my judgment.** Appendix C lists it. Six intents fit no team, so the team questions skip them.


**Each ticket can get 169 answers: our 47 plus ABCD's 122, of which 96 work from the opening and 14 more from the whole chat.**

## Appendix A: the 122 new questions, by what they ask

Scores are balanced accuracy on the 1,004 test chats. "Answer from" names the first input that reaches 80%.

### The chat's topic: 63 of 68 work from the opening

| Question                                   | Asks                                                   | Opening | Whole chat | Chance | Answer from   |
|--------------------------------------------|--------------------------------------------------------|---------|------------|--------|---------------|
| `flow`                                     | Which of ABCD's 10 topics the chat is about            | 88%     | 97%        | 10%    | Opening       |
| `intent`                                   | Which of ABCD's 55 intents the chat is about           | 76%     | 92%        | 2%     | Whole chat    |
| `faq_answer`                               | Which of 48 FAQ entries answers the question           | 61%     | 89%        | 2%     | Whole chat    |
| `flow:account_access`                      | The chat is about account_access                       | 95%     | 99%        | 50%    | Opening       |
| `flow:manage_account`                      | The chat is about manage_account                       | 92%     | 97%        | 50%    | Opening       |
| `flow:order_issue`                         | The chat is about order_issue                          | 92%     | 97%        | 50%    | Opening       |
| `flow:product_defect`                      | The chat is about product_defect                       | 94%     | 98%        | 50%    | Opening       |
| `flow:purchase_dispute`                    | The chat is about purchase_dispute                     | 93%     | 98%        | 50%    | Opening       |
| `flow:shipping_issue`                      | The chat is about shipping_issue                       | 92%     | 97%        | 50%    | Opening       |
| `flow:single_item_query`                   | The chat is about single_item_query                    | 94%     | 98%        | 50%    | Opening       |
| `flow:storewide_query`                     | The chat is about storewide_query                      | 93%     | 98%        | 50%    | Opening       |
| `flow:subscription_inquiry`                | The chat is about subscription_inquiry                 | 94%     | 99%        | 50%    | Opening       |
| `flow:troubleshoot_site`                   | The chat is about troubleshoot_site                    | 93%     | 98%        | 50%    | Opening       |
| `intent:recover_username`                  | The chat's intent is recover_username                  | 93%     | 98%        | 50%    | Opening       |
| `intent:recover_password`                  | The chat's intent is recover_password                  | 91%     | 97%        | 50%    | Opening       |
| `intent:reset_2fa`                         | The chat's intent is reset_2fa                         | 90%     | 100%       | 50%    | Opening       |
| `intent:status_service_added`              | The chat's intent is status_service_added              | 89%     | 87%        | 50%    | Opening       |
| `intent:status_service_removed`            | The chat's intent is status_service_removed            | 89%     | 96%        | 50%    | Opening       |
| `intent:status_shipping_question`          | The chat's intent is status_shipping_question          | 83%     | 100%       | 50%    | Opening       |
| `intent:status_credit_missing`             | The chat's intent is status_credit_missing             | 89%     | 95%        | 50%    | Opening       |
| `intent:manage_change_address`             | The chat's intent is manage_change_address             | 98%     | 80%        | 50%    | Opening       |
| `intent:manage_change_name`                | The chat's intent is manage_change_name                | 78%     | 86%        | 50%    | Whole chat    |
| `intent:manage_change_phone`               | The chat's intent is manage_change_phone               | 98%     | 100%       | 50%    | Opening       |
| `intent:manage_payment_method`             | The chat's intent is manage_payment_method             | 99%     | 100%       | 50%    | Opening       |
| `intent:status_mystery_fee`                | The chat's intent is status_mystery_fee                | 83%     | 100%       | 50%    | Opening       |
| `intent:status_delivery_time`              | The chat's intent is status_delivery_time              | 87%     | 97%        | 50%    | Opening       |
| `intent:status_payment_method`             | The chat's intent is status_payment_method             | 93%     | 83%        | 50%    | Opening       |
| `intent:status_quantity`                   | The chat's intent is status_quantity                   | 86%     | 96%        | 50%    | Opening       |
| `intent:manage_upgrade`                    | The chat's intent is manage_upgrade                    | 94%     | 96%        | 50%    | Opening       |
| `intent:manage_downgrade`                  | The chat's intent is manage_downgrade                  | 90%     | 100%       | 50%    | Opening       |
| `intent:manage_create`                     | The chat's intent is manage_create                     | 86%     | 95%        | 50%    | Opening       |
| `intent:manage_cancel`                     | The chat's intent is manage_cancel                     | 88%     | 100%       | 50%    | Opening       |
| `intent:refund_initiate`                   | The chat's intent is refund_initiate                   | 80%     | 89%        | 50%    | Opening       |
| `intent:refund_update`                     | The chat's intent is refund_update                     | 83%     | 93%        | 50%    | Opening       |
| `intent:refund_status`                     | The chat's intent is refund_status                     | 93%     | 96%        | 50%    | Opening       |
| `intent:return_stain`                      | The chat's intent is return_stain                      | 86%     | 100%       | 50%    | Opening       |
| `intent:return_color`                      | The chat's intent is return_color                      | 79%     | 95%        | 50%    | Whole chat    |
| `intent:return_size`                       | The chat's intent is return_size                       | 95%     | 92%        | 50%    | Opening       |
| `intent:bad_price_competitor`              | The chat's intent is bad_price_competitor              | 88%     | 100%       | 50%    | Opening       |
| `intent:bad_price_yesterday`               | The chat's intent is bad_price_yesterday               | 84%     | 94%        | 50%    | Opening       |
| `intent:out_of_stock_general`              | The chat's intent is out_of_stock_general              | 95%     | 100%       | 50%    | Opening       |
| `intent:out_of_stock_one_item`             | The chat's intent is out_of_stock_one_item             | 86%     | 91%        | 50%    | Opening       |
| `intent:promo_code_invalid`                | The chat's intent is promo_code_invalid                | 83%     | 96%        | 50%    | Opening       |
| `intent:promo_code_out_of_date`            | The chat's intent is promo_code_out_of_date            | 92%     | 93%        | 50%    | Opening       |
| `intent:mistimed_billing_already_returned` | The chat's intent is mistimed_billing_already_returned | 84%     | 100%       | 50%    | Opening       |
| `intent:mistimed_billing_never_bought`     | The chat's intent is mistimed_billing_never_bought     | 94%     | 100%       | 50%    | Opening       |
| `intent:status`                            | The chat's intent is status                            | 86%     | 95%        | 50%    | Opening       |
| `intent:manage`                            | The chat's intent is manage                            | 91%     | 94%        | 50%    | Opening       |
| `intent:missing`                           | The chat's intent is missing                           | 89%     | 96%        | 50%    | Opening       |
| `intent:cost`                              | The chat's intent is cost                              | 92%     | 94%        | 50%    | Opening       |
| `intent:boots`                             | The chat's intent is boots                             | 95%     | 98%        | 50%    | Opening       |
| `intent:shirt`                             | The chat's intent is shirt                             | 91%     | 97%        | 50%    | Opening       |
| `intent:jeans`                             | The chat's intent is jeans                             | 95%     | 100%       | 50%    | Opening       |
| `intent:jacket`                            | The chat's intent is jacket                            | 97%     | 98%        | 50%    | Opening       |
| `intent:pricing`                           | The chat's intent is pricing                           | 88%     | 96%        | 50%    | Opening       |
| `intent:membership`                        | The chat's intent is membership                        | 91%     | 100%       | 50%    | Opening       |
| `intent:timing`                            | The chat's intent is timing                            | 90%     | 98%        | 50%    | Opening       |
| `intent:policy`                            | The chat's intent is policy                            | 94%     | 100%       | 50%    | Opening       |
| `intent:status_active`                     | The chat's intent is status_active                     | —       | —          | 50%    | No test chats |
| `intent:status_due_amount`                 | The chat's intent is status_due_amount                 | 84%     | 93%        | 50%    | Opening       |
| `intent:status_due_date`                   | The chat's intent is status_due_date                   | 84%     | 88%        | 50%    | Opening       |
| `intent:manage_pay_bill`                   | The chat's intent is manage_pay_bill                   | 89%     | 96%        | 50%    | Opening       |
| `intent:manage_extension`                  | The chat's intent is manage_extension                  | 90%     | 98%        | 50%    | Opening       |
| `intent:manage_dispute_bill`               | The chat's intent is manage_dispute_bill               | 97%     | 97%        | 50%    | Opening       |
| `intent:credit_card`                       | The chat's intent is credit_card                       | 92%     | 96%        | 50%    | Opening       |
| `intent:shopping_cart`                     | The chat's intent is shopping_cart                     | 94%     | 100%       | 50%    | Opening       |
| `intent:search_results`                    | The chat's intent is search_results                    | 90%     | 96%        | 50%    | Opening       |
| `intent:slow_speed`                        | The chat's intent is slow_speed                        | 94%     | 100%       | 50%    | Opening       |

The intent questions have 3 to 38 yes chats among the 1,004 test chats, so one more hit moves the smallest by several points.

### The agent's actions: 26 of 34 work from the opening

| Question                     | Asks                                              | Opening | Whole chat | Chance | Answer from |
|------------------------------|---------------------------------------------------|---------|------------|--------|-------------|
| `action:verify-identity`     | The agent runs verify-identity                    | 82%     | 92%        | 50%    | Opening     |
| `action:ask-the-oracle`      | The agent runs ask-the-oracle                     | 84%     | 91%        | 50%    | Opening     |
| `action:validate-purchase`   | The agent runs validate-purchase                  | 89%     | 96%        | 50%    | Opening     |
| `action:make-password`       | The agent runs make-password                      | 91%     | 99%        | 50%    | Opening     |
| `action:promo-code`          | The agent runs promo-code                         | 82%     | 90%        | 50%    | Opening     |
| `action:subscription-status` | The agent runs subscription-status                | 89%     | 93%        | 50%    | Opening     |
| `action:offer-refund`        | The agent runs offer-refund                       | 78%     | 88%        | 50%    | Whole chat  |
| `action:make-purchase`       | The agent runs make-purchase                      | 78%     | 95%        | 50%    | Whole chat  |
| `action:record-reason`       | The agent runs record-reason                      | 82%     | 88%        | 50%    | Opening     |
| `action:enter-details`       | The agent runs enter-details                      | 82%     | 90%        | 50%    | Opening     |
| `action:shipping-status`     | The agent runs shipping-status                    | 81%     | 92%        | 50%    | Opening     |
| `action:update-order`        | The agent runs update-order                       | 80%     | 90%        | 50%    | Opening     |
| `action:pull-up-account`     | The agent runs pull-up-account                    | 88%     | 94%        | 50%    | Opening     |
| `action:update-account`      | The agent runs update-account                     | 83%     | 91%        | 50%    | Opening     |
| `action:send-link`           | The agent runs send-link                          | 83%     | 91%        | 50%    | Opening     |
| `action:notify-team`         | The agent runs notify-team                        | 83%     | 91%        | 50%    | Opening     |
| `action:membership`          | The agent runs membership                         | 84%     | 92%        | 50%    | Opening     |
| `action:search-faq`          | The agent runs search-faq                         | 87%     | 94%        | 50%    | Opening     |
| `action:try-again`           | The agent runs try-again                          | 85%     | 91%        | 50%    | Opening     |
| `action:log-out-in`          | The agent runs log-out-in                         | 92%     | 100%       | 50%    | Opening     |
| `action:instructions`        | The agent runs instructions                       | 91%     | 95%        | 50%    | Opening     |
| `action:search-jeans`        | The agent runs search-jeans                       | 89%     | 98%        | 50%    | Opening     |
| `action:search-shirt`        | The agent runs search-shirt                       | 90%     | 98%        | 50%    | Opening     |
| `action:search-boots`        | The agent runs search-boots                       | 97%     | 100%       | 50%    | Opening     |
| `action:search-jacket`       | The agent runs search-jacket                      | 88%     | 94%        | 50%    | Opening     |
| `action:search-pricing`      | The agent runs search-pricing                     | 80%     | 86%        | 50%    | Whole chat  |
| `action:search-timing`       | The agent runs search-timing                      | 88%     | 94%        | 50%    | Opening     |
| `action:search-membership`   | The agent runs search-membership                  | 80%     | 86%        | 50%    | Whole chat  |
| `action:search-policy`       | The agent runs search-policy                      | 82%     | 86%        | 50%    | Opening     |
| `action:select-faq`          | The agent runs select-faq                         | 86%     | 92%        | 50%    | Opening     |
| `first_action`               | Which action the agent runs first                 | 23%     | 21%        | 5%     | Weak        |
| `identity_check`             | The strongest identity check the agent runs       | 78%     | 90%        | 25%    | Whole chat  |
| `agent_changes_something`    | The agent changes something, not only looks it up | 74%     | 86%        | 50%    | Whole chat  |
| `action_count`               | How many actions the agent runs: 1-2, 3, 4 or 5+  | 46%*    | 55%*       | 35%*   | Weak        |

\* Ordered levels, so plain accuracy, and the chance column shows the most common level.

### Details the agent will need: 7 of 10 work from the opening

| Question               | Asks                                                | Opening | Whole chat | Chance | Answer from |
|------------------------|-----------------------------------------------------|---------|------------|--------|-------------|
| `needs:account_id`     | The customer will have to give their account_id     | 85%     | 93%        | 50%    | Opening     |
| `needs:amount`         | The customer will have to give their amount         | 69%     | 93%        | 50%    | Whole chat  |
| `needs:email`          | The customer will have to give their email          | 86%     | 98%        | 50%    | Opening     |
| `needs:name`           | The customer will have to give their name           | 86%     | 96%        | 50%    | Opening     |
| `needs:order_id`       | The customer will have to give their order_id       | 89%     | 96%        | 50%    | Opening     |
| `needs:phone`          | The customer will have to give their phone          | 77%     | 98%        | 50%    | Whole chat  |
| `needs:pin_number`     | The customer will have to give their pin_number     | 88%     | 95%        | 50%    | Opening     |
| `needs:street_address` | The customer will have to give their street_address | 80%     | 100%       | 50%    | Whole chat  |
| `needs:username`       | The customer will have to give their username       | 89%     | 95%        | 50%    | Opening     |
| `needs:zip_code`       | The customer will have to give their zip_code       | 84%     | 98%        | 50%    | Opening     |

### Values the agent enters: 1 of 7 reaches 80% with the whole chat

| Question          | Asks                                            | Opening | Whole chat | Chance | Answer from |
|-------------------|-------------------------------------------------|---------|------------|--------|-------------|
| `escalation_team` | Which company_team the agent enters, or none    | 73%     | 89%        | 25%    | Whole chat  |
| `account_change`  | Which account_slotval the agent enters, or none | 72%     | 75%        | 20%    | Weak        |
| `order_change`    | Which change_option the agent enters, or none   | 54%     | 66%        | 14%    | Weak        |
| `order_option`    | Which order_slotval the agent enters, or none   | 68%     | 76%        | 12%    | Weak        |
| `recorded_reason` | Which reason_slotval the agent enters, or none  | 46%     | 73%        | 9%     | Weak        |
| `refund_method`   | Which refund_target the agent enters, or none   | 49%     | 50%        | 20%    | Weak        |
| `product`         | Which product the agent enters, or none         | 11%     | 62%        | 6%     | Weak        |

### Facts about the customer: the text predicts both weakly

| Question                 | Asks                                         | Opening | Whole chat | Chance | Answer from |
|--------------------------|----------------------------------------------|---------|------------|--------|-------------|
| `member_level`           | The customer's membership level              | 26%     | 44%        | 25%    | Weak        |
| `payment_method_on_file` | How the customer paid, where ABCD records it | 45%     | 53%        | 25%    | Weak        |

### The chat's length: the text predicts it weakly

| Question      | Asks                                                         | Opening | Whole chat | Chance | Answer from |
|---------------|--------------------------------------------------------------|---------|------------|--------|-------------|
| `chat_length` | How many turns the chat takes: under 18, 18-21, 22-26 or 27+ | 41%*    | 52%*       | 28%*   | Weak        |

\* Ordered levels, so plain accuracy, and the chance column shows the most common level.

## Appendix B: our 47 questions on ABCD's 1,004 test chats

"Ours" is the pipeline trained only on synthetic tickets. "Retrained" is the same model type trained on ABCD's 8,034 training chats.

### Choice questions: ours beat guessing on 3 of 5, and retrained on all 5

Accuracy, against always guessing the most common answer.

| Question             | Test chats | Most common | Ours, opening | Ours, whole chat | Retrained, opening | Retrained, whole chat |
|----------------------|------------|-------------|---------------|------------------|--------------------|-----------------------|
| `category`           | 424        | 54%         | 73%           | 78%              | 93%                | 99%                   |
| `category_reordered` | 424        | 54%         | 74%           | 78%              | 93%                | 99%                   |
| `message_type`       | 965        | 41%         | 27%           | 23%              | 88%                | 96%                   |
| `primary_team`       | 915        | 54%         | 67%           | 67%              | 89%                | 97%                   |
| `journey_stage`      | 490        | 67%         | 10%           | 36%              | 92%                | 99%                   |

### Two-way yes/no questions: ours score a median 68% from the opening, retrained 93%

Balanced accuracy, where chance is 50%.

| Question                 | Test chats | Ours, opening | Ours, whole chat | Retrained, opening | Retrained, whole chat |
|--------------------------|------------|---------------|------------------|--------------------|-----------------------|
| `refund_requested`       | 658        | 63%           | 78%              | 91%                | 100%                  |
| `mentions_billing_issue` | 772        | 66%           | 74%              | 94%                | 98%                   |
| `mentions_account_issue` | 912        | 80%           | 75%              | 94%                | 98%                   |
| `mentions_bug`           | 915        | 53%           | 62%              | 93%                | 99%                   |
| `needs_staff_action`     | 1004       | 54%           | 56%              | 74%                | 86%                   |
| `refund_implied`         | 767        | 53%           | 57%              | 92%                | 98%                   |
| `charge_amount_given`    | 1004       | 88%           | 89%              | 89%                | 93%                   |
| `cannot_log_in`          | 915        | 70%           | 73%              | 95%                | 99%                   |
| `security_related`       | 915        | 72%           | 76%              | 95%                | 99%                   |
| `identity_details_given` | 1004       | 77%           | 71%              | 98%                | 97%                   |
| `sensitive_data_present` | 1004       | —             | 90%              | —                  | 100%                  |

`sensitive_data_present` has no yes in any opening, so only the whole chat is graded.

### One-way questions: we rarely give a false yes, but miss many true ones

The share of test chats we answer right.

| Question                   | ABCD knows       | Test chats | Ours, opening | Ours, whole chat |
|----------------------------|------------------|------------|---------------|------------------|
| `mentions_feature_request` | Only no          | 1004       | 99%           | 92%              |
| `issue_count`              | Always one issue | 1004       | 62%           | 1%               |
| `duplicate_charge`         | Only no          | 951        | 100%          | 100%             |
| `chargeback_mentioned`     | Only no          | 984        | 100%          | 100%             |
| `wants_new_payment_method` | Only no          | 968        | 100%          | 99%              |
| `existing_customer`        | Only yes         | 329        | 15%           | 54%              |
| `has_account`              | Only yes         | 214        | 61%           | 87%              |

### 24 questions have no ABCD label

`bug_severity`, `has_reproducible_steps`, `frustration`, `priority`, `time_pressure`, `needs_more_info`, `payment_method_used`, `order_date_given`, `problem_after_update`, `blames_update`, `error_message_given`, `device_or_browser_given`, `tried_fix_themselves`, `possible_widespread_issue`, `feature_request_strength`, `sentiment`, `politeness`, `frustration_rising`, `threatens_to_leave`, `churn_risk`, `asks_for_human`, `prior_contact`, `likely_apple_user`, `issues_related`.

## Appendix C: I mapped ABCD's intents to our answers by hand

The groups cover ABCD's 55 intents, plus `status_questions`, which 30 chats use but the ontology leaves out. A question skips any intent that fits none of its answers.

| Group                        | Intents | Members                                                                                                                                                                                                                          | `primary_team`       |
|------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| Can't get in                 | 3       | `recover_password`, `recover_username`, `reset_2fa`                                                                                                                                                                              | account_support      |
| Profile changes              | 3       | `manage_change_address`, `manage_change_name`, `manage_change_phone`                                                                                                                                                             | account_support      |
| Money problems               | 10      | `bad_price_competitor`, `bad_price_yesterday`, `manage_dispute_bill`, `mistimed_billing_already_returned`, `mistimed_billing_never_bought`, `refund_initiate`, `refund_status`, `refund_update`, `status_credit_missing`, `status_mystery_fee` | billing_and_payments |
| Billing with no problem      | 8       | `manage_extension`, `manage_pay_bill`, `manage_payment_method`, `status_active`, `status_due_amount`, `status_due_date`, `status_payment_method`, `status_questions`                                                              | billing_and_payments |
| Site problems                | 3       | `search_results`, `shopping_cart`, `slow_speed`                                                                                                                                                                                  | engineering          |
| Returns                      | 3       | `return_color`, `return_size`, `return_stain`                                                                                                                                                                                    | other                |
| Orders                       | 12      | `manage`, `manage_cancel`, `manage_create`, `manage_downgrade`, `manage_upgrade`, `missing`, `out_of_stock_general`, `out_of_stock_one_item`, `status`, `status_delivery_time`, `status_quantity`, `status_shipping_question`     | other                |
| Product and policy questions | 8       | `boots`, `jacket`, `jeans`, `membership`, `policy`, `pricing`, `shirt`, `timing`                                                                                                                                                 | other                |
| Unclear                      | 6       | `cost`, `credit_card`, `promo_code_invalid`, `promo_code_out_of_date`, `status_service_added`, `status_service_removed`                                                                                                          | skipped              |

- **`category` follows the team:** account, billing or bug_report. Our category has no answer for "other", so only 424 test chats are graded.
- **`journey_stage`:** the 4 clothing queries and both out-of-stock intents are browsing, and `shopping_cart` is checkout. Intents about a past order are post_purchase.
- **`message_type`:** product questions and status checks are questions. Changes, returns and lost logins are requests. Site problems, missing items and disputed bills are problem reports.
- **Yes/no questions** take yes from one set of groups and no from the groups that rule it out. For example, `cannot_log_in` is yes for "Can't get in" and no for every other group except "Unclear".
- **4 questions can only be no,** because ABCD never raises them: `mentions_feature_request`, `wants_new_payment_method`, `duplicate_charge` and `chargeback_mentioned`. Each skips the intents where a yes might hide, such as `manage_dispute_bill`.
- **2 questions can only be yes:** `existing_customer` for intents about a past order, and `has_account` for account and subscription intents.
- **4 questions read the chat itself:** `needs_staff_action` is yes when the agent changes something. `identity_details_given`, `sensitive_data_present` and `charge_amount_given` read which details the customer typed.
- **`issue_count` is always one,** because every ABCD chat holds one issue.

## Appendix D: method, and one command reruns it all

- **Data:** `abcd_v1.1.json.gz` holds 8,034 training, 1,004 dev and 1,004 test chats. The dev chats are unused. `ontology.json` lists the 10 flows, 55 intents, 30 actions and each action's values. `kb.json` lists each intent's playbook.
- **Inputs:** each model reads the opening or every customer message. Messages are joined one per line, so our splitter cuts at message ends.
- **Our 9 models:** loaded from `models/`, trained only on synthetic tickets, and run through `Pipeline.run` on the 1,004 test chats. `rules.answer` builds the 47 answers.
- **Retrained and new questions:** one head per question, of the same type as our 7 trained models. Word 1–2-grams and character 2–5-grams go through TF-IDF into a logistic regression with balanced class weights. Each head fits on the training chats and is graded on the test chats.
- **Scores:** yes/no and choice questions get balanced accuracy, and chance is 1 over the number of answers. Ordered levels get plain accuracy.
- **Typed details:** ABCD masks most details as placeholders such as `<order_id>`, so a placeholder marks a typed detail. Names stay unmasked, so a typed name is a match on the chat's customer name or first name.
- **Entered values:** the value an agent typed is matched to ABCD's option list with `difflib` at a 0.6 cutoff, and only from actions that take that value.
- **Playbook measures:** "only playbook actions" means every action is in the intent's playbook. "Keeps its order" also needs them in playbook order. "Every step" means every playbook action appears, in order.
- **Entity finder:** it runs on each customer message, and a find counts as real when that message holds the detail's placeholder.
- **Rerun:** download ABCD as the docstring of `micro_llm/abcd.py` shows, then run `python -m micro_llm.abcd`. It takes 109 s on 10 cores and writes `data/abcd_analysis.json`. `tests/test_abcd.py` holds 7 tests.
