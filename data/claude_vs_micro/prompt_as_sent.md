# The prompt Claude Haiku 5.5 saw in the ABCD comparison

`save_prompt.py` rebuilt it from `compare.py` without calling Claude, using the run's arguments (`300 16 haiku`).

- **Why it matches the run:** this folder's `compare.py` is byte-identical to the copy that ran on 2026-10-26.
- **Its inputs predate the run:** on 2026-10-26, `micro_llm/abcd.py` had last changed at 10:12, before the run, and `micro_llm/questions.json` on 2026-10-24.
- **How each chat was sent:** the user message went to the CLI on stdin, once per chat, with up to 2 retries and 16 chats at a time.

## Command

```
claude -p --model haiku --effort low --output-format json --tools "" --strict-mcp-config --no-session-persistence --setting-sources "" --system-prompt <SYSTEM>
```

## System prompt (`SYSTEM`)

```
You triage support chats for an online clothing store. Reply only with the JSON object asked for, with no other text.
```

## User message (`prompt(c)`), with `{opening}` standing for the customer's opening

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

## The full user message for the first sampled chat (ABCD id 8354)

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
Hi there just checking on the status of my refund.
>>>
```
