# micro_llm

**micro_llm builds small specialist models that each answer a few narrow questions about a support ticket, and runs them in parallel.**

The first use case is support-ticket triage: a rule-based splitter cuts a ticket into pieces, a router sends each piece to the right specialist models, and code rules turn their answers into 47 ticket-level answers.

## How it works

| Wave | What runs                                                    | Kind of model                       |
|------|--------------------------------------------------------------|-------------------------------------|
| 1    | Splitter cuts the ticket into pieces; entity finder runs     | Rules (regex)                       |
| 2    | Router and tone model read every piece                       | TF-IDF + logistic regression        |
| 3    | Billing, account, bug and feature specialists read the pieces routed to them; the linker reads each pair of issues | TF-IDF + logistic regression |
| 4    | 21 ticket answers come from rules over the model answers     | Rules                               |

Training data is synthetic. Code samples every label first, then the Claude CLI writes text that shows those labels, so the truth is known before any text exists.

## Steps 2 and 3: decide, then act

The 47 answers feed rule deciders that pick actions: a refund review, login help and a feature vote. Each action works through stand-in tools whose records are in `examples/support_ticket_98423/systems.json`.

- **An unknown account runs only verification.** A link goes to the email on file, never to an address typed into the ticket.
- **The other steps wait in SQLite** until a message arrives from the verified account, then run once.
- **No tool moves money.** A repeated charge becomes a refund request for staff to approve.

## Results on synthetic tickets (v1, 2026-10-24)

479 tickets written by Claude Haiku for $6.79, with 96 held out by ticket. Each accuracy sits beside the majority-class baseline, the score of always giving the most common answer.

| Answers come from                  | Questions | Accuracy | Baseline |
|------------------------------------|-----------|----------|----------|
| 7 trained models                   | 25        | 0.916    | 0.718    |
| Code rules over the model answers  | 15        | 0.942    | 0.593    |
| Entity rules                       | 7         | 0.969    | 0.798    |

- **Speed:** 3.9 ms median per ticket on a laptop CPU, including 15 model runs for the sample ticket.
- **Weakest:** the tone levels (frustration 0.71 against a 0.66 baseline) and the splitter on tickets with no punctuation (91% of pieces cut exactly).

## Run it

Requires Python 3.11+ and the [Claude CLI](https://docs.claude.com/en/docs/claude-code) for data generation.

```bash
pip install numpy==2.3.5 scikit-learn==1.7.2 pytest
python3 -m micro_llm.generate --tickets 480 --batch 2 --workers 12   # writes data/tickets.jsonl
python3 -m micro_llm.train                                           # saves models/*.pkl
python3 -m micro_llm.evaluate                                        # scores held-out tickets
python3 -m micro_llm.run "Hi, I was charged twice for order #123."
python3 -m micro_llm.resolve                                         # steps 1-3 on the sample ticket
python3 -m micro_llm.resolve --thread 1 --account A-1001 "I confirmed it's me."
python3 -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
