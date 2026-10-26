# CLAUDE.md — micro_llm

**micro_llm builds small specialist models that each answer a few narrow questions about a support ticket, and runs them in parallel the way Jev's Playground splits a ticket.**

The repo is public: https://github.com/DrewWasem/micro_llm

## The first use case is support-ticket triage with 9 models

- **The scope was set on 2026-10-24:** 9 model types answer the 47 questions in `examples/support_ticket_98423/playground_questions.json`. The plan is `model_plan.json` beside it.
- **2 models are rules and 7 are trained:** the splitter and entity finder use regexes. The other 7 are TF-IDF + logistic regression, trained on tickets the Claude CLI writes from labels sampled in code.
- **Steps 2 and 3 were added on 2026-10-25:** rule deciders pick actions from the 47 answers, and actions run through stand-in tools in `examples/support_ticket_98423/systems.json`.
- **An unknown account runs only verification:** a link goes to the email on file, and the other steps wait until a message arrives from the verified account.
- **C-MAPSS is parked, not dropped.**
- **Domain rules live in the SME:** `.claude/smes/ticket-triage-models/knowledge/`.

## Run it from the repo root

| Step                       | Command                                                              | Note                                                |
|----------------------------|----------------------------------------------------------------------|-----------------------------------------------------|
| Write training tickets     | `python3 -m micro_llm.generate --tickets 480 --batch 2 --workers 12` | Haiku, ~$0.013 per ticket; batches cache in `data/` |
| Train the 7 models         | `python3 -m micro_llm.train`                                         | Saves to `models/`, which is gitignored             |
| Score held-out tickets     | `python3 -m micro_llm.evaluate`                                      | Writes `models/eval.json`                           |
| Answer one ticket          | `python3 -m micro_llm.run "ticket text"`                             | With no text it runs the sample ticket              |
| Decide and act (steps 2–3) | `python3 -m micro_llm.resolve "ticket text"`                         | Unknown account: only verification runs             |
| Continue after verifying   | `python3 -m micro_llm.resolve --thread 1 --account A-1001 "text"`    | Runs the steps that waited in `data/threads.sqlite` |
| Analyze ABCD's real chats  | `python3 -m micro_llm.abcd`                                          | Download ABCD first, per the module's docstring     |
| Test                       | `python3 -m pytest -q`                                               | The pipeline test skips until models exist          |

## Nothing under `.claude/` or `.tasks/` may reach GitHub

- Both folders are gitignored because the repo is public. They hold the memory tree, task tickets, hooks and subagent output.
- Run `git status` before every commit and confirm only project files are staged.
- Never commit secrets, `.env` files or model weights. `.gitignore` blocks the common weight formats.

## The Claude Code setup lives on this machine only

- **Settings baseline v1.0.0** in `.claude/settings.json`. Refresh with `/skill-sync settings`.
- **Memory tree** in `.claude/memory/`, loaded index-only by the session hook. Use `/recall` and `/remember`.
- **Task tickets** in `.tasks/`, managed by `/plan-tasks`, `/next` and `/done`.
- **14 commands, 5 agents and 3 skills**, identical to the workspace's canonical copies on 2026-10-23.
- **Secret scan** runs before every Write and Edit through `.claude/hooks/secret-scan.sh`.

## Conventions

- Every implementation needs a verification loop: a test, lint or type-check.
- Commit prefixes: FEAT, FIX, ENH, PERF, REFACTOR, TEST, DOC, STYLE, CHORE, WIP.

## Jev estimate: about 0.3 s and under a tenth of a cent per ticket

Rough estimates for the sample ticket in `examples/support_ticket_98423/`. Replace them with the Playground's measured numbers.

| Plan                                 | Requests | Input tokens  | Cost              | Wall time   |
|--------------------------------------|----------|---------------|-------------------|-------------|
| Docs' 5 questions, whole ticket      | 1        | ~600–900      | ~$0.00002–0.00004 | ~0.1–0.35 s |
| All 47 questions, whole ticket       | 1        | ~2,700–6,000  | ~$0.0001–0.0003   | ~0.1–0.35 s |
| 4 pieces + whole ticket, in parallel | 5        | ~7,200–15,000 | ~$0.0003–0.0006   | ~0.2–0.5 s  |

- **Low end:** my fit to 4 billed docs examples, 241 + 26 per question + characters ÷ 4.
- **High end:** an audit billed a fixed 122 tokens per extra question (Rafe & Das 2026).
- **Price:** $0.042 per 1M input tokens. Output is free.
- **One call takes ~0.1–0.35 s:** from TypeSafe's ~0.11 s mean to an audit's 0.33 s p90.
- **Drew saw sub-second answers** in the Playground on 2026-10-24.
- **Extra questions add tokens, not time,** per TypeSafe's docs.
- **Parallel requests finish with the slowest one,** so 5 requests take about one call's time.
- **Splitting is code, not Jev:** 4.6 µs per ticket. The pieces are in `context_pieces.json`.
