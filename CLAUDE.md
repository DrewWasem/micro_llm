# CLAUDE.md — micro_llm

**micro_llm is a small language model project. Nothing is built yet, and the scope is still open.**

The repo is public: https://github.com/DrewWasem/micro_llm

## Set the scope before writing code

- Record the scope with `/remember` as a decision when it is set.
- Then rewrite the first line of this file and of `README.md` to say what the project does.
- Bootstrap a domain SME at the same time: `~/claude_projects/scripts/bootstrap-sme-system.sh . <sme-name> "<domain>"`.

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
