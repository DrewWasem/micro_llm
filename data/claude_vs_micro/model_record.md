# Model record for the Claude run in this folder

Facts about the Claude side of `compare.py` that the result files do not hold themselves.

- **Date of the run:** 2026-10-26, from the file times of `claude_haiku.jsonl` and `results_haiku.json`.
- **CLI version:** `claude --version` prints `2.1.310 (Claude Code)`.
- **Model:** `compare.py` passes `--model haiku`. The cached rows in `claude_haiku.jsonl` don't store the model name, so I checked the alias separately on 2026-10-26: `claude -p --model haiku --output-format json` under CLI 2.1.310 reported `claude-haiku-5-5` as its `modelUsage` key.
- **Thinking:** `claude --help` lists `--effort <level>` and no option that turns extended thinking off. The run used `--effort low`.
- **Settings and concurrency:** the full command, the system prompt, the retry count and the number of concurrent calls are in `compare.py` (`CMD`, `SYSTEM`, the retry loop and `WORKERS`). The run passed `300 16 haiku` as sample size, workers and model, so 16 calls ran at once. The first line of `run_haiku.log` records it: "asking haiku about 300 chats (0 cached), 16 at a time".
- **The exact prompt:** `prompt_as_sent.md` holds it, rebuilt without calling Claude by `save_prompt.py`.
