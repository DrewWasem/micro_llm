"""Rebuild the prompt compare.py sent Claude, without calling Claude, and save it as prompt_as_sent.md.

python3 -I save_prompt.py <repo> <out_dir>
Runs only compare.py's code above `def ask(`, which loads ABCD and builds SYSTEM, QUESTIONS,
prompt() and CMD, with the run's own arguments: 300 chats, 16 workers, model haiku.
"""

import sys
from pathlib import Path

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
source = Path(__file__).with_name("compare.py").read_text()
head = source.split("\ndef ask(", 1)[0]

sys.argv = ["compare.py", str(REPO), str(OUT), "300", "16", "haiku"]
ns = {"__name__": "compare_head"}
exec(compile(head, "compare.py", "exec"), ns)

system, questions, cmd = ns["SYSTEM"], ns["QUESTIONS"], ns["CMD"]
first = ns["sample"][0]
flags = " ".join(
    '""' if arg == "" else ("<SYSTEM>" if arg == system else arg) for arg in cmd
)
template = f"{questions}\n\nThe customer's opening message:\n<<<\n{{opening}}\n>>>"
opening = ns["abcd"].customer_text(first, "opening")
assert ns["prompt"](first) == template.replace("{opening}", opening)

out = OUT / "prompt_as_sent.md"
out.write_text(
    "\n".join(
        [
            "# The prompt Claude Haiku 5.5 saw in the ABCD comparison",
            "",
            "`save_prompt.py` rebuilt it from `compare.py` without calling Claude, using the run's arguments (`300 16 haiku`).",
            "",
            "- **Why it matches the run:** this folder's `compare.py` is byte-identical to the copy that ran on 2026-10-26.",
            "- **Its inputs predate the run:** on 2026-10-26, `micro_llm/abcd.py` had last changed at 10:12, before the run, and `micro_llm/questions.json` on 2026-10-24.",
            "- **How each chat was sent:** the user message went to the CLI on stdin, once per chat, with up to 2 retries and 16 chats at a time.",
            "",
            "## Command",
            "",
            "```",
            flags,
            "```",
            "",
            "## System prompt (`SYSTEM`)",
            "",
            "```",
            system,
            "```",
            "",
            "## User message (`prompt(c)`), with `{opening}` standing for the customer's opening",
            "",
            "```",
            template,
            "```",
            "",
            f"## The full user message for the first sampled chat (ABCD id {first['convo_id']})",
            "",
            "```",
            ns["prompt"](first),
            "```",
            "",
        ]
    )
)
print(f"wrote {out}")
