"""Answers the 47 questions for one ticket and prints how it was split and routed.

python -m micro_llm.run "Hi, I was charged twice for order #123."
python -m micro_llm.run            # the sample ticket
"""

import json
import sys

from micro_llm.config import SAMPLE_TICKET
from micro_llm.pipeline import Pipeline


def main():
    text = (
        " ".join(sys.argv[1:])
        or json.loads(SAMPLE_TICKET.read_text())["ticket"]["text"]
    )
    pipeline = Pipeline()
    pipeline.run("warm up")
    result = pipeline.run(text)
    for p in result["pieces"]:
        print(
            f"{p['id']} {p['kind']:8} {p['team']:21} {'+'.join(p['models']):28} {p['text']}"
        )
    for e in result["entities"]:
        print(f"entity {e['kind']}: {e['text']}")
    for pair in result["pairs"]:
        print(
            f"linker {pair['a']}-{pair['b']}: issues_related {pair['issues_related']}"
        )
    for q, a in result["answers"].items():
        value = a if isinstance(a, float) else a.get("answer", a.get("level"))
        print(f"{q:26} {value}")
    print(f"{result['model_runs']} model runs, {result['ms']}")


if __name__ == "__main__":
    main()
