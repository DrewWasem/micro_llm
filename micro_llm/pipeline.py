"""Runs one ticket through the models in waves, as `model_plan.json` lays out.

Wave 1 splits the ticket and finds entities with rules. Wave 2 runs the router and the tone
model on every piece. Wave 3 sends each issue piece to its specialists and each pair of issue
pieces to the linker. Rules then compute the 47 ticket answers.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations

from micro_llm import entities, rules, splitter
from micro_llm.models import load
from micro_llm.routing import specialists

# A piece goes to every team the router gives at least this chance, so one piece can reach two specialists
ROUTE_MIN = 0.3


def routes(router):
    """The specialists that read one piece, given the router's answers for it."""
    kind = router["piece_kind"]["p"]
    if kind.get("problem", 0) + kind.get("request", 0) < 0.5:
        return []
    k = "problem" if kind.get("problem", 0) >= kind.get("request", 0) else "request"
    names = []
    for team, p in router["primary_team"]["p"].items():
        if p >= ROUTE_MIN:
            names += [n for n in specialists(k, team) if n not in names]
    return names


class Pipeline:
    def __init__(self, models=None, workers=8):
        self.models = models or load()
        self.pool = ThreadPoolExecutor(workers)

    def run(self, text):
        clock = time.perf_counter
        t0 = clock()
        pieces = splitter.split(text) or [splitter.Piece("P1", text, 0, len(text))]
        found = entities.find(text)
        texts = [p.text for p in pieces]
        t1 = clock()

        router = self.pool.submit(self.models["router"].predict, texts)
        tone = self.pool.submit(self.models["tone"].predict, texts)
        outs = [
            {"router": r, "tone": t} for r, t in zip(router.result(), tone.result())
        ]
        t2 = clock()

        # One run per piece and model, so a piece can reach several models and a model several pieces
        jobs = [(i, name) for i, o in enumerate(outs) for name in routes(o["router"])]
        issues = [i for i, o in enumerate(outs) if rules.issue_p(o) >= 0.5]
        pairs = list(combinations(issues, 2))
        runs = [
            self.pool.submit(self.models[name].predict, [texts[i]]) for i, name in jobs
        ]
        links = [
            self.pool.submit(self.models["linker"].predict, [(texts[a], texts[b])])
            for a, b in pairs
        ]
        for (i, name), run in zip(jobs, runs):
            outs[i][name] = run.result()[0]
        related = [link.result()[0]["issues_related"] for link in links]
        t3 = clock()

        kinds = [
            {e.kind for e in found if e.start < p.end and p.start < e.end}
            for p in pieces
        ]
        apple = any(entities.is_apple(e) for e in found)
        answers = rules.answer(outs, related, kinds, apple)
        t4 = clock()
        ms = lambda a, b: round((b - a) * 1000, 2)
        return {
            "answers": answers,
            "pieces": [
                {
                    "id": p.id,
                    "text": p.text,
                    "kind": o["router"]["piece_kind"]["answer"],
                    "team": o["router"]["primary_team"]["answer"],
                    "models": list(o),
                }
                for p, o in zip(pieces, outs)
            ],
            "entities": [asdict(e) for e in found],
            "pairs": [
                {"a": pieces[a].id, "b": pieces[b].id, "issues_related": p}
                for (a, b), p in zip(pairs, related)
            ],
            "model_runs": 2 * len(pieces) + len(jobs) + len(pairs),
            "ms": {
                "split_and_entities": ms(t0, t1),
                "router_and_tone": ms(t1, t2),
                "specialists_and_linker": ms(t2, t3),
                "rules": ms(t3, t4),
                "total": ms(t0, t4),
            },
            "outputs": outs,
        }
