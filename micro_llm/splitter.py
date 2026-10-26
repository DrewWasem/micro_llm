"""Cuts a ticket into pieces with rules: sentence ends, then ', and' / ', but' / ', plus' joins.

A greeting that opens a sentence, such as "Hi,", is dropped because it asks for nothing.
"""

import re
from dataclasses import dataclass

# A sentence ends at . ! ? followed by a space or the end; "$49.99" and "example.com" stay whole
SENTENCE = re.compile(r"(?:[^.!?\n]|[.!?]+(?=[^\s.!?]))+(?:[.!?]+|$)", re.M)
GREETING = re.compile(
    r"(?:hi|hello|hey|good\s(?:morning|afternoon|evening)|dear\s[^,\n]+)"
    r"(?:\s(?:there|team|all|support|folks))?\s*[,!.]\s*",
    re.I,
)
JOIN = re.compile(r",\s+(?:and|but|plus)\s+", re.I)


@dataclass(frozen=True)
class Piece:
    id: str
    text: str
    start: int
    end: int


def split(text):
    """The ticket's pieces in order, each with its character span."""
    spans = []
    for m in SENTENCE.finditer(text):
        start, end = m.start(), m.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        greeting = GREETING.match(text, start, end)
        if greeting:
            start = greeting.end()
        cut = start
        for join in JOIN.finditer(text, start, end):
            spans.append((cut, join.start()))
            cut = join.end()
        spans.append((cut, end))
    spans = [(s, e) for s, e in spans if re.search(r"\w", text[s:e])]
    return [Piece(f"P{i}", text[s:e], s, e) for i, (s, e) in enumerate(spans, 1)]
