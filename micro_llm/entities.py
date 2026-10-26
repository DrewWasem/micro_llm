"""Finds exact spans with rules: order numbers, money, dates, devices, error text and personal data.

Jev can't extract text, so this is the one step that returns words from the ticket itself.
"""

import re
from dataclasses import dataclass

DAYS = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
PATTERNS = {
    "order_number": (
        r"#\s?[A-Z]{0,3}-?\d{3,}\b"
        r"|\b(?:order|ref|reference|confirmation|invoice|booking)\s*(?:no\.?|number|num|id)?\s*[:#]?\s*[A-Z]{0,3}-?\d{4,}\b"
    ),
    "money": r"[$€£]\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars|bucks|usd|eur|euros|gbp|pounds)\b",
    "date": (
        rf"\b(?:(?:last|this|next|on)\s+)?{DAYS}\b"
        r"|\b(?:yesterday|today|last night|this morning|last week|last month)\b"
        rf"|\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
        rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}(?=\W|$)"
        r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b|\b\d+\s+(?:days?|weeks?)\s+ago\b"
    ),
    "device": (
        r"\b(?:iphone|ipad|macbook|imac|mac|ios|ipados|macos|android|pixel|samsung|galaxy|windows|linux"
        r"|chromebook|chrome|firefox|safari|microsoft edge|opera)(?:\s?\d+)?\b"
    ),
    "payment_method": (
        r"\b(?:apple pay|google pay|paypal|venmo|klarna|visa|mastercard|amex|american express"
        r"|debit card|credit card|bitcoin|crypto)\b"
    ),
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "phone": r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)",
    "card_number": r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
    # A password must hold a digit or symbol, so "my password is wrong" is not one
    "password": r"\bpassword\s*(?:is|:)\s*(?=\S*[\d!@#$%^&*])\S+",
    "username": r"\b(?:username|user name|user id|handle)\s*(?::\s*@?|(?:is\s+)?@)[\w.-]+",
    "error_text": r"\berror(?:\s+code)?\s*[:#]?\s*[A-Z]*-?\d{2,}\b",
}
COMPILED = {kind: re.compile(p, re.I) for kind, p in PATTERNS.items()}
# Case-sensitive, so "3 times to place an order" is not a street address
COMPILED["address"] = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Dr|Drive|Way|Court|Ct|Place|Pl)\b\.?"
)
QUOTES = [
    re.compile(r"[\"“]([^\"”\n]{3,120})[\"”]"),
    re.compile(
        r"(?:(?<=\s)|(?<=^)|(?<=[:(]))['‘]([^'’\n]{3,120})['’](?![A-Za-z])", re.M
    ),
]
ERROR_WORDS = re.compile(
    r"\b(?:error|errors|says|said|message|shows|showing|displays|popup|pop-up|warning|failed|fails|exception)\b",
    re.I,
)
# The sign-off words ignore case and the name must be capitalised, so "Thanks in advance" has no name
SIGNOFF = re.compile(
    r"(?i:thanks|thank you|thx|cheers|regards|best|sincerely)[\s,!.]*(?:[-–—]\s*)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)[\s.!]*$"
)
MY_NAME = re.compile(r"\bmy name is\s+([A-Za-z][a-z]+)", re.I)
NOT_NAMES = {
    "again",
    "guys",
    "team",
    "all",
    "everyone",
    "so",
    "much",
    "advance",
    "folks",
    "you",
    "anyway",
    "though",
    "lot",
    "support",
    "bye",
}
APPLE = re.compile(
    r"^(?:iphone|ipad|macbook|imac|mac|ios|ipados|macos|safari|apple pay)", re.I
)


@dataclass(frozen=True)
class Entity:
    kind: str
    text: str
    start: int
    end: int


def luhn(digits):
    total = 0
    for i, d in enumerate(reversed(digits)):
        d = int(d) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def is_apple(entity):
    return APPLE.match(entity.text) is not None


def find(text):
    """Every entity in the ticket, sorted by position. Later kinds never overlap earlier ones."""
    found = []

    def add(kind, start, end):
        if not any(start < e.end and e.start < end for e in found):
            found.append(Entity(kind, text[start:end], start, end))

    for m in COMPILED["card_number"].finditer(text):
        if luhn(re.sub(r"\D", "", m.group())):
            add("card_number", m.start(), m.end())
    for kind, pattern in COMPILED.items():
        if kind != "card_number":
            for m in pattern.finditer(text):
                add(kind, m.start(), m.end())
    for quote in QUOTES:
        for m in quote.finditer(text):
            context = text[max(0, m.start() - 60) : m.end() + 20]
            if ERROR_WORDS.search(context):
                add("error_text", m.start(), m.end())
    for m in [SIGNOFF.search(text), MY_NAME.search(text)]:
        if m and m.group(1).split()[0].lower() not in NOT_NAMES:
            add("name", m.start(1), m.end(1))
    return sorted(found, key=lambda e: e.start)
