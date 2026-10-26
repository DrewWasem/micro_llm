"""Keeps each thread's steps and an audit log in SQLite, so a later message on the thread can
run the steps that waited for a verified account.

A thread holds each action once, so a repeated message never runs an action twice.
"""

import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY,
    account TEXT,
    opened TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    thread INTEGER NOT NULL REFERENCES threads (id),
    action TEXT NOT NULL,
    -- ready or waiting until it runs, then who acts next: done, with_staff or with_customer
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    inputs TEXT NOT NULL,
    result TEXT,
    PRIMARY KEY (thread, action)
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    thread INTEGER NOT NULL,
    action TEXT,
    event TEXT NOT NULL,
    detail TEXT NOT NULL,
    at TEXT NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    def log(self, thread, action, event, **detail):
        self.db.execute(
            "INSERT INTO audit (thread, action, event, detail, at) VALUES (?, ?, ?, ?, ?)",
            (thread, action, event, json.dumps(detail), now()),
        )

    def open(self, thread=None):
        """The thread's id, after checking it exists. A new thread when none is given."""
        with self.db:
            if thread is None:
                thread = self.db.execute(
                    "INSERT INTO threads (opened) VALUES (?)", (now(),)
                ).lastrowid
                self.log(thread, None, "opened")
            elif not self.db.execute(
                "SELECT 1 FROM threads WHERE id = ?", (thread,)
            ).fetchone():
                raise ValueError(f"no thread {thread}")
        return thread

    def account(self, thread):
        return self.db.execute(
            "SELECT account FROM threads WHERE id = ?", (thread,)
        ).fetchone()[0]

    def connect(self, thread, account):
        """Ties the thread to the account the customer verified. A thread keeps its first account."""
        current = self.account(thread)
        if current == account:
            return
        if current is not None:
            raise ValueError(f"thread {thread} is tied to another account")
        with self.db:
            self.db.execute(
                "UPDATE threads SET account = ? WHERE id = ?", (account, thread)
            )
            self.db.execute(
                "UPDATE steps SET status = 'done' WHERE thread = ? AND action = 'verify_identity'"
                " AND status = 'with_customer'",
                (thread,),
            )
            self.log(thread, None, "account connected", account=account)

    def add(self, thread, step, status):
        """Stores a step unless the thread already has its action. True when it was new."""
        with self.db:
            added = self.db.execute(
                "INSERT OR IGNORE INTO steps (thread, action, status, reason, inputs)"
                " VALUES (?, ?, ?, ?, ?)",
                (thread, step.action, status, step.reason, json.dumps(step.inputs)),
            ).rowcount
            if added:
                self.log(
                    thread, step.action, "planned", status=status, reason=step.reason
                )
        return bool(added)

    def release(self, thread):
        """Makes every waiting step ready and returns their actions."""
        with self.db:
            actions = [
                a
                for (a,) in self.db.execute(
                    "SELECT action FROM steps WHERE thread = ? AND status = 'waiting' ORDER BY rowid",
                    (thread,),
                )
            ]
            self.db.execute(
                "UPDATE steps SET status = 'ready' WHERE thread = ? AND status = 'waiting'",
                (thread,),
            )
            for a in actions:
                self.log(thread, a, "released")
        return actions

    def ready(self, thread):
        """The ready steps' actions and inputs, in the order they were planned."""
        rows = self.db.execute(
            "SELECT action, inputs FROM steps WHERE thread = ? AND status = 'ready' ORDER BY rowid",
            (thread,),
        )
        return [(a, json.loads(i)) for a, i in rows]

    def finish(self, thread, action, outcome, calls):
        """Saves what an action did, with every tool call it made."""
        result = {"did": outcome.did, "reply": outcome.reply, "calls": calls}
        with self.db:
            self.db.execute(
                "UPDATE steps SET status = ?, result = ? WHERE thread = ? AND action = ?",
                (outcome.status, json.dumps(result), thread, action),
            )
            self.log(thread, action, outcome.status, did=outcome.did, calls=calls)

    def waiting(self, thread):
        return self.db.execute(
            "SELECT COUNT(*) FROM steps WHERE thread = ? AND status = 'waiting'",
            (thread,),
        ).fetchone()[0]
