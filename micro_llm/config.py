"""Paths and constants shared by the ticket models."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BATCHES_DIR = DATA_DIR / "batches"
TICKETS_FILE = DATA_DIR / "tickets.jsonl"
MODELS_DIR = ROOT / "models"
SAMPLE_TICKET = ROOT / "examples" / "support_ticket_98423" / "ticket_breakdown.json"
# Stand-in records for the systems the step-3 actions call
SAMPLE_SYSTEMS = ROOT / "examples" / "support_ticket_98423" / "systems.json"
# Each thread's planned steps and audit log, so a later message can run the steps that waited
STORE_FILE = DATA_DIR / "threads.sqlite"

# Claude CLI model alias that writes the synthetic training tickets
GENERATOR_MODEL = "haiku"

# Share of tickets held out for testing, split by ticket so no ticket's pieces leak into training
TEST_SHARE = 0.2
SEED = 7
