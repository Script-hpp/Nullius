"""Admission for paid requests. Uncertain requests retain their reservation on disk.

Reservations are estimates, not provider invoices. A shared file lock coordinates
clients/processes; reservations survive restarts and must never expire automatically.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as dt
import fcntl
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .costs import BudgetExceeded, gross, logged_gross_total, read_log


@dataclass
class JobBudget:
    limit_usd: float
    live: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    spent_usd: float = 0.0

    def check(self):
        if self.cancelled.is_set():
            raise BudgetExceeded("Job stopped; no further paid requests are allowed.")
        if not math.isfinite(self.limit_usd) or self.limit_usd <= 0:
            raise BudgetExceeded("A positive finite job budget is required.")


current_budget = contextvars.ContextVar("job_budget", default=None)


def context_map(pool, function, items):
    """Keep job budget/cancellation context in each executor task.

    A Context cannot be entered concurrently, so each item needs its own copy.
    Lazily constructed clients must capture the same budget as their caller.
    """
    tasks = [(contextvars.copy_context(), item) for item in items]
    return pool.map(lambda task: task[0].run(function, task[1]), tasks)


def number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (ValueError, TypeError) as exc:
        raise BudgetExceeded(f"Invalid spending limit: {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise BudgetExceeded(f"Invalid spending limit: {name}")
    return value


@contextlib.contextmanager
def ledger(path: Path):
    try:
        with _locked_ledger(path) as state:
            yield state
    except (OSError, ValueError) as exc:
        raise BudgetExceeded("Spending ledger unavailable; paid requests are blocked.") from exc


@contextlib.contextmanager
def _locked_ledger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_path = path.with_suffix(".reservations.json")
    with path.with_suffix(".spend.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        yield state
        # Replace atomically; a crash before settlement leaves the reservation in place.
        temp = state_path.with_suffix(".tmp")
        with temp.open("w") as out:
            json.dump(state, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, state_path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def pending(path: Path) -> float:
    with ledger(path) as state:
        return sum(r["amount"] for r in state.values())


def reserve_model(path: Path, settings, scope: JobBudget, amount: float) -> str:
    scope.check()
    if not math.isfinite(settings.budget_total_usd) or settings.budget_total_usd < 0 or not math.isfinite(settings.vat_rate) or settings.vat_rate < 0:
        raise BudgetExceeded("Invalid model budget or VAT rate.")
    if not math.isfinite(amount) or amount <= 0:
        raise BudgetExceeded("Cannot bound the price of this model request.")
    with ledger(path) as state:
        scope.check()
        outstanding = sum(r["amount"] for r in state.values())
        job_pending = sum(r["amount"] for r in state.values() if r["job"] == scope.id)
        charge = gross(amount, settings.vat_rate)
        if scope.limit_usd <= 0 or scope.spent_usd + gross(job_pending, settings.vat_rate) + charge > scope.limit_usd:
            raise BudgetExceeded("Job budget cannot cover another model request.")
        total = settings.budget_total_usd
        spent = logged_gross_total(path, settings.vat_rate, settings.budget_since)
        if total > 0 and spent + gross(outstanding + amount, settings.vat_rate) > total * .95:
            raise BudgetExceeded("Budget cannot cover another model request (95% ceiling).")
        if scope.live:
            if total <= 0 or not settings.budget_since:
                raise BudgetExceeded("Live budget needs an explicit total and start date.")
            if total - spent - gross(outstanding + amount, settings.vat_rate) < number("LIVE_MIN_BUDGET_USD", 3):
                raise BudgetExceeded("Live model budget floor reached.")
            cap = number("MAX_DAILY_COST_USD", 0)
            midnight = dt.datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            daily = sum(e["cost_usd"] for e in read_log(path) if e["ts"] >= midnight)
            if cap and gross(daily + outstanding + amount, settings.vat_rate) > cap:
                raise BudgetExceeded("Daily model budget cannot cover another request.")
        token = uuid.uuid4().hex
        state[token] = {"amount": amount, "job": scope.id, "ts": time.time()}
    return token


def settle_model(path: Path, settings, scope: JobBudget, token: str, response, book):
    if not math.isfinite(response.cost_usd) or response.cost_usd < 0:
        scope.cancelled.set()
        raise BudgetExceeded("Invalid provider cost; reservation retained.")
    with ledger(path) as state:
        book(response)
        scope.spent_usd += gross(response.cost_usd, settings.vat_rate)
        reserved = state.pop(token)["amount"]
        if response.cost_usd > reserved + 1e-9:
            scope.cancelled.set()  # Provider usage exceeded our conservative estimate.


def release(path: Path, token: str):
    with ledger(path) as state:
        state.pop(token, None)
