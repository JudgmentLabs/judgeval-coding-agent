r"""A value Hermes hands a callback -> a frozen, size-accounted snapshot, in one bounded pass. Pure.

Hermes keeps the objects it passes a hook (tool arguments, provider tool calls) and may change them after the
hook returns, so a record keeps a copy, never a reference. Copying a large tree in Python is what made the
callbacks slow (a 10,000-row argument: 5.5 ms p95 to copy, then a second walk to size it). Here one walk measures
the tree against a budget before any copy is made, and stops as soon as the budget is spent; only a tree within
budget is copied, by ``marshal`` in C.

Budgets are data (``Budget``); ``CALLBACK`` is the one a callback spends. One field, one pass:

    value --take--> the value itself         str, int, float, bool, None: immutable, nothing to copy
                \-> bytes (marshal)          a dict/list within budget: plain JSON types, copied in C
                \-> Omitted(reason)          over budget (nodes, chars, depth): dropped whole, never cut
    a tree with other types (provider objects, tuples, non-str keys)
        --copy--> the same budget, spent by a plain Python copy (unknown objects as their str), then marshal

Every budget dimension ends the same way, in Omitted: a tree nested too deep is not cut at its depth (a
deep ``{"path": ".env"}`` would lose the evidence that makes its turn sensitive), it is omitted whole.

Impossible: a snapshot over its budget; part of a value (a cut, at any budget, depth included); a reference to Hermes's object in a record;
an Omitted crossing the pipe (spooler.bound turns it into the OMITTED marker, named in ``omitted``, and the
helper treats an omitted input as sensitive: its turn is blanked, as for a .env read).

A record field holding ``bytes`` is a snapshot: records hold no other bytes. ``unpack`` (in the helper) turns
it back into the value, with the rest of the normalization callbacks no longer do (tool call arguments as JSON).
"""

from __future__ import annotations

import json
import marshal
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Budget:
    nodes: int  # members of every dict and list, counted before they are visited
    chars: int  # characters of every string and key (numbers, bools, null: 8 each)
    depth: int  # containers nested below the root a snapshot may hold; deeper, the value is omitted whole


# What one callback may spend on one field. 8192 members is ~0.6 ms of walk plus ~0.1 ms of copy at p95 on
# Python 3.13 (tests/bench_hermes.py); 512 KB is the most the helper redacts in one field (spooler.FIELD_CAP).
CALLBACK = Budget(nodes=8192, chars=512 * 1024, depth=12)


class Omitted:
    """A field over budget. ``reason``: which budget ran out. Never sent: spooler.bound replaces it."""

    __slots__ = ("reason",)

    def __init__(self, reason: str):
        self.reason = reason

    def __repr__(self) -> str:
        return f"Omitted({self.reason!r})"


_SCALAR = (int, float, bool, type(None))
_PLAIN = 0  # all plain: marshal can copy it as is
_COPY = -1  # holds another type: copy it in Python


def _measure(value: Any, budget: Budget) -> int | Omitted:
    """One walk, level by level, stopping as soon as a budget runs out. Characters when the tree is plain
    JSON types within budget; _COPY when it must be copied in Python; Omitted when it is over budget."""
    nodes = chars = 0
    level = [value]
    for _depth in range(budget.depth + 1):
        deeper = []
        push = deeper.append
        for item in level:
            nodes += len(item)
            if nodes > budget.nodes:
                return Omitted("nodes")
            if type(item) is dict:
                try:
                    chars += sum(map(len, item))
                except TypeError:  # a key that is not a string
                    return _COPY
                members = item.values()
            else:
                members = item
            for member in members:
                kind = type(member)
                if kind is str:
                    chars += len(member)
                elif kind is dict or kind is list:
                    push(member)
                elif kind is int or kind is float or kind is bool or member is None:
                    chars += 8
                else:
                    return _COPY
            if chars > budget.chars:
                return Omitted("chars")
        if not deeper:
            return chars
        level = deeper
    return Omitted("depth")


def _copy(value: Any, budget: Budget) -> Any:
    """A plain copy, spending the same budget (raises _Over): unknown objects as their str."""
    spent = [0, 0]  # nodes, chars

    def visit(item: Any, depth: int) -> Any:
        if isinstance(item, str):
            spent[1] += len(item)
            out = str(item)
        elif isinstance(item, _SCALAR):
            spent[1] += 8
            out = item if type(item) in _SCALAR else (bool(item) if isinstance(item, bool) else item.real)
        elif not isinstance(item, (dict, list, tuple)):
            out = str(item)
            spent[1] += len(out)
        elif depth > budget.depth:  # the same limit _measure applies
            raise _Over("depth")
        else:
            spent[0] += len(item)
            if spent[0] > budget.nodes:
                raise _Over("nodes")
            if isinstance(item, dict):
                out = {}
                for key, member in item.items():
                    key = key if type(key) is str else str(key)
                    spent[1] += len(key)
                    out[key] = visit(member, depth + 1)
            else:
                out = [visit(member, depth + 1) for member in item]
        if spent[1] > budget.chars:
            raise _Over("chars")
        return out

    return visit(value, 0)


class _Over(Exception):
    pass


def take(value: Any, budget: Budget = CALLBACK) -> Any:
    """The snapshot a record keeps of ``value``: itself when immutable, marshal bytes of a copy, or Omitted."""
    kind = type(value)
    if kind is str:
        return value if len(value) <= budget.chars else Omitted("chars")
    if value is None or kind is int or kind is float or kind is bool:
        return value
    if kind is dict or kind is list:
        measured = _measure(value, budget)
        if isinstance(measured, Omitted):
            return measured
        if measured != _COPY:
            try:
                return marshal.dumps(value)
            except (ValueError, RuntimeError):  # changed under us by another thread: copy it the slow way
                pass
    try:
        plain = _copy(value, budget)
    except _Over as over:
        return Omitted(str(over))
    except (RuntimeError, RecursionError):  # changed while copied, or nested beyond Python's own limit
        return Omitted("changed")
    if isinstance(plain, str):
        return plain if len(plain) <= budget.chars else Omitted("chars")
    return marshal.dumps(plain) if isinstance(plain, (dict, list)) else plain


def weight(value: Any) -> int:
    """What a snapshot weighs against the queue and the helper's budgets: its characters, or its marshal bytes
    (never fewer than its characters). Omitted weighs more than any budget."""
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, Omitted):
        return 1 << 62
    return 8


def text(value: Any, budget: Budget = CALLBACK) -> Any:
    """A tool result or message content that is not text, as text (its JSON), within budget; else Omitted."""
    snap = take(value, budget)
    if isinstance(snap, bytes):
        return json.dumps(marshal.loads(snap), ensure_ascii=False)
    if isinstance(snap, Omitted) or isinstance(snap, str):
        return snap
    return json.dumps(snap)


# ---- in the helper ---------------------------------------------------------------------------------


def _arguments(value: Any) -> Any:
    if isinstance(value, bytes):
        return marshal.loads(value)
    if isinstance(value, str):  # the model's JSON text, kept whole in the callback and parsed here
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def unpack(record: dict) -> dict:
    """A record as it was before snapshotting: each snapshot loaded, each tool call's arguments parsed."""
    out = {key: marshal.loads(value) if isinstance(value, bytes) else value for key, value in record.items()}
    calls = out.get("tool_calls")
    if isinstance(calls, list):
        out["tool_calls"] = [{**call, "arguments": _arguments(call.get("arguments"))} if isinstance(call, dict) else call for call in calls]
    return out
