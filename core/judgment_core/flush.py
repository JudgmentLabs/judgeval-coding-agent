"""Route hook events into turns and send the turns that are ready. Runs detached from the agent.

Hooks drop one file per event in ``inbox/``. One flusher at a time folds new
events through ``assemble.Fold`` in the order they happened, appends each to
its turn's file under ``turns/``, and sends every turn the fold marked ready.
Persistent state is the fold's small routing table (``fold.json``) and one
record per turn (``index.json``): its file, whether it is waiting to be
sent, the last ``judgment.update_id`` sent, when it started, and whether a
later turn may replay its text (``replayable``). A turn whose llm inputs were
not read from a transcript also keeps the messages it added to the thread
(``<turn>.messages.json``), so later turns replay them whole.

A transcript replays every earlier turn of the session in each request,
including turns this sender never saw (traced off) or sent without text
(metadata). A request read from a transcript names the earlier turns
it holds (``replay_turns``); it keeps its transcript input only when each of
them is replayable and went to the same project. Otherwise it is rebuilt from
the turn's own events and the kept messages of earlier turns, as for a harness
without a transcript.

Each turn's events pass through plain stages: inbox -> fold -> turn file -> enrich (transcript
text) -> capture policy (metadata strips text; sanitized redacts it) -> spans -> post. The turn file and the index keep
only what the policy allows, so no secret the rules know stays on disk after a send.
"""

from __future__ import annotations

import os
import signal
import sys

if __name__ == "__main__" and "--detach" in sys.argv[1:]:
    # First, before the imports below: the hook waits for this process to leave its process group (judgment-hook).
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass

import fcntl
import gzip
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Iterable
from pathlib import Path

from judgment_core import redact
from judgment_core.assemble import Fold, turn_key
from judgment_core.claude_transcript import enrich_claude_events
from judgment_core.codex_transcript import enrich_codex_events
from judgment_core.cursor_transcript import enrich_cursor_events
from judgment_core.config import DEFAULT_API_URL, load_credentials, traces_url
from judgment_core.messages import text_message
from judgment_core.optin import load_project, safe_id, state_dir, tracing_forced_off
from judgment_core.spans import build_request, turn_start

ATTEMPTS = 3
TIMEOUT_SECONDS = 10
DEADLINE_SECONDS = 20
# Long enough for a running flusher to reach its deadline and finish its last post.
WAIT_SECONDS = 60
# A hook writes its file in one go. One that still does not parse after this long was cut off.
PARTIAL_SECONDS = 10
# A turn with no single export stamp can never be sent: nothing records which project it was meant for.
ORPHAN_SECONDS = 24 * 3600
# Wait this long after a turn's last event before its first send, so a child that starts just after the
# turn ends goes out with it.
GRACE_MS = int(os.environ.get("JUDGMENT_FLUSH_GRACE_MS", "1500"))
# A turn with a child still running waits for it, but not forever: the child may never report its end.
HOLD_MS = 10 * 60 * 1000
# A turn whose agent died before it ended gets no stop. Send what it has once it has been quiet this long.
STALE_MS = 24 * 3600 * 1000
# A sent turn that has been quiet this long is dropped from disk.
KEEP_SECONDS = 7 * 24 * 3600
# One request stays well under the server's body limit. A larger turn goes out in several, root last.
REQUEST_BYTES = 2 * 1024 * 1024
# Earlier turns' kept messages ride along in every llm input of a turn, and nothing compacts them. Past this size,
# only the newest whole turns that fit are replayed.
HISTORY_BYTES = 1024 * 1024
_LOG_MAX = 64 * 1024


# The state holds prompts, answers, and tool output. Only its owner may read it.
def _private_state(state: Path) -> None:
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state, 0o700)


def _private_append(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(text)


def _private_write(path: Path, text: str) -> None:
    """Replace path in one rename, so a reader sees the old file or the new one."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    scratch = path.with_name(path.name + ".tmp")
    fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(scratch, path)


def _log(state: Path, message: str) -> None:
    try:
        path = state / "flush.log"
        if path.is_file() and path.stat().st_size > _LOG_MAX:
            path.replace(state / "flush.log.1")
        _private_append(path, message.rstrip() + "\n")
        # The hook creates this file when it redirects the flusher's output.
        os.chmod(path, 0o600)
    except OSError:
        return


def _acquire(state: Path, key: str) -> int | None:
    """Become the one flusher. Return the lock's descriptor, held until it is closed.

    The kernel drops a flock when its process dies, so there is no stale lock.
    The hook starts a flusher at every turn end. When one is already running,
    one more per API key waits for it and then runs, so a turn that ended after
    the running flusher's last scan still goes out, and a flusher busy with one
    key's turns never strands another key's. Any further process leaves.
    """
    folder = state / "locks"
    folder.mkdir(mode=0o700, exist_ok=True)
    run = os.open(folder / "flush.run", os.O_RDWR | os.O_CREAT, 0o600)
    if _try_lock(run):
        return run
    wait = os.open(folder / f"{_fingerprint(key) if key else 'no-key'}.wait", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _try_lock(wait):
            deadline = time.time() + WAIT_SECONDS
            while time.time() < deadline:
                time.sleep(0.05)
                if _try_lock(run):
                    return run
    finally:
        os.close(wait)
    os.close(run)
    return None


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _drop_export_refs(state: Path) -> None:
    """Older builds stored the API key in this directory. Remove it."""
    refdir = state / "export-refs"
    if refdir.exists():
        import shutil

        shutil.rmtree(refdir, ignore_errors=True)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict]:
    """A turn's events. An event stored twice (a crash before its inbox file was deleted) counts once."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    seen = set()
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if not isinstance(item, dict):
            continue
        ident = item.get("_id")
        if ident:
            if ident in seen:
                continue
            seen.add(ident)
        events.append(item)
    return events


def _stamp(event: dict) -> int:
    try:
        return int(event.get("ts") or 0)
    except (TypeError, ValueError):
        return 0


def _turn_file(state: Path, key: str) -> Path:
    harness, session, turn = key.split("\t")
    return state / "turns" / safe_id(harness) / safe_id(session) / f"{safe_id(turn)}.jsonl"


def _messages_file(state: Path, key: str) -> Path:
    return _turn_file(state, key).with_suffix(".messages.json")


def _forget(state: Path, key: str, item: dict) -> None:
    """No later turn replays this one: not its kept messages, nor its text in a transcript."""
    item.pop("summary", None)  # written by earlier versions
    item["messages"] = False
    item["replayable"] = False
    _messages_file(state, key).unlink(missing_ok=True)


def _inbox(state: Path) -> list[tuple[Path, dict, list]]:
    """Event files in the order hooks wrote them, up to the first one still being written."""
    folder = state / "inbox"
    if not folder.is_dir():
        return []
    now = time.time()
    files = []
    for path in folder.iterdir():
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((stat.st_mtime_ns, path.name, path))
    found = []
    for mtime_ns, name, path in sorted(files):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(event, dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            if now - mtime_ns / 1e9 > PARTIAL_SECONDS:
                path.unlink(missing_ok=True)  # cut off when its hook died; it will never parse
                continue
            break  # still being written: what follows waits, so a stop is never folded before its start
        # One id per file write, so an event replayed after a crash is stored once.
        event["_id"] = f"{name}.{mtime_ns}"
        found.append((path, event, [mtime_ns, name]))
    return found


def ingest(state: Path, env: dict | None = None) -> dict[str, dict]:
    """Fold new inbox events into turns. Return the turn index."""
    legacy = _Redactor(env)
    index = _load_json(state / "index.json")
    fold = Fold(_load_json(state / "fold.json"))
    fold.sent = {key for key, item in index.items() if item.get("update_id") is not None}
    fold.unsent_keys = {key for key, item in index.items() if item.get("update_id") is None}
    consumed = []
    # The inbox files folded in, by name and mtime, saved with the fold: a crash before they were deleted
    # replays none of them. A file that appears late, with an older mtime than one already folded, is new.
    done = set(fold.s["inbox"].get("done") or [])
    # State from a build that kept only the newest file folded: what sorts at or before it was folded.
    mark = fold.s["inbox"].get("last") if "done" not in fold.s["inbox"] else None
    folded = []
    for path, event, order in _inbox(state):
        folded.append(event["_id"])
        if event["_id"] in done or (mark and order <= mark):
            consumed.append(path)
            continue
        if "capture" not in event and event.get("kind") != "command":
            # An older build wrote this event without redacting it. Redact it before it is stored.
            event = legacy.redact(event)
        children = set(fold.s["child"])
        try:
            routed, ready, extra = fold.step(event)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            _log(state, f"drop an event: {type(exc).__name__}")
            consumed.append(path)
            continue
        for number, item in enumerate(extra):
            item["_id"] = f"{event['_id']}.close{number}"
        for item in extra + ([routed] if routed else []):
            _store(state, index, fold, item)
        for key in ready:
            if key in index and not index[key].get("ready"):
                index[key]["ready"] = True
                index[key]["ready_at"] = int(time.time() * 1000)
        for child in set(fold.s["child"]) - children:
            _adopt(state, index, fold, child)
        consumed.append(path)
    retained = []
    for ident in done | set(folded):
        name, _, mtime = ident.rpartition(".")
        for folder in ("inbox.tmp", "inbox"):
            try:
                if str((state / folder / name).stat().st_mtime_ns) == mtime:
                    retained.append(ident)
                    break
            except OSError:
                pass
    fold.s["inbox"] = {"done": sorted(retained)}
    if consumed or set(retained) != done:
        _private_write(state / "fold.json", json.dumps(fold.state(int(time.time() * 1000), set(index))))
        _private_write(state / "index.json", json.dumps(index))
        for path in consumed:
            path.unlink(missing_ok=True)
    return index


def _store(state: Path, index: dict, fold: Fold, event: dict) -> None:
    """Append one event to its turn's file."""
    key = turn_key(event)
    path = _turn_file(state, key)
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    try:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                line = "\n" + line  # a crash cut the last line short; start this one on its own line
    except OSError:
        pass
    _private_append(path, line)
    if key not in index:
        index[key] = {"first": _stamp(event), "update_id": None, "ready": False}
        fold.unsent_keys.add(key)
    index[key]["last"] = _stamp(event)
    # Event times come from the agent and can sit in the future (a tool's end). Grace counts from arrival here.
    index[key]["stored_at"] = int(time.time() * 1000)


def _adopt(state: Path, index: dict, fold: Fold, child: str) -> None:
    """A Cursor child conversation stored before its Task bound it moves onto the parent turn."""
    parent_session, parent_turn = fold.s["child"][child][0], fold.s["child"][child][1]
    prefix = f"cursor\t{child}\t"
    moved = False
    for key in [key for key in index if key.startswith(prefix)]:
        for event in _read_jsonl(_turn_file(state, key)):
            event.update(session_id=parent_session, turn_id=parent_turn)
            event["agent_id"] = event.get("agent_id") or child
            _store(state, index, fold, event)
            moved = True
        _turn_file(state, key).unlink(missing_ok=True)
        index.pop(key, None)
        fold.unsent_keys.discard(key)
    parent = f"cursor\t{parent_session}\t{parent_turn}"
    if moved and parent in index and (fold.s["closed"].get(parent) or index[parent].get("update_id") is not None):
        index[parent]["ready"] = True


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _destination(events: list[dict], env: dict | None) -> tuple[str, dict[str, str] | None]:
    """Where this turn goes: ``("send", creds)``, ``("other", None)``, or ``("orphan", None)``.

    The hook stamps every event with the key fingerprint, project, URL, and org
    it had. The state never holds the key, so the flusher sends a turn only
    when every event carries one stamp and that stamp is this process's key. A
    turn for another key is that key's flusher's work. An event without a
    stamp, or two different stamps in one turn, makes the turn an orphan.
    """
    stamps = {
        tuple(str(event.get(field) or "") for field in ("export_fp", "export_project", "export_url", "export_org"))
        for event in events
        if not event.get("interrupted")
    }
    if len(stamps) != 1:
        return "orphan", None
    fp, project, url, org = stamps.pop()
    if not fp or not project:
        return "orphan", None
    key = load_credentials(env).get("JUDGMENT_API_KEY", "")
    if not key or _fingerprint(key) != fp:
        return "other", None
    return "send", {"JUDGMENT_API_KEY": key, "JUDGMENT_PROJECT_ID": project, "JUDGMENT_API_URL": url or DEFAULT_API_URL, "JUDGMENT_ORG_ID": org}


def turn_capture(events: list[dict]) -> str:
    """The strictest capture any event was recorded under. An event with no stamp, from an older build, is sanitized."""
    modes = {event.get("capture") for event in events}
    if "metadata" in modes:
        return "metadata"
    if modes == {"full"}:
        return "full"
    return "sanitized"


class _Redactor:
    """The rules for each project's turns, and the secret values in this process's environment."""

    def __init__(self, env: dict | None = None):
        self.env = os.environ if env is None else env
        self._secrets: list[tuple[str, str]] | None = None
        self._rules: dict[str, redact.Ruleset] = {}

    @property
    def secrets(self) -> list[tuple[str, str]]:
        if self._secrets is None:
            self._secrets = redact.env_secrets(self.env, load_credentials(self.env).get("JUDGMENT_API_KEY", ""))
        return self._secrets

    @property
    def names(self) -> set[str]:
        """The secret env names this process holds a value for."""
        return {name for name, _value in self.secrets}

    def rules(self, events: list[dict]) -> redact.Ruleset:
        root = next((str(event.get("project_root")) for event in events if event.get("project_root")), "")
        if root not in self._rules:
            self._rules[root] = redact.ruleset(load_project(Path(root) / ".judgment" / "tracing.json") if root else {})
        return self._rules[root]

    def checked_environment(self, events: list[dict]) -> bool:
        for event in events:
            snapshot = event.get("env_capture")
            if snapshot is None:
                continue
            if not isinstance(snapshot, dict) or snapshot.get("error"):
                return False
            names = snapshot.get("names")
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                return False
            if len(names) > redact.hook_budget() or len(redact.jq_json(names).encode("utf-8")) > 4096:
                return False
            pairs = sorted({(name, value) for name, value in self.secrets if name in names})
            digest = hashlib.sha256(redact.jq_json(pairs).encode("utf-8")).hexdigest()[:32] if pairs else ""
            if set(names) != {name for name, _ in pairs} or snapshot.get("digest") != digest:
                return False
        return True

    def redact(self, event: dict) -> dict:
        return redact.redact_event(event, self.rules([event]), self.secrets)

    def history(self, messages: list[dict], rs: redact.Ruleset) -> tuple[list[dict], list[dict]]:
        """One earlier turn's messages redacted, and what that replaced as field history. None of them at a limit."""
        try:
            scrubbed, records = redact.scrub_fields({"history": messages}, rs, self.secrets, redact.sender_budget())
        except redact.RedactionLimit:
            return [], []
        return scrubbed["history"], records


def _sanitize(state: Path, key: str, events: list[dict], prepared: list[dict], redactor: _Redactor) -> tuple[list[dict], bool]:
    """Redact the enriched turn, and whether this process could check it: the environment the hook redacted with is the
    one here. When it is not, every event loses its text, as when its rules fail. The turn file is rewritten when this
    changes what it holds, so the redacted version is all that stays on disk.
    """
    rs = redactor.rules(events)
    checked = redactor.checked_environment(events)
    prepared = redact.redact_events(carry_records(events, prepared, redactor.names), rs, redactor.secrets)
    if not checked:
        # Last, so carry_records cannot put the hook's text back.
        prepared = [redact._unredactable(event) for event in prepared]
    stored = [redact.redact_event(event, rs, redactor.secrets) for event in events]
    if stored != events:
        _private_write(_turn_file(state, key), "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in stored))
    return prepared, checked


def carry_records(stored: list[dict], prepared: list[dict], names: Iterable[str] | None = None) -> list[dict]:
    """The enriched events, each with the records its stored event carried for the fields enrichment left as they were,
    and the hook's text kept wherever this process cannot redact the transcript text that would replace it.

    Enrichment can replace a field with transcript text (a tool's output) or build new events. A record describes the
    text it was counted in, so a replaced field's records go; the sender counts what it replaces in the new text.

    Env values are the one rule input a process may not share: the hook redacted with its agent's environment, and the
    sender that reads the queue may have been started by another agent. So (``names``: the secret env names this
    process holds; None skips the second step):
      - a field the hook redacted an env value in keeps the hook's sanitized text, whatever the transcript says;
      - when the hook redacted an env value this process does not hold, no transcript text is kept at all: a changed
        or added text field goes back to the hook's text or is dropped, and an event built from the transcript alone
        loses its text. The env value may be anywhere in the transcript, and nothing here could find it.
    """
    by_id = {event["_id"]: event for event in stored if event.get("_id")}
    # Which env values the hook's text depended on: its env records, and its provenance (env_found), which says which
    # env values each field held even when another rule replaced them or the event's rules failed.
    hook_env = {name for event in stored for name, _field in _env_refs(event)}
    missing = bool(hook_env - set(names)) if names is not None else False
    seen: set[str] = set()
    out = []
    for event in prepared:
        base = by_id.get(event.get("_id")) if event.get("_id") not in seen else None
        if event.get("_id"):
            seen.add(event["_id"])
        event = dict(event)
        if base is not None:
            env_fields = {field for _name, field in _env_refs(base)}
            for field in env_fields | (set(event) | set(base) if missing else set()):
                if (field in env_fields or redact.field_class(field)["redact"]) and event.get(field) != base.get(field):
                    if field in base:
                        event[field] = base[field]
                    else:
                        event.pop(field, None)
        elif missing:
            event = redact.strip_text(event)
        carried = [item for item in redact.census(base) if event.get(item["field"]) == base.get(item["field"])] if base else []
        event = {key: value for key, value in event.items() if key != "redactions"}
        out.append({**event, "redactions": carried} if carried else event)
    return out


def _env_refs(event: dict) -> set[tuple[str, str]]:
    """(name, field) for each env value the hook found in the event's text: from its records and its provenance."""
    refs = {(item["rule"][4:], item["field"]) for item in redact.census(event) if item["rule"].startswith("env:")}
    found = event.get("env_found")
    for item in found if isinstance(found, list) else []:
        if isinstance(item, dict) and item.get("name") and item.get("field"):
            refs.add((str(item["name"]), str(item["field"])))
    return refs


def _redact_history(history: list[list[dict]], rs: redact.Ruleset, redactor: _Redactor) -> tuple[list[list[dict]], list[dict]]:
    """Earlier turns' messages redacted, and the records of what that replaced, as field history."""
    out, records = [], []
    for turn in history:
        messages, found = redactor.history(turn, rs)
        out.append(messages)
        records += found
    return out, records


def _enrich(events: list[dict]) -> list[dict]:
    return enrich_cursor_events(enrich_codex_events(enrich_claude_events(events)))


def _post(url: str, body: bytes, headers: dict[str, str]) -> int:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _batches(spans: list[dict]) -> list[list[dict]]:
    """The turn's spans in batches of at most REQUEST_BYTES of JSON each, in order, so the root, listed last, goes in
    the last request after every child. A span larger than that is alone in its batch."""
    batches: list[list[dict]] = [[]]
    size = 0
    for span in spans:
        span_size = len(json.dumps(span))
        if batches[-1] and size + span_size > REQUEST_BYTES:
            batches.append([])
            size = 0
        batches[-1].append(span)
        size += span_size
    return batches


def _body(payload: dict, spans: list[dict]) -> bytes:
    resource = payload["resourceSpans"][0]
    scope = resource["scopeSpans"][0]
    body = {"resourceSpans": [{"resource": resource["resource"], "scopeSpans": [{"scope": scope["scope"], "spans": spans}]}]}
    return gzip.compress(json.dumps(body).encode("utf-8"), compresslevel=5)


def _post_retrying(state: Path, url: str, body: bytes, headers: dict[str, str]) -> int:
    status = 0
    for attempt in range(ATTEMPTS):
        try:
            status = _post(url, body, headers)
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
            _log(state, f"post attempt {attempt + 1}: {type(exc).__name__}")
            status = 0
        if status == 200 or status in (400, 401, 403, 404, 413):
            break
        _log(state, f"post status {status}")
        time.sleep(0.2 * (2**attempt))
    return status


def _send(state: Path, url: str, headers: dict[str, str], payload: dict) -> tuple[int, int]:
    """Post the turn's spans, root last. Return the status that stopped the send (200 when it went through) and how
    many requests the server took.

    A request the server refuses as too large (413) is split in two and each half sent, so one large span does not
    cost the spans that shared its request. A span refused on its own is dropped, and only it.
    """
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root = spans[-1] if spans else None
    queue = _batches(spans)
    taken = 0
    while queue:
        batch = queue.pop(0)
        status = _post_retrying(state, url, _body(payload, batch), headers)
        if status == 413 and len(batch) > 1:
            half = len(batch) // 2
            queue[:0] = [batch[:half], batch[half:]]
            continue
        if status == 413:
            _log(state, "drop the turn's root span: post status 413" if batch[0] is root else "drop a span: post status 413")
            continue
        if status != 200:
            return status, taken
        taken += 1
    return 200, taken


def _untraced(index: dict, key: str, dest: str, turns: list) -> bool:
    """Whether any of these earlier turns of the session is not one a later turn may replay into this project: traced
    off (no record), sent without text (metadata, failed rules), not sent yet, or sent to another project."""
    harness, session, _turn = key.split("\t")
    for turn in turns:
        item = index.get(f"{harness}\t{session}\t{turn}")
        if item is None or not item.get("replayable") or item.get("dest") != dest:
            return True
    return False


def _replayable(index: dict, key: str, dest: str, prepared: list[dict]) -> list[dict]:
    """The events, each request read from a transcript keeping its input only when every earlier turn it replays may be
    replayed. One that may not is rebuilt from the turn's own events and the kept messages of earlier turns.

    A subagent's requests are checked on their own: its transcript holds only the turns that gave it a task.
    """
    out = []
    for event in prepared:
        turns = event.get("replay_turns")
        if "input_messages" in event and (not isinstance(turns, list) or _untraced(index, key, dest, turns)):
            event = {name: value for name, value in event.items() if name != "input_messages"}
        out.append({name: value for name, value in event.items() if name != "replay_turns"})
    return out


def _history(state: Path, index: dict, key: str, dest: str) -> list[list[dict]]:
    """The messages each earlier turn of the same session, sent to the same project, added to the thread. Oldest first."""
    harness, session, _turn = key.split("\t")
    prefix = f"{harness}\t{session}\t"
    start = int(index[key].get("start") or index[key].get("first") or 0)
    earlier = sorted(
        (int(item.get("start") or 0), other)
        for other, item in index.items()
        if other != key and other.startswith(prefix) and item.get("messages") and item.get("dest") == dest and int(item.get("start") or 0) < start
    )
    turns: list[list[dict]] = []
    budget = HISTORY_BYTES
    skipped = 0
    # Newest first: a long session keeps its latest turns whole and leaves out the oldest.
    for _start, other in reversed(earlier):
        try:
            turn = json.loads(_messages_file(state, other).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(turn, list):
            continue
        messages = [message for message in turn if isinstance(message, dict)]
        size = len(json.dumps(messages, ensure_ascii=False))
        if skipped or size > budget:
            skipped += 1
            continue
        budget -= size
        turns.append(messages)
    turns.reverse()
    if skipped:
        turns.insert(0, [text_message("system", f"[{skipped} earlier turns omitted]")])
    return turns


def _waiting(state: Path, index: dict) -> set[str]:
    """Ready turns not to send yet: still inside the grace period, or with a child still running."""
    now = int(time.time() * 1000)
    fold = Fold(_load_json(state / "fold.json"))
    waiting = set()
    for key, item in index.items():
        if not item.get("ready"):
            continue
        arrived = max(int(item.get("stored_at") or 0), int(item.get("ready_at") or 0))
        if item.get("update_id") is None and now - arrived < GRACE_MS:
            waiting.add(key)
        elif fold.running(key) and now - int(item.get("ready_at") or now) < HOLD_MS:
            waiting.add(key)
    return waiting


def export_once(state: Path | None = None, env: dict | None = None) -> int:
    """Fold new events, then post ready turns that match this process's key.

    Return how many requests returned 200, or -1 when the process should stop:
    the key is missing, or a post failed with a status retrying will not fix or
    exhausted its retries. Unsent turns stay ready either way.
    """
    if tracing_forced_off(env):
        return 0
    # mkdir(mode=...) does not apply to parents it creates. The umask does.
    os.umask(0o077)
    state = state or state_dir(env)
    _private_state(state)
    _drop_export_refs(state)
    index = ingest(state, env)
    redactor = _Redactor(env)
    now = int(time.time() * 1000)
    for item in index.values():
        if not item.get("ready") and item.get("update_id") is None and now - int(item.get("stored_at") or now) > STALE_MS:
            item["ready"] = True
            item["ready_at"] = int(item.get("stored_at") or 0)
    waiting = _waiting(state, index)
    ready = sorted((key for key, item in index.items() if item.get("ready") and key not in waiting), key=lambda key: index[key].get("first") or 0)
    if not ready:
        _prune(state, index)
        return 0
    if not load_credentials(env).get("JUDGMENT_API_KEY"):
        _log(state, "skip: JUDGMENT_API_KEY is missing")
        return -1
    sent = 0
    for key in ready:
        item = index[key]
        events = _read_jsonl(_turn_file(state, key))
        verdict, creds = _destination(events, env)
        if verdict == "other":
            continue
        if verdict == "orphan":
            if events and max(map(_stamp, events)) < (time.time() - ORPHAN_SECONDS) * 1000:
                _drop(state, index, key)
                _log(state, "drop a turn: no export stamp names one project for every event")
            continue
        try:
            prepared = _enrich(events)
            # Transcript events carry no stamp. The hook's events say how the turn was recorded.
            capture = turn_capture(events)
            checked = True
            redaction = None
            rs = redactor.rules(events)
            assert creds is not None
            dest = f"{creds['JUDGMENT_API_URL']} {creds['JUDGMENT_ORG_ID']} {creds['JUDGMENT_PROJECT_ID']}"
            # A request keeps its transcript input only when every earlier turn it replays may be replayed here.
            prepared = _replayable(index, key, dest, prepared)
            if capture == "metadata":
                # Transcript enrichment has already run, so this is what gets sent and kept.
                prepared = [redact.metadata_event(event, rs, redactor.secrets) for event in carry_records(events, prepared, redactor.names)]
                if not redactor.checked_environment(events):
                    prepared = [redact._unredactable(event) for event in prepared]
            elif capture == "sanitized":
                prepared, checked = _sanitize(state, key, events, prepared, redactor)
            # A metadata turn is no later turn's context. Nor is one whose rules failed or whose environment did not check.
            quiet = capture == "metadata" or (capture != "full" and (bool(rs.error) or not checked))
            if quiet and (item.get("messages") or item.get("replayable")):
                # Settled before any send, so a failed post cannot leave the old text as context.
                _forget(state, key, item)
                _private_write(state / "index.json", json.dumps(index))
            started = turn_start(prepared)
            if started is not None:
                item["start"] = started
            update_id = None if item.get("update_id") is None else int(item["update_id"]) + 1
            # Rules that failed drop every text in the request, the earlier turns' included.
            history = [] if quiet else _history(state, index, key, dest)
            if capture != "full":
                # Earlier turns may have been captured in full, or stored by an older build. What is replaced in them now
                # is this request's redaction, so it joins the root's audit as field history.
                history, extra = _redact_history(history, rs, redactor)
                redaction = redact.turn_redaction(prepared, extra)
            turn = build_request(prepared, update_id=update_id, history=[message for messages in history for message in messages], redaction=redaction)
        except (OSError, ValueError) as exc:
            # A ValueError's message can quote the transcript value it choked on.
            _log(state, f"skip a turn: {type(exc).__name__}")
            continue
        if turn is None:
            item["ready"] = False
            continue
        headers = {
            "Authorization": f"Bearer {creds['JUDGMENT_API_KEY']}",
            "X-Project-Id": creds["JUDGMENT_PROJECT_ID"],
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        }
        if creds["JUDGMENT_ORG_ID"]:
            headers["X-Organization-Id"] = creds["JUDGMENT_ORG_ID"]
        url = traces_url(creds["JUDGMENT_API_URL"])
        status, taken = _send(state, url, headers, turn.payload)
        if status == 400:
            # The server will never take this turn as built. Retrying it would hold back every turn after it.
            _log(state, f"skip a turn: post status {status}")
            item["ready"] = False
            _private_write(state / "index.json", json.dumps(index))
            continue
        if status != 200:
            _log(state, f"post status {status}")
            _private_write(state / "index.json", json.dumps(index))
            return -1
        item["update_id"] = 0 if update_id is None else update_id
        item["ready"] = False
        if quiet:
            _forget(state, key, item)
        else:
            item.pop("summary", None)  # written by earlier versions
            item["replayable"] = True
            item["messages"] = turn.messages is not None
            if turn.messages is None:
                _messages_file(state, key).unlink(missing_ok=True)
            else:
                _private_write(_messages_file(state, key), json.dumps(turn.messages, ensure_ascii=False))
        item["dest"] = dest
        _private_write(state / "index.json", json.dumps(index))
        if not taken:
            # Nothing of the turn is in Judgment. Later turns may still replay it: what decides that is how it was captured.
            _log(state, "drop a turn: post status 413 for every span")
            continue
        sent += 1
    return sent


def _drop(state: Path, index: dict, key: str) -> None:
    _turn_file(state, key).unlink(missing_ok=True)
    _messages_file(state, key).unlink(missing_ok=True)
    index.pop(key, None)
    _private_write(state / "index.json", json.dumps(index))


def _prune(state: Path, index: dict) -> None:
    cutoff = (time.time() - KEEP_SECONDS) * 1000
    old = [key for key, item in index.items() if not item.get("ready") and int(item.get("last") or 0) < cutoff]
    for key in old:
        _turn_file(state, key).unlink(missing_ok=True)
        _messages_file(state, key).unlink(missing_ok=True)
        index.pop(key, None)
    if old:
        _private_write(state / "index.json", json.dumps(index))
    # The person a hook saved for each session. A session back after this long is resolved again.
    for path in (state / "users").glob("*"):
        try:
            if path.stat().st_mtime * 1000 < cutoff:
                path.unlink()
        except OSError:
            pass


def _detach() -> None:
    """Leave the hook's session, so an agent that kills its hooks' process group or session misses this.

    The hook ignores SIGHUP before it starts this, and starts it in its own process group where it can.
    setsid() fails for a process group leader, which is then already out of the hook's group.
    """
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    if "--detach" in (sys.argv[1:] if argv is None else argv):
        _detach()
    if tracing_forced_off():
        return
    state = state_dir()
    _private_state(state)
    held = _acquire(state, load_credentials().get("JUDGMENT_API_KEY", ""))
    if held is None:
        return
    deadline = time.time() + DEADLINE_SECONDS
    try:
        idle = 0
        while idle < 2 and time.time() < deadline:
            count = export_once(state)
            if count < 0:
                return
            if count == 0 and _waiting(state, _load_json(state / "index.json")):
                time.sleep(0.2)
                continue
            idle = idle + 1 if count == 0 else 0
            time.sleep(0.05 if count == 0 else 0)
    finally:
        os.close(held)


if __name__ == "__main__":
    main()
