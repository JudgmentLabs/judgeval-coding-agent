"""Route hook events into turns and send the turns that are ready. Runs detached from the agent.

Hooks drop one file per event in ``inbox/``. One flusher at a time folds new
events through ``assemble.Fold`` in the order they happened, appends each to
its turn's file under ``turns/``, and sends every turn the fold marked ready.
Persistent state is the fold's small routing table (``fold.json``) and one
record per turn (``index.json``): its file, whether it is waiting to be
sent, the last ``judgment.update_id`` sent, and a short summary for later
turns' history.

Each turn's events pass through plain stages: inbox -> fold -> turn file -> enrich (transcript
text) -> capture policy (metadata strips text; sanitized redacts it, and blanks the whole turn
when a tool call touched a sensitive file) -> spans -> post. The turn file and the index keep
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
from pathlib import Path

from judgment_core import redact
from judgment_core.assemble import Fold, turn_key
from judgment_core.claude_transcript import enrich_claude_events
from judgment_core.codex_transcript import enrich_codex_events
from judgment_core.cursor_transcript import enrich_cursor_events
from judgment_core.config import DEFAULT_API_URL, load_credentials, traces_url
from judgment_core.optin import load_project, safe_id, state_dir, tracing_forced_off
from judgment_core.spans import build_request, turn_summary

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
    """Fold new inbox events into turns. Return the turn index.

    A turn is blanked here, as the event that makes it sensitive is stored: before any wait, history, or send. See
    ``_store``.
    """
    legacy = _Redactor(env)
    index = _load_json(state / "index.json")
    fold = Fold(_load_json(state / "fold.json"))
    fold.sent = {key for key, item in index.items() if item.get("update_id") is not None}
    fold.unsent_keys = {key for key, item in index.items() if item.get("update_id") is None}
    consumed = []
    # The last inbox file folded in, saved with the fold. A crash before its files were deleted replays nothing.
    mark = fold.s["inbox"].get("last")
    for path, event, order in _inbox(state):
        if mark and order <= mark:
            consumed.append(path)
            continue
        fold.s["inbox"]["last"] = order
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
            _store(state, index, fold, item, legacy)
        for key in ready:
            if key in index and not index[key].get("ready"):
                index[key]["ready"] = True
                index[key]["ready_at"] = int(time.time() * 1000)
        for child in set(fold.s["child"]) - children:
            _adopt(state, index, fold, child, legacy)
        consumed.append(path)
    if consumed:
        _private_write(state / "fold.json", json.dumps(fold.state(int(time.time() * 1000), set(index))))
        _private_write(state / "index.json", json.dumps(index))
        for path in consumed:
            path.unlink(missing_ok=True)
    return index


def _store(state: Path, index: dict, fold: Fold, event: dict, redactor: "_Redactor") -> None:
    """Append one event to its turn's file.

    A turn's sensitivity is sticky and set as soon as one of its events shows it: the turn is marked blanked, its
    summary cleared, and its file rewritten without text, before any wait or history. A later event of a blanked turn,
    such as a child's that arrives after the turn was sent, is stored without text.
    """
    key = turn_key(event)
    path = _turn_file(state, key)
    item = index.get(key)
    blank = None if (item and item.get("blanked")) or event.get("capture") in ("full", "metadata") else _sensitive(event, redactor)
    if (item and item.get("blanked")) or blank:
        event = redact.strip_text(event)
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
    if blank:
        _blank(state, index, key, blank)


def _sensitive(event: dict, redactor: "_Redactor") -> dict | None:
    """Why this event blanks its turn: the hook's mark, or the sender's own check of its tool input."""
    return redact.first_sensitive([event], redactor.rules([event]), redactor.secrets, redactor.home)


def _blank(state: Path, index: dict, key: str, blank: dict) -> None:
    """Mark a turn blanked, for good: no summary, and a turn file without text. Safe to repeat."""
    item = index[key]
    if not item.get("blanked"):
        item["blanked"] = blank
    item["summary"] = None
    path = _turn_file(state, key)
    events = _read_jsonl(path)
    stripped = [redact.strip_text(event) for event in events]
    if stripped != events:
        _private_write(path, "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in stripped))


def _adopt(state: Path, index: dict, fold: Fold, child: str, redactor: "_Redactor") -> None:
    """A Cursor child conversation stored before its Task bound it moves onto the parent turn."""
    parent_session, parent_turn = fold.s["child"][child][0], fold.s["child"][child][1]
    prefix = f"cursor\t{child}\t"
    moved = False
    for key in [key for key in index if key.startswith(prefix)]:
        blank = index[key].get("blanked")
        for event in _read_jsonl(_turn_file(state, key)):
            event.update(session_id=parent_session, turn_id=parent_turn)
            event["agent_id"] = event.get("agent_id") or child
            _store(state, index, fold, event, redactor)
            moved = True
        parent_key = f"cursor\t{parent_session}\t{parent_turn}"
        if blank and parent_key in index:
            # A child conversation blanked before its Task bound it blanks the parent it joins.
            _blank(state, index, parent_key, blank)
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
    def home(self) -> str:
        return str(self.env.get("HOME") or "")

    def rules(self, events: list[dict]) -> redact.Ruleset:
        root = next((str(event.get("project_root")) for event in events if event.get("project_root")), "")
        if root not in self._rules:
            self._rules[root] = redact.ruleset(load_project(Path(root) / ".judgment" / "tracing.json") if root else {})
        return self._rules[root]

    def redact(self, event: dict) -> dict:
        return redact.redact_event(event, self.rules([event]), self.secrets)

    def text(self, value: str, rs: redact.Ruleset, field: str) -> tuple[str, list[dict]]:
        return redact.redact_text(value, rs, self.secrets, field)


def _sanitize(state: Path, key: str, item: dict, events: list[dict], prepared: list[dict], redactor: _Redactor) -> tuple[list[dict], dict | None]:
    """Find whether the enriched turn touched a sensitive file, then redact it. A blanked turn stays blanked.

    Detection runs first, on the text as read: a placeholder can hide what made a path sensitive
    (.env.<token> becomes .env.[REDACTED:...]). The turn file is rewritten when this changes what
    it holds, so the redacted or blanked version is all that stays on disk.
    """
    rs = redactor.rules(events)
    blank = item.get("blanked") or redact.first_sensitive(prepared, rs, redactor.secrets, redactor.home)
    prepared = [redact.redact_event(event, rs, redactor.secrets) for event in carry_records(events, prepared)]
    stored = [redact.redact_event(event, rs, redactor.secrets) for event in events]
    if blank:
        item["blanked"] = blank
        item["summary"] = None
        stored = [redact.strip_text(event) for event in stored]
        # Spans are built from the whole turn and lose their text there. A free-form name would be a span name.
        prepared = [redact.unblank(event) for event in prepared]
    if stored != events:
        _private_write(_turn_file(state, key), "".join(json.dumps(event, ensure_ascii=False, default=str) + "\n" for event in stored))
    return prepared, blank


def carry_records(stored: list[dict], prepared: list[dict]) -> list[dict]:
    """The enriched events, each with the records its stored event carried for the fields enrichment left as they were.

    Enrichment can replace a field with transcript text (a tool's output) or build new events. A record describes the
    text it was counted in, so a replaced field's records go; the sender counts what it replaces in the new text.
    """
    by_id = {event["_id"]: event for event in stored if event.get("_id")}
    seen: set[str] = set()
    out = []
    for event in prepared:
        base = by_id.get(event.get("_id")) if event.get("_id") not in seen else None
        if event.get("_id"):
            seen.add(event["_id"])
        carried = [item for item in redact.census(base) if event.get(item["field"]) == base.get(item["field"])] if base else []
        event = {key: value for key, value in event.items() if key != "redactions"}
        out.append({**event, "redactions": carried} if carried else event)
    return out


def _redact_history(history: list[dict], rs: redact.Ruleset, redactor: _Redactor) -> tuple[list[dict], list[dict]]:
    """Earlier turns' prompts and answers redacted, and the records of what that replaced, as field history."""
    out, records = [], []
    for turn in history:
        prompt, found_prompt = redactor.text(str(turn.get("prompt") or ""), rs, "history")
        answer, found_answer = redactor.text(str(turn.get("answer") or ""), rs, "history")
        out.append({**turn, "prompt": prompt, "answer": answer})
        records += found_prompt + found_answer
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


def _requests(payload: dict) -> list[bytes]:
    """The turn as gzipped request bodies of at most REQUEST_BYTES each before compression.

    Spans keep their order, so the root, listed last, goes in the last request after every child.
    """
    resource = payload["resourceSpans"][0]
    scope = resource["scopeSpans"][0]
    batches: list[list[dict]] = [[]]
    size = 0
    for span in scope["spans"]:
        span_size = len(json.dumps(span))
        if batches[-1] and size + span_size > REQUEST_BYTES:
            batches.append([])
            size = 0
        batches[-1].append(span)
        size += span_size
    bodies = []
    for spans in batches:
        body = {"resourceSpans": [{"resource": resource["resource"], "scopeSpans": [{"scope": scope["scope"], "spans": spans}]}]}
        bodies.append(gzip.compress(json.dumps(body).encode("utf-8"), compresslevel=5))
    return bodies


def _history(index: dict, key: str, dest: str) -> list[dict]:
    """Earlier turns of the same session sent to the same project, oldest first."""
    harness, session, _turn = key.split("\t")
    prefix = f"{harness}\t{session}\t"
    start = index[key].get("start") or index[key].get("first") or 0
    # A turn with events not yet sent (ready again: a late child, a failed resend) is left out: those events can still
    # blank it.
    earlier = [
        item["summary"]
        for other, item in index.items()
        if other != key and other.startswith(prefix) and item.get("summary") and not item.get("blanked") and not item.get("ready") and item.get("dest") == dest
    ]
    return sorted((turn for turn in earlier if int(turn.get("start") or 0) < start), key=lambda turn: int(turn.get("start") or 0))


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
            blank = None
            redaction = None
            rs = redactor.rules(events)
            if capture == "metadata":
                # Transcript enrichment has already run, so this is what gets sent and kept.
                prepared = [redact.metadata_event(event, rs, redactor.secrets) for event in carry_records(events, prepared)]
            elif capture == "sanitized":
                prepared, blank = _sanitize(state, key, item, events, prepared, redactor)
            # A blanked or metadata turn is no later turn's context. Nor is one whose rules failed.
            quiet = capture == "metadata" or bool(blank) or (capture != "full" and bool(rs.error))
            if quiet and item.get("summary") is not None:
                # Settled before any send, so a failed post cannot leave the old text as context.
                item["summary"] = None
                _private_write(state / "index.json", json.dumps(index))
            elif blank:
                _private_write(state / "index.json", json.dumps(index))
            summary = turn_summary(prepared)
            if summary:
                item["start"] = summary["start"]
            update_id = None if item.get("update_id") is None else int(item["update_id"]) + 1
            # Earlier turns may hold text from before the project switched to metadata.
            assert creds is not None
            dest = f"{creds['JUDGMENT_API_URL']} {creds['JUDGMENT_ORG_ID']} {creds['JUDGMENT_PROJECT_ID']}"
            # Rules that failed drop every text in the request, the earlier turns' included.
            history = [] if quiet else _history(index, key, dest)
            if capture != "full":
                # Earlier turns may have been captured in full, or stored by an older build. What is replaced in them now
                # is this request's redaction, so it joins the root's audit as field history.
                history, extra = _redact_history(history, rs, redactor)
                redaction = redact.turn_redaction(prepared, blank, extra)
            payload = build_request(prepared, update_id=update_id, history=history, redaction=redaction)
        except (OSError, ValueError) as exc:
            # A ValueError's message can quote the transcript value it choked on.
            _log(state, f"skip a turn: {type(exc).__name__}")
            continue
        if payload is None:
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
        status = 200
        for body in _requests(payload):
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
            if status != 200:
                break
        if status in (400, 413):
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
        item["summary"] = None if quiet else summary
        item["dest"] = dest
        _private_write(state / "index.json", json.dumps(index))
        sent += 1
    return sent


def _drop(state: Path, index: dict, key: str) -> None:
    _turn_file(state, key).unlink(missing_ok=True)
    index.pop(key, None)
    _private_write(state / "index.json", json.dumps(index))


def _prune(state: Path, index: dict) -> None:
    cutoff = (time.time() - KEEP_SECONDS) * 1000
    old = [key for key, item in index.items() if not item.get("ready") and int(item.get("last") or 0) < cutoff]
    for key in old:
        _turn_file(state, key).unlink(missing_ok=True)
        index.pop(key, None)
    if old:
        _private_write(state / "index.json", json.dumps(index))


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
