"""Hermes plugin callback latency, in-process, the way Hermes calls the callbacks.

    python3 tests/bench_hermes.py [ROUNDS] [--json]

Cases (every fake credential is built at runtime; the env is the sandbox's, never this process's):

  short        a tool turn with ordinary short text, every hook, sender off
  tool4k       post_tool_call with a 4 KB tool output
  dense56k     post_api_request, post_tool_call, and turn end (post_llm_call + on_session_end) with a
               56 KB payload holding 1,000 fake credentials
  concurrent8  8 threads, each running its own sessions of 4 KB tool turns at once
  control8     the same 8 threads calling a callback that does nothing: the GIL's share, not the plugin's
  paced8       8 threads, 0-2 ms apart, mixing 4 KB and 56 KB secret-dense turns
  stalled      the helper stopped (SIGSTOP): its pipe fills, then the queue; callbacks must not notice
  saturated    56 KB turns back to back into a 4 MB queue, faster than the helper redacts: drops, never waits
  turnend      on_session_end with the real sender launch (sink down, so each sender exits quickly)
  handoff      not a callback: ms from on_session_end returning to its records being in the helper's pipe

Prints p50 / p95 / max in ms for each case and hook, and the queue's drop counts.
"""

from __future__ import annotations

import json
import os
import random
import signal
import statistics
import string
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hermes_support import Turn, later, load_plugin, new_session_id, plugin_env  # noqa: E402
from hermes_support import Sandbox  # noqa: E402

rng = random.Random(4242)


def tok(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n - 1)) + rng.choice(string.digits)


def fake_credential(i: int) -> str:
    kind = i % 8
    if kind == 0:
        return "gh" + "p_" + tok(36)
    if kind == 1:
        return "sk-" + "proj-" + tok(40)
    if kind == 2:
        return "AK" + "IA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    if kind == 3:
        return "xo" + "xb-" + tok(11, string.digits) + "-" + tok(20)
    if kind == 4:
        return "AI" + "za" + tok(35)
    if kind == 5:
        return "sk" + "_live_" + tok(24)
    if kind == 6:
        return "Authorization: Bearer " + tok(24)
    return "DB_PASS" + "WORD=" + tok(16)


def dense_payload(count: int = 1000, size: int = 56 * 1024) -> str:
    lines = [f"k{i} {fake_credential(i)}" for i in range(count)]
    text = "\n".join(lines)
    while len(text) < size:
        text += "\nfiller " + tok(40)
    return text[:size].rsplit("\n", 1)[0]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class Timer:
    def __init__(self):
        self.times: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def fire(self, ctx, plugin, events, label: str | None = None, only: set | None = None):
        for hook, kwargs, at in events:
            kwargs.setdefault("telemetry_schema_version", "hermes.observer.v1")
            (callback,) = ctx.hooks[hook]
            start = time.perf_counter()
            callback(**kwargs)
            took = (time.perf_counter() - start) * 1000
            if only is None or hook in only:
                with self.lock:
                    self.times.setdefault(label or hook, []).append(took)


def tool_turn(sid: str, n: int, result: str, answer: str, api_text: str | None = None) -> Turn:
    turn = Turn(sid, f"prompt {n} with some ordinary text", first=n == 0)
    call = turn.api(tool=("read_file", {"path": "notes.txt"}))
    turn.tool(call, "read_file", {"path": "notes.txt"}, result)
    turn.api(text=api_text if api_text is not None else answer)
    turn.end(answer)
    return turn


def run(rounds: int) -> dict[str, dict[str, tuple]]:
    report: dict[str, dict[str, tuple]] = {}
    sb = Sandbox()
    try:
        plain4k = ("line of ordinary tool output text\n" * 128)[:4096]
        dense = dense_payload()

        def summarize(case: str, timer: Timer) -> None:
            report[case] = {hook: (statistics.median(v), pct(v, 0.95), max(v), len(v)) for hook, v in sorted(timer.times.items())}
            everything = [x for v in timer.times.values() for x in v]
            report[case]["ALL"] = (statistics.median(everything), pct(everything, 0.95), max(everything), len(everything))

        with plugin_env(sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            # Warm up: compile everything once, as plugin load and the first turn would.
            sid = new_session_id()
            Timer().fire(ctx, plugin, tool_turn(sid, 0, "warm", "warm").events)

            timer = Timer()
            for n in range(rounds):
                sid = new_session_id()
                turn = tool_turn(sid, 0, '{"content": "1|hello"}', f"answer {n}")
                timer.fire(ctx, plugin, turn.events + [("on_session_finalize", {"session_id": sid, "platform": "cli"}, later(turn.events, 0.1))])
            summarize("short", timer)

            timer = Timer()
            for n in range(rounds):
                sid = new_session_id()
                turn = tool_turn(sid, 0, plain4k, f"answer {n}")
                timer.fire(ctx, plugin, turn.events, only={"post_tool_call"})
            summarize("tool4k", timer)

            timer = Timer()
            for n in range(max(rounds // 2, 50)):
                sid = new_session_id()
                turn = tool_turn(sid, 0, dense, dense, api_text=dense)
                timer.fire(ctx, plugin, turn.events, only={"post_tool_call", "post_api_request", "post_llm_call", "on_session_end"})
            summarize("dense56k", timer)

            timer = Timer()

            def worker(w: int) -> None:
                for n in range(max(rounds // 4, 25)):
                    sid = new_session_id() + f"w{w}"
                    turn = tool_turn(sid, 0, plain4k, f"answer {w} {n}")
                    timer.fire(ctx, plugin, turn.events)

            threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            summarize("concurrent8", timer)

            # Control: the same 8 busy threads calling a callback that does nothing. What is left is
            # Python's GIL handing the threads round, not the plugin.
            timer = Timer()
            noop = {hook: [lambda **kw: None] for hook in ctx.hooks}

            class Noop:
                hooks = noop

            def control(w: int) -> None:
                for n in range(max(rounds // 4, 25)):
                    sid = new_session_id() + f"n{w}"
                    turn = tool_turn(sid, 0, plain4k, f"answer {w} {n}")
                    timer.fire(Noop, plugin, turn.events, label="noop")

            threads = [threading.Thread(target=control, args=(w,)) for w in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            summarize("control8", timer)

            # Paced: 8 threads, each waiting 0-2 ms between hooks, as agents waiting on a model or tool do.
            timer = Timer()

            def paced(w: int) -> None:
                local = random.Random(w)
                for n in range(max(rounds // 4, 25)):
                    sid = new_session_id() + f"p{w}"
                    turn = tool_turn(sid, 0, plain4k if n % 4 else dense, f"answer {w} {n}")
                    for event in turn.events:
                        timer.fire(ctx, plugin, [event])
                        time.sleep(local.random() * 0.002)

            threads = [threading.Thread(target=paced, args=(w,)) for w in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            summarize("paced8", timer)
            errors = plugin.errors["count"]

        spooler = sys.modules[load_plugin().__name__ + ".spooler"]
        with plugin_env(sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            Timer().fire(ctx, plugin, tool_turn(new_session_id(), 0, "warm", "warm").events)
            deadline = time.time() + 10
            while (plugin.spool.proc is None or not plugin.spool.handed_over()) and time.time() < deadline:
                time.sleep(0.005)
            helper = plugin.spool.proc
            os.kill(helper.pid, signal.SIGSTOP)
            try:
                timer = Timer()
                for n in range(max(rounds // 4, 100)):
                    turn = tool_turn(new_session_id(), 0, dense, dense, api_text=dense)
                    timer.fire(ctx, plugin, turn.events)
                summarize("stalled", timer)
                report["stalled"]["_queue"] = (plugin.spool.bytes / 1e6, sum(plugin.spool.dropped.values()), 0, 0)
                start = time.perf_counter()
                plugin.spool.close()
                report["stalled"]["_exit_wait"] = ((time.perf_counter() - start) * 1000, 0, 0, 0)
            finally:
                os.kill(helper.pid, signal.SIGCONT)
            errors += plugin.errors["count"]

        old = spooler.QUEUE_BYTES
        spooler.QUEUE_BYTES = 4 * 1024 * 1024
        try:
            with plugin_env(sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
                timer = Timer()
                for n in range(max(rounds // 2, 100)):
                    turn = tool_turn(new_session_id(), 0, dense, dense, api_text=dense)
                    timer.fire(ctx, plugin, turn.events)
                summarize("saturated", timer)
                report["saturated"]["_queue"] = (plugin.spool.bytes / 1e6, sum(plugin.spool.dropped.values()), 0, 0)
                errors += plugin.errors["count"]
        finally:
            spooler.QUEUE_BYTES = old

        with plugin_env(sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            Timer().fire(ctx, plugin, tool_turn(new_session_id(), 0, "warm", "warm").events)
            handoff = []
            for n in range(max(rounds // 4, 100)):
                turn = tool_turn(new_session_id(), 0, plain4k if n % 2 else dense, f"answer {n}")
                Timer().fire(ctx, plugin, turn.events)
                start = time.perf_counter()
                while not plugin.spool.handed_over():
                    time.sleep(0.0001)
                handoff.append((time.perf_counter() - start) * 1000)
                time.sleep(0.005)
            report["handoff"] = {"ALL": (statistics.median(handoff), pct(handoff, 0.95), max(handoff), len(handoff))}

        with plugin_env(sb, JUDGMENT_FLUSH=None, JUDGMENT_API_URL="http://127.0.0.1:9", JUDGMENT_FLUSH_GRACE_MS="0") as (plugin, ctx):
            timer = Timer()
            for n in range(max(rounds // 4, 40)):
                sid = new_session_id()
                turn = tool_turn(sid, 0, '{"content": "1|hello"}', f"answer {n}")
                timer.fire(ctx, plugin, turn.events, label=None, only={"on_session_end"})
                time.sleep(0.02)  # turns are not back to back; let the sender start between them
            summarize("turnend", timer)
            errors += plugin.errors["count"]
            plugin.spool.drain(60)
            for proc in list(plugin.core._senders):
                try:
                    proc.wait(timeout=30)
                except Exception:
                    pass
        report["_errors"] = {"count": (errors, 0, 0, 0)}
    finally:
        sb.cleanup()
    return report


def main() -> None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    rounds = int(args[0]) if args else 400
    report = run(rounds)
    if "--json" in sys.argv:
        print(json.dumps(report))
        return
    print(f"{'case':12s} {'hook':22s} {'n':>6s} {'p50':>8s} {'p95':>8s} {'max':>8s}  (ms)")
    for case, hooks in report.items():
        for hook, (p50, p95, mx, n) in hooks.items():
            if hook == "_queue":
                print(f"{case:12s} {'queue at end':22s} {p50:.1f} MB held, {p95} records dropped whole")
            elif hook == "_exit_wait":
                print(f"{case:12s} {'exit wait':22s} {p50:.1f} ms (bound {1000 * 0.5:.0f} ms)")
            else:
                print(f"{case:12s} {hook:22s} {n:6d} {p50:8.3f} {p95:8.3f} {mx:8.3f}")


if __name__ == "__main__":
    main()
