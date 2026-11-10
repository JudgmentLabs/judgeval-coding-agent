"""Real Hermes Agent, end to end. Not part of ``unittest discover`` (no test_ prefix) and not run in CI.

    python3 tests/e2e_hermes.py [-v]

Needs an installed Hermes: HERMES_BIN, else /workspace/hermes-env/.venv/bin/hermes. Skipped without one.
Each run gets a temp HOME and HERMES_HOME, installs this repo's hermes/ as the user plugin
``judgment-tracing``, enables it with ``hermes plugins enable``, points Hermes at a local mock
OpenAI-compatible server (tests/mock_openai.py), and points the plugin at a local fake Judgment
sink. Hermes runs with ``hermes chat -q`` (one-shot) and only PATH, HOME, HERMES_HOME, LANG and
the dummy Judgment settings in its env, so no real key reaches it.

What is mocked: the model provider (scripted replies, fixed token counts) and the Judgment API.
What is real: Hermes 0.19.x, its plugin loader, hook firing, read_file and delegate_task tools,
the plugin, the vendored core, and the detached sender.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_sink import Sink, attrs, roots  # noqa: E402
from mock_openai import MockLLM  # noqa: E402
from support import ENV_SECRET_NAME, REPO, SECRETS, leaks, wait_for  # noqa: E402

HERMES = Path(os.environ.get("HERMES_BIN", "/workspace/hermes-env/.venv/bin/hermes"))
PLUGIN = "judgment-tracing"
GITHUB = SECRETS["github_token"]["core"]
OPENAI = SECRETS["openai_key"]["core"]
ENV_VALUE = SECRETS["env_value"]["core"]
DOTENV = "DATABASE_URL=postgres://app:" + "Lq" + "9xT" * 4 + "@db/app\nSTRIPE=" + SECRETS["stripe_key"]["core"] + "\n"
REPORT: dict = {}


def last_user(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def has_tool_result(body: dict) -> bool:
    return any(message.get("role") == "tool" for message in body.get("messages") or [])


class Home:
    """A temp HOME with HERMES_HOME, the plugin installed and enabled, and a project."""

    def __init__(self, llm_url: str, sink_url: str, opt_in: bool = True):
        self.dir = Path(tempfile.mkdtemp(prefix="jt-hermes-e2e-"))
        self.home = self.dir / "home"
        self.hermes_home = self.home / ".hermes"
        self.state = self.dir / "state"
        self.project = self.home / "proj"
        self.project.mkdir(parents=True)
        (self.hermes_home / "plugins").mkdir(parents=True)
        shutil.copytree(REPO / "hermes", self.hermes_home / "plugins" / PLUGIN, ignore=shutil.ignore_patterns("__pycache__"))
        if opt_in:
            (self.project / ".judgment").mkdir()
            (self.project / ".judgment" / "tracing.json").write_text('{"enabled": true}\n')
        (self.project / "notes.txt").write_text(f"deploy notes\ntoken: {GITHUB}\nservice: {ENV_VALUE}\nbuild 4f1e2d3c9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d\n")
        (self.project / ".env").write_text(DOTENV)
        (self.hermes_home / "config.yaml").write_text(
            f"model:\n  default: mock-model-1\n  provider: custom\n  base_url: {llm_url}\n  api_key: dummy-local\n"
        )
        self.stats = self.dir / "stats.jsonl"
        self.env = {
            "PATH": f"{HERMES.parent}:/usr/bin:/bin",
            "HOME": str(self.home),
            "HERMES_HOME": str(self.hermes_home),
            "LANG": "C.UTF-8",
            "JUDGMENT_API_URL": sink_url,
            "JUDGMENT_API_KEY": SECRETS["judgment_key"]["core"],
            "JUDGMENT_PROJECT_ID": "proj-hermes-e2e",
            "JUDGMENT_STATE_DIR": str(self.state),
            "JUDGMENT_HERMES_STATS": str(self.stats),
            ENV_SECRET_NAME: ENV_VALUE,
        }

    def hermes(self, *args: str, timeout: int = 240) -> subprocess.CompletedProcess:
        return subprocess.run([str(HERMES), *args], cwd=self.project, env=self.env, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)

    def enable(self) -> None:
        result = self.hermes("plugins", "enable", PLUGIN, "--no-allow-tool-override", timeout=120)
        assert result.returncode == 0, result.stdout[-500:] + result.stderr[-500:]

    def ask(self, prompt: str) -> tuple[subprocess.CompletedProcess, float]:
        start = time.time()
        result = self.hermes("chat", "-q", prompt, "-Q", "--yolo")
        return result, time.time()

    def state_text(self) -> str:
        if not self.state.exists():
            return ""
        return "\n".join(path.read_text(errors="replace") for path in sorted(self.state.rglob("*")) if path.is_file())

    def timings(self) -> dict[str, list[int]]:
        found: dict[str, list[int]] = {}
        if self.stats.exists():
            for line in self.stats.read_text().splitlines():
                for hook, values in json.loads(line)["ns"].items():
                    found.setdefault(hook, []).extend(values)
        return found

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def pct(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] / 1e6


@unittest.skipUnless(HERMES.exists(), f"Hermes is not installed at {HERMES}")
class RealHermes(unittest.TestCase):
    def wait_spans(self, sink: Sink, count: int, timeout: float = 30) -> bool:
        return wait_for(lambda: len(roots(sink.spans())) >= count, timeout=timeout, step=0.2)

    def test_1_tool_turn_with_secrets_and_a_retry(self):
        notes = None

        def reply(body, n):
            if has_tool_result(body):
                # The first answer attempt fails once with a 500, so Hermes retries.
                if not getattr(reply, "failed", False):
                    reply.failed = True
                    return {"status": 500, "message": "upstream exploded, key was " + OPENAI}
                return {"text": f"The notes hold a token {GITHUB} and build 4f1e2d3c9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d."}
            return {"tool": "read_file", "args": {"path": notes}}

        with Sink() as sink, MockLLM(reply) as llm:
            home = Home(llm.url, sink.url)
            notes = str(home.project / "notes.txt")
            try:
                home.enable()
                result, exited = home.ask(f"Read notes.txt. My key is {OPENAI}, do not share it.")
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                landed = self.wait_spans(sink, 1, timeout=30)
                landed_at = time.time()
                self.assertTrue(landed, "no turn reached the sink within 30 s of Hermes exiting")
                spans = sink.spans()
                root = roots(spans)[0]
                ra = attrs(root)
                self.assertEqual(root["name"], "hermes turn")
                self.assertEqual(ra["judgment.span_kind"], "agent")
                sid = ra["judgment.session_id"]
                self.assertTrue(all(attrs(s).get("judgment.session_id") == sid and attrs(s).get("gen_ai.conversation.id") == sid for s in spans))
                kinds = sorted(attrs(s)["judgment.span_kind"] for s in spans)
                llms = [s for s in spans if attrs(s)["judgment.span_kind"] == "llm"]
                tools = [s for s in spans if attrs(s)["judgment.span_kind"] == "tool"]
                errored = [s for s in llms if s.get("status", {}).get("code") == 2]
                self.assertEqual([attrs(s)["gen_ai.tool.name"] for s in tools], ["read_file"])
                self.assertGreaterEqual(len(llms), 3, kinds)  # tool request, failed attempt, answer
                self.assertGreaterEqual(len(errored), 1)
                ok_llms = [s for s in llms if s not in errored]
                self.assertTrue(all(attrs(s)["judgment.llm.model"] == "mock-model-1" for s in llms))
                self.assertTrue(all(int(attrs(s).get("judgment.usage.non_cached_input_tokens", 0)) > 1000 for s in ok_llms))
                self.assertTrue(all(s["parentSpanId"] == root["spanId"] for s in spans if s is not root))
                body = sink.text()
                self.assertEqual(leaks(body, hashes=False), [], "a fake secret reached the sink")
                self.assertEqual(leaks(home.state_text(), hashes=False), [], "a fake secret stayed on disk")
                self.assertIn("[REDACTED:", body)
                self.assertIn("4f1e2d3c9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d", body)  # a git SHA is not a secret
                self.assertTrue(ra.get("judgment.redacted"))
                REPORT["run1"] = {"spans": len(spans), "kinds": kinds, "llm_errors": len(errored), "redaction_count": ra.get("judgment.redaction_count"),
                                  "landed_after_exit_s": round(landed_at - exited, 2), "hermes_rc": result.returncode, "llm_requests": len(llm.requests)}
                REPORT["timings"] = home.timings()
            finally:
                home.cleanup()

    def test_2_dotenv_turn_is_blanked(self):
        with Sink() as sink, MockLLM(lambda body, n: {"text": "It sets DATABASE_URL and STRIPE."} if has_tool_result(body) else {"tool": "read_file", "args": {"path": ".env"}}) as llm:
            home = Home(llm.url, sink.url)
            try:
                home.enable()
                result, exited = home.ask("What is in .env?")
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertTrue(self.wait_spans(sink, 1))
                spans = sink.spans()
                ra = attrs(roots(spans)[0])
                self.assertEqual(ra.get("judgment.redacted_reason"), "sensitive_path")
                self.assertEqual(ra.get("judgment.redacted_rule"), "dotenv")
                self.assertTrue(all(attrs(s).get("judgment.input", "") == "" and attrs(s).get("judgment.output", "") == "" for s in spans))
                self.assertEqual(leaks(sink.text(), hashes=False), [])
                self.assertEqual(leaks(home.state_text(), hashes=False), [])
                self.assertNotIn("Lq9xT", sink.text() + home.state_text())
                REPORT["run2"] = {"spans": len(spans), "blanked": True, "trigger": ra.get("judgment.redacted_trigger")}
            finally:
                home.cleanup()

    def test_3_background_subagent(self):
        def reply(body, n):
            if "count to three" in last_user(body) and "delegate" not in last_user(body):
                return {"text": "one two three"}
            if has_tool_result(body):
                return {"text": "Started a helper to count."}
            return {"tool": "delegate_task", "args": {"goal": "count to three", "context": "be brief"}}

        with Sink() as sink, MockLLM(reply) as llm:
            home = Home(llm.url, sink.url)
            try:
                home.enable()
                result, exited = home.ask("delegate counting to a helper")
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertTrue(self.wait_spans(sink, 1))
                time.sleep(3)  # a resend with the child, if the child finished first
                spans = sink.spans()
                latest = {}
                for span in spans:
                    latest[span["spanId"]] = span
                spans = list(latest.values())
                root = roots(spans)[0]
                agents = [s for s in spans if attrs(s)["judgment.span_kind"] == "agent" and s is not root]
                self.assertEqual(len(agents), 1, [s["name"] for s in spans])
                agent = agents[0]
                self.assertEqual(agent["parentSpanId"], root["spanId"])
                self.assertEqual(attrs(agent)["judgment.input"], "count to three")
                REPORT["run3"] = {"spans": len(spans), "subagent": agent["name"], "child_spans": sum(1 for s in spans if s.get("parentSpanId") == agent["spanId"]),
                                  "subagent_output": attrs(agent).get("judgment.output", "")[:40], "requests": len(sink.requests)}
            finally:
                home.cleanup()

    def test_4_opted_out_project_sends_nothing(self):
        with Sink() as sink, MockLLM([{"text": "hello"}]) as llm:
            home = Home(llm.url, sink.url, opt_in=False)
            try:
                home.enable()
                result, _ = home.ask("say hello")
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                time.sleep(4)
                self.assertEqual(sink.requests, [])
                inbox = home.state / "inbox"
                self.assertFalse(inbox.exists() and any(inbox.iterdir()))
                REPORT["run4"] = {"requests": len(sink.requests), "state_files": sum(1 for p in home.state.rglob("*") if p.is_file()) if home.state.exists() else 0}
            finally:
                home.cleanup()

    def test_5_plugin_installed_not_enabled_sends_nothing(self):
        with Sink() as sink, MockLLM([{"text": "hello"}]) as llm:
            home = Home(llm.url, sink.url)
            try:
                result, _ = home.ask("say hello")
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                time.sleep(4)
                self.assertEqual(sink.requests, [])
                self.assertFalse(home.state.exists())
                REPORT["run5"] = {"requests": len(sink.requests), "state_dir": home.state.exists()}
            finally:
                home.cleanup()

    def test_6_judgment_command_in_the_interactive_cli(self):
        """/judgment off, a message, /judgment on, a message: only the second is traced. Drives `hermes chat --cli` in a pty."""
        import pty
        import re
        import select

        def drain(fd: int, seconds: float) -> str:
            out = b""
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if ready:
                    try:
                        out += os.read(fd, 65536)
                    except OSError:
                        break
            return re.sub(r"\x1b\[[0-9;?]*[ a-zA-Z]|\x1b\][^\x07]*\x07|\r", "", out.decode(errors="replace"))

        with Sink() as sink, MockLLM(lambda body, n: {"text": f"reply {n}"}) as llm:
            home = Home(llm.url, sink.url)
            try:
                home.enable()
                pid, fd = pty.fork()
                if pid == 0:
                    os.chdir(home.project)
                    os.execve(str(HERMES), [str(HERMES), "chat", "--cli"], {**home.env, "TERM": "xterm", "COLUMNS": "120", "LINES": "40"})
                drain(fd, 15)
                said = {}
                for line, wait in (("/judgment off", 4), ("first message, not traced", 8), ("/judgment on", 4), ("second message, traced", 8)):
                    os.write(fd, line.encode() + b"\r")
                    said[line] = drain(fd, wait)
                os.write(fd, b"/exit\r")
                drain(fd, 5)
                try:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                os.close(fd)
                self.assertIn("off for this session", said["/judgment off"])
                self.assertIn("on for this session", said["/judgment on"])
                self.assertTrue(self.wait_spans(sink, 1))
                time.sleep(3)
                inputs = sorted({attrs(root)["judgment.input"] for root in roots(sink.spans())})
                self.assertEqual(inputs, ["second message, traced"])
                REPORT["run6"] = {"traced_turns": inputs, "command_replies": [said["/judgment off"].count("Judgment tracing"), said["/judgment on"].count("Judgment tracing")]}
            finally:
                home.cleanup()


    def test_7_judgment_down_hermes_exits_normally(self):
        with MockLLM(lambda body, n: {"text": "hello"}) as llm:
            timings = {}
            for label, url in (("up", None), ("down", "http://127.0.0.1:9")):
                with Sink() as sink:
                    home = Home(llm.url, url or sink.url)
                    try:
                        home.enable()
                        start = time.time()
                        result, exited = home.ask("say hello")
                        timings[label] = round(exited - start, 2)
                        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                        if label == "down":
                            log = home.state / "flush.log"
                            self.assertTrue(wait_for(lambda: log.exists() and "post attempt" in log.read_text(), timeout=30))
                            self.assertEqual(leaks(log.read_text(), hashes=False), [])
                    finally:
                        home.cleanup()
            self.assertLess(timings["down"], timings["up"] + 3, timings)
            REPORT["run7"] = {"hermes_wall_s": timings}



def print_report() -> None:
    timings = REPORT.pop("timings", {})
    print("\nreal Hermes e2e report")
    for key, value in REPORT.items():
        print(f"  {key}: {json.dumps(value)}")
    if timings:
        print("  in-process hook time (ms), run 1:")
        every = []
        for hook, values in sorted(timings.items()):
            every += values
            print(f"    {hook:22s} n={len(values):3d} p50={pct(values, 0.5):.3f} p95={pct(values, 0.95):.3f} max={max(values) / 1e6:.3f}")
        print(f"    {'all':22s} n={len(every):3d} p50={pct(every, 0.5):.3f} p95={pct(every, 0.95):.3f} mean={statistics.mean(every) / 1e6:.3f}")


if __name__ == "__main__":
    program = unittest.main(exit=False, verbosity=2)
    print_report()
    sys.exit(0 if program.result.wasSuccessful() else 1)
